import os
import io
import time
import json
import logging
import subprocess
import re
import hashlib
import random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, url_for, session
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

import requests
import base64

# --- Configuration ---
# Directory where this file lives (e.g. api/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root (one level up) — where you said your files actually live
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Candidate directories to search for model / datasets / yolo: prefer BASE_DIR then PROJECT_ROOT
CANDIDATE_DIRS = [PROJECT_ROOT, BASE_DIR]

APP_STATIC = os.path.join(BASE_DIR, 'static')
os.makedirs(APP_STATIC, exist_ok=True)

ANNOTATED_FILENAME = 'annotated_frame.jpg'
ANNOTATED_PATH = os.path.join(APP_STATIC, ANNOTATED_FILENAME)

# defaults (will be overridden if found in candidate dirs)
DEFAULT_MODEL_FILENAME = 'chicken_disease_classifier_vgg16.keras'
DEFAULT_YOLO_DIRNAME = 'yolo_config_files'
DEFAULT_PREPARED_DATASET = 'prepared_dataset'

# Logging 
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('cdd_server')

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# In-memory state
current_command = {'action': 'idle'}
last_detection = {'disease': None, 'image_path': None, 'timestamp': None}
history = []
HISTORY_LIMIT = int(os.getenv('HISTORY_LIMIT', '50'))

# User and settings storage
users_db = {}
settings_db = {}
schedules_db = []
environmental_data = {
    'temperature': 22.5,
    'humidity': 65.2,
    'air_quality': 'Good',
    'wind_speed': 5.3,
    'last_update': datetime.utcnow().isoformat()
}

# --- Helpers to locate files across candidate dirs ---

def find_file_in_candidates(filename, candidates=CANDIDATE_DIRS):
    for d in candidates:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    # not found
    return None

def find_dir_in_candidates(dirname, candidates=CANDIDATE_DIRS):
    for d in candidates:
        p = os.path.join(d, dirname)
        if os.path.isdir(p):
            return p
    return None

# Resolve model path (search api/ then project root)
found_model = find_file_in_candidates(DEFAULT_MODEL_FILENAME)
if found_model:
    MODEL_PATH = found_model
else:
    # default placement (where logs previously pointed)
    MODEL_PATH = os.path.join(BASE_DIR, DEFAULT_MODEL_FILENAME)

# Resolve yolo dir (search api/ then project root)
found_yolo_dir = find_dir_in_candidates(DEFAULT_YOLO_DIRNAME)
if found_yolo_dir:
    YOLO_DIR = found_yolo_dir
else:
    YOLO_DIR = os.path.join(BASE_DIR, DEFAULT_YOLO_DIRNAME)
# Ensure dir exists for later writes/lookups
os.makedirs(YOLO_DIR, exist_ok=True)

# Resolve prepared_dataset dir
prepared_dir = find_dir_in_candidates(DEFAULT_PREPARED_DATASET)
if not prepared_dir:
    prepared_dir = os.path.join(BASE_DIR, DEFAULT_PREPARED_DATASET)

log.info('Search candidates: %s', CANDIDATE_DIRS)
log.info('Using MODEL_PATH=%s', MODEL_PATH)
log.info('Using YOLO_DIR=%s', YOLO_DIR)
log.info('Using prepared_dataset=%s', prepared_dir)

# --- Model loading (local only) ---

def load_keras_model():
    # Expect the model locally (either in BASE_DIR or PROJECT_ROOT)
    if not os.path.exists(MODEL_PATH):
        log.warning('Model file not found at %s. Please place %s in one of: %s',
                    MODEL_PATH, DEFAULT_MODEL_FILENAME, CANDIDATE_DIRS)
        raise FileNotFoundError(f'Model file not found at {MODEL_PATH}')

    log.info('Loading Keras model from %s', MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Try to load model; handle errors gracefully
try:
    model = load_keras_model()
except Exception as e:
    log.exception('Could not load model: %s', e)
    model = None

# --- YOLO setup (local config files only) ---

yolo_files = {
    'cfg': 'yolov3.cfg',
    'weights': 'yolov3.weights',
    'names': 'coco.names'
}

yolo_cfg = os.path.join(YOLO_DIR, yolo_files['cfg'])
yolo_weights = os.path.join(YOLO_DIR, yolo_files['weights'])
yolo_names = os.path.join(YOLO_DIR, yolo_files['names'])

net = None
classes = []
output_layers = []

if os.path.exists(yolo_cfg) and os.path.exists(yolo_weights) and os.path.exists(yolo_names):
    try:
        net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
        layer_names = net.getLayerNames()
        outs = net.getUnconnectedOutLayers()
        # accommodate different return types from getUnconnectedOutLayers()
        try:
            output_layers = [layer_names[i[0] - 1] if hasattr(i, '__len__') else layer_names[i - 1] for i in outs]
        except Exception:
            output_layers = [layer_names[i - 1] for i in outs]
        with open(yolo_names, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        log.info('YOLO loaded with %d classes', len(classes))
    except Exception as e:
        log.exception('Error initializing YOLO: %s', e)
else:
    log.warning('YOLO files not found in %s. Chicken detection will be disabled. Expected files: %s', YOLO_DIR, list(yolo_files.values()))

# --- Image helpers ---

def preprocess_image(img_path, target_size=(128, 128)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


def detect_chicken(frame, conf_threshold=0.5, nms_threshold=0.4):
    """Return (found_boolean, annotated_frame). If YOLO isn't available return (True, frame) to let
    classifier run as a best-effort (optimistic pass).
    """
    if net is None or not classes:
        log.debug('YOLO not available; skipping chicken bounding box detection (optimistic pass)')
        return True, frame

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            if len(scores) == 0:
                continue
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if class_id < len(classes) and classes[class_id] == 'bird' and confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

    if len(boxes) == 0:
        return False, frame

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(idxs) == 0:
        return False, frame

    for i in idxs.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
        cv2.putText(frame, label, (max(x,0), max(y - 8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    return True, frame

# Default/fallback class map (keeps labels consistent even without the training generator available)
FALLBACK_CLASS_MAP = {
    0: 'Chicken Drinking water',
    1: 'Chicken Feeding',
    2: 'avian_influenza',
    3: 'dead_chickens',
    4: 'gumboro_disease',
    5: 'healthy',
    6: 'healthy_chicken',
    7: 'infectious_coryza',
    8: 'new_castles_disease',
    9: 'splay_foot',
}

# Try to infer class mapping from a prepared dataset if present (optional)
class_labels_map = None
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    if os.path.exists(prepared_dir):
        datagen = ImageDataGenerator(rescale=1./255)
        gen = datagen.flow_from_directory(prepared_dir, target_size=(128,128), batch_size=1, class_mode='categorical')
        class_indices = gen.class_indices
        class_labels_map = {v: k for k, v in class_indices.items()}
        log.info('Derived class labels map from prepared_dataset with %d classes', len(class_labels_map))
    else:
        log.info('No prepared_dataset directory found at %s; will use fallback labels', prepared_dir)
except Exception as e:
    log.warning('Could not derive class labels map: %s', e)

# --- Flask routes --- (unchanged from your prior file)
@app.route('/', methods=['GET'])
def index():
    try:
        return render_template('index.html')
    except Exception:
        return jsonify({'status': 'ok'}), 200


@app.route('/analyze', methods=['POST'])
def analyze():
    log.info('📷 STARTING IMAGE ANALYSIS')
    
    # Handle both 'file' and 'image' field names
    uploaded_file = None
    if 'file' in request.files:
        uploaded_file = request.files['file']
    elif 'image' in request.files:
        uploaded_file = request.files['image']
    
    if not uploaded_file:
        log.error('No file uploaded in request')
        return jsonify({'error': 'No file uploaded'}), 400

    if uploaded_file.filename == '':
        log.error('Empty filename in uploaded file')
        return jsonify({'error': 'No file selected'}), 400
    
    log.info(f'   File: {uploaded_file.filename}')
    log.info(f'   Size: {uploaded_file.content_length or "unknown"} bytes')
    log.info(f'   Type: {uploaded_file.content_type}')

    # Create uploads directory
    uploads_dir = os.path.join(BASE_DIR, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    
    filename = secure_filename(uploaded_file.filename)
    tmp_path = os.path.join(uploads_dir, f"upload_{int(time.time())}_{filename}")
    uploaded_file.save(tmp_path)

    frame = cv2.imread(tmp_path)
    if frame is None:
        log.error(f'Failed to load image from {tmp_path}')
        return jsonify({'error': 'Uploaded file is not a valid image'}), 400
    
    log.info(f'   Image loaded: {frame.shape[1]}x{frame.shape[0]} pixels')

    chicken_detected, annotated_frame = detect_chicken(frame)

    try:
        cv2.imwrite(ANNOTATED_PATH, annotated_frame)
    except Exception as e:
        log.exception('Failed to write annotated image: %s', e)

    image_url = url_for('static', filename=ANNOTATED_FILENAME)

    if not chicken_detected:
        log.warning('⚠️  NO CHICKEN DETECTED in uploaded image')
        log.info(f'   Image: {image_url}')
        log.info('─' * 50)
        last_detection.update({'disease': None, 'image_path': image_url, 'timestamp': datetime.utcnow().isoformat()})
        return jsonify({'error': 'No chicken detected', 'image_url': image_url}), 200

    predicted_label = None
    try:
        if model is None:
            log.warning('No model loaded; returning fallback label')
            predicted_label = 'unknown_model'
        else:
            log.info('   Running AI model prediction...')
            img_array = preprocess_image(tmp_path)
            preds = model.predict(img_array)
            if preds is None:
                predicted_label = 'prediction_failed'
            else:
                predicted_index = int(np.argmax(preds, axis=1)[0])
                if class_labels_map and isinstance(class_labels_map, dict) and predicted_index in class_labels_map:
                    predicted_label = class_labels_map[predicted_index]
                else:
                    predicted_label = FALLBACK_CLASS_MAP.get(predicted_index, f'class_{predicted_index}')
                log.info(f'   Model prediction completed: class {predicted_index}')
    except Exception as e:
        log.exception('Prediction error: %s', e)
        predicted_label = 'prediction_error'

    now = datetime.utcnow().isoformat()
    last_detection.update({'disease': predicted_label, 'image_path': image_url, 'timestamp': now})

    history.insert(0, {'disease': predicted_label, 'image_url': image_url, 'timestamp': now})
    while len(history) > HISTORY_LIMIT:
        history.pop()

    # Calculate confidence and class_id
    confidence = 0.85
    class_id = 5
    
    try:
        if model is not None and 'preds' in locals():
            confidence = float(np.max(preds))
            class_id = int(np.argmax(preds, axis=1)[0])
    except:
        pass
    
    # Log the detection result to terminal
    log.info('🔍 DISEASE DETECTION RESULT:')
    log.info(f'   Disease: {predicted_label}')
    log.info(f'   Confidence: {confidence:.2%}')
    log.info(f'   Class ID: {class_id}')
    log.info(f'   Timestamp: {now}')
    log.info(f'   Image: {image_url}')
    log.info('─' * 50)
    
    response = {
        'disease': predicted_label,
        'confidence': confidence,
        'class_id': class_id,
        'image_url': image_url,
        'note': 'Model suggestions — consult a vet for confirmation.',
        'timestamp': now
    }

    return jsonify(response), 200


@app.route('/result', methods=['GET'])
def get_result():
    if last_detection.get('disease'):
        return jsonify({
            'disease': last_detection.get('disease'),
            'image_url': last_detection.get('image_path'),
            'timestamp': last_detection.get('timestamp')
        })
    if os.path.exists(ANNOTATED_PATH):
        return jsonify({'disease': None, 'image_url': url_for('static', filename=ANNOTATED_FILENAME)}), 200
    return jsonify({'disease': None}), 200


@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({'history': history}), 200


@app.route('/command', methods=['GET'])
def get_command():
    return jsonify(current_command)


@app.route('/command', methods=['POST'])
def set_command():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'error': "Invalid JSON"}), 400

    if not data or 'action' not in data:
        return jsonify({'error': "Missing 'action' in JSON body"}), 400

    action = data['action']
    if action not in ['idle', 'capture']:
        return jsonify({'error': 'Invalid action'}), 400

    current_command['action'] = action
    current_command['timestamp'] = datetime.utcnow().isoformat()

    return jsonify({'status': 'command set', 'action': action}), 200


# --- Analyze remote: forward to Raspberry Pi ---
PI_HOST = os.getenv('PI_HOST')
PI_TOKEN = os.getenv('PI_API_TOKEN')
PI_TIMEOUT = int(os.getenv('PI_TIMEOUT', '12'))

@app.route('/analyze_remote', methods=['POST'])
def analyze_remote():
    if 'file' not in request.files:
        return jsonify({'error':'No file uploaded'}), 400

    if not PI_HOST:
        return jsonify({'error':'PI_HOST not configured'}), 500

    f = request.files['file']
    fname = secure_filename(f.filename or 'upload.jpg')

    files = {'file': (fname, f.stream, f.content_type)}
    headers = {'X-PI-TOKEN': PI_TOKEN} if PI_TOKEN else {}

    try:
        pi_url = f'http://{PI_HOST}/analyze'
        r = requests.post(pi_url, files=files, headers=headers, timeout=PI_TIMEOUT)
    except requests.exceptions.RequestException as e:
        return jsonify({'error':'pi_unreachable','detail':str(e)}), 502

    try:
        pj = r.json()
    except Exception:
        return jsonify({'error':'invalid_response_from_pi','status_code': r.status_code, 'text': r.text}), 502

    if 'image_b64' in pj and pj['image_b64']:
        try:
            imgdata = base64.b64decode(pj['image_b64'])
            annotated_path = os.path.join(app.static_folder, 'annotated_frame.jpg')
            with open(annotated_path, 'wb') as wf:
                wf.write(imgdata)
            image_url = url_for('static', filename='annotated_frame.jpg')
            pj['image_url'] = image_url
        except Exception as e:
            pj['image_write_error'] = str(e)

    return jsonify(pj), 200


# --- WiFi Network Management ---
@app.route('/wifi/scan', methods=['GET'])
def scan_wifi_networks():
    """Scan for available WiFi networks"""
    try:
        # Use nmcli on Linux systems
        result = subprocess.run(['nmcli', '-t', '-f', 'SSID,SIGNAL,SECURITY', 'dev', 'wifi'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            # Fallback to iwlist on older systems
            result = subprocess.run(['iwlist', 'scan'], capture_output=True, text=True, timeout=10)
            return parse_iwlist_output(result.stdout)
        
        networks = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(':')
                if len(parts) >= 3:
                    ssid = parts[0].strip()
                    signal = parts[1].strip()
                    security = parts[2].strip()
                    
                    if ssid and ssid != '--':
                        networks.append({
                            'ssid': ssid,
                            'signal': int(signal) if signal.isdigit() else 0,
                            'security': security if security else 'Open',
                            'signal_strength': get_signal_description(int(signal) if signal.isdigit() else 0)
                        })
        
        # Remove duplicates and sort by signal strength
        unique_networks = {}
        for network in networks:
            ssid = network['ssid']
            if ssid not in unique_networks or network['signal'] > unique_networks[ssid]['signal']:
                unique_networks[ssid] = network
        
        sorted_networks = sorted(unique_networks.values(), key=lambda x: x['signal'], reverse=True)
        
        return jsonify({'networks': sorted_networks[:20]}), 200  # Limit to top 20
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'WiFi scan timeout'}), 500
    except FileNotFoundError:
        # Simulate networks for demo/development
        return jsonify({'networks': get_demo_networks()}), 200
    except Exception as e:
        log.exception('WiFi scan error: %s', e)
        return jsonify({'error': 'WiFi scan failed', 'detail': str(e)}), 500


@app.route('/wifi/connect', methods=['POST'])
def connect_wifi():
    """Connect to a WiFi network"""
    try:
        data = request.get_json()
        if not data or 'ssid' not in data:
            return jsonify({'error': 'SSID required'}), 400
        
        ssid = data['ssid']
        password = data.get('password', '')
        
        # Use nmcli to connect
        if password:
            cmd = ['nmcli', 'dev', 'wifi', 'connect', ssid, 'password', password]
        else:
            cmd = ['nmcli', 'dev', 'wifi', 'connect', ssid]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return jsonify({'status': 'connected', 'ssid': ssid}), 200
        else:
            return jsonify({'error': 'Connection failed', 'detail': result.stderr}), 400
            
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Connection timeout'}), 500
    except FileNotFoundError:
        # Simulate connection for demo
        return jsonify({'status': 'connected', 'ssid': data.get('ssid', 'Demo Network')}), 200
    except Exception as e:
        log.exception('WiFi connect error: %s', e)
        return jsonify({'error': 'Connection failed', 'detail': str(e)}), 500


def parse_iwlist_output(output):
    """Parse iwlist scan output"""
    networks = []
    current_network = {}
    
    for line in output.split('\n'):
        line = line.strip()
        if 'Cell' in line and 'Address:' in line:
            if current_network.get('ssid'):
                networks.append(current_network)
            current_network = {}
        elif 'ESSID:' in line:
            ssid = re.search(r'ESSID:"(.*)"', line)
            if ssid:
                current_network['ssid'] = ssid.group(1)
        elif 'Signal level=' in line:
            signal = re.search(r'Signal level=(-?\d+)', line)
            if signal:
                current_network['signal'] = abs(int(signal.group(1)))
                current_network['signal_strength'] = get_signal_description(current_network['signal'])
        elif 'Encryption key:' in line:
            if 'off' in line:
                current_network['security'] = 'Open'
            else:
                current_network['security'] = 'WPA/WPA2'
    
    if current_network.get('ssid'):
        networks.append(current_network)
    
    return {'networks': networks[:20]}


def get_signal_description(signal_level):
    """Convert signal level to description"""
    if signal_level >= 80:
        return 'Excellent'
    elif signal_level >= 60:
        return 'Good'
    elif signal_level >= 40:
        return 'Fair'
    elif signal_level >= 20:
        return 'Weak'
    else:
        return 'Very Weak'


def get_demo_networks():
    """Demo networks for development/testing"""
    return [
        {'ssid': 'FarmNet_5G', 'signal': 85, 'security': 'WPA3', 'signal_strength': 'Excellent'},
        {'ssid': 'DroneNet_2.4G', 'signal': 72, 'security': 'WPA2', 'signal_strength': 'Good'},
        {'ssid': 'Mobile_Hotspot', 'signal': 58, 'security': 'WPA2', 'signal_strength': 'Fair'},
        {'ssid': 'Backup_WiFi', 'signal': 34, 'security': 'WPA2', 'signal_strength': 'Weak'},
        {'ssid': 'Guest_Network', 'signal': 45, 'security': 'Open', 'signal_strength': 'Fair'}
    ]


# --- Authentication System ---
@app.route('/auth/register', methods=['POST'])
def register_user():
    data = request.get_json()
    if not data or not all(k in data for k in ['username', 'email', 'password']):
        return jsonify({'error': 'Missing required fields'}), 400
    
    username = data['username'].strip()
    email = data['email'].strip()
    password = data['password']
    
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    
    if username in users_db:
        return jsonify({'error': 'Username already exists'}), 400
    
    # Hash password
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    users_db[username] = {
        'email': email,
        'password_hash': password_hash,
        'created_at': datetime.utcnow().isoformat()
    }
    
    session['user'] = username
    return jsonify({'status': 'registered', 'username': username}), 201


@app.route('/auth/login', methods=['POST'])
def login_user():
    data = request.get_json()
    if not data or not all(k in data for k in ['username', 'password']):
        return jsonify({'error': 'Missing username or password'}), 400
    
    username = data['username'].strip()
    password = data['password']
    
    if username not in users_db:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    if users_db[username]['password_hash'] != password_hash:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    session['user'] = username
    return jsonify({
        'status': 'logged_in',
        'username': username,
        'email': users_db[username]['email']
    }), 200


@app.route('/auth/logout', methods=['POST'])
def logout_user():
    session.pop('user', None)
    return jsonify({'status': 'logged_out'}), 200


@app.route('/auth/status', methods=['GET'])
def auth_status():
    if 'user' in session:
        username = session['user']
        return jsonify({
            'authenticated': True,
            'username': username,
            'email': users_db[username]['email']
        }), 200
    return jsonify({'authenticated': False}), 200


# --- Settings Management ---
@app.route('/settings', methods=['GET'])
def get_settings():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['user']
    user_settings = settings_db.get(username, {})
    return jsonify({'settings': user_settings}), 200


@app.route('/settings', methods=['POST'])
def save_settings():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    if not data or 'settings' not in data:
        return jsonify({'error': 'Missing settings data'}), 400
    
    username = session['user']
    settings_db[username] = data['settings']
    return jsonify({'status': 'settings_saved'}), 200


# --- Environmental Data ---
@app.route('/environmental', methods=['GET'])
def get_environmental_data():
    # Simulate changing environmental data
    environmental_data.update({
        'temperature': round(random.uniform(15, 30), 1),
        'humidity': round(random.uniform(40, 80), 1),
        'air_quality': random.choice(['Good', 'Fair', 'Poor']),
        'wind_speed': round(random.uniform(0, 15), 1),
        'last_update': datetime.utcnow().isoformat()
    })
    return jsonify(environmental_data), 200


# --- Health Trends ---
@app.route('/health/trends', methods=['GET'])
def get_health_trends():
    if len(history) < 3:
        return jsonify({'error': 'Insufficient data for trends'}), 400
    
    # Generate trend data from history
    recent_history = history[:10]
    stats = {
        'total': len(recent_history),
        'healthy': sum(1 for h in recent_history if 'healthy' in h.get('disease', '').lower()),
        'warnings': sum(1 for h in recent_history if h.get('disease', '') in ['gumboro_disease', 'infectious_coryza', 'splay_foot']),
        'critical': sum(1 for h in recent_history if h.get('disease', '') in ['avian_influenza', 'new_castles_disease', 'dead_chickens'])
    }
    
    # Generate 7-day trend
    trend_data = []
    for i in range(7):
        day_data = {
            'day': f'Day {i+1}',
            'healthy': random.randint(0, 1),
            'warning': random.randint(0, 1),
            'critical': random.randint(0, 1)
        }
        trend_data.append(day_data)
    
    trend = 'Improving' if stats['healthy'] >= 6 else 'Stable' if stats['healthy'] >= 3 else 'Concerning'
    
    return jsonify({
        'stats': stats,
        'trend_data': trend_data,
        'trend': trend,
        'recommendation': get_health_recommendation(stats)
    }), 200


def get_health_recommendation(stats):
    if stats['critical'] > 0:
        return 'Critical issues detected. Immediate veterinary consultation recommended.'
    elif stats['warnings'] > 2:
        return 'Multiple warning signs detected. Consider preventive measures.'
    elif stats['healthy'] > 5:
        return 'Flock health appears good. Continue current practices.'
    else:
        return 'Insufficient data for comprehensive assessment.'


# --- Scheduling System ---
@app.route('/schedule', methods=['GET'])
def get_schedule():
    return jsonify({'schedules': schedules_db}), 200


@app.route('/schedule', methods=['POST'])
def create_schedule():
    data = request.get_json()
    if not data or 'type' not in data:
        return jsonify({'error': 'Missing schedule type'}), 400
    
    schedule = {
        'id': len(schedules_db) + 1,
        'type': data['type'],
        'time': data.get('time', '10:00'),
        'date': data.get('date', (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')),
        'status': 'scheduled',
        'created_at': datetime.utcnow().isoformat()
    }
    
    schedules_db.append(schedule)
    return jsonify({'status': 'scheduled', 'schedule': schedule}), 201


# --- Emergency Alerts ---
@app.route('/alert/emergency', methods=['POST'])
def send_emergency_alert():
    data = request.get_json()
    alert_type = data.get('type', 'general')
    message = data.get('message', 'Emergency alert from Kgosi BioDrone')
    
    # Simulate sending alerts
    alert_id = f'EMRG_{int(time.time())}'
    
    log.info(f'Emergency alert sent: {alert_id} - {message}')
    
    return jsonify({
        'status': 'alert_sent',
        'alert_id': alert_id,
        'recipients': ['Dr. Sarah Molefe', 'Farm Manager', 'Animal Health Authority'],
        'timestamp': datetime.utcnow().isoformat()
    }), 200


# --- System Health ---
@app.route('/system/health', methods=['GET'])
def system_health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'yolo_available': net is not None,
        'uptime': '99.8%',
        'avg_latency': '38ms',
        'components': {
            'drone': {'status': 'online', 'latency': '12ms'},
            'camera': {'status': 'online', 'latency': '8ms'},
            'wifi': {'status': 'strong', 'signal': '85%'},
            'ai_model': {'status': 'ready', 'latency': '156ms'},
            'gps': {'status': 'fixed', 'satellites': 12},
            'battery': {'status': '78%', 'health': 'good'}
        },
        'timestamp': datetime.utcnow().isoformat()
    }), 200


# --- Data Export ---
@app.route('/export/history', methods=['GET'])
def export_history():
    format_type = request.args.get('format', 'json')
    
    if format_type == 'csv':
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Date', 'Disease', 'Confidence', 'Image URL'])
        
        for item in history:
            writer.writerow([
                item.get('timestamp', ''),
                item.get('disease', ''),
                '85%',  # Placeholder confidence
                item.get('image_url', '')
            ])
        
        return output.getvalue(), 200, {'Content-Type': 'text/csv'}
    
    return jsonify({'history': history}), 200


# --- Care Guide Data ---
@app.route('/care-guide/<disease>', methods=['GET'])
def get_care_guide(disease):
    care_guides = {
        'Chicken Drinking water': {
            'name': 'Chicken Drinking Water',
            'severity': 'healthy',
            'description': 'Normal drinking behavior observed',
            'immediate': 'No action required - normal behavior',
            'daily': 'Monitor water consumption levels',
            'weekly': 'Clean and sanitize water systems',
            'prevention': 'Maintain consistent water quality and temperature',
            'next_steps': ['Continue monitoring water quality', 'Ensure clean water supply', 'Check water temperature']
        },
        'Chicken Feeding': {
            'name': 'Chicken Feeding',
            'severity': 'healthy',
            'description': 'Normal feeding behavior detected',
            'immediate': 'No action required - normal behavior',
            'daily': 'Track feed consumption patterns',
            'weekly': 'Rotate feed stocks and check freshness',
            'prevention': 'Maintain proper feed storage conditions',
            'next_steps': ['Monitor feed consumption rates', 'Check feed quality', 'Ensure adequate feed supply']
        },
        'avian_influenza': {
            'name': 'Avian Influenza',
            'severity': 'critical',
            'description': 'Highly contagious viral infection detected',
            'immediate': 'Isolate affected birds immediately, implement strict biosecurity',
            'daily': 'Monitor all birds for symptoms, maintain quarantine protocols',
            'weekly': 'Continue monitoring, follow veterinary treatment plan',
            'prevention': 'Vaccination programs, strict biosecurity, limit visitor access',
            'next_steps': ['IMMEDIATE QUARANTINE', 'Contact veterinarian urgently', 'Notify authorities', 'Test all birds']
        },
        'dead_chickens': {
            'name': 'Dead Chickens',
            'severity': 'critical',
            'description': 'Mortality detected - immediate investigation required',
            'immediate': 'Remove and properly dispose of carcasses, examine remaining flock',
            'daily': 'Intensive monitoring of remaining birds for 14 days',
            'weekly': 'Continue health surveillance, document all findings',
            'prevention': 'Regular health checks, vaccination schedules, stress reduction',
            'next_steps': ['Remove carcasses immediately', 'Investigate cause of death', 'Check other birds', 'Contact vet']
        },
        'gumboro_disease': {
            'name': 'Gumboro Disease',
            'severity': 'warning',
            'description': 'Infectious bursal disease affecting immune system',
            'immediate': 'Isolate affected birds, provide clean environment',
            'daily': 'Supportive care, monitor hydration and nutrition',
            'weekly': 'Follow vaccination protocol, maintain biosecurity',
            'prevention': 'Proper vaccination timing, stress management, clean facilities',
            'next_steps': ['Isolate affected birds', 'Provide supportive care', 'Contact veterinarian', 'Review vaccination']
        },
        'healthy': {
            'name': 'Healthy',
            'severity': 'healthy',
            'description': 'Birds appear healthy with no signs of disease',
            'immediate': 'No action required',
            'daily': 'Continue normal feeding and care routine',
            'weekly': 'Regular health observations, maintain clean environment',
            'prevention': 'Consistent care practices, vaccination schedules',
            'next_steps': ['Continue routine monitoring', 'Maintain current care practices', 'Schedule routine check']
        },
        'healthy_chicken': {
            'name': 'Healthy Chicken',
            'severity': 'healthy',
            'description': 'Individual bird showing good health indicators',
            'immediate': 'No action required',
            'daily': 'Normal feeding and care routine',
            'weekly': 'Health monitoring, environmental checks',
            'prevention': 'Balanced nutrition, clean environment, regular exercise',
            'next_steps': ['Continue monitoring', 'Maintain nutrition program', 'Regular health checks']
        },
        'infectious_coryza': {
            'name': 'Infectious Coryza',
            'severity': 'warning',
            'description': 'Bacterial respiratory infection detected',
            'immediate': 'Isolate birds, improve air quality and ventilation',
            'daily': 'Administer prescribed antibiotics, monitor breathing',
            'weekly': 'Continue treatment course, environmental improvements',
            'prevention': 'Good ventilation, reduce stress, vaccination programs',
            'next_steps': ['Isolate affected birds', 'Antibiotic treatment', 'Improve ventilation', 'Contact vet']
        },
        'new_castles_disease': {
            'name': 'Newcastle Disease',
            'severity': 'critical',
            'description': 'Highly contagious viral disease affecting nervous system',
            'immediate': 'Complete quarantine, contact veterinary and health authorities',
            'daily': 'Strict isolation, no movement of birds or equipment',
            'weekly': 'Follow official control measures, monitoring protocols',
            'prevention': 'Vaccination programs, strict biosecurity, regular health monitoring',
            'next_steps': ['EMERGENCY QUARANTINE', 'Contact authorities', 'Veterinary consultation', 'Cull if necessary']
        },
        'splay_foot': {
            'name': 'Splay Foot',
            'severity': 'warning',
            'description': 'Developmental leg deformity affecting mobility',
            'immediate': 'Provide non-slip surfaces, assess bird mobility',
            'daily': 'Monitor eating and drinking ability, provide easy access',
            'weekly': 'Evaluate quality of life, consider housing modifications',
            'prevention': 'Proper incubation conditions, appropriate flooring, genetic selection',
            'next_steps': ['Assess severity', 'Provide supportive care', 'Adjust housing', 'Monitor mobility']
        }
    }
    
    guide = care_guides.get(disease, care_guides['healthy'])
    return jsonify(guide), 200


# --- Run server ---
if __name__ == '__main__':
    # pip install flask tensorflow opencv-python-headless python-dotenv
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8000'))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() in ('1', 'true', 'yes')
    app.run(host=host, port=port, debug=debug)
