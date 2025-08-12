import os
import io
import time
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, url_for
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from werkzeug.utils import secure_filename
from azure.storage.blob import BlobClient
from dotenv import load_dotenv

load_dotenv()

import requests
import base64

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_STATIC = os.path.join(BASE_DIR, 'static')
os.makedirs(APP_STATIC, exist_ok=True)

AZURE_CONN_STR = os.getenv('AZURE_STORAGE_CONN_STRING')
CONTAINER_NAME = os.getenv('AZURE_CONTAINER', 'models')
BLOB_NAME = os.getenv('AZURE_BLOB_NAME', 'model.keras')

YOLO_CONTAINER_NAME = os.getenv('YOLO_CONTAINER_NAME', 'yolo')
YOLO_DIR = os.path.join('/tmp', 'yolo')
os.makedirs(YOLO_DIR, exist_ok=True)

MODEL_PATH = os.path.join('/tmp', 'model.keras')
ANNOTATED_FILENAME = 'annotated_frame.jpg'
ANNOTATED_PATH = os.path.join(APP_STATIC, ANNOTATED_FILENAME)

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('cdd_server')

app = Flask(__name__, static_folder='static', template_folder='templates')

# In-memory state
current_command = {'action': 'idle'}
last_detection = {'disease': None, 'image_path': None, 'timestamp': None}
history = []
HISTORY_LIMIT = int(os.getenv('HISTORY_LIMIT', '50'))

# --- Helpers: Azure blob download (if configured) ---

def download_blob_to_local(container_name, blob_name, local_path, timeout=300):
    if not AZURE_CONN_STR:
        log.info('No AZURE_CONN_STR configured; skipping blob download for %s', blob_name)
        return False

    log.info('Downloading blob %s/%s -> %s', container_name, blob_name, local_path)
    blob = BlobClient.from_connection_string(conn_str=AZURE_CONN_STR,
                                            container_name=container_name,
                                            blob_name=blob_name)
    with open(local_path, 'wb') as f:
        f.write(blob.download_blob(timeout=timeout).readall())
    return True

# --- Model loading ---

def load_keras_model():
    # If model doesn't exist on disk try to download from Azure
    if not os.path.exists(MODEL_PATH):
        if AZURE_CONN_STR:
            try:
                download_blob_to_local(CONTAINER_NAME, BLOB_NAME, MODEL_PATH)
            except Exception as e:
                log.exception('Failed to download model from Azure: %s', e)
        else:
            log.warning('MODEL_PATH missing and no Azure connection string; expecting a local model at %s', MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
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

# --- YOLO setup ---

yolo_files = {
    'cfg': 'yolov3.cfg',
    'weights': 'yolov3.weights',
    'names': 'coco.names'
}

def download_yolo_files():
    for key, fname in yolo_files.items():
        dst = os.path.join(YOLO_DIR, fname)
        if not os.path.exists(dst):
            try:
                download_blob_to_local(YOLO_CONTAINER_NAME, fname, dst)
            except Exception as e:
                log.warning('Could not download YOLO file %s: %s', fname, e)

# attempt download (safe if AZURE not configured)
download_yolo_files()

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
        # getUnconnectedOutLayers may return shape (N,1) or a list of ints
        outs = net.getUnconnectedOutLayers()
        output_layers = [layer_names[i[0] - 1] if hasattr(i, '__len__') else layer_names[i - 1] for i in outs]
        with open(yolo_names, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        log.info('YOLO loaded with %d classes', len(classes))
    except Exception as e:
        log.exception('Error initializing YOLO: %s', e)
else:
    log.warning('YOLO files not found in %s. Chicken detection will be disabled.', YOLO_DIR)

# --- Image helpers ---

def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


def detect_chicken(frame, conf_threshold=0.5, nms_threshold=0.4):
    """Return (found_boolean, annotated_frame). If YOLO isn't available return (True, frame) to let
    classifier run as a best-effort (you can change the default behavior).
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
    prepared_dir = os.path.join(BASE_DIR, 'prepared_dataset')
    if os.path.exists(prepared_dir):
        datagen = ImageDataGenerator(rescale=1./255)
        gen = datagen.flow_from_directory(prepared_dir, target_size=(128,128), batch_size=1, class_mode='categorical')
        class_indices = gen.class_indices
        class_labels_map = {v: k for k, v in class_indices.items()}
        log.info('Derived class labels map from prepared_dataset with %d classes', len(class_labels_map))
    else:
        log.info('No prepared_dataset directory found; will use fallback labels')
except Exception as e:
    log.warning('Could not derive class labels map: %s', e)

# --- Flask routes ---

@app.route('/', methods=['GET'])
def index():
    # simple health endpoint + the UI
    try:
        return render_template('index.html')
    except Exception:
        # If no template present (during quick tests) return basic JSON health
        return jsonify({'status': 'ok'}), 200


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(uploaded_file.filename)
    tmp_path = os.path.join('/tmp', f"upload_{int(time.time())}_{filename}")
    uploaded_file.save(tmp_path)

    frame = cv2.imread(tmp_path)
    if frame is None:
        return jsonify({'error': 'Uploaded file is not a valid image'}), 400

    # Run chicken detection (YOLO). If YOLO not available the function currently returns True.
    chicken_detected, annotated_frame = detect_chicken(frame)

    # save annotated image to static
    try:
        cv2.imwrite(ANNOTATED_PATH, annotated_frame)
    except Exception as e:
        log.exception('Failed to write annotated image: %s', e)

    image_url = url_for('static', filename=ANNOTATED_FILENAME)

    if not chicken_detected:
        # Update last detection (no chicken)
        last_detection.update({'disease': None, 'image_path': image_url, 'timestamp': datetime.utcnow().isoformat()})
        return jsonify({'error': 'No chicken detected', 'image_url': image_url}), 200

    # If we have a model, run classifier
    predicted_label = None
    try:
        if model is None:
            log.warning('No model loaded; returning fallback label')
            predicted_label = 'unknown_model'
        else:
            img_array = preprocess_image(tmp_path)
            preds = model.predict(img_array)
            if preds is None:
                predicted_label = 'prediction_failed'
            else:
                predicted_index = int(np.argmax(preds, axis=1)[0])
                # map index to label
                if class_labels_map and isinstance(class_labels_map, dict) and predicted_index in class_labels_map:
                    predicted_label = class_labels_map[predicted_index]
                else:
                    predicted_label = FALLBACK_CLASS_MAP.get(predicted_index, f'class_{predicted_index}')
    except Exception as e:
        log.exception('Prediction error: %s', e)
        predicted_label = 'prediction_error'

    # record detection
    now = datetime.utcnow().isoformat()
    last_detection.update({'disease': predicted_label, 'image_path': image_url, 'timestamp': now})

    # add to history (keep most recent first)
    history.insert(0, {'disease': predicted_label, 'image_url': image_url, 'timestamp': now})
    while len(history) > HISTORY_LIMIT:
        history.pop()

    response = {
        'disease': predicted_label,
        'image_url': image_url,
        'note': 'Model suggestions — consult a vet for confirmation.'
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
    # still return an image path for UI fallbacks
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

    # If capture, optionally we could enqueue something. For now just acknowledge.
    return jsonify({'status': 'command set', 'action': action}), 200


# --- Analyze remote: forward to Raspberry Pi ---
PI_HOST = os.getenv('PI_HOST')            # e.g. '192.168.1.55:5000' OR 'pi.example.com:5000'
PI_TOKEN = os.getenv('PI_API_TOKEN')      # must match Pi's token
PI_TIMEOUT = int(os.getenv('PI_TIMEOUT', '12'))  # seconds

@app.route('/analyze_remote', methods=['POST'])
def analyze_remote():
    """
    Forward image to a Raspberry Pi running the pi_server.py service.
    The Pi is expected to return JSON with 'disease' and 'image_b64' fields.
    """
    if 'file' not in request.files:
        return jsonify({'error':'No file uploaded'}), 400

    if not PI_HOST:
        return jsonify({'error':'PI_HOST not configured'}), 500

    f = request.files['file']
    fname = secure_filename(f.filename or 'upload.jpg')

    # forward as multipart/form-data to the Pi
    # requests needs file-like object; use stream
    files = {'file': (fname, f.stream, f.content_type)}
    headers = {'X-PI-TOKEN': PI_TOKEN} if PI_TOKEN else {}

    try:
        pi_url = f'http://{PI_HOST}/analyze'
        r = requests.post(pi_url, files=files, headers=headers, timeout=PI_TIMEOUT)
    except requests.exceptions.RequestException as e:
        return jsonify({'error':'pi_unreachable','detail':str(e)}), 502

    # if pi returned failure code, pass that through
    try:
        pj = r.json()
    except Exception:
        return jsonify({'error':'invalid_response_from_pi','status_code': r.status_code, 'text': r.text}), 502

    # If the Pi returns a base64 image, write it to static annotated file for UI compatibility
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

    # mirror expected fields for frontend
    return jsonify(pj), 200


# --- Run server ---
if __name__ == '__main__':
    # recommended Python packages:
    # pip install flask tensorflow opencv-python-headless azure-storage-blob python-dotenv
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8000'))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() in ('1', 'true', 'yes')
    app.run(host=host, port=port, debug=debug)
