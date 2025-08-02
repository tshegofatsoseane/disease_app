import os
import time
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import BlobClient
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

AZURE_CONN_STR = os.getenv("AZURE_STORAGE_CONN_STRING")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER", "models")
BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "model.keras")

MODEL_PATH = "/tmp/model.keras"

def load_model():
    if not os.path.exists(MODEL_PATH):
        blob = BlobClient.from_connection_string(
            conn_str=AZURE_CONN_STR,
            container_name=CONTAINER_NAME,
            blob_name=BLOB_NAME
        )
        content = blob.download_blob().readall()
        with open(MODEL_PATH, "wb") as f:
            f.write(content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Preprocess the input image
def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale to [0, 1]
    return img_array

# Predict the disease in the chicken image
def predict_disease(model, img_path, class_indices):
    preprocessed_img = preprocess_image(img_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_indices[predicted_class_index]
    return predicted_class_label

# Mapping class indices to class names
prepared_directory = os.path.join(ROOT_DIR, 'prepared_dataset')

datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    prepared_directory,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}


YOLO_CONTAINER_NAME = os.getenv("YOLO_CONTAINER_NAME", "yolo")

YOLO_DIR = "/tmp/yolo"
os.makedirs(YOLO_DIR, exist_ok=True)

def download_yolo_file(filename):
    local_path = os.path.join(YOLO_DIR, filename)
    if not os.path.exists(local_path):
        blob = BlobClient.from_connection_string(
            conn_str=AZURE_CONN_STR,
            container_name=YOLO_CONTAINER_NAME,
            blob_name=filename
        )
        with open(local_path, "wb") as f:
            f.write(blob.download_blob(timeout=300).readall())
    return local_path


# List of YOLO files to download
yolo_filenames = ["yolov3.cfg", "yolov3.weights", "coco.names"]

# Download in parallel using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=3) as executor:
    paths = list(executor.map(download_yolo_file, yolo_filenames))

# Assign the returned paths
yolo_config_path, yolo_weights_path, yolo_labels_path = paths


# Load YOLO
net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open(yolo_labels_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


# Function to detect chickens using YOLO
def detect_chicken(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == classes.index('bird') and confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return True, frame
    return False, frame

# Capture image from camera using libcamera
def capture_image_with_libcamera(output_path="captured_image.jpg"):
    capture_command = f"libcamera-still -o {output_path} --nopreview -t 1000"
    exit_code = os.system(capture_command)
    if exit_code != 0:
        print("Error: Failed to capture image with libcamera.")
        return None
    else:
        print(f"Image successfully captured and saved to {output_path}.")
        return output_path

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def detect():
    # === Capture Image ===
    img_path = capture_image_with_libcamera()
    if not img_path:
        return render_template('index.html', message="Failed to capture image")

    frame = cv2.imread(img_path)

    # === Lazy-load YOLO ===
    def get_yolo():
        yolo_config_path = download_yolo_file("yolov3.cfg")
        yolo_weights_path = download_yolo_file("yolov3.weights")
        yolo_labels_path = download_yolo_file("coco.names")

        net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        with open(yolo_labels_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return net, output_layers, classes

    net, output_layers, classes = get_yolo()

    # === Detect Chicken ===
    def detect_chicken(frame):
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids, confidences, boxes = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id < len(classes) and classes[class_id] == 'bird' and confidence > 0.5:
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return True, frame
        return False, frame

    chicken_detected, annotated_frame = detect_chicken(frame)
    annotated_path = "static/annotated_frame.jpg"
    cv2.imwrite(annotated_path, annotated_frame)

    # === Lazy-load Model & Predict ===
    if chicken_detected:
        model = tf.keras.models.load_model(MODEL_PATH)

        def preprocess_image(img_path, target_size=(128, 128)):
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            return img_array / 255.0

        # Static class labels (no need to load dataset)
        class_labels = {
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

        img_array = preprocess_image(img_path)
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels.get(predicted_index, "Unknown")

        return render_template('index.html', disease=predicted_label)

    return render_template('index.html', message="No chicken detected")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
