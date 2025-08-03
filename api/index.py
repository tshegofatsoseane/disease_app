import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template
from tensorflow.keras.preprocessing import image
from azure.storage.blob import BlobClient
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

AZURE_CONN_STR = os.getenv("AZURE_STORAGE_CONN_STRING")
MODEL_CONTAINER = os.getenv("AZURE_CONTAINER", "models")
MODEL_BLOB_NAME = os.getenv("AZURE_BLOB_NAME", "model.keras")
YOLO_CONTAINER = os.getenv("YOLO_CONTAINER_NAME", "yolo")

MODEL_PATH = "/tmp/model.keras"
YOLO_DIR = "/tmp/yolo"
os.makedirs(YOLO_DIR, exist_ok=True)


# === Helpers ===

def download_blob_to_file(container, blob_name, local_path):
    if not os.path.exists(local_path):
        blob = BlobClient.from_connection_string(
            conn_str=AZURE_CONN_STR,
            container_name=container,
            blob_name=blob_name
        )
        with open(local_path, "wb") as f:
            f.write(blob.download_blob(timeout=300).readall())
    return local_path


def capture_image(output_path="captured_image.jpg"):
    if os.system(f"libcamera-still -o {output_path} --nopreview -t 1000") != 0:
        print("Failed to capture image")
        return None
    return output_path


def load_model():
    download_blob_to_file(MODEL_CONTAINER, MODEL_BLOB_NAME, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0


def load_yolo():
    config = download_blob_to_file(YOLO_CONTAINER, "yolov3.cfg", os.path.join(YOLO_DIR, "yolov3.cfg"))
    weights = download_blob_to_file(YOLO_CONTAINER, "yolov3.weights", os.path.join(YOLO_DIR, "yolov3.weights"))
    labels = download_blob_to_file(YOLO_CONTAINER, "coco.names", os.path.join(YOLO_DIR, "coco.names"))

    net = cv2.dnn.readNet(weights, config)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    with open(labels, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return net, output_layers, classes


def detect_chicken(frame, net, output_layers, classes):
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
            label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return True, frame
    return False, frame


# === Routes ===

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def detect():
    img_path = capture_image()
    if not img_path:
        return render_template('index.html', message="Failed to capture image")

    frame = cv2.imread(img_path)

    net, output_layers, classes = load_yolo()
    chicken_detected, annotated_frame = detect_chicken(frame, net, output_layers, classes)

    annotated_path = "static/annotated_frame.jpg"
    cv2.imwrite(annotated_path, annotated_frame)

    if chicken_detected:
        model = load_model()
        img_array = preprocess_image(img_path)

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

        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions, axis=1)[0])
        predicted_label = class_labels.get(predicted_index, "Unknown")

        return render_template('index.html', disease=predicted_label)

    return render_template('index.html', message="No chicken detected")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

