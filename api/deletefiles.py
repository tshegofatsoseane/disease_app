import os
import time

YOLO_DIR = "/tmp/yolo"

def cleanup_downloaded_files():
    files_to_remove = [
        os.path.join(YOLO_DIR, "yolov3.cfg"),
        os.path.join(YOLO_DIR, "yolov3.weights"),
        os.path.join(YOLO_DIR, "coco.names")
    ]
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")

cleanup_downloaded_files()
