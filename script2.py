from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv
load_dotenv()

conn_str = os.getenv("AZURE_STORAGE_CONN_STRING")
container_name = "yolo" 

yolo_files = ["yolov3.cfg", "yolov3.weights", "coco.names"]

service_client = BlobServiceClient.from_connection_string(conn_str)
container_client = service_client.get_container_client(container_name)

for filename in yolo_files:
    with open(f"yolo_config_files/{filename}", "rb") as data:
        blob_client = container_client.get_blob_client(blob=filename)
        blob_client.upload_blob(data, overwrite=True)
        print(f"Uploaded {filename}")

