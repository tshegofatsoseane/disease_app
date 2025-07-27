import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import BlobClient

# Load environment variables from .env file
load_dotenv()

conn_str = os.getenv("AZURE_STORAGE_CONN_STRING")
if not conn_str:
    raise ValueError("AZURE_STORAGE_CONN_STRING not set")

# Now you can use the connection string safely
service_client = BlobServiceClient.from_connection_string(conn_str)

# Example: Create a container
container_name = "models"

# Path to your model file
local_file_path = "chicken_disease_classifier_vgg16.keras"
blob_name = "chicken_disease_classifier_vgg16.keras"

# Create a blob client
blob_client = service_client.get_blob_client(container=container_name, blob=blob_name)

# Upload the file
with open(local_file_path, "rb") as data:
    blob_client.upload_blob(data, overwrite=True)
    print(f"Model uploaded to blob storage as '{blob_name}'.")
