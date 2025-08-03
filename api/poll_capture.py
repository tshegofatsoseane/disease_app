import os
import requests
import time

RENDER_API_URL = "https://disease-app-r13b.onrender.com"  # Your Flask server URL

while True:
    try:
        res = requests.get(f"{RENDER_API_URL}/command")
        command = res.json().get("action")

        if command == "capture":
            print("Capturing image...")
            # Capture image locally with libcamera
            os.system("libcamera-still -o captured.jpg --nopreview -t 1000")
            
            # Upload to analyze endpoint using key 'file' to match Flask expectation
            with open("captured.jpg", "rb") as f:
                response = requests.post(f"{RENDER_API_URL}/analyze", files={"file": f})
            
            try:
                result = response.json()
            except Exception:
                result = {"error": "Invalid response from server"}

            print("Analysis result:", result)
            
            # Reset command to idle
            requests.post(f"{RENDER_API_URL}/command", json={"action": "idle"})

    except Exception as e:
        print("Error:", e)

    time.sleep(5)  # Poll every 5 seconds

