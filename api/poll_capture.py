import os
import requests
import time

RENDER_API_URL = "https://disease-app-r13b.onrender.com"

while True:
    try:
        # Ask the server for a command
        res = requests.get(f"{RENDER_API_URL}/command")
        command = res.json().get("action")

        if command == "capture":
            print("[Pi] Capturing image...")

            # Capture image using libcamera-still
            image_path = "captured.jpg"
            os.system(f"libcamera-still -o {image_path} --nopreview -t 1000")

            if os.path.exists(image_path):
                # Send image to /analyze route
                with open(image_path, "rb") as f:
                    response = requests.post(f"{RENDER_API_URL}/analyze", files={"file": f})

                try:
                    print("[Pi] Analysis result:", response.json())
                except Exception as e:
                    print("[Pi] Failed to decode response:", e)
                    print(response.text)

                # Reset command to idle
                requests.post(f"{RENDER_API_URL}/command", json={"action": "idle"})
            else:
                print("[Pi] Image capture failed. File not found.")

    except Exception as e:
        print("[Pi] Error:", e)

    time.sleep(5)  # Poll every 5 seconds

