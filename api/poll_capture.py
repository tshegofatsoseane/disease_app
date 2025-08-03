import requests
import time

RENDER_API_URL = "https://disease-app-r13b.onrender.com"

while True:
    try:
        res = requests.get(f"{RENDER_API_URL}/command")
        command = res.json().get("action")

        if command == "capture":
            print("Capturing image...")
            # Capture image on the Pi
            os.system("libcamera-still -o captured.jpg --nopreview -t 1000")
            # Send to /analyze
            with open("captured.jpg", "rb") as f:
                response = requests.post(f"{RENDER_API_URL}/analyze", files={"image": f})
                print("Analysis result:", response.json())
            # Reset command
            requests.post(f"{RENDER_API_URL}/command", json={"action": "idle"})
    except Exception as e:
        print("Error:", e)

    time.sleep(5)  # Poll every 5 seconds

