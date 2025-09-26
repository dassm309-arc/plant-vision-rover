from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np
import asyncio
import websockets
import threading

# --- CONFIGURATION ---
ESP32_IP = "YOUR_ESP32_IP"  # <-- IMPORTANT: UPDATE THIS
WEBSOCKET_URI = f"ws://{ESP32_IP}:81/"

app = Flask(__name__)

# --- Load the AI Model ---
model = tf.keras.models.load_model('models/corn_disease_classifier.h5')
img_size = (128, 128)
print("Model loaded.")
camera = cv2.VideoCapture(0)

# --- WebSocket Client to talk to ESP32 ---
async def send_to_esp32(command):
    try:
        async with websockets.connect(WEBSOCKET_URI) as websocket:
            await websocket.send(command)
            print(f"Sent command to ESP32: {command}")
    except Exception as e:
        print(f"Failed to send command to ESP32: {e}")

def run_command(command):
    asyncio.run(send_to_esp32(command))

# --- Video Generation with AI ---
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # --- AI Prediction ---
            img_resized = cv2.resize(frame, img_size)
            img_preprocessed = np.expand_dims(img_resized, axis=0) / 255.0
            prediction = model.predict(img_preprocessed, verbose=0)
            
            status = "Healthy"
            color = (0, 255, 0)
            
            if prediction[0][0] <= 0.5:
                status = "Diseased Detected"
                color = (0, 0, 255)
                # Automatically send spray command
                threading.Thread(target=run_command, args=('P',)).start() # Pump ON
                threading.Timer(2.0, run_command, args=('p',)).start() # Pump OFF after 2s
            
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control/<command>')
def control(command):
    # Run the send_to_esp32 in a separate thread to not block the web server
    threading.Thread(target=run_command, args=(command,)).start()
    return {"status": "command sent", "command": command}

# --- HTML Template (will be created automatically) ---
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Agri-Sense Rover Control</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; text-align: center; background: #f0f0f0; }
        .container { max-width: 800px; margin: auto; padding: 10px; }
        img { border: 2px solid #333; margin-bottom: 10px; width: 100%; max-width: 640px; }
        .controls, .actions { display: grid; max-width: 400px; margin: 10px auto; gap: 10px; }
        .controls { grid-template-columns: 1fr 1fr 1fr; }
        .actions { grid-template-columns: 1fr 1fr; }
        button { padding: 15px; font-size: 18px; border-radius: 8px; border: none; background: #007bff; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Agri-Sense Rover Control</h1>
        <img src="/video_feed">
        <div class="controls">
            <div></div>
            <button onmousedown="sendCommand('F')" onmouseup="sendCommand('S')">▲ Fwd</button>
            <div></div>
            <button onmousedown="sendCommand('L')" onmouseup="sendCommand('S')">◄ Left</button>
            <button onclick="sendCommand('S')">STOP</button>
            <button onmousedown="sendCommand('R')" onmouseup="sendCommand('S')">► Right</button>
            <div></div>
            <button onmousedown="sendCommand('B')" onmouseup="sendCommand('S')">▼ Bwd</button>
            <div></div>
        </div>
        <div class="actions">
            <button onclick="sendCommand('E')">Toggle Frame</button>
            <button onclick="sendCommand('A')">Toggle Wheel Angle</button>
        </div>
    </div>
    <script>
        function sendCommand(cmd) {
            fetch('/control/' + cmd);
        }
    </script>
</body>
</html>
"""

import os
if not os.path.exists('templates'):
    os.makedirs('templates')
with open('templates/index.html', 'w') as f:
    f.write(html_template)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)