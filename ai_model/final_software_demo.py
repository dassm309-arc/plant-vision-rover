from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# --- Load the AI Model ---
try:
    model = tf.keras.models.load_model('models/corn_disease_classifier.h5')
    img_size = (128, 128)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Camera Setup ---
camera = cv2.VideoCapture(0)

def generate_frames():
    """Generates frames from the webcam with AI predictions."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            status = "Point Leaf at Camera"
            color = (255, 255, 255) # White text for default state

            if model:
                # Prepare the frame for the model
                img_resized = cv2.resize(frame, img_size)
                img_preprocessed = np.expand_dims(img_resized, axis=0) / 255.0
                prediction = model.predict(img_preprocessed, verbose=0)
                
                prediction_value = prediction[0][0]

                # --- NEW: CONFIDENCE THRESHOLD LOGIC ---
                # Only show a result if the model is >90% confident
                
                if prediction_value < 0.1:  # High confidence it's DISEASED (class 0)
                    status = "Diseased Detected"
                    color = (0, 0, 255) # Red
                    cv2.putText(frame, "SPRAYING...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                elif prediction_value > 0.9: # High confidence it's HEALTHY (class 1)
                    status = "Healthy"
                    color = (0, 255, 0) # Green
                
                # If the prediction is between 0.1 and 0.9, it's uncertain,
                # so we do nothing and the status remains "Point Leaf at Camera".

            # Draw the status text on the frame
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            # Encode the frame as a JPEG image
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control/<command>')
def control(command):
    """Receives control commands from the webpage."""
    print(f"--- SIMULATED ACTION: Received command '{command}' ---")
    return {"status": "command received"}

# --- Create the HTML file for the webpage ---
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Agri-Sense Rover - Software Simulation</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; text-align: center; background: #f4f4f4; }
        .container { max-width: 800px; margin: 1rem auto; padding: 10px; background: white; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        img { border: 2px solid #ddd; margin-bottom: 15px; width: 100%; max-width: 640px; border-radius: 5px; }
        .controls, .actions { display: grid; max-width: 400px; margin: 10px auto; gap: 10px; }
        .controls { grid-template-columns: 1fr 1fr 1fr; }
        .actions { grid-template-columns: 1fr 1fr; }
        button { padding: 15px; font-size: 18px; border-radius: 8px; border: none; background: #007bff; color: white; cursor: pointer; }
        button:active { background-color: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Agri-Sense Rover - Software Simulation</h1>
        <img src="/video_feed">
        <div class="controls">
            <div></div>
            <button onclick="sendCommand('Move Forward')">▲</button>
            <div></div>
            <button onclick="sendCommand('Turn Left')">◄</button>
            <button onclick="sendCommand('Stop')">STOP</button>
            <button onclick="sendCommand('Turn Right')">►</button>
            <div></div>
            <button onclick="sendCommand('Move Backward')">▼</button>
            <div></div>
        </div>
        <div class="actions">
            <button onclick="sendCommand('Toggle Frame')">Frame</button>
            <button onclick="sendCommand('Toggle Wheel Angle')">Wheel Angle</button>
        </div>
    </div>
    <script>
        function sendCommand(cmd) {
            fetch('/control/' + cmd);
            console.log('Simulated command:', cmd);
        }
    </script>
</body>
</html>
"""

# Create a 'templates' folder and the index.html file if they don't exist
if not os.path.exists('templates'):
    os.makedirs('templates')
with open('templates/index.html', 'w') as f:
    f.write(html_template)

# --- Run the Web Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)