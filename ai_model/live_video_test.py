import cv2
import tensorflow as tf
import numpy as np
import time

# --- 1. LOAD THE TRAINED MODEL ---
model_path = 'models/corn_disease_classifier.h5'
print("Loading model...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Define image size that the model expects
img_size = (128, 128)

# --- 2. SETUP THE WEBCAM ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- 3. VARIABLES FOR SPRAY SIMULATION ---
is_spraying = False
spray_start_time = 0
spray_duration = 2 # How long the "SPRAYING..." message should appear (in seconds)

print("Webcam started. Press 'q' to quit.")

# --- 4. REAL-TIME PREDICTION LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # --- PREPROCESS THE FRAME for the model ---
    img_resized = cv2.resize(frame, img_size)
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = img_array / 255.0

    # --- MAKE A PREDICTION ---
    prediction = model.predict(img_array)

    # --- DISPLAY THE RESULT ---
    if prediction[0][0] > 0.5:
        label = 'Healthy'
        color = (0, 255, 0) # Green
    else:
        label = 'Diseased'
        color = (0, 0, 255) # Red
        # If disease is detected, start the spray simulation timer
        if not is_spraying:
            is_spraying = True
            spray_start_time = time.time()

    # Write the main prediction text on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # --- DISPLAY THE SPRAY SIMULATION ---
    if is_spraying:
        # Show the "SPRAYING..." message
        cv2.putText(frame, 'SPRAYING...', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Check if the spray duration has passed
        if time.time() - spray_start_time > spray_duration:
            is_spraying = False # Reset the spray simulation

    # Display the final frame in a window
    cv2.imshow('Live Plant Disease Detection', frame)

    # --- EXIT CONDITION ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
print("Closing webcam.")
cap.release()
cv2.destroyAllWindows()