# plant-vision-rover
A CV-powered rover to detect and spray plant diseases for SIH 2025.

# 🌽 Plant-Vision-Rover

An autonomous rover project designed for the Smart India Hackathon (SIH) that uses computer vision to detect diseases in corn crops and simulate a targeted spray response.

![Demo GIF](https://i.imgur.com/your-demo-gif-url.gif)  ---

## 🎯 Project Goal

The objective of this project is to create a low-cost, efficient solution for precision agriculture. By identifying and spraying only affected plants, we aim to reduce pesticide usage, minimize environmental impact, and improve crop health. This repository contains the complete software for the AI model and the control logic.

## ✨ Key Features

* **Real-Time Disease Detection:** Uses a Convolutional Neural Network (CNN) to classify corn leaves as 'Healthy' or 'Diseased' from a live video feed.
* **Simulated Spray Response:** When a disease is detected, the system displays a visual "SPRAYING..." cue on-screen, simulating the activation of hardware.
* **Efficient Model:** The model is saved in the `.h5` format, ready to be tested and deployed.
* **Large File Handling:** Uses Git LFS to manage large model files, ensuring repository health.

---

## 🛠️ Technology Stack

* **AI / Machine Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV
* **Programming Language:** Python
* **Environment:** Jupyter Notebook (for training), Python script (for execution)

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### **Prerequisites**

* Python 3.9+
* Git and Git LFS installed

### **Installation & Setup**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/dassm309-sys/plant-vision-rover.git](https://github.com/dassm309-sys/plant-vision-rover.git)
    cd plant-vision-rover
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### **Usage**

1.  **Run the live detection script:**
    ```bash
    cd ai_model
    python live_video_test.py
    ```

2.  **Point your webcam** at an image of a corn leaf (healthy or diseased). The application will display the prediction in real time.

3.  Press the **'q'** key to quit the application.

---

## 🤝 Contributing

Contributions from team members are welcome! Please read our `CONTRIBUTING.md` file for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
