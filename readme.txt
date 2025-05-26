🔐 Face Recognition Web App

A full-fledged **Face Recognition System** built using **Python, OpenCV, and Flask**, designed with a sleek **dark-themed, glassmorphism-inspired UI** using **Tailwind CSS**.

🚀 Features

* 🎥 Real-time face detection using webcam
* 🧠 Face recognition with LBPH algorithm (OpenCV)
* ➕ Add new users by capturing 20 facial images automatically
* 🌐 Fully responsive and modern web interface
* ✨ Glassmorphism UI with smooth animations and dark mode
* 🗂️ Dynamic training of face dataset and model
* 🔁 Live recognition feed displays user names or “Unknown”

🛠️ Tech Stack

* **Frontend:** HTML, Tailwind CSS, Jinja2
* **Backend:** Python (Flask), OpenCV, NumPy
* **Model:** LBPH Face Recognizer
* **UI Design:** Minimal, dark mode with glassmorphism and animated transitions

📁 Project Structure

```
├── app.py                  # Flask backend
├── dataset/                # Stores face images
├── face-trainer.yml        # Trained model file
├── haarcascade_frontalface_default.xml  # Face detection model
├── templates/              # HTML files (index, add, result)
└── static/                 # (Optional) Custom CSS/JS or images
```

✅ How It Works

1. On launch, choose to **add a new face** or **recognize an existing user**.
2. If adding a new user, the system captures 20 images and trains the model.
3. On recognition, the webcam feed displays the name of the recognized user or shows "Unknown".


💡 Ideal For:

* College or internship project demos
* Hands-on learning of AI/ML + full-stack development
* Real-world application of computer vision & model training
