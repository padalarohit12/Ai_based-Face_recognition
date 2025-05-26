from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
from PIL import Image
import numpy as np

app = Flask(__name__)

face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
users_file = "users.json"
dataset_path = "dataset"
trainer_file = "face-trainer.yml"


# PHASE 1: Dataset Creation
def create_dataset(user_id):
    cam = cv2.VideoCapture(1)
    detector = cv2.CascadeClassifier(face_cascade_path)
    sample_num = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sample_num += 1
            os.makedirs(dataset_path, exist_ok=True)
            cv2.imwrite(f"{dataset_path}/User.{user_id}.{sample_num}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Creating Dataset', img)
        if cv2.waitKey(100) & 0xFF == ord('q') or sample_num >= 20:
            break

    cam.release()
    cv2.destroyAllWindows()

# PHASE 2: Train Model
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(face_cascade_path)

    def get_images_and_labels(path=dataset_path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for imagePath in image_paths:
            pil_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(pil_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)

        return face_samples, ids

    faces, ids = get_images_and_labels()
    recognizer.train(faces, np.array(ids))
    recognizer.save(trainer_file)

# PHASE 3: Face Recognition
def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_file)
    faceCascade = cv2.CascadeClassifier(face_cascade_path)

    names = ["Unknown", "ROHIT", "PRAVIN" , "SOWMYA"]

    cam = cv2.VideoCapture(1)
    cam.set(3, 640)
    cam.set(4, 480)

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            name = "Unknown"
            if confidence < 70 and id < len(names):
                name = names[id]

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{name} {round(100 - confidence)}%", (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Face Recognition', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


# FLASK ROUTES
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/collect', methods=['POST'])
def collect():
    user_id = int(request.form['user_id'])
    create_dataset(user_id)
    return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
def train():
    train_model()
    return redirect(url_for('index'))

@app.route('/recognize', methods=['POST'])
def recognize():
    recognize_faces()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)