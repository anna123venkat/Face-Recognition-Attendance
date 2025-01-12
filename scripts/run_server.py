import os
from flask import Flask, request, render_template
from datetime import datetime, date
import cv2
import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Global Variables
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
datetoday = date.today().strftime("%m_%d_%y")
attendance_file = f'Attendance/Attendance-{datetoday}.csv'

# Ensure directories exist
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

# Create today's attendance file if it doesn't exist
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Entry Time,Exit Time,Status\n')


# Utility Functions
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return faces


def identify_face(face_array):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(face_array)


def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(attendance_file)

    if int(userid) not in list(df['Roll']):
        # First recognition (entry)
        new_entry = f"{username},{userid},{current_time},-,-\n"
        with open(attendance_file, 'a') as f:
            f.write(new_entry)
    else:
        # Subsequent recognition (exit)
        df.loc[df['Roll'] == int(userid), 'Exit Time'] = current_time
        entry_time = df.loc[df['Roll'] == int(userid), 'Entry Time'].values[0]
        status = calculate_status(entry_time, current_time)
        df.loc[df['Roll'] == int(userid), 'Status'] = status
        df.to_csv(attendance_file, index=False)


def calculate_status(entry_time, exit_time):
    entry_time = datetime.strptime(entry_time, "%H:%M:%S")
    exit_time = datetime.strptime(exit_time, "%H:%M:%S")
    duration = (exit_time - entry_time).total_seconds() / 60
    if duration < 6:
        return "Absent"
    elif 6 <= duration < 10:
        return "Notable"
    else:
        return "Present"


# Routes
@app.route('/')
def home():
    df = pd.read_csv(attendance_file)
    return render_template(
        'home.html',
        names=df['Name'].tolist(),
        rolls=df['Roll'].tolist(),
        entry=df['Entry Time'].tolist(),
        exit=df['Exit Time'].tolist(),
        status=df['Status'].tolist(),
        totalreg=len(os.listdir('static/faces')),
        datetoday2=date.today().strftime("%d-%B-%Y")
    )


@app.route('/start', methods=['GET'])
def start():
    model_path = 'static/face_recognition_model.pkl'
    if not os.path.exists(model_path):
        return render_template('home.html', mess="Model not found! Add a user to train the model.")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50)).reshape(1, -1)
            name = identify_face(face)[0]
            add_attendance(name)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Face Recognition Attendance", frame)
        if cv2.waitKey(1) == 27:  # Exit on pressing 'ESC'
            break

    cap.release()
    cv2.destroyAllWindows()
    return home()


@app.route('/add', methods=['POST'])
def add_user():
    new_username = request.form['newusername']
    new_userid = request.form['newuserid']
    user_dir = os.path.join('static/faces', f"{new_username}_{new_userid}")
    os.makedirs(user_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    for i in range(10):  # Capture 10 images
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(user_dir, f"{new_username}_{i}.jpg"), face)
            break

    cap.release()
    train_model()
    return home()


def train_model():
    faces, labels = [], []
    for user in os.listdir('static/faces'):
        for img_name in os.listdir(os.path.join('static/faces', user)):
            img_path = os.path.join('static/faces', user, img_name)
            img = cv2.imread(img_path)
            resized = cv2.resize(img, (50, 50)).ravel()
            faces.append(resized)
            labels.append(user)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(np.array(faces), labels)
    joblib.dump(model, 'static/face_recognition_model.pkl')


if __name__ == '__main__':
    app.run(debug=True)
