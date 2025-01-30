import cv2
import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request
from datetime import datetime, date
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
datetoday = date.today().strftime("%m_%d_%y")
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ensure necessary directories exist
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Entry Time,Exit Time,Status\n')

# Load trained model
def load_model():
    if os.path.exists('face_recognition_model.pkl'):
        return joblib.load('face_recognition_model.pkl')
    return None

model = load_model()

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    return faces

def identify_face(face_array):
    if model:
        return model.predict(face_array)
    return None

def add_attendance(name):
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    
    if int(userid) not in df['Roll'].values:
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'{name},{userid},{current_time},-,Present\n')
    else:
        df.loc[df['Roll'] == int(userid), 'Exit Time'] = current_time
        df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)

@app.route('/')
def home():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    return render_template('home.html', records=df.to_dict(orient='records'))

@app.route('/start')
def start_attendance():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50)).reshape(1, -1)
            person = identify_face(face)
            if person:
                add_attendance(person[0])
                cv2.putText(frame, person[0], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return home()

if __name__ == '__main__':
    app.run(debug=True)