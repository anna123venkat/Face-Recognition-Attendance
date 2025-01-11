import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)

# Constants
nimgs = 10
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Directory setup
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')

if not os.path.isdir('static'):
    os.makedirs('static')

if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Entry Time,Exit Time,Status')

# Utility functions
def totalreg():
    return len(os.listdir('static/faces'))

def calculate_status(entry_time, exit_time):
    entry_time = datetime.strptime(entry_time, "%H:%M:%S")
    exit_time = datetime.strptime(exit_time, "%H:%M:%S")
    duration = (exit_time - entry_time).total_seconds() / 60  # in minutes
    if duration <= 6:
        return "Absent"
    elif 7 <= duration < 10:
        return "Notable"
    else:
        return "Present"

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    return face_points

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    entry = df['Entry Time']
    exit = df['Exit Time']
    status = df['Status']
    l = len(df)
    return names, rolls, entry, exit, status, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time},-,-')
    else:
        df.loc[df['Roll'] == int(userid), 'Exit Time'] = current_time
        entry_time = df.loc[df['Roll'] == int(userid), 'Entry Time'].values[0]
        status = calculate_status(entry_time, current_time)
        df.loc[df['Roll'] == int(userid), 'Status'] = status
        df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)

# Routes
@app.route('/')
def home():
    names, rolls, entry, exit, status, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, entry=entry, exit=exit, status=status, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    names, rolls, entry, exit, status, l = extract_attendance()
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, entry=entry, exit=exit, status=status, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, entry, exit, status, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, entry=entry, exit=exit, status=status, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i = 0
    while i < nimgs:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.imwrite(f'{userimagefolder}/{newusername}_{i}.jpg', frame[y:y+h, x:x+w])
            i += 1
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    train_model()
    names, rolls, entry, exit, status, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, entry=entry, exit=exit, status=status, l=l, totalreg=totalreg(), datetoday2=datetoday2, new_user_added=True)

# Main
if __name__ == '__main__':
    app.run(debug=True)
