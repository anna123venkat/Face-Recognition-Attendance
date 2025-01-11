import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load the Haarcascade face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def extract_faces(img):
    """
    Extracts face regions from an image.
    :param img: Input image.
    :return: List of face coordinates (x, y, w, h).
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except Exception as e:
        print(f"Error in face extraction: {e}")
        return []

def identify_face(facearray):
    """
    Identifies a face using the trained KNN model.
    :param facearray: Flattened face array.
    :return: Predicted label (user ID).
    """
    try:
        model = joblib.load('static/face_recognition_model.pkl')
        return model.predict(facearray)
    except Exception as e:
        print(f"Error in face identification: {e}")
        return None

def train_model():
    """
    Trains the KNN model using stored face data.
    """
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
    print("Model trained and saved successfully.")
