import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

def train_model():
    """
    Trains a K-Nearest Neighbors (KNN) model using stored face data.
    Saves the trained model as 'face_recognition_model.pkl'.
    """
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    
    for user in userlist:
        user_folder = os.path.join('static/faces', user)
        for imgname in os.listdir(user_folder):
            img_path = os.path.join(user_folder, imgname)
            img = cv2.imread(img_path)
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())  # Flatten the image
            labels.append(user)  # Add the folder name as the label
    
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    
    # Save the trained model to a file
    model_path = 'static/face_recognition_model.pkl'
    joblib.dump(knn, model_path)
    print(f"Model trained and saved at {model_path}")

if __name__ == "__main__":
    train_model()
