import os

# General Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
FACES_DIR = os.path.join(STATIC_DIR, 'faces')
ATTENDANCE_DIR = os.path.join(BASE_DIR, 'Attendance')

# Model and Data Paths
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
MODEL_PATH = os.path.join(STATIC_DIR, 'face_recognition_model.pkl')
ATTENDANCE_FILE = os.path.join(ATTENDANCE_DIR, f'Attendance-{date.today().strftime("%m_%d_%y")}.csv')

# Flask Configuration
FLASK_DEBUG = True
SECRET_KEY = 'your_secret_key_here'

# Face Recognition Settings
IMAGE_SIZE = (50, 50)
KNN_NEIGHBORS = 5
