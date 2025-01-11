# Face Recognition-Based Attendance System

## Description
This project is a real-time, automated attendance system leveraging face recognition technology. It combines Flask, OpenCV, and K-Nearest Neighbors (KNN) to detect and recognize faces, track attendance, and generate detailed reports.

## Features
- **Contactless Attendance**: Utilizes a webcam to capture faces and log attendance.
- **Real-Time Recognition**: Identifies and classifies faces in real-time.
- **Dynamic Status Updates**: Categorizes attendees as "Present," "Notable," or "Absent" based on time spent.
- **User Management**: Provides an interface to add, manage, and display users.
- **Daily Logs**: Stores attendance records in CSV files for easy access.

## Technologies Used
- **Programming Language**: Python
- **Framework**: Flask
- **Libraries**: OpenCV, Pandas, NumPy, scikit-learn
- **Frontend**: HTML, CSS

## Project Overview
This system automates the attendance process, ensuring:
1. Improved accuracy by eliminating manual errors.
2. Reduced administrative workload.
3. Adaptability for various environments such as educational institutions and workplaces.

The application is structured to:
- Detect faces using Haar Cascade Classifier.
- Recognize faces with KNN for reliable identification.
- Generate visual feedback and attendance logs via a user-friendly Flask interface.

## Installation and Setup
1. Clone this repository:
    ```bash
    git clone <repository-url>
    ```
2. Navigate to the project directory:
    ```bash
    cd Face-Recognition-Attendance
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the application:
    ```bash
    python app.py
    ```
5. Access the application in your browser at `http://127.0.0.1:5000`.

## Usage
1. **Take Attendance**: Click on "Take Attendance" to begin face detection and recognition.
2. **Add New User**: Enter a new user's name and ID, and the system will capture their face and train the model.
3. **View Logs**: Attendance logs are automatically stored as CSV files in the `Attendance` directory.

## Project Report
For a detailed explanation of the system design, methodology, results, and references, see the attached project report (`CV Report.pdf`).

## Authors
- Abishek S
- Manoj S
- Prasanna Venkatesh S

## Acknowledgments
This project was completed as part of the B.Tech program in Artificial Intelligence and Data Science at Mepco Schlenk Engineering College, under the guidance of Dr. E. Emerson Nithiyaraj and Dr. J. Angela Jennifa Sujana.

## License
This project is licensed under the [MIT License](LICENSE).
