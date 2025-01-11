import pandas as pd
from datetime import datetime, date
import os

# Initialize today's date
datetoday = date.today().strftime("%m_%d_%y")

# Function to initialize attendance CSV file
def initialize_attendance_file():
    if not os.path.isdir('Attendance'):
        os.makedirs('Attendance')
    attendance_file = f'Attendance/Attendance-{datetoday}.csv'
    if attendance_file not in os.listdir('Attendance'):
        with open(attendance_file, 'w') as f:
            f.write('Name,Roll,Entry Time,Exit Time,Status')
    return attendance_file

# Function to calculate attendance status based on time spent
def calculate_status(entry_time, exit_time):
    entry_time = datetime.strptime(entry_time, "%H:%M:%S")
    exit_time = datetime.strptime(exit_time, "%H:%M:%S")
    duration = (exit_time - entry_time).total_seconds() / 60  # Convert to minutes
    if duration <= 6:
        return "Absent"
    elif 6 < duration < 10:
        return "Notable"
    else:
        return "Present"

# Function to extract attendance data
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name'].tolist()
    rolls = df['Roll'].tolist()
    entry = df['Entry Time'].tolist()
    exit = df['Exit Time'].tolist()
    status = df['Status'].tolist()
    l = len(df)
    return names, rolls, entry, exit, status, l

# Function to add or update attendance
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')

    if int(userid) not in df['Roll'].values:  # First recognition (entry)
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time},-,-')
    else:  # Subsequent recognition (exit)
        df.loc[df['Roll'] == int(userid), 'Exit Time'] = current_time
        entry_time = df.loc[df['Roll'] == int(userid), 'Entry Time'].values[0]
        status = calculate_status(entry_time, current_time)
        df.loc[df['Roll'] == int(userid), 'Status'] = status
        df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)
