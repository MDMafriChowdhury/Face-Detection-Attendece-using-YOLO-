

# Web-Based Face Recognition Attendance System Using Flask App and YOLO model

## Project Description

Odoo Face Attendance is a web-based kiosk application that uses facial recognition to manage employee attendance, posting all check-in and check-out events directly to your Odoo database in real-time.

This project consists of a Python Flask server that handles face detection (using OpenCV) and communicates with Odoo (using odoorpc). It provides a simple web interface for training new employees, a kiosk interface for daily attendance, and a reporting dashboard that pulls live data from Odoo.

## Installation

Clone or download this repository to your computer.
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required dependencies.
```bash
pip install Flask opencv-python numpy Pillow pyOpenSSL
```

## Download Haar Cascade File

Download the Haar Cascade file required for face detection.

- File name: haarcascade_frontalface_default.xml

- Source: OpenCV GitHub (data/haarcascades)

- Click Raw, then right-click and select Save As...

- Save it in the same directory as app.py.

# File Structure

```bash
Face_Attendance_App/
│
├── app.py                                # Main Flask server (run this file)
│
├── haarcascade_frontalface_default.xml   # OpenCV face detection model
│
├── templates/                            # Flask templates folder
│   ├── index.html                        # Main attendance page
│   └── train.html                        # User training page
│   └── reports.html                      # Admin attendence reports page
│
├── dataset/                              # (Auto-created) Stores training images
│
├── attendance.db                         # (Auto-created) SQLite database
│
└── trainer.yml                           # (Auto-created) Trained recognition model

```

## Usage
```bash
python app.py
```
After starting, the terminal will show a message similar to:
```pgsql
[INFO] To access from your phone, use https://<ip>:5000
```
# Access the Application

- Make sure your phone is on the same Wi-Fi network as your computer.

- Open your browser and go to the HTTPS address shown in the terminal, for example:
https://<ip>:5000

- Accept the browser's security warning by clicking:
Advanced → Proceed to [your IP address] (unsafe)

# Train a New User(RealTime training)
```text
1. Click "Train New User" on the main page.
2. Enter the user’s name and click "Start Training".
3. Follow the 7-stage guided capture process:
   - Look Straight
   - Look Left
   - Look Right
   - Look Up
   - Look Down
   - Close
   - Far
4. Wait while the model trains automatically (10–30 seconds).
5. Once complete, click "Back to App".
```

# Take Attendance
```text
1. Open the main page again.
2. Point your camera at your face.
3. When the UI displays:
   Detected: [Your Name] (xx%)
4. Click "Check In" or "Check Out" to record attendance.

```
Smart attendance logic ensures:
- A user cannot check in twice.
- A user cannot check out without checking in.

# View Reports:

- Go to the reports page: https://<YOUR_PC_IP>:5000/reports

- Use the filters to select a user or date range.

- Click "Fetch Report" to see the data on the page.

- Click "Download PDF" to save the same report as a PDF.



# Technologies Used

Flask – Web framework for backend logic

- OpenCV (LBPH) – Face detection and recognition

- SQLite – Local database for users and attendance logs

- HTML5 / JavaScript – Frontend interface

- pyOpenSSL – Enables HTTPS for browser camera access

- Realtime face training
