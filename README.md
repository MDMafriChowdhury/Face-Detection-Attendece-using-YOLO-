Web-Based Face Recognition Attendance System

A self-contained Flask web application for facial recognition attendance. It's accessible from mobile devices and includes a web-based interface for training new users.

Installation

Clone or download this repository to your computer.

Install the required Python libraries using pip:

pip install Flask opencv-python numpy Pillow pyOpenSSL


Download the haarcascade_frontalface_default.xml file.

You can find it on the OpenCV GitHub repository.

Click "Raw", then right-click and "Save As...".

Save this file in the same directory as app.py.

File Structure

Your project directory must be set up as follows for the application to work correctly:

Face_Attendance_App/
│
├── app.py                  <-- The main web server (run this file)
│
├── haarcascade_frontalface_default.xml  <-- OpenCV model for *finding* faces
│
├── templates/              <-- Flask folder for HTML files
│   ├── index.html          <-- The main attendance page
│   └── train.html          <-- The user training page
│
├── dataset/                <-- (Created automatically) Stores training images
│
├── attendance.db           <-- (Created automatically) SQLite database
│
└── trainer.yml             <-- (Created automatically) The trained recognition model


Usage

Start the Server
Open your terminal, navigate to your project folder, and run:

python app.py


Access the App

Open your phone's browser (make sure it's on the same Wi-Fi as the server).

Go to the https://<YOUR_PC_IP_ADDRESS>:5000 address shown in the terminal.

You must accept the browser's security warning (this is normal for a self-signed certificate). Click "Advanced" -> "Proceed".

Train a New User

From the main page, click the "Train New User" button.

Enter the user's name and click "Start Training".

Follow the 7-stage guided process (Look Straight, Look Left, etc.).

The model will train automatically after capturing all images.

Take Attendance

On the main page, point your phone's camera at your face.

The status will change to Detected: [Your Name] (xx%).

Press the "Check In" or "Check Out" button to record the event.

Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

License

MIT
