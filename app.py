import cv2
import os
import numpy as np
import sqlite3
import time
import datetime
from flask import Flask, Response, render_template, request, jsonify, make_response
import base64
import io
from PIL import Image
from fpdf import FPDF

# --- Configuration ---
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
TRAINER_FILE = 'trainer.yml'
DATABASE_FILE = 'attendance.db'
DATASET_PATH = 'dataset' # Added for training
CONFIDENCE_THRESHOLD = 50 

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Helper: Create dataset directory ---
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# --- Global Variables for Detection ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
font = cv2.FONT_HERSHEY_SIMPLEX

# --- Global State for API ---
app.last_action_time = 0
COOLDOWN_SECONDS = 5
user_map = {} # Will be loaded at startup

# --- Database Helper Functions (Moved from train_tool.py) ---

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL DEFAULT 'checked-out' 
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (user_id)
    )
    ''')
    conn.commit()
    conn.close()
    print(f"Database '{DATABASE_FILE}' initialized.")

def load_user_map_from_db():
    """Loads the user ID-to-name mapping from the database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT user_id, name FROM users")
        rows = cursor.fetchall()
        return {str(row[0]): row[1] for row in rows}
    except sqlite3.Error as e:
        print(f"Error loading user map from DB: {e}")
        return {}
    finally:
        conn.close()

def get_user_by_name_from_db(name):
    """Finds a user by name (case-insensitive) and returns their ID."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT user_id FROM users WHERE LOWER(name) = LOWER(?)", (name,))
        row = cursor.fetchone()
        return row[0] if row else None
    except sqlite3.Error as e:
        print(f"Error fetching user by name: {e}")
        return None
    finally:
        conn.close()

def get_next_user_id_from_db():
    """Finds the highest user ID and returns the next available ID."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT MAX(user_id) FROM users")
        row = cursor.fetchone()
        max_id = row[0] if row and row[0] is not None else 0
        return max_id + 1
    except sqlite3.Error as e:
        print(f"Error fetching next user ID: {e}")
        return 1
    finally:
        conn.close()

def create_user_in_db(user_id, name):
    """Saves a new user to the database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (user_id, name) VALUES (?, ?)", (user_id, name))
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"Error: User name '{name}' or ID '{user_id}' already exists.")
        return False
    except sqlite3.Error as e:
        print(f"Error creating user: {e}")
        return False
    finally:
        conn.close()
    return True

def record_attendance_db(user_id, user_name, event_type):
    """Records a check-in or check-out event to the database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    time_str = now.strftime("%H:%M:%S")
    
    try:
        cursor.execute("SELECT status FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if not row:
            return (f"Error: User ID {user_id} not found.", False)
            
        current_status = row[0]
        is_success = False
        
        if event_type == "check-in":
            if current_status == "checked-in":
                message = f"'{user_name}' is already checked in."
            else:
                cursor.execute("UPDATE users SET status = 'checked-in' WHERE user_id = ?", (user_id,))
                message = f"'{user_name}' checked-in at {time_str}"
                is_success = True
                
        elif event_type == "check-out":
            if current_status == "checked-out":
                message = f"'{user_name}' must check in first."
            else:
                cursor.execute("UPDATE users SET status = 'checked-out' WHERE user_id = ?", (user_id,))
                message = f"'{user_name}' checked-out at {time_str}"
                is_success = True
        
        if is_success:
            cursor.execute(
                "INSERT INTO events (user_id, event_type, timestamp) VALUES (?, ?, ?)",
                (user_id, event_type, timestamp_str)
            )
            conn.commit()
            print(f"[Attendance] {message}")
        
        return (message, is_success)
        
    except sqlite3.Error as e:
        conn.rollback()
        print(f"Error recording attendance: {e}")
        return ("Database Error", False)
    finally:
        conn.close()

def decode_base64_image(base64_string):
    """Decodes a base64 string into a CV2-compatible image."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(',')[1]
        img_bytes = base64.b64decode(base64_string)
        pil_img = Image.open(io.BytesIO(img_bytes))
        open_cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return open_cv_image
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

# --- Training Helper (from train_tool.py) ---
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []
    
    for image_path in image_paths:
        if not image_path.endswith('.jpg'):
            continue
        try:
            pil_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if pil_img is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue
            
            img_numpy = np.array(pil_img, 'uint8')
            id_str = os.path.split(image_path)[-1].split(".")[1]
            id = int(id_str)
            
            ids.append(id)
            face_samples.append(img_numpy)
        except Exception as e:
            print(f"Warning: Skipping file {image_path}. Error: {e}")
            
    return face_samples, ids

# --- Helper to load recognizer ---
def load_recognizer():
    global recognizer, user_map
    if not os.path.exists(TRAINER_FILE):
        print(f"Warning: '{TRAINER_FILE}' not found. Training needed.")
        return
    try:
        recognizer.read(TRAINER_FILE)
        user_map = load_user_map_from_db()
        print(f"[INFO] Models reloaded. {len(user_map)} user(s) found.")
    except cv2.error as e:
        print(f"Error loading recognizer: {e}. Is trainer.yml valid?")

# --- NEW HELPER: Format Duration ---
def format_duration(seconds):
    """Converts seconds into a human-readable Hh Mm Ss format."""
    if seconds is None or seconds < 0:
        return "N/A"
    if seconds == 0:
        return "0m 0s"
        
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
        
    return " ".join(parts)

# --- Flask API Routes ---

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/recognize_frame', methods=['POST'])
def recognize_frame():
    """Receives a frame from the client, recognizes it, and returns the user info."""
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"status": "error", "message": "No image data provided"}), 400

    img = decode_base64_image(data['image'])
    if img is None:
        return jsonify({"status": "error", "message": "Invalid image data"}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({"status": "no_face"})

    (x, y, w, h) = faces[0]
    id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
    
    if confidence < CONFIDENCE_THRESHOLD:
        name = user_map.get(str(id), "Unknown ID")
        return jsonify({
            "status": "known", 
            "user_id": id, 
            "user_name": name,
            "confidence": round(100 - confidence)
        })
    else:
        return jsonify({"status": "unknown", "confidence": round(100 - confidence)})

@app.route('/attendance_action', methods=['POST'])
def attendance_action():
    """Handles check-in and check-out button presses."""
    now = time.time()
    if now - app.last_action_time < COOLDOWN_SECONDS:
        return jsonify({"success": False, "message": "Please wait..."})
    
    data = request.json
    if data.get('user_id') is None:
        return jsonify({"success": False, "message": "No known face detected!"})

    message, is_success = record_attendance_db(data['user_id'], data['user_name'], data['action'])
    if is_success:
        app.last_action_time = now
    
    return jsonify({"success": is_success, "message": message})


# --- NEW TRAINING ROUTES ---

@app.route('/train')
def train_page():
    """Serves the new training webpage."""
    return render_template('train.html')

@app.route('/start_training', methods=['POST'])
def start_training():
    """Creates a new user (or finds existing) and returns the user ID."""
    data = request.json
    user_name = data.get('name', '').strip()
    if not user_name:
        return jsonify({"success": False, "message": "Name cannot be empty."})

    user_id = get_user_by_name_from_db(user_name)
    if user_id is not None:
        print(f"User '{user_name}' (ID: {user_id}) already exists. Adding more images.")
        # Clear old images for this user
        for f in os.listdir(DATASET_PATH):
            if f.startswith(f"User.{user_id}."):
                os.remove(os.path.join(DATASET_PATH, f))
        print(f"Removed old images for user ID {user_id}.")
    else:
        user_id = get_next_user_id_from_db()
        if not create_user_in_db(user_id, user_name):
            return jsonify({"success": False, "message": "Error creating user in DB."})
        print(f"Creating new user '{user_name}' with ID {user_id}.")
        
    return jsonify({"success": True, "user_id": user_id, "user_name": user_name})

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    """Receives and saves a single frame for training."""
    data = request.json
    user_id = data.get('user_id')
    count = data.get('count')
    image_data = data.get('image')

    if not all([user_id, count, image_data]):
        return jsonify({"success": False, "message": "Missing data."})

    img = decode_base64_image(image_data)
    if img is None:
        return jsonify({"success": False, "message": "Invalid image data."})

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({"success": False, "message": "No face detected in frame."})
    
    (x, y, w, h) = faces[0]
    face_img = gray[y:y+h, x:x+w]
    
    img_path = os.path.join(DATASET_PATH, f"User.{user_id}.{count}.jpg")
    cv2.imwrite(img_path, face_img)
    
    return jsonify({"success": True, "message": f"Saved {img_path}"})

@app.route('/run_model_training', methods=['POST'])
def run_model_training():
    """Triggers the backend to train the LBPH model."""
    print("[INFO] Training model on all captured images...")
    
    try:
        faces, ids = get_images_and_labels(DATASET_PATH)
        if not faces:
            print("Error: No faces found to train.")
            return jsonify({"success": False, "message": "No faces found in dataset."})

        recognizer.train(faces, np.array(ids))
        recognizer.write(TRAINER_FILE)
        
        # Reload the recognizer and user map in memory
        load_recognizer()
        
        print(f"\n[INFO] {len(np.unique(ids))} user(s) trained. Model saved to {TRAINER_FILE}.")
        return jsonify({"success": True, "message": "Training complete! Model updated."})

    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({"success": False, "message": f"Error during training: {e}"})

# --- PDF Helper Class (MODIFIED) ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Attendance Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def report_table(self, data, col_widths):
        self.set_font('Arial', 'B', 10)
        
        # Headers (MODIFIED)
        headers = ['Name', 'Date', 'First In', 'Last Out', 'Total Duration', 'All Sessions']
        col_keys = ['name', 'date', 'first_in', 'last_out', 'duration', 'sessions']
        
        for i, header in enumerate(headers):
            self.cell(col_widths[col_keys[i]], 10, header, 1, 0, 'C')
        self.ln()
        
        # Data (MODIFIED - Using MultiCell for robust row height)
        self.set_font('Arial', '', 9)
        if not data:
            self.cell(sum(col_widths.values()), 10, 'No data found for this selection.', 1, 1, 'C')
            return
            
        for row in data:
            start_y = self.get_y()
            
            # Create a list of all data points for the row
            row_data_points = [
                str(row['name']),
                str(row['event_date']),
                str(row['first_check_in']),
                str(row['last_check_out']),
                str(row['total_duration_formatted']),
                str(row['all_sessions']) # The new multi-line data
            ]
            
            max_y = start_y
            
            # Draw all cells as MultiCells to handle wrapping and get max height
            cell_heights = []
            
            # First pass: Draw MultiCells and find max height
            current_x = self.get_x()
            for i, key in enumerate(col_keys):
                width = col_widths[key]
                self.multi_cell(width, 8, row_data_points[i], 0, 'L')
                cell_heights.append(self.get_y())
                current_x += width
                self.set_xy(current_x, start_y) # Reset Y for next cell
            
            max_y = max(cell_heights)
            
            # Second pass: Draw the borders based on max_y
            current_x = self.l_margin
            self.set_xy(current_x, start_y) # Go back to start of row
            for i, key in enumerate(col_keys):
                width = col_widths[key]
                # Draw border for the full height of the row
                self.rect(current_x, start_y, width, max_y - start_y)
                current_x += width
            
            self.set_xy(self.l_margin, max_y) # Move to the next line

# --- Internal Helper for fetching report data (MODIFIED) ---

def _get_report_data(user_id, start_date, end_date):
    """Internal function to query the DB for a smart summary report."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row # This lets us access columns by name
    cursor = conn.cursor()
    
    params = []
    conditions = []
    
    # Build WHERE conditions for the CTE
    if user_id:
        conditions.append("e.user_id = ?")
        params.append(user_id)
    if start_date:
        conditions.append("DATE(e.timestamp) >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("DATE(e.timestamp) <= ?")
        params.append(end_date)
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # This is the new "smart" query to calculate daily summaries
    query = f"""
    WITH PairedEvents AS (
        SELECT
            e.user_id,
            e.event_type,
            e.timestamp,
            DATE(e.timestamp) AS event_date,
            TIME(e.timestamp) AS event_time,
            LEAD(e.event_type, 1) OVER (
                PARTITION BY e.user_id, DATE(e.timestamp) 
                ORDER BY e.timestamp
            ) AS next_event_type,
            LEAD(e.timestamp, 1) OVER (
                PARTITION BY e.user_id, DATE(e.timestamp) 
                ORDER BY e.timestamp
            ) AS next_timestamp
        FROM events e
        WHERE {where_clause}
    ),
    
    Calculations AS (
        SELECT
            user_id,
            event_date,
            
            -- Calculate duration for valid pairs
            CASE
                WHEN event_type = 'check-in' AND next_event_type = 'check-out'
                THEN (strftime('%s', next_timestamp) - strftime('%s', timestamp))
                ELSE 0
            END AS duration_seconds,
            
            -- Create a string for each valid session
            CASE
                WHEN event_type = 'check-in' AND next_event_type = 'check-out'
                THEN TIME(timestamp) || ' - ' || TIME(next_timestamp)
                ELSE NULL
            END AS session_string,
            
            -- Flag for first check-in
            CASE
                WHEN event_type = 'check-in'
                THEN event_time
                ELSE NULL
            END AS check_in_time,
            
            -- Flag for last check-out
            CASE
                WHEN event_type = 'check-out'
                THEN event_time
                ELSE NULL
            END AS check_out_time
            
        FROM PairedEvents
    )
    
    SELECT
        u.name,
        c.event_date,
        MIN(c.check_in_time) AS first_check_in,
        MAX(c.check_out_time) AS last_check_out,
        SUM(c.duration_seconds) AS total_duration_seconds,
        -- Use GROUP_CONCAT to combine all session strings, separated by a newline
        -- MODIFIED: Replaced FILTER clause with a CASE statement for wider SQLite compatibility
        GROUP_CONCAT(CASE WHEN c.session_string IS NOT NULL THEN c.session_string ELSE NULL END, CHAR(10)) AS all_sessions
    FROM Calculations c
    JOIN users u ON u.user_id = c.user_id
    GROUP BY u.name, c.event_date
    HAVING first_check_in IS NOT NULL -- Only show days where user was present
    ORDER BY c.event_date DESC, u.name;
    """

    try:
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        # Convert rows to dicts and format the duration
        report_data = []
        for row in rows:
            row_dict = dict(row)
            row_dict['total_duration_formatted'] = format_duration(row_dict.get('total_duration_seconds'))
            # Handle NULLs from DB
            if not row_dict['first_check_in']:
                row_dict['first_check_in'] = "---"
            if not row_dict['last_check_out']:
                row_dict['last_check_out'] = "---"
            if not row_dict['all_sessions']:
                row_dict['all_sessions'] = "No sessions"
            report_data.append(row_dict)
            
        return report_data, None
    except sqlite3.Error as e:
        print(f"Error fetching smart report: {e}")
        return None, str(e)
    finally:
        conn.close()


# --- NEW REPORTING & ADMIN ROUTES ---

@app.route('/reports')
def reports_page():
    """Serves the new reports/admin page."""
    return render_template('reports.html')

@app.route('/api/users', methods=['GET'])
def get_all_users():
    """Fetches all users for the report filter dropdown."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT user_id, name FROM users ORDER BY name")
        users = cursor.fetchall()
        # Convert list of tuples to list of dicts for easier JSON
        user_list = [{"user_id": row[0], "name": row[1]} for row in users]
        return jsonify(user_list)
    except sqlite3.Error as e:
        print(f"Error fetching all users: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/attendance_report', methods=['GET'])
def get_attendance_report_json():
    """Provides the attendance report data as JSON for the web table."""
    user_id = request.args.get('user_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    data, error = _get_report_data(user_id, start_date, end_date)
    
    if error:
        return jsonify({"error": error}), 500
    return jsonify(data)

@app.route('/api/download_pdf', methods=['GET'])
def download_pdf_report():
    """Generates and serves the attendance report as a PDF download."""
    user_id = request.args.get('user_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    data, error = _get_report_data(user_id, start_date, end_date)
    
    if error:
        return f"Error generating report: {error}", 500
    
    # --- FIX: Instantiate the PDF object ---
    # MODIFIED: Set orientation and format in the constructor, not add_page()
    pdf = PDF(orientation='L', format='A4')
    pdf.add_page() # Use Landscape mode for more space
    
    # Define column widths (MODIFIED)
    col_widths = {
        'name': 45,
        'date': 25,
        'first_in': 25,
        'last_out': 25,
        'duration': 30,
        'sessions': 127 # Total 277, fits A4-L page
    }
    
    pdf.report_table(data, col_widths)

    # Generate a dynamic filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"attendance_report_{timestamp}.pdf"
    
    # Create a response and send the PDF data
    response = make_response(pdf.output(dest='S').encode('latin-1'))
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return response


# --- Main ---
if __name__ == "__main__":
    init_db() # Ensure DB exists
    load_recognizer() # Load models on startup
    
    print("[INFO] Starting Flask server...")
    print("[INFO] To access from your phone, use https://<YOUR_PC_IP_ADDRESS>:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False, ssl_context='adhoc')