from flask import Flask, render_template, request, jsonify, Response, send_file
import cv2
import numpy as np
import os
import sqlite3
import pandas as pd
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
from deepface import DeepFace
import json
import threading
from werkzeug.utils import secure_filename
import gc  # Garbage collector
import time

# Fix for dotenv issue
import sys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    def patch_flask_dotenv():
        try:
            import flask.cli
            flask.cli.load_dotenv = lambda *args, **kwargs: None
        except ImportError:
            pass
    patch_flask_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/faces'
app.config['SECRET_KEY'] = 'your_secret_key_here'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class AttendanceSystem:
    def __init__(self):
        self.db_path = 'attendance.db'
        self.create_tables()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.camera = None
        self.is_capturing = False
        
        # Face recognition settings - keeping consistent across the app
        self.face_model = "Facenet512"
        self.face_detector = "opencv"
        self.similarity_threshold = 0.65  # More lenient threshold for better recognition
        
        # Email configuration
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender_email = "your_email@gmail.com"
        self.sender_password = "your_app_password"

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def create_tables(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT UNIQUE,
                face_encoding BLOB,
                model_name TEXT,
                encoding_size INTEGER,
                registration_date TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                timestamp TIMESTAMP,
                activity TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Check if we need to add the new columns to an existing table
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'model_name' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN model_name TEXT")
        if 'encoding_size' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN encoding_size INTEGER")
            
        conn.commit()
        conn.close()

    def validate_email(self, email):
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return re.match(pattern, email) is not None

    def generate_user_id(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        conn.close()
        return f"EMP{count + 1:04d}"

    def send_email(self, to_email, activity):
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = to_email
            msg['Subject'] = f"Attendance {activity} Notification"
            
            body = f"Your attendance has been marked as {activity} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()
            return True
        except Exception as e:
            print(f"Email error: {str(e)}")
            return False

    def preprocess_face(self, image):
        """Preprocess face image to reduce memory usage and improve recognition"""
        # Resize image to reduce memory usage
        max_size = 224  # Smaller size for processing
        h, w = image.shape[:2]
        if h > max_size or w > max_size:
            if h > w:
                new_h, new_w = max_size, int(w * max_size / h)
            else:
                new_h, new_w = int(h * max_size / w), max_size
            image = cv2.resize(image, (new_w, new_h))
        
        # Apply histogram equalization for better recognition in different lighting
        if len(image.shape) == 3:  # Color image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        
        return image

    def extract_face_encoding(self, face_image):
        """Extract face encoding using memory-efficient settings"""
        try:
            # Preprocess image
            face_image = self.preprocess_face(face_image)
            
            # Force garbage collection before heavy operation
            gc.collect()
            
            # Use lighter model with memory-efficient settings
            embedding_obj = DeepFace.represent(
                face_image, 
                model_name=self.face_model,
                detector_backend=self.face_detector,
                enforce_detection=False,
                align=True
            )
            
            if not embedding_obj or len(embedding_obj) == 0:
                raise Exception("No face embedding generated")
                
            embedding = embedding_obj[0]["embedding"]
            
            # Normalize the embedding vector
            embedding_array = np.array(embedding)
            normalized_embedding = embedding_array / np.linalg.norm(embedding_array)
            
            # Force garbage collection after heavy operation
            gc.collect()
            
            return normalized_embedding
        except Exception as e:
            print(f"Face encoding error: {str(e)}")
            raise

    def register_user(self, name, email, face_encoding):
        if not self.validate_email(email):
            return False, "Invalid email format"
        
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        if cursor.fetchone():
            conn.close()
            return False, "Email already registered"
        
        user_id = self.generate_user_id()
        cursor.execute("""
            INSERT INTO users (user_id, name, email, face_encoding, model_name, encoding_size, registration_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, name, email, face_encoding.tobytes(), self.face_model, len(face_encoding), datetime.now()))
        conn.commit()
        conn.close()
        
        return True, user_id

    def verify_face(self, face_encoding):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, email, face_encoding, model_name, encoding_size FROM users")
        result = None, None
        best_match = 0
        best_user = None
        
        for row in cursor.fetchall():
            try:
                stored_encoding = np.frombuffer(row[2])
                stored_model = row[3] if row[3] else "unknown"
                stored_size = row[4] if row[4] else len(stored_encoding)
                
                # Basic size check
                if len(stored_encoding) != len(face_encoding):
                    print(f"Encoding size mismatch for user {row[0]}: {len(stored_encoding)} vs {len(face_encoding)}")
                    continue
                
                # Normalize stored encoding if needed
                stored_norm = np.linalg.norm(stored_encoding)
                if stored_norm > 0:
                    stored_encoding = stored_encoding / stored_norm
                
                # Calculate cosine similarity
                similarity = np.dot(face_encoding, stored_encoding)
                print(f"User {row[0]} similarity: {similarity:.4f}")
                
                if similarity > best_match:
                    best_match = similarity
                    best_user = (row[0], row[1])
                
            except Exception as e:
                print(f"Error comparing face with user {row[0]}: {str(e)}")
                continue
        
        conn.close()
        
        # Return the best match if it's above our threshold
        if best_match > self.similarity_threshold:
            print(f"Best match: User {best_user[0]} with similarity {best_match:.4f}")
            return best_user
        else:
            print(f"No match found. Best similarity was {best_match:.4f}")
            return None, None

    def mark_attendance(self, user_id, email, activity):
        conn = self.get_connection()
        cursor = conn.cursor()
        timestamp = datetime.now()
        cursor.execute("""
            INSERT INTO attendance (user_id, timestamp, activity)
            VALUES (?, ?, ?)
        """, (user_id, timestamp, activity))
        conn.commit()
        conn.close()
        
        # Send email in a separate thread to not block the main thread
        email_thread = threading.Thread(target=self.send_email, args=(email, activity))
        email_thread.daemon = True
        email_thread.start()
        
        return True

    def export_attendance_report(self):
        conn = self.get_connection()
        query = """
            SELECT 
                u.user_id,
                u.name,
                u.email,
                a.timestamp,
                a.activity,
                CASE 
                    WHEN a.activity = 'check-out' THEN 
                        strftime('%s', a.timestamp) - strftime('%s', (
                            SELECT timestamp 
                            FROM attendance a2 
                            WHERE a2.user_id = a.user_id 
                            AND a2.activity = 'check-in' 
                            AND date(a2.timestamp) = date(a.timestamp)
                            AND a2.timestamp < a.timestamp
                            ORDER BY a2.timestamp DESC 
                            LIMIT 1
                        ))
                    ELSE 0 
                END as duration_seconds
            FROM attendance a
            JOIN users u ON a.user_id = u.user_id
            ORDER BY a.timestamp
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        df['duration_hours'] = df['duration_seconds'] / 3600
        
        filename = f"attendance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(filename, index=False)
        return filename

    def update_user_face(self, user_id, face_encoding):
        """Update a user's face encoding in the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users 
            SET face_encoding = ?, model_name = ?, encoding_size = ?
            WHERE user_id = ?
        """, (face_encoding.tobytes(), self.face_model, len(face_encoding), user_id))
        conn.commit()
        conn.close()
        return True

# Initialize attendance system
attendance_system = AttendanceSystem()

def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera")
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = attendance_system.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['POST'])
def register():
    try:
        name = request.form['name']
        email = request.form['email']
        
        if not attendance_system.validate_email(email):
            return jsonify({'success': False, 'message': 'Invalid email format'})
        
        # Capture and process face
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return jsonify({'success': False, 'message': 'Could not open camera'})
        
        # Take multiple frames to get a better image - increase from 3 to 5 frames
        frames = []
        for _ in range(5):  # Capture 5 frames instead of 3
            ret, frame = camera.read()
            if ret:
                frames.append(frame)
            # Add a small delay to stabilize camera
            time.sleep(0.1)
            
        camera.release()
        
        if not frames:
            return jsonify({'success': False, 'message': 'Camera error - no frames captured'})
        
        # Process the best frame (or average them)
        best_frame = frames[-1]  # Use the last frame which is likely more stable
        
        gray = cv2.cvtColor(best_frame, cv2.COLOR_BGR2GRAY)
        faces = attendance_system.face_cascade.detectMultiScale(gray, 1.1, 4)  # Adjust parameters to be more sensitive
        
        if len(faces) == 0:
            # Try again with more sensitive parameters if no face detected
            faces = attendance_system.face_cascade.detectMultiScale(gray, 1.05, 3)
        
        if len(faces) != 1:
            return jsonify({'success': False, 'message': 'Please ensure exactly one face is visible and well-lit'})
        
        x, y, w, h = faces[0]
        face_frame = best_frame[y:y+h, x:x+w]
        
        # Save face image for reference
        face_filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f"{email}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"))
        cv2.imwrite(face_filename, face_frame)
        
        try:
            encoding = attendance_system.extract_face_encoding(face_frame)
            if encoding is None or len(encoding) == 0:
                return jsonify({'success': False, 'message': 'Could not extract face features. Please try again with better lighting'})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Face analysis error: {str(e)}'})
        
        success, result = attendance_system.register_user(name, email, encoding)
        
        if success:
            return jsonify({'success': True, 'message': f'Registration successful! Your ID is {result}'})
        else:
            return jsonify({'success': False, 'message': result})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Registration error: {str(e)}'})

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    try:
        activity = request.form['activity']
        
        if activity not in ['check-in', 'check-out', 'break-in', 'break-out']:
            return jsonify({'success': False, 'message': 'Invalid activity type'})
        
        # Capture and process face
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return jsonify({'success': False, 'message': 'Could not open camera'})
        
        # Take multiple frames to get a better image
        frames = []
        for _ in range(5):  # Capture 5 frames instead of 3
            ret, frame = camera.read()
            if ret:
                frames.append(frame)
            # Add a small delay to stabilize camera
            time.sleep(0.1)
            
        camera.release()
        
        if not frames:
            return jsonify({'success': False, 'message': 'Camera error - no frames captured'})
        
        # Process the best frame (or average them)
        best_frame = frames[-1]  # Use the last frame which is likely more stable
        
        gray = cv2.cvtColor(best_frame, cv2.COLOR_BGR2GRAY)
        faces = attendance_system.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            # Try again with more sensitive parameters if no face detected
            faces = attendance_system.face_cascade.detectMultiScale(gray, 1.05, 3)
        
        if len(faces) != 1:
            return jsonify({'success': False, 'message': 'Please ensure exactly one face is visible and well-lit'})
        
        x, y, w, h = faces[0]
        face_frame = best_frame[y:y+h, x:x+w]
        
        try:
            encoding = attendance_system.extract_face_encoding(face_frame)
            if encoding is None or len(encoding) == 0:
                return jsonify({'success': False, 'message': 'Could not extract face features. Please try again with better lighting'})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Face analysis error: {str(e)}'})
        
        user_id, email = attendance_system.verify_face(encoding)
        
        if user_id is None:
            return jsonify({'success': False, 'message': 'Face not recognized. Please register first.'})
        
        # Update the face encoding on successful recognition to adapt to changes
        if activity == 'check-in':
            try:
                attendance_system.update_user_face(user_id, encoding)
            except Exception as e:
                print(f"Warning: Could not update face encoding: {str(e)}")
        
        attendance_system.mark_attendance(user_id, email, activity)
        return jsonify({'success': True, 'message': f'Attendance marked as {activity} for {email}'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Attendance error: {str(e)}'})

@app.route('/update_face', methods=['POST'])
def update_face():
    try:
        email = request.form['email']
        
        # Verify email exists
        conn = attendance_system.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM users WHERE email=?", (email,))
        user_data = cursor.fetchone()
        conn.close()
        
        if not user_data:
            return jsonify({'success': False, 'message': 'Email not found'})
            
        user_id = user_data[0]
        
        # Capture and process face
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return jsonify({'success': False, 'message': 'Could not open camera'})
            
        # Take multiple frames to get a better image
        frames = []
        for _ in range(3):  # Capture 3 frames
            ret, frame = camera.read()
            if ret:
                frames.append(frame)
            
        camera.release()
        
        if not frames:
            return jsonify({'success': False, 'message': 'Camera error - no frames captured'})
        
        # Process the best frame
        best_frame = frames[-1]  # Use the last frame which is likely more stable
        
        gray = cv2.cvtColor(best_frame, cv2.COLOR_BGR2GRAY)
        faces = attendance_system.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) != 1:
            return jsonify({'success': False, 'message': 'Please ensure exactly one face is visible'})
        
        x, y, w, h = faces[0]
        face_frame = best_frame[y:y+h, x:x+w]
        
        # Save face image for reference
        face_filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f"{email}_update_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"))
        cv2.imwrite(face_filename, face_frame)
        
        try:
            encoding = attendance_system.extract_face_encoding(face_frame)
        except Exception as e:
            return jsonify({'success': False, 'message': f'Face analysis error: {str(e)}'})
        
        attendance_system.update_user_face(user_id, encoding)
        return jsonify({'success': True, 'message': 'Face updated successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Update error: {str(e)}'})

@app.route('/export_report')
def export_report():
    try:
        filename = attendance_system.export_attendance_report()
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({'success': False, 'message': f'Report generation error: {str(e)}'})

if __name__ == '__main__':
    # Direct fix for dotenv issue - bypass Flask's dotenv loading
    import flask.cli
    flask.cli.load_dotenv = lambda *args, **kwargs: None
    
    app.run(debug=True)