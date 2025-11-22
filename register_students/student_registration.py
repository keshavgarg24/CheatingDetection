"""
Student Face Registration and Database Management

This module provides functionality for registering students with face recognition
and managing the student face database for identity verification during exams.
"""

import sqlite3
import pickle
import cv2
import numpy as np
from datetime import datetime


class StudentFaceDatabase:
    """
    Manages student face encodings and registration for identity verification.
    
    Stores face encodings as pickled numpy arrays and registration photos as JPEG bytes
    in a SQLite database for efficient retrieval and matching.
    """
    
    def __init__(self, db_path="student_faces.sqlite"):
        """
        Initialize the StudentFaceDatabase.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """
        Create the students table if it doesn't exist.
        
        Schema includes face_encoding as BLOB (pickled numpy array) and
        registration_photo as BLOB (JPEG bytes).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                roll_number TEXT UNIQUE,
                seat_number TEXT,
                face_encoding BLOB NOT NULL,
                registration_photo BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"[INFO] Student database initialized at {self.db_path}")
    
    def register_student_from_camera(self, student_id, name, roll_number, seat_number):
        """
        Register a student by capturing their face from webcam.
        
        Interactive process that displays live camera feed and captures face
        when user presses 'c'. Extracts face encoding and saves to database.
        
        Args:
            student_id: Unique student identifier
            name: Student's full name
            roll_number: Student's roll number
            seat_number: Assigned seat number
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            import face_recognition
        except ImportError:
            print("[ERROR] face_recognition library not installed.")
            print("Install with: pip install face-recognition")
            return False
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam")
            return False
        
        print(f"\n=== Registering Student: {name} ===")
        print("Position your face in the camera frame")
        print("Press 'c' to capture, 'q' to quit")
        
        captured_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read from camera")
                break
            
            # Display frame
            display_frame = frame.copy()
            cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Registering: {name}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Student Registration", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                captured_frame = frame.copy()
                print("[INFO] Frame captured!")
                break
            elif key == ord('q'):
                print("[INFO] Registration cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured_frame is None:
            print("[ERROR] No frame captured")
            return False
        
        # Save student encoding
        success = self._save_student_encoding(
            captured_frame, student_id, name, roll_number, seat_number
        )
        
        if success:
            print(f"[SUCCESS] Student {name} registered successfully!")
        else:
            print(f"[ERROR] Failed to register student {name}")
        
        return success
    
    def _save_student_encoding(self, frame, student_id, name, roll_number, seat_number):
        """
        Extract face encoding from frame and save to database.
        
        Args:
            frame: OpenCV frame containing student's face
            student_id: Unique student identifier
            name: Student's full name
            roll_number: Student's roll number
            seat_number: Assigned seat number
            
        Returns:
            bool: True if encoding extracted and saved successfully
        """
        try:
            import face_recognition
        except ImportError:
            print("[ERROR] face_recognition library not installed")
            return False
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract face encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if len(face_encodings) == 0:
            print("[ERROR] No face detected in frame")
            return False
        
        if len(face_encodings) > 1:
            print("[WARNING] Multiple faces detected, using the first one")
        
        # Use the first face encoding
        face_encoding = face_encodings[0]
        
        # Pickle the face encoding
        encoding_blob = pickle.dumps(face_encoding)
        
        # Encode frame as JPEG
        _, jpeg_buffer = cv2.imencode('.jpg', frame)
        photo_blob = jpeg_buffer.tobytes()
        
        # Save to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO students 
                (student_id, name, roll_number, seat_number, face_encoding, registration_photo)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (student_id, name, roll_number, seat_number, encoding_blob, photo_blob))
            
            conn.commit()
            conn.close()
            
            print(f"[INFO] Face encoding saved for {name}")
            return True
            
        except sqlite3.Error as e:
            print(f"[ERROR] Database error: {e}")
            return False
    
    def identify_student(self, frame):
        """
        Identify a student from a frame by comparing face encoding.
        
        Args:
            frame: OpenCV frame containing a face
            
        Returns:
            dict: {
                'identified': bool,
                'student_id': str or None,
                'name': str or None,
                'roll_number': str or None,
                'seat_number': str or None,
                'confidence': float (0.0-1.0)
            }
        """
        try:
            import face_recognition
        except ImportError:
            return {
                'identified': False,
                'student_id': None,
                'name': None,
                'roll_number': None,
                'seat_number': None,
                'confidence': 0.0,
                'error': 'face_recognition not installed'
            }
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract face encodings from frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if len(face_encodings) == 0:
            return {
                'identified': False,
                'student_id': None,
                'name': None,
                'roll_number': None,
                'seat_number': None,
                'confidence': 0.0,
                'error': 'No face detected'
            }
        
        # Use the first face encoding
        unknown_encoding = face_encodings[0]
        
        # Load all registered students
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT student_id, name, roll_number, seat_number, face_encoding FROM students")
        students = cursor.fetchall()
        conn.close()
        
        if len(students) == 0:
            return {
                'identified': False,
                'student_id': None,
                'name': None,
                'roll_number': None,
                'seat_number': None,
                'confidence': 0.0,
                'error': 'No students registered'
            }
        
        # Compare against all registered encodings
        best_match = None
        best_distance = float('inf')
        
        for student_id, name, roll_number, seat_number, encoding_blob in students:
            # Unpickle the face encoding
            registered_encoding = pickle.loads(encoding_blob)
            
            # Calculate face distance (lower is better)
            distance = face_recognition.face_distance([registered_encoding], unknown_encoding)[0]
            
            if distance < best_distance:
                best_distance = distance
                best_match = {
                    'student_id': student_id,
                    'name': name,
                    'roll_number': roll_number,
                    'seat_number': seat_number
                }
        
        # Match threshold (configurable, default 0.5)
        match_threshold = 0.5
        
        if best_distance < match_threshold:
            confidence = 1.0 - best_distance  # Convert distance to confidence
            return {
                'identified': True,
                'student_id': best_match['student_id'],
                'name': best_match['name'],
                'roll_number': best_match['roll_number'],
                'seat_number': best_match['seat_number'],
                'confidence': round(confidence, 2)
            }
        else:
            return {
                'identified': False,
                'student_id': None,
                'name': None,
                'roll_number': None,
                'seat_number': None,
                'confidence': 0.0,
                'error': 'No match found'
            }
    
    def list_registered_students(self):
        """
        Get list of all registered students.
        
        Returns:
            list: List of dicts containing student information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT student_id, name, roll_number, seat_number, created_at 
            FROM students 
            ORDER BY created_at DESC
        """)
        
        students = []
        for row in cursor.fetchall():
            students.append({
                'student_id': row[0],
                'name': row[1],
                'roll_number': row[2],
                'seat_number': row[3],
                'created_at': row[4]
            })
        
        conn.close()
        return students


# Example usage
if __name__ == "__main__":
    db = StudentFaceDatabase()
    
    print("\n=== Student Face Registration System ===")
    print("1. Register new student")
    print("2. List registered students")
    print("3. Test identification")
    print("4. Exit")
    
    choice = input("\nEnter choice: ")
    
    if choice == "1":
        student_id = input("Enter student ID: ")
        name = input("Enter name: ")
        roll_number = input("Enter roll number: ")
        seat_number = input("Enter seat number: ")
        
        db.register_student_from_camera(student_id, name, roll_number, seat_number)
    
    elif choice == "2":
        students = db.list_registered_students()
        print(f"\n=== Registered Students ({len(students)}) ===")
        for student in students:
            print(f"ID: {student['student_id']}, Name: {student['name']}, "
                  f"Roll: {student['roll_number']}, Seat: {student['seat_number']}")
    
    elif choice == "3":
        print("\nStarting camera for identification test...")
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = db.identify_student(frame)
            
            display_frame = frame.copy()
            if result['identified']:
                text = f"{result['name']} ({result['confidence']:.2f})"
                color = (0, 255, 0)
            else:
                text = "Unknown"
                color = (0, 0, 255)
            
            cv2.putText(display_frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Identification Test", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
