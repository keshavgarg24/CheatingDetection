"""
Identity Monitoring for Exam Sessions

This module provides real-time identity verification during exams by comparing
detected faces against registered students and verifying seat assignments.
"""

import time
from register_students.student_registration import StudentFaceDatabase


class IdentityMonitor:
    """
    Monitors student identity during exams with seat assignment verification.
    
    Performs periodic identity checks (default every 30 seconds) to ensure
    the correct student is in the assigned seat and detects unauthorized individuals.
    """
    
    def __init__(self, seat_assignments=None, verification_interval=30, db_path="student_faces.sqlite"):
        """
        Initialize the IdentityMonitor.
        
        Args:
            seat_assignments: Dict mapping seat_id to expected student_id
                             Example: {"SEAT_01": "STU001", "SEAT_02": "STU002"}
            verification_interval: Seconds between identity verification checks (default 30)
            db_path: Path to student face database
        """
        self.face_db = StudentFaceDatabase(db_path)
        self.seat_assignments = seat_assignments or {}
        self.verification_interval = verification_interval
        self.last_verification_time = 0
        self.current_student = None
        
        print(f"[INFO] IdentityMonitor initialized with {len(self.seat_assignments)} seat assignments")
        print(f"[INFO] Verification interval: {verification_interval} seconds")
    
    def verify_student_identity(self, frame, seat_id):
        """
        Verify student identity and check seat assignment.
        
        Implements 30-second throttling to avoid excessive verification calls.
        Compares detected face against registered students and validates seat assignment.
        
        Args:
            frame: OpenCV frame containing student's face
            seat_id: Current seat identifier being monitored
            
        Returns:
            dict: {
                'status': str - 'verified', 'unidentified', or 'wrong_seat'
                'alert_level': str - 'none', 'medium', or 'high'
                'student': dict or None - Identified student info if verified
                'message': str - Human-readable status message
                'confidence': float - Identification confidence (0.0-1.0)
                'expected_student_id': str or None - Expected student for this seat
                'identified_student_id': str or None - Actually identified student
            }
        """
        # Throttling: only verify every verification_interval seconds
        current_time = time.time()
        if current_time - self.last_verification_time < self.verification_interval:
            # Return cached result if available
            if self.current_student:
                return {
                    'status': 'verified',
                    'alert_level': 'none',
                    'student': self.current_student,
                    'message': f"Verified: {self.current_student['name']}",
                    'confidence': self.current_student.get('confidence', 0.0),
                    'expected_student_id': self.seat_assignments.get(seat_id),
                    'identified_student_id': self.current_student['student_id'],
                    'throttled': True
                }
            else:
                return {
                    'status': 'pending',
                    'alert_level': 'none',
                    'student': None,
                    'message': 'Verification pending',
                    'confidence': 0.0,
                    'expected_student_id': self.seat_assignments.get(seat_id),
                    'identified_student_id': None,
                    'throttled': True
                }
        
        # Update last verification time
        self.last_verification_time = current_time
        
        # Identify student from frame
        identification_result = self.face_db.identify_student(frame)
        
        # Check if face was identified
        if not identification_result['identified']:
            self.current_student = None
            return {
                'status': 'unidentified',
                'alert_level': 'medium',
                'student': None,
                'message': 'Unidentified person detected',
                'confidence': 0.0,
                'expected_student_id': self.seat_assignments.get(seat_id),
                'identified_student_id': None,
                'throttled': False
            }
        
        # Student identified successfully
        identified_student_id = identification_result['student_id']
        expected_student_id = self.seat_assignments.get(seat_id)
        
        # Check seat assignment if configured
        if expected_student_id:
            if identified_student_id != expected_student_id:
                # Wrong student in this seat
                self.current_student = None
                return {
                    'status': 'wrong_seat',
                    'alert_level': 'high',
                    'student': identification_result,
                    'message': f"Wrong seat: {identification_result['name']} in seat assigned to {expected_student_id}",
                    'confidence': identification_result['confidence'],
                    'expected_student_id': expected_student_id,
                    'identified_student_id': identified_student_id,
                    'throttled': False
                }
        
        # Correct student verified
        self.current_student = identification_result
        return {
            'status': 'verified',
            'alert_level': 'none',
            'student': identification_result,
            'message': f"Verified: {identification_result['name']}",
            'confidence': identification_result['confidence'],
            'expected_student_id': expected_student_id,
            'identified_student_id': identified_student_id,
            'throttled': False
        }
    
    def get_current_student(self):
        """
        Get the currently verified student information.
        
        Returns:
            dict or None: Current student info if verified, None otherwise
        """
        return self.current_student
    
    def reset_verification(self):
        """
        Reset verification state, forcing a new check on next call.
        """
        self.last_verification_time = 0
        self.current_student = None
        print("[INFO] Identity verification state reset")
    
    def update_seat_assignment(self, seat_id, student_id):
        """
        Update or add a seat assignment.
        
        Args:
            seat_id: Seat identifier
            student_id: Expected student ID for this seat
        """
        self.seat_assignments[seat_id] = student_id
        print(f"[INFO] Seat assignment updated: {seat_id} -> {student_id}")
    
    def remove_seat_assignment(self, seat_id):
        """
        Remove a seat assignment.
        
        Args:
            seat_id: Seat identifier to remove
        """
        if seat_id in self.seat_assignments:
            del self.seat_assignments[seat_id]
            print(f"[INFO] Seat assignment removed: {seat_id}")


# Example usage
if __name__ == "__main__":
    import cv2
    
    # Example seat assignments
    seat_assignments = {
        "SEAT_01": "STU001",
        "SEAT_02": "STU002",
        "SEAT_03": "STU003"
    }
    
    # Initialize monitor
    monitor = IdentityMonitor(seat_assignments=seat_assignments, verification_interval=5)
    
    # Test with webcam
    print("\n=== Identity Verification Test ===")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    seat_id = "SEAT_01"  # Simulating monitoring of SEAT_01
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Verify identity
        result = monitor.verify_student_identity(frame, seat_id)
        
        # Display result on frame
        display_frame = frame.copy()
        
        # Color code based on alert level
        if result['alert_level'] == 'none':
            color = (0, 255, 0)  # Green
        elif result['alert_level'] == 'medium':
            color = (0, 165, 255)  # Orange
        else:  # high
            color = (0, 0, 255)  # Red
        
        # Display status
        cv2.putText(display_frame, f"Status: {result['status']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_frame, result['message'], 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if result['student']:
            cv2.putText(display_frame, f"Confidence: {result['confidence']:.2f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if result.get('throttled'):
            cv2.putText(display_frame, "(Cached)", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        cv2.imshow("Identity Monitor Test", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
