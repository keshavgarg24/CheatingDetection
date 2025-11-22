"""
COMPLETE EXAM MONITORING SYSTEM - ALL-IN-ONE
Single file with all detection modules, violation system, and clean UI
Implements all requirements without external dependencies
"""

import cv2
import numpy as np
import time
import sqlite3
import mediapipe as mp
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any

# Import configurations
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, SHOW_WINDOW,
    MAX_STRIKES, LOOKING_AWAY_TIMEOUT,
    BOUNDARY_MARGIN, SOFT_SCORE_TO_STRIKE, EXAM_VIOLATIONS,
    VIOLATION_COOLDOWNS, USE_YOLO, EYE_GAZE_ENABLED, POSTURE_DETECTION_ENABLED,
    MOTION_DETECTION_ENABLED, WRITING_CONFIDENCE_THRESHOLD, FRAME_QUALITY_THRESHOLD,
    HEAD_TURN_RATIO, LOOK_DOWN_RATIO, HAND_NEAR_FACE_RATIO, LAP_ZONE_BOTTOM_RATIO,
    HAND_EVENT_REPEAT, HAND_GESTURE_WINDOW, HAND_GESTURE_X_PIXELS,
    USE_AUDIO_ALERTS, AUDIO_ALERTS, AUDIO_VOICE_SPEED, AUDIO_VOLUME,
    SHOW_STRIKE_ALERTS, SHOW_INSTANT_FLAG_ALERT, ALERT_DISPLAY_DURATION,
    STRIKE_ALERT_COLOR, INSTANT_FLAG_COLOR, ALERT_FONT_SIZE
)
from db import init_db, ensure_student, log_violation, set_strikes, get_status

# Try to import optional modules with graceful fallbacks
try:
    from register_students.identity_monitor import IdentityMonitor
    IDENTITY_AVAILABLE = True
except ImportError:
    print("[WARNING] Identity verification not available - install face_recognition")
    IDENTITY_AVAILABLE = False
    IdentityMonitor = None

if USE_YOLO:
    try:
        from detection.yolo_phone import load_model, detect_phones
        YOLO_AVAILABLE = True
    except ImportError:
        print("[WARNING] YOLO phone detection not available")
        YOLO_AVAILABLE = False
else:
    YOLO_AVAILABLE = False

# ==================== SESSION MANAGEMENT ====================

class ExamSession:
    """Manages exam session state and initialization"""
    
    def __init__(self, student_id: str = None):
        self.student_id = student_id
        self.session_start_time = None
        self.session_active = False
        self.recognition_locked = False
        self.verified_student_name = None
        self.current_strikes = 0
        self.current_soft_score = 0
        self.is_debarred = False
        
    def initialize_with_student(self, student_id: str, student_name: str = None):
        """Initialize fresh exam session for detected student"""
        self.student_id = student_id
        print(f"[SESSION] Initializing fresh exam session for {student_name or student_id}...")
        
        # Database cleanup and initialization
        init_db()
        ensure_student(self.student_id, name=student_name, seat_no=None)
        self._clear_previous_session_data()
        set_strikes(self.student_id, 0)
        
        # Reset session state
        self.session_start_time = time.monotonic()
        self.session_active = True
        self.recognition_locked = True
        self.verified_student_name = student_name or f"Student_{student_id}"
        self.current_strikes = 0
        self.current_soft_score = 0
        self.is_debarred = False
        
        print(f"[SESSION] Fresh session initialized for {self.verified_student_name} ({self.student_id})")
        return True
        
    def initialize_fresh_session(self):
        """Initialize session without specific student (legacy method)"""
        if not self.student_id:
            print("[SESSION] Waiting for student detection before initializing session...")
            return False
        return self.initialize_with_student(self.student_id)
        
    def _clear_previous_session_data(self):
        """Clear all previous session data from database"""
        if not self.student_id:
            return
        try:
            with sqlite3.connect("monitoring.sqlite") as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM violations WHERE student_id = ?", (self.student_id,))
                cursor.execute("""
                    UPDATE monitoring_status 
                    SET strikes = 0, status = 'normal', last_update = ?
                    WHERE student_id = ?
                """, (datetime.now(timezone.utc).isoformat(), self.student_id))
                conn.commit()
        except Exception as e:
            print(f"[ERROR] Failed to clear session data: {e}")
    
    def lock_student_identity(self, student_name: str):
        """Lock student identity after recognition"""
        if not self.recognition_locked:
            self.recognition_locked = True
            self.verified_student_name = student_name
            if self.student_id:
                print(f"[SESSION] Identity locked: {student_name} ({self.student_id})")
            return True
        return False

# ==================== VIOLATION SYSTEM ====================

class ViolationSystem:
    """Enhanced violation system with instant flagging and proximity zones"""
    
    def __init__(self, student_id: str = None):
        self.student_id = student_id
        self.last_violation_time = {}
        self.violation_start_times = {}
        self.current_violation_reasons = []  # Track active violation reasons
        self.instant_flag_triggered = False  # For phone/material detection
        self.proximity_violations = set()  # Track faces too close to student
        
    def set_student_id(self, student_id: str):
        """Set student ID after detection"""
        self.student_id = student_id
        
    def reset_session(self):
        """Reset for new session"""
        self.last_violation_time = {}
        self.violation_start_times = {}
        
    def _check_cooldown(self, violation_type: str) -> bool:
        """Check if violation is not in cooldown period"""
        if violation_type not in self.last_violation_time:
            return True
        cooldown_period = VIOLATION_COOLDOWNS.get(violation_type, 10.0)
        time_since_last = time.monotonic() - self.last_violation_time[violation_type]
        return time_since_last >= cooldown_period
    
    def process_violation(self, session: ExamSession, violation_type: str, details: str = "", 
                         severity: str = "moderate") -> Dict:
        """Process a violation with proper cooldown"""
        if session.is_debarred or not self._check_cooldown(violation_type):
            return {'action': 'ignored'}
        
        violation_category = EXAM_VIOLATIONS.get(violation_type, "soft")
        
        if violation_category == "hard":
            session.current_strikes += 1
            set_strikes(self.student_id, session.current_strikes)
            
            # Add strike reason for display
            self.add_strike_reason(violation_type, details)
            
            # Play audio alert for strike
            self._play_audio_alert(session.current_strikes)
            
            result = {
                'action': 'strike_added',
                'strikes': session.current_strikes,
                'soft_score': session.current_soft_score
            }
            if session.current_strikes >= MAX_STRIKES:
                session.is_debarred = True
                result['action'] = 'debarred'
        
        elif violation_category == "debar":
            session.is_debarred = True
            result = {'action': 'debarred'}
        
        else:  # soft violation
            session.current_soft_score += 1
            result = {
                'action': 'soft_violation',
                'strikes': session.current_strikes,
                'soft_score': session.current_soft_score
            }
            if session.current_soft_score >= SOFT_SCORE_TO_STRIKE:
                session.current_soft_score = 0
                session.current_strikes += 1
                set_strikes(self.student_id, session.current_strikes)
                
                # Add strike reason for display
                self.add_strike_reason(f"soft_to_{violation_type}", "Multiple soft violations")
                
                # Play audio alert for strike
                self._play_audio_alert(session.current_strikes)
                
                result['action'] = 'soft_to_strike'
                result['strikes'] = session.current_strikes
                if session.current_strikes >= MAX_STRIKES:
                    session.is_debarred = True
                    result['action'] = 'debarred'
        
        # Log violation
        try:
            log_violation(self.student_id, violation_type, f"{details} | Action: {result['action']}")
        except Exception as e:
            print(f"[ERROR] Failed to log violation: {e}")
        
        self.last_violation_time[violation_type] = time.monotonic()
        return result
    
    def process_instant_flag(self, session: ExamSession, violation_type: str, details: str = "") -> Dict:
        """Process violations that immediately flag/debar student (phone, materials)"""
        if session.is_debarred:
            return {'action': 'ignored'}
        
        # Instant flag = immediate disbarment
        session.is_debarred = True
        self.instant_flag_triggered = True
        
        # Play instant flag audio alert
        self._play_instant_flag_alert(violation_type)
        
        # Add violation reason to current display
        self.current_violation_reasons.append({
            'type': 'FLAGGED',
            'reason': f"{violation_type.replace('_', ' ').title()} detected",
            'timestamp': time.time(),
            'severity': 'critical',
            'is_instant_flag': True
        })
        
        result = {'action': 'instant_flagged', 'reason': violation_type}
        
        # Log violation
        try:
            log_violation(self.student_id, f"INSTANT_FLAG_{violation_type}", f"{details} | IMMEDIATELY FLAGGED")
        except Exception as e:
            print(f"[ERROR] Failed to log instant flag: {e}")
        
        return result
    
    def add_strike_reason(self, violation_type: str, details: str = ""):
        """Add strike with specific reason for display"""
        self.current_violation_reasons.append({
            'type': 'STRIKE',
            'reason': f"{violation_type.replace('_', ' ').title()}: {details}",
            'timestamp': time.time(),
            'severity': 'warning'
        })
        
        # Keep only last 5 violation reasons
        if len(self.current_violation_reasons) > 5:
            self.current_violation_reasons.pop(0)
    
    def check_face_proximity(self, student_bbox: List[int], other_faces: List, frame_dims: Tuple) -> List:
        """Check if other faces are too close to registered student's zone"""
        proximity_violations = []
        if not student_bbox or not other_faces:
            return proximity_violations
        
        frame_w, frame_h = frame_dims
        sx1, sy1, sx2, sy2 = student_bbox
        
        # Define student's personal zone (expanded around face)
        zone_expansion = 0.3  # 30% larger zone around student face
        zone_w = sx2 - sx1
        zone_h = sy2 - sy1
        expand_x = int(zone_w * zone_expansion)
        expand_y = int(zone_h * zone_expansion)
        
        zone_x1 = max(0, sx1 - expand_x)
        zone_y1 = max(0, sy1 - expand_y)
        zone_x2 = min(frame_w, sx2 + expand_x)
        zone_y2 = min(frame_h, sy2 + expand_y)
        
        # Check each other face against student zone
        for i, other_face in enumerate(other_faces):
            ox1, oy1, ox2, oy2 = other_face
            
            # Check for overlap with student's zone
            if not (ox2 < zone_x1 or ox1 > zone_x2 or oy2 < zone_y1 or oy1 > zone_y2):
                # Calculate overlap percentage
                overlap_x1 = max(zone_x1, ox1)
                overlap_y1 = max(zone_y1, oy1)
                overlap_x2 = min(zone_x2, ox2)
                overlap_y2 = min(zone_y2, oy2)
                
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                other_face_area = (ox2 - ox1) * (oy2 - oy1)
                
                if overlap_area > 0 and other_face_area > 0:
                    overlap_percent = overlap_area / other_face_area
                    
                    if overlap_percent > 0.1:  # 10% overlap threshold
                        proximity_violations.append({
                            'face_id': i,
                            'bbox': other_face,
                            'overlap_percent': overlap_percent,
                            'distance': self._calculate_face_distance(student_bbox, other_face)
                        })
        
        return proximity_violations
    
    def _calculate_face_distance(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate distance between face centers"""
        c1_x = (bbox1[0] + bbox1[2]) / 2
        c1_y = (bbox1[1] + bbox1[3]) / 2
        c2_x = (bbox2[0] + bbox2[2]) / 2
        c2_y = (bbox2[1] + bbox2[3]) / 2
        
        return ((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2) ** 0.5
    
    def _play_audio_alert(self, strike_count: int):
        """Play audio alert for strikes"""
        if not USE_AUDIO_ALERTS:
            return
            
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', int(200 * AUDIO_VOICE_SPEED))
            engine.setProperty('volume', AUDIO_VOLUME)
            
            if strike_count == 1:
                engine.say(AUDIO_ALERTS["first_strike"])
            elif strike_count == 2:
                engine.say(AUDIO_ALERTS["second_strike"])
            elif strike_count >= 3:
                engine.say(AUDIO_ALERTS["third_strike"])
            
            engine.runAndWait()
        except ImportError:
            print("[INFO] pyttsx3 not available for audio alerts")
        except Exception as e:
            print(f"[ERROR] Audio alert failed: {e}")
    
    def _play_instant_flag_alert(self, violation_type: str):
        """Play audio alert for instant flagging"""
        if not USE_AUDIO_ALERTS:
            return
            
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', int(200 * AUDIO_VOICE_SPEED))
            engine.setProperty('volume', AUDIO_VOLUME)
            
            if violation_type == "phone_detected":
                engine.say(AUDIO_ALERTS["phone_detected"])
            else:
                engine.say(AUDIO_ALERTS["instant_flag"])
            
            engine.runAndWait()
        except ImportError:
            print("[INFO] pyttsx3 not available for audio alerts")
        except Exception as e:
            print(f"[ERROR] Instant flag audio alert failed: {e}")
    
    def start_continuous_violation(self, violation_type: str):
        """Start tracking continuous violation"""
        if violation_type not in self.violation_start_times:
            self.violation_start_times[violation_type] = time.monotonic()
    
    def end_continuous_violation(self, session: ExamSession, violation_type: str):
        """End continuous violation and check duration"""
        if violation_type not in self.violation_start_times:
            return None
        
        start_time = self.violation_start_times.pop(violation_type)
        duration = time.monotonic() - start_time
        
        thresholds = {
            'looking_away': LOOKING_AWAY_TIMEOUT,
            'head_turned': LOOKING_AWAY_TIMEOUT,
            'looking_down': 15.0
        }
        
        threshold = thresholds.get(violation_type, 10.0)
        if duration >= threshold:
            return self.process_violation(
                session, violation_type,
                f"Sustained {violation_type.replace('_', ' ')} for {duration:.1f}s"
            )
        return None

# ==================== DETECTION SYSTEM ====================

class AllInOneDetector:
    """Complete detection system with all modules"""
    
    def __init__(self, student_id: str = None):
        self.student_id = student_id
        self.frame_count = 0
        
        # Initialize MediaPipe
        self.mp_face = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        # Initialize YOLO if available
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = load_model()
                print("[DETECTION] YOLO phone detector loaded")
            except Exception as e:
                print(f"[WARNING] YOLO failed to load: {e}")
        
        # Initialize identity monitor if available (without seat assignments - will identify any registered student)
        self.identity_monitor = None
        if IDENTITY_AVAILABLE:
            try:
                # Initialize without seat assignments so it can identify any registered student
                # Seat assignments are optional and only used for verification, not identification
                self.identity_monitor = IdentityMonitor(
                    seat_assignments=None,  # No seat assignments - identify any registered student
                    verification_interval=5  # Check more frequently during detection phase
                )
                print("[DETECTION] Identity monitor loaded (will identify any registered student)")
                
                # Check if there are registered students
                try:
                    from register_students.student_registration import StudentFaceDatabase
                    face_db = StudentFaceDatabase()
                    registered_students = face_db.list_registered_students()
                    if registered_students:
                        print(f"[DETECTION] Found {len(registered_students)} registered student(s) in database:")
                        for student in registered_students[:5]:  # Show first 5
                            print(f"  - {student['name']} (ID: {student['student_id']}, Roll: {student['roll_number']})")
                        if len(registered_students) > 5:
                            print(f"  ... and {len(registered_students) - 5} more")
                    else:
                        print("[WARNING] No registered students found in database!")
                        print("[WARNING] Register students using: python register_students/student_registration.py")
                except Exception as e:
                    print(f"[WARNING] Could not check registered students: {e}")
                    
            except Exception as e:
                print(f"[WARNING] Identity monitor failed to initialize: {e}")
                import traceback
                traceback.print_exc()
        
        # Motion detection variables
        self.prev_frame = None
        self.motion_history = deque(maxlen=10)
        
        # Hand gesture tracking
        self.hand_events = 0
        self.last_hand_near_face = False
        self.last_hand_in_lap = False
        self.wrist_x_history = deque(maxlen=HAND_GESTURE_WINDOW)
        
        print(f"[DETECTION] All-in-one detector initialized for {student_id or 'dynamic detection'}")
    
    def set_student_id(self, student_id: str):
        """Set student ID after detection"""
        self.student_id = student_id
        # Identity monitor doesn't need reinitialization - it already works without seat assignments
        if IDENTITY_AVAILABLE and self.identity_monitor:
            print(f"[DETECTION] Student ID set to {student_id} - identity monitor will continue verification")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame through ALL detection modules simultaneously"""
        
        self.frame_count += 1
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            # Initialize results
        results = {
            'frame_number': self.frame_count,
            'timestamp': time.time(),
            'faces_detected': 0,
            'face_bbox': None,
            'face_center': None,
            'keypoints': None,
            'student_id': None,  # Will be set from identity verification
            'student_name': None,
            'student_verified': False,
            'confidence': 0.0,
            'head_turned': False,
            'looking_down': False,
            'looking_away': False,
            'phone_detected': False,
            'hands_detected': 0,
            'hand_near_face': False,
            'hand_in_lap': False,
            'suspicious_gestures': False,
            'excessive_motion': False,
            'writing_detected': False,
            'frame_quality': self._assess_frame_quality(frame),
            'violations': []
        }
        
        # 1. FACE DETECTION
        with self.mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_det:
            face_results = face_det.process(rgb_frame)
            detections = face_results.detections if face_results and face_results.detections else []
            
            results['faces_detected'] = len(detections)
            results['all_face_bboxes'] = []  # Store all detected faces
            
            # Process all detected faces
            for detection in detections:
                face_bbox = self._bbox_from_detection(detection, w, h)
                results['all_face_bboxes'].append(face_bbox)
            
            if detections:
                # Get main face (largest - assumed to be the student)
                main_face = self._get_main_face(detections, w, h)
                if main_face:
                    bbox = self._bbox_from_detection(main_face, w, h)
                    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    keypoints = self._extract_keypoints(main_face, w, h)
                    
                    results.update({
                        'face_bbox': bbox,
                        'face_center': center,
                        'keypoints': keypoints
                    })
                    
                    # HEAD POSE ANALYSIS
                    head_pose_results = self._analyze_head_pose(bbox, keypoints, w, h)
                    results.update(head_pose_results)
        
        # 2. HAND DETECTION
        with self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=0,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        ) as hands:
            hand_results = hands.process(rgb_frame)
            hand_landmarks = hand_results.multi_hand_landmarks or []
            
            results['hands_detected'] = len(hand_landmarks)
            
            if hand_landmarks:
                hand_analysis = self._analyze_hands(hand_landmarks, results.get('face_center'), w, h)
                results.update(hand_analysis)
        
        # 3. IDENTITY VERIFICATION (identify registered students from database)
        if self.identity_monitor and results['faces_detected'] > 0 and results.get('face_bbox'):
            try:
                # Extract face region for better recognition
                x1, y1, x2, y2 = results['face_bbox']
                # Add padding around face
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                face_region = frame[y1:y2, x1:x2]
                
                # Skip if face region is too small
                if face_region.shape[0] < 50 or face_region.shape[1] < 50:
                    results['identity_status'] = 'face_too_small'
                    return results
                
                # Pass seat_id - use student_id if available, otherwise use a placeholder
                # This allows the identity monitor to work without seat assignments
                seat_id = self.student_id or "DEFAULT_SEAT"
                
                # Force fresh verification during detection phase (bypass throttling for first few seconds)
                # Reset verification if we haven't identified yet
                if not self.student_id and self.frame_count % 30 == 1:
                    # Force a fresh check every 30 frames during detection phase
                    self.identity_monitor.reset_verification()
                
                identity_result = self.identity_monitor.verify_student_identity(face_region, seat_id)
                
                # Extract student information from identity result
                status = identity_result.get('status', 'unknown')
                student_info = identity_result.get('student')
                throttled = identity_result.get('throttled', False)
                
                if student_info and status in ['verified', 'wrong_seat']:
                    # Student was identified (either verified or wrong seat)
                    identified_student_id = identity_result.get('identified_student_id') or student_info.get('student_id')
                    results.update({
                        'student_verified': status == 'verified',  # Only verified if status is 'verified'
                        'student_id': identified_student_id,  # Extract student_id properly
                        'student_name': student_info.get('name'),
                        'confidence': identity_result.get('confidence', 0.0),
                        'identity_status': status
                    })
                else:
                    # No student identified yet
                    results.update({
                        'student_verified': False,
                        'student_id': None,
                        'student_name': None,
                        'confidence': 0.0,
                        'identity_status': status
                    })
                    # Only log significant identity verification events (not debug frame-by-frame)
                    if self.frame_count % 180 == 0 and not throttled and status == 'verified':
                        error = identity_result.get('error', '')
                        message = identity_result.get('message', '')
                        # Only log significant events - not debug frame-by-frame
                        if status == 'unidentified' and self.frame_count % 300 == 0:
                            if error == 'No students registered':
                                print(f"[INFO] No registered students found. Register students first.")
                            elif error == 'No face detected':
                                pass  # Don't log face detection issues repeatedly
                        elif status == 'pending':
                            pass  # Don't log throttled checks
                        
            except Exception as e:
                print(f"[ERROR] Identity verification failed: {e}")
                import traceback
                traceback.print_exc()
                results['identity_status'] = 'error'
        
        # 4. OBJECT DETECTION (YOLO) - Only cheating devices
        if self.yolo_model:  
            try:
                cheating_devices = detect_phones(self.yolo_model, frame, confidence=0.45)  # Higher confidence
                
                # Filter for actual cheating devices only
                phones_detected = []
                for device in cheating_devices:
                    device_name = device.get('class_name', '').lower()
                    # Only flag phones, laptops, and tablets (not wires, chargers, etc.)
                    if any(keyword in device_name for keyword in ['phone', 'laptop', 'tablet']):
                        phones_detected.append(device)
                
                results['phone_detected'] = len(phones_detected) > 0
                results['detected_devices'] = phones_detected
                
                if phones_detected:
                    device_names = [d.get('class_name', 'device') for d in phones_detected]
                    print(f"[CHEATING DEVICE] Detected: {', '.join(device_names)}")
                    
            except Exception as e:
                print(f"[ERROR] Device detection failed: {e}")
        
        # 5. MOTION ANALYSIS
        motion_results = self._analyze_motion(frame)
        results.update(motion_results)
        
        # 6. POSTURE DETECTION
        if POSTURE_DETECTION_ENABLED:
            posture_results = self._analyze_posture(rgb_frame)
            results.update(posture_results)
        
        return results
    
    def _assess_frame_quality(self, frame: np.ndarray) -> Dict:
        """Assess frame quality"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        
        blur_quality = min(blur_score / 100, 1.0)
        brightness_quality = 1.0 - abs(brightness - 128) / 128
        overall_quality = (blur_quality + brightness_quality) / 2
        
        return {
            'blur_score': blur_score,
            'brightness': brightness,
            'quality_score': overall_quality,
            'is_acceptable': overall_quality >= FRAME_QUALITY_THRESHOLD
        }
    
    def _get_main_face(self, detections: List, w: int, h: int):
        """Get largest face from detections"""
        if not detections:
            return None
        
        det_areas = []
        for det in detections:
            bbox = self._bbox_from_detection(det, w, h)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            det_areas.append((area, det))
        
        return sorted(det_areas, key=lambda x: x[0], reverse=True)[0][1]
    
    def _bbox_from_detection(self, detection, w: int, h: int) -> List[int]:
        """Extract bounding box from MediaPipe detection"""
        rel = detection.location_data.relative_bounding_box
        x1 = int(rel.xmin * w)
        y1 = int(rel.ymin * h)
        x2 = int((rel.xmin + rel.width) * w)
        y2 = int((rel.ymin + rel.height) * h)
        return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
    
    def _extract_keypoints(self, detection, w: int, h: int) -> Dict:
        """Extract facial keypoints"""
        keypoints = {}
        for idx, kp in enumerate(detection.location_data.relative_keypoints):
            keypoints[idx] = (int(kp.x * w), int(kp.y * h))
        return keypoints
    
    def _analyze_head_pose(self, bbox: List[int], keypoints: Dict, w: int, h: int) -> Dict:
        """Enhanced head pose and eye gaze analysis"""
        results = {
            'head_turned': False, 
            'looking_down': False,
            'looking_away': False,
            'eye_gaze_direction': 'center',
            'head_turn_angle': 0,
            'gaze_deviation_score': 0.0
        }
        
        if not keypoints or len(keypoints) < 3:
            return results
        
        x1, y1, x2, y2 = bbox
        bbox_w = max(1, x2 - x1)
        bbox_h = max(1, y2 - y1)
        
        # MediaPipe face keypoints
        RIGHT_EYE, LEFT_EYE, NOSE_TIP = 0, 1, 2
        if all(k in keypoints for k in [RIGHT_EYE, LEFT_EYE, NOSE_TIP]):
            ex_r, ey_r = keypoints[RIGHT_EYE]
            ex_l, ey_l = keypoints[LEFT_EYE]
            nx, ny = keypoints[NOSE_TIP]
            
            # Enhanced head turn detection
            mid_eye_x = (ex_r + ex_l) / 2.0
            mid_eye_y = (ey_r + ey_l) / 2.0
            horiz_offset = abs(nx - mid_eye_x) / float(bbox_w)
            head_turn_angle = horiz_offset * 90  # Convert to approximate degrees
            
            results['head_turn_angle'] = head_turn_angle
            results['head_turned'] = horiz_offset > HEAD_TURN_RATIO
            
            # Enhanced looking down detection
            bbox_cy = (y1 + y2) / 2.0
            nose_eye_offset = (ny - mid_eye_y) / float(bbox_h)
            down_offset = (ny - bbox_cy) / float(bbox_h)
            results['looking_down'] = nose_eye_offset > LOOK_DOWN_RATIO or down_offset > 0.15
            
            # Real eye gaze analysis
            gaze_vector_x = nx - mid_eye_x
            gaze_vector_y = ny - mid_eye_y
            
            # Normalize by face size
            if bbox_w > 0 and bbox_h > 0:
                norm_gaze_x = gaze_vector_x / bbox_w
                norm_gaze_y = gaze_vector_y / bbox_h
                gaze_deviation = math.sqrt(norm_gaze_x**2 + norm_gaze_y**2)
                results['gaze_deviation_score'] = gaze_deviation
                
                # Determine gaze direction
                threshold = 0.12
                if abs(norm_gaze_x) > threshold:
                    if norm_gaze_x > 0:
                        results['eye_gaze_direction'] = 'right'
                    else:
                        results['eye_gaze_direction'] = 'left'
                elif abs(norm_gaze_y) > threshold:
                    if norm_gaze_y > 0:
                        results['eye_gaze_direction'] = 'down'
                    else:
                        results['eye_gaze_direction'] = 'up'
                
                # Enhanced looking away detection (combine multiple factors)
                looking_away_indicators = [
                    results['head_turned'],           # Head turned
                    gaze_deviation > 0.4,            # Eyes looking significantly away
                    head_turn_angle > 35,            # Strong head turn
                    results['eye_gaze_direction'] in ['left', 'right']  # Eyes looking sideways
                ]
                
                results['looking_away'] = sum(looking_away_indicators) >= 2
        
        return results
    
    def _analyze_hands(self, hand_landmarks: List, face_center: Tuple, w: int, h: int) -> Dict:
        """Analyze hand behavior"""
        results = {
            'hand_near_face': False,
            'hand_in_lap': False,
            'suspicious_gestures': False,
            'writing_detected': False
        }
        
        if not hand_landmarks:
            return results
        
        for hand in hand_landmarks:
            wrist = hand.landmark[0]
            px = int(wrist.x * w)
            py = int(wrist.y * h)
            
            # Check hand near face
            if face_center:
                distance = np.hypot(px - face_center[0], py - face_center[1])
                if distance < 150:  # pixels
                    results['hand_near_face'] = True
            
            # Check hand in lap zone
            if py > int(LAP_ZONE_BOTTOM_RATIO * h):
                results['hand_in_lap'] = True
            
            # Track wrist movement for gesture detection
            self.wrist_x_history.append(px)
            if len(self.wrist_x_history) >= self.wrist_x_history.maxlen:
                span = max(self.wrist_x_history) - min(self.wrist_x_history)
                if span >= HAND_GESTURE_X_PIXELS:
                    results['suspicious_gestures'] = True
                    self.wrist_x_history.clear()
        
        # Track hand events for violations
        if results['hand_near_face'] and not self.last_hand_near_face:
            self.hand_events += 1
        if results['hand_in_lap'] and not self.last_hand_in_lap:
            self.hand_events += 1
        
        self.last_hand_near_face = results['hand_near_face']
        self.last_hand_in_lap = results['hand_in_lap']
        
            # Check if writing (simplified detection)
        if results['hand_in_lap'] and not results['hand_near_face']:
            results['writing_detected'] = True
            
        # Hand analysis - no verbose logging
        
        return results
    
    def _analyze_motion(self, frame: np.ndarray) -> Dict:
        """Analyze motion patterns"""
        results = {'excessive_motion': False, 'motion_level': 0.0}
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(gray, self.prev_frame)
                motion_level = np.mean(diff) / 255.0
                
                self.motion_history.append(motion_level)
                
                # Check for excessive motion
                if len(self.motion_history) >= 5:
                    avg_motion = np.mean(self.motion_history)
                    results['motion_level'] = avg_motion
                    results['excessive_motion'] = avg_motion > 0.15
            
            self.prev_frame = gray.copy()
            
        except Exception as e:
            print(f"[ERROR] Motion analysis failed: {e}")
        
        return results
    
    def _analyze_posture(self, rgb_frame: np.ndarray) -> Dict:
        """Analyze body posture"""
        results = {'posture_violations': []}
        
        try:
            with self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5
            ) as pose:
                pose_results = pose.process(rgb_frame)
                
                if pose_results.pose_landmarks:
                    # Simple posture analysis - check shoulder alignment
                    landmarks = pose_results.pose_landmarks.landmark
                    
                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]
                    
                    # Calculate shoulder tilt
                    shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
                    if shoulder_diff > 0.1:  # Threshold for excessive lean
                        results['posture_violations'].append({
                            'type': 'excessive_lean',
                            'severity': 'moderate'
                        })
        
        except Exception as e:
            print(f"[ERROR] Posture analysis failed: {e}")
        
        return results

# ==================== UI DISPLAY SYSTEM ====================

class CleanUI:
    """Clean, non-overlapping UI display"""
    
    def __init__(self, frame_width=1280, frame_height=720):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            'normal': (102, 255, 102),
            'warning': (0, 255, 255),
            'critical': (0, 0, 255),
            'info': (255, 255, 255),
            'boundary': (255, 255, 0)
        }
    
    def draw_interface(self, frame: np.ndarray, session: ExamSession, detection_results: Dict, 
                      elapsed_time: float, violations_display: List = None) -> np.ndarray:
        """Draw complete clean interface with alerts"""
        
        # Draw student boundary if face detected
        if detection_results.get('face_bbox'):
            self._draw_boundary(frame, detection_results['face_bbox'])
        
        # Draw student info above head
        if session.verified_student_name and detection_results.get('face_bbox'):
            # Format: "Name (ID)" instead of just ID
            student_display = f"{session.verified_student_name} ({session.student_id})"
            self._draw_student_info(frame, detection_results['face_bbox'], 
                                  student_display, session.student_id)
        
        # Draw status panel
        self._draw_status_panel(frame, session, elapsed_time)
        
        # Draw big alert messages for instant flags and strikes
        if violations_display:
            self._draw_big_alerts(frame, violations_display, session)
        
        # Draw current violations
        if violations_display:
            self._draw_violations(frame, violations_display)
        
        # Draw violation reasons from violation system
        self._draw_violation_reasons(frame, violations_display)
        
        # Draw lap zone line
        self._draw_lap_zone(frame)
        
        return frame
    
    def _draw_violation_reasons(self, frame: np.ndarray, violations_display: List):
        """Draw specific violation reasons on screen"""
        if not violations_display:
            return
        
        # Find violation system messages
        violation_reasons = []
        for violation in violations_display:
            if hasattr(violation, 'get') and 'reason' in violation:
                violation_reasons.append(violation)
        
        if not violation_reasons:
            return
        
        # Draw violation reasons in top-right corner
        start_y = 30
        for i, violation in enumerate(violation_reasons[-3:]):  # Show last 3 reasons
            reason_text = f"{violation['type']}: {violation['reason']}"
            color = (0, 0, 255) if violation['severity'] == 'critical' else (0, 255, 255)
            
            # Get text size for background
            (text_w, text_h), baseline = cv2.getTextSize(
                reason_text, self.font, 0.6, 2
            )
            
            # Draw background rectangle
            bg_x1 = self.frame_width - text_w - 20
            bg_y1 = start_y + (i * 35) - text_h - 5
            bg_x2 = self.frame_width - 10
            bg_y2 = start_y + (i * 35) + 10
            
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)
            
            # Draw text
            cv2.putText(frame, reason_text, 
                       (bg_x1 + 5, start_y + (i * 35)), 
                       self.font, 0.6, color, 2, cv2.LINE_AA)
    
    def _draw_big_alerts(self, frame: np.ndarray, violations_display: List, session: ExamSession):
        """Draw big alert messages for strikes and instant flags"""
        if not violations_display:
            return
        
        current_time = time.time()
        alerts_to_show = []
        
        # Find recent violations that need big alerts
        for violation in violations_display:
            if hasattr(violation, 'get'):
                if violation.get('is_instant_flag', False):
                    if current_time - violation.get('timestamp', 0) < ALERT_DISPLAY_DURATION:
                        alerts_to_show.append({
                            'text': f"INSTANT FLAG: {violation['reason']}",
                            'color': INSTANT_FLAG_COLOR,
                            'size': ALERT_FONT_SIZE * 1.5
                        })
                elif violation.get('type') == 'STRIKE':
                    if current_time - violation.get('timestamp', 0) < ALERT_DISPLAY_DURATION:
                        alerts_to_show.append({
                            'text': f"STRIKE {session.current_strikes}: {violation['reason']}",
                            'color': STRIKE_ALERT_COLOR,
                            'size': ALERT_FONT_SIZE
                        })
        
        # Draw big center alerts
        for i, alert in enumerate(alerts_to_show[-2:]):  # Show max 2 alerts
            text = alert['text']
            color = alert['color']
            font_size = alert['size']
            
            # Get text dimensions
            (text_w, text_h), baseline = cv2.getTextSize(
                text, self.font, font_size, 3
            )
            
            # Center position
            center_x = self.frame_width // 2
            center_y = (self.frame_height // 2) + (i * 80) - 40
            
            # Background rectangle
            bg_x1 = center_x - text_w // 2 - 20
            bg_y1 = center_y - text_h - 10
            bg_x2 = center_x + text_w // 2 + 20
            bg_y2 = center_y + 10
            
            # Draw background with transparency effect
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Draw border
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 4)
            
            # Draw text
            cv2.putText(frame, text, 
                       (center_x - text_w // 2, center_y), 
                       self.font, font_size, color, 3, cv2.LINE_AA)
    
    def _draw_boundary(self, frame: np.ndarray, bbox: List[int]):
        """Draw clean boundary around student"""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['boundary'], 2)
        
        # Corner markers
        corner_length = 15
        corner_thickness = 3
        
        # Draw corner markers
        corners = [
            (x1, y1), (x2, y1), (x1, y2), (x2, y2)
        ]
        
        for i, (cx, cy) in enumerate(corners):
            if i == 0:  # Top-left
                cv2.line(frame, (cx, cy), (cx + corner_length, cy), self.colors['boundary'], corner_thickness)
                cv2.line(frame, (cx, cy), (cx, cy + corner_length), self.colors['boundary'], corner_thickness)
            elif i == 1:  # Top-right
                cv2.line(frame, (cx, cy), (cx - corner_length, cy), self.colors['boundary'], corner_thickness)
                cv2.line(frame, (cx, cy), (cx, cy + corner_length), self.colors['boundary'], corner_thickness)
            elif i == 2:  # Bottom-left
                cv2.line(frame, (cx, cy), (cx + corner_length, cy), self.colors['boundary'], corner_thickness)
                cv2.line(frame, (cx, cy), (cx, cy - corner_length), self.colors['boundary'], corner_thickness)
            elif i == 3:  # Bottom-right
                cv2.line(frame, (cx, cy), (cx - corner_length, cy), self.colors['boundary'], corner_thickness)
                cv2.line(frame, (cx, cy), (cx, cy - corner_length), self.colors['boundary'], corner_thickness)
    
    def _draw_student_info(self, frame: np.ndarray, bbox: List[int], 
                          student_name: str, student_id: str):
        """Draw student info above head"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        info_y = max(y1 - 50, 30)
        
        info_text = f"{student_name} | {student_id}"
        (text_width, text_height), _ = cv2.getTextSize(info_text, self.font, 0.6, 2)
        
        # Background rectangle
        bg_x1 = center_x - text_width // 2 - 10
        bg_y1 = info_y - text_height - 10
        bg_x2 = center_x + text_width // 2 + 10
        bg_y2 = info_y + 10
        
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), self.colors['info'], 1)
        
        text_x = center_x - text_width // 2
        cv2.putText(frame, info_text, (text_x, info_y), self.font, 0.6, self.colors['info'], 2)
    
    def _draw_status_panel(self, frame: np.ndarray, session: ExamSession, elapsed_time: float):
        """Draw status panel"""
        panel_x = 15
        panel_y = 30
        
        # Determine status color
        if session.is_debarred:
            status_color = self.colors['critical']
            status_text = "DEBARRED"
        elif session.current_strikes >= 2:
            status_color = self.colors['critical']
            status_text = "CRITICAL"
        elif session.current_strikes >= 1:
            status_color = self.colors['warning']
            status_text = "WARNING"
        else:
            status_color = self.colors['normal']
            status_text = "MONITORING"
        
        # Status lines
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        lines = [
            f"STATUS: {status_text}",
            f"STRIKES: {session.current_strikes}/3",
            f"SOFT: {session.current_soft_score}/3",
            f"TIME: {minutes:02d}:{seconds:02d}"
        ]
        
        for i, line in enumerate(lines):
            y_pos = panel_y + i * 25
            cv2.putText(frame, line, (panel_x, y_pos), self.font, 0.6, status_color, 2)
    
    def _draw_violations(self, frame: np.ndarray, violations: List):
        """Draw recent violations"""
        if not violations:
            return
        
        y_start = 150
        for i, violation in enumerate(violations[-3:]):  # Show last 3
            y_pos = y_start + i * 20
            color = self.colors['warning'] if violation.get('severity') == 'minor' else self.colors['critical']
            cv2.putText(frame, violation.get('message', ''), (15, y_pos), self.font, 0.45, color, 1)
    
    def _draw_lap_zone(self, frame: np.ndarray):
        """Draw lap zone line"""
        lap_y = int(LAP_ZONE_BOTTOM_RATIO * self.frame_height)
        cv2.line(frame, (0, lap_y), (self.frame_width, lap_y), (128, 128, 128), 1)
        cv2.putText(frame, "LAP ZONE", (10, lap_y - 6), self.font, 0.5, (128, 128, 128), 1)

# ==================== LIVE OUTPUT SYSTEM ====================

class LiveOutput:
    """Structured live output system"""
    
    def __init__(self, output_file: str = "live_monitoring.json"):
        self.output_file = output_file
        self.last_output_time = 0
        self.output_interval = 1.0
    
    def generate_output(self, session: ExamSession, detection_results: Dict, 
                       elapsed_time: float, force: bool = False) -> Optional[str]:
        """Generate live output in specified format"""
        
        current_time = time.monotonic()
        if not force and (current_time - self.last_output_time) < self.output_interval:
            return None
        
        # Create structured output
        output_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "student_name": session.verified_student_name or "Unknown",
            "student_roll": session.student_id,
            "student_present": "yes" if detection_results.get('faces_detected', 0) > 0 else "no",
            "boundary_violation": "yes" if detection_results.get('faces_detected', 0) > 1 else "no",
            "second_face_inside_boundary": "yes" if detection_results.get('faces_detected', 0) > 1 else "no",
            "eye_gaze_status": "looking_away" if detection_results.get('looking_away') else "normal",
            "head_pose_status": "suspicious" if detection_results.get('head_turned') or detection_results.get('looking_down') else "normal",
            "object_detected": "phone" if detection_results.get('phone_detected') else "none",
            "motion_status": "suspicious" if detection_results.get('excessive_motion') else "normal",
            "soft_score": session.current_soft_score,
            "strike_count": session.current_strikes,
            "decision": "debarred" if session.is_debarred else ("strike_issued" if session.current_strikes > 0 else "monitoring")
        }
        
        # Write to file
        try:
            with open(self.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to write live output: {e}")
        
        self.last_output_time = current_time
        return json.dumps(output_data, indent=2)

# ==================== MAIN MONITORING FUNCTION ====================

def initialize_camera():
    """Initialize camera with error handling"""
    camera_backends = [
        (cv2.CAP_AVFOUNDATION, "AVFoundation (macOS)"),
        (cv2.CAP_ANY, "Default backend")
    ]
    
    for backend, backend_name in camera_backends:
        try:
            print(f"[INFO] Trying {backend_name}...")
            cap = cv2.VideoCapture(CAMERA_INDEX, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                print(f"[INFO] Camera initialized with {backend_name}")
                return cap
            cap.release()
        except Exception as e:
            print(f"[ERROR] {backend_name} failed: {e}")
    
    print("[ERROR] Could not initialize camera")
    return None

def main():
    """Main monitoring function - COMPLETE EXAM MONITORING SYSTEM WITH DYNAMIC STUDENT DETECTION"""
    
    print("=" * 60)
    print("COMPLETE EXAM MONITORING SYSTEM - STARTING")
    print("=" * 60)
    
    # Initialize camera first
    cap = initialize_camera()
    if cap is None:
        print("[ERROR] Camera initialization failed")
        return
    
    # Initialize systems without specific student ID - will be set after detection
    session = ExamSession()  # No student ID initially
    violation_system = ViolationSystem()  # No student ID initially
    detector = AllInOneDetector()  # No student ID initially
    ui = CleanUI(FRAME_WIDTH, FRAME_HEIGHT)
    live_output = LiveOutput()
    
    print("[EXAM] Camera and detection systems ready")
    print("[EXAM] Waiting for student detection and identification...")
    
    try:
        # Student detection phase variables
        student_detected = False
        student_id = None
        student_name = None
        
        # Monitoring variables (will be used after student is detected)
        exam_start_time = None
        last_face_seen_at = time.monotonic()
        absence_start_time = None
        frame_count = 0
        violations_display = deque(maxlen=3)
        continuous_violations = {}
        
        while True:
            frame_start = time.perf_counter()
            frame_count += 1
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read camera frame")
                break
            
            h, w = frame.shape[:2]
            current_time = time.monotonic()
            
            # Process frame through detection modules
            detection_results = detector.process_frame(frame)
            
            # PHASE 1: STUDENT DETECTION AND IDENTIFICATION
            if not student_detected:
                # Check if student was identified through face recognition
                identified_student_id = detection_results.get('student_id')
                identity_status = detection_results.get('identity_status', 'unknown')
                
                if identified_student_id and detection_results.get('student_verified'):
                    # Student identified and verified through face recognition
                    student_id = identified_student_id
                    student_name = detection_results.get('student_name') or f"Student_{student_id}"
                    confidence = detection_results.get('confidence', 0.0)
                    student_detected = True
                    print(f"[EXAM] ✓ Student identified via face recognition: {student_name} ({student_id}) [confidence: {confidence:.2f}]")
                    
                elif identified_student_id and identity_status == 'wrong_seat':
                    # Student identified but wrong seat - still use this student
                    student_id = identified_student_id
                    student_name = detection_results.get('student_name') or f"Student_{student_id}"
                    confidence = detection_results.get('confidence', 0.0)
                    student_detected = True
                    print(f"[EXAM] ⚠ Student identified (seat mismatch): {student_name} ({student_id}) [confidence: {confidence:.2f}]")
                    print(f"[EXAM] Proceeding with monitoring...")
                    
                elif detection_results['faces_detected'] > 0:
                    # Face detected but not identified - provide helpful messages
                    if frame_count % 90 == 0:  # Print every 90 frames (every ~3 seconds at 30fps)
                        status_msg = f"Status: {identity_status}"
                        if IDENTITY_AVAILABLE and detector.identity_monitor:
                            print(f"[DETECTION] Face detected but not yet identified ({status_msg})")
                            print(f"[DETECTION] Waiting for face recognition match against registered students...")
                        else:
                            print(f"[DETECTION] Face detected but identity verification unavailable")
                            print(f"[DETECTION] Install face_recognition: pip install face-recognition")
                    
                    # Don't auto-generate student ID - wait for proper identification
                    # Only proceed if no identity system is available
                    if not IDENTITY_AVAILABLE:
                        # Fallback: generate ID only if face_recognition is not available
                        student_id = f"ID_{int(time.time())}"
                        student_name = f"Student_{student_id}"
                        student_detected = True
                        print(f"[EXAM] Face detected (identity verification unavailable): {student_name} ({student_id})")
                
                if student_detected:
                    # Initialize session with detected student
                    session.initialize_with_student(student_id, student_name)
                    violation_system.set_student_id(student_id)
                    detector.set_student_id(student_id)
                    exam_start_time = time.monotonic()
                    print("[EXAM] *** MONITORING ACTIVE - STRIKES ENABLED ***")
                
                # Show detection UI during student detection phase
                if frame_count % 30 == 0:
                    # Waiting for student detection - only log periodically
                    if frame_count % 300 == 0:  # Every 10 seconds at 30fps
                        print(f"[INFO] Waiting for student detection... (Faces: {detection_results['faces_detected']})")
                
                # Simple UI during detection phase
                if detection_results['faces_detected'] > 0:
                    cv2.putText(frame, "Student detected - initializing...", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Looking for student...", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if SHOW_WINDOW:
                    cv2.imshow("Exam Monitor", frame)
                
            # PHASE 2: ACTIVE MONITORING (after student is detected and session initialized)
            else:
                # Only log significant events (violations, strikes, debarment)
                
                # Update presence tracking
                if detection_results['faces_detected'] > 0:
                    last_face_seen_at = current_time
                    absence_start_time = None
                else:
                    if absence_start_time is None:
                        absence_start_time = current_time
                
                # Process violations
                
                # 1. Cheating device detection (INSTANT FLAG - immediate debar)
                if detection_results['phone_detected']:
                    detected_devices = detection_results.get('detected_devices', [])
                    device_names = [d.get('class_name', 'device') for d in detected_devices]
                    device_list = ', '.join(device_names)
                    
                    result = violation_system.process_instant_flag(
                        session, "cheating_device_detected", 
                        f"Cheating device detected: {device_list} - INSTANT FLAGGING"
                    )
                    print(f"[INSTANT FLAG] Cheating device detected ({device_list}): {result}")
                    break  # Stop monitoring immediately
                
                # 2. Face proximity violations (enhanced multiple faces check)
                if detection_results['faces_detected'] > 1 and detection_results['face_bbox']:
                    # Get other faces (exclude the main student face)
                    other_faces = [bbox for bbox in detection_results['all_face_bboxes'] 
                                 if bbox != detection_results['face_bbox']]
                    
                    proximity_violations = violation_system.check_face_proximity(
                        detection_results['face_bbox'], other_faces, (frame.shape[1], frame.shape[0])
                    )
                    
                    for prox_violation in proximity_violations:
                        result = violation_system.process_violation(
                            session, "face_proximity",
                            f"Face proximity violation: {prox_violation}"
                        )
                        if result['action'] not in ['ignored']:
                            print(f"[STRIKE] Face proximity: Strike {session.current_strikes}/3")
                
                # 3. Head pose violations (looking away)
                head_turn_angle = detection_results.get('head_turn_angle', 0)
                gaze_deviation = detection_results.get('gaze_deviation_score', 0)
                
                looking_away = (
                    detection_results['head_turned'] or 
                    detection_results['looking_away'] or
                    head_turn_angle > 45
                )
                
                if looking_away:
                    if 'looking_away' not in continuous_violations:
                        continuous_violations['looking_away'] = current_time
                else:
                    if 'looking_away' in continuous_violations:
                        duration = current_time - continuous_violations['looking_away']
                        if duration >= 1.8:  # 1.8 seconds - more sensitive
                            result = violation_system.process_violation(
                                session, "looking_away_sustained",
                                f"Looking away for {duration:.1f}s (deviation: {gaze_deviation:.2f})"
                            )
                            if result['action'] not in ['ignored']:
                                print(f"[STRIKE] Looking away violation: Strike {session.current_strikes}/3")
                        del continuous_violations['looking_away']
                
                # 4. Talking/collaboration detection
                talking_indicators = [
                    detection_results.get('faces_detected', 0) > 1,  # Other person present
                    detection_results['head_turned'],                # Head turned toward person
                    detection_results['looking_away'],               # Eyes looking away
                    head_turn_angle > 35                            # Significant head turn
                ]
                
                if sum(talking_indicators) >= 3:
                    if 'talking_attempt' not in continuous_violations:
                        continuous_violations['talking_attempt'] = current_time
                    elif current_time - continuous_violations['talking_attempt'] >= 2.5:
                        result = violation_system.process_violation(
                            session, "face_proximity",
                            f"Potential talking/collaboration detected"
                        )
                        if result['action'] not in ['ignored']:
                            print(f"[STRIKE] Collaboration detected: Strike {session.current_strikes}/3")
                        del continuous_violations['talking_attempt']
                else:
                    if 'talking_attempt' in continuous_violations:
                        del continuous_violations['talking_attempt']
                
                # 5. Looking down violations
                if detection_results['looking_down']:
                    if 'looking_down' not in continuous_violations:
                        violation_system.start_continuous_violation('looking_down')
                        continuous_violations['looking_down'] = True
                else:
                    if 'looking_down' in continuous_violations:
                        result = violation_system.end_continuous_violation(session, 'looking_down')
                        del continuous_violations['looking_down']
                        if result and result['action'] not in ['ignored']:
                            print(f"[STRIKE] Extended looking down: Strike {session.current_strikes}/3")
                
                # 6. Hand violations (soft violations)
                if detector.hand_events >= HAND_EVENT_REPEAT and not detection_results['writing_detected']:
                    result = violation_system.process_violation(
                        session, "hand_suspicious", 
                        f"Suspicious hand activity (events={detector.hand_events})", "moderate"
                    )
                    if result['action'] not in ['ignored']:
                        print(f"[SOFT] Hand behavior: {session.current_soft_score}/3 soft violations")
                        detector.hand_events = 0  # Reset counter
                
                # 7. Motion violations
                if detection_results['excessive_motion']:
                    result = violation_system.process_violation(
                        session, "excessive_motion", 
                        f"Motion level: {detection_results.get('motion_level', 0):.2f}", "moderate"
                    )
                    if result['action'] not in ['ignored']:
                        print(f"[SOFT] Excessive motion: {session.current_soft_score}/3 soft violations")
                
                # 8. Posture violations
                for posture_violation in detection_results.get('posture_violations', []):
                    result = violation_system.process_violation(
                        session, f"posture_{posture_violation['type']}", 
                        "Suspicious body posture", posture_violation.get('severity', 'moderate')
                    )
                    if result['action'] != 'ignored':
                        print(f"[VIOLATION] Posture: {result}")
                
                # Add current violations to display
                if session.current_strikes > 0 or session.current_soft_score > 0:
                    violation_msg = f"Strikes: {session.current_strikes}/3, Soft: {session.current_soft_score}/3"
                    violations_display.append({
                        'message': violation_msg,
                        'severity': 'critical' if session.current_strikes >= 2 else 'warning',
                        'timestamp': current_time
                    })
                
                # Calculate elapsed time for active monitoring
                elapsed_time = current_time - exam_start_time
                
                # Combine violation display with violation system reasons
                all_violations = list(violations_display) + violation_system.current_violation_reasons
                
                # Draw complete clean UI
                display_frame = ui.draw_interface(
                    frame.copy(), session, detection_results, 
                    elapsed_time, all_violations
                )
                
                # Generate live output
                live_output.generate_output(session, detection_results, elapsed_time)
                
                # Show frame with monitoring UI
                if SHOW_WINDOW:
                    cv2.imshow("COMPLETE EXAM MONITORING SYSTEM (q=quit)", display_frame)
                
                # Check if student is debarred
                if session.is_debarred:
                    print("\n" + "=" * 60)
                    print("*** STUDENT DEBARRED - EXAM TERMINATED ***")
                    print("=" * 60)
                    
                    # Show debarred message
                    cv2.putText(display_frame, "STUDENT DEBARRED - EXAM TERMINATED", 
                               (50, FRAME_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.2, (0, 0, 255), 3, cv2.LINE_AA)
                    
                    if SHOW_WINDOW:
                        cv2.imshow("COMPLETE EXAM MONITORING SYSTEM (q=quit)", display_frame)
                        cv2.waitKey(5000)  # Show for 5 seconds
                    break
            
            # Performance info (every 150 frames - reduced logging)
            if frame_count % 150 == 0:
                frame_time = (time.perf_counter() - frame_start) * 1000
                fps = 1000 / frame_time if frame_time > 0 else 0
                # Performance logging - only periodically
                if frame_count % 600 == 0:  # Every 20 seconds at 30fps
                    print(f"[INFO] Performance: {fps:.1f} FPS")
            
            # Check for quit key
            if SHOW_WINDOW:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except Exception as e:
        print(f"[ERROR] Monitoring failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n[EXAM] Monitoring session completed")
        
        # Final session summary
        print("\n" + "=" * 60)
        print("EXAM SESSION SUMMARY")
        print("=" * 60)
        if student_detected and session.student_id:
            student_name = session.verified_student_name if session.verified_student_name else "Not recognized"
            print(f"Student: {student_name} ({session.student_id})")
            final_status = 'DEBARRED' if session.is_debarred else 'COMPLETED'
            if session.is_debarred and violation_system.instant_flag_triggered:
                final_status = 'FLAGGED (Instant)'
            print(f"Final Status: {final_status}")
            print(f"Final Strikes: {session.current_strikes}/3")
            print(f"Final Soft Score: {session.current_soft_score}/3")
        else:
            print("Student: Not detected")
            print("Final Status: NO STUDENT DETECTED")
        print("=" * 60)

if __name__ == "__main__":
    main()