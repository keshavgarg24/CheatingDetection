"""
Advanced Body Posture Detection for Exam Monitoring

This module analyzes student body posture and movement patterns to detect
suspicious behavior including excessive leaning, turning away, and
unusual body positioning during exams.
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time


class AdvancedPostureDetector:
    """
    Advanced body posture detection for suspicious behavior monitoring.
    
    Analyzes:
    - Body lean angles and direction
    - Shoulder positioning and orientation
    - Torso stability and movement
    - Arm positioning relative to body
    - Overall posture stability over time
    """
    
    def __init__(self):
        """Initialize body posture detector."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Key pose landmarks for analysis
        self.KEY_LANDMARKS = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'nose': 0
        }
        
        # Posture history tracking
        self.posture_history = deque(maxlen=150)  # 5 seconds at 30 FPS
        self.lean_history = deque(maxlen=90)      # 3 seconds for lean detection
        self.arm_position_history = deque(maxlen=60)  # 2 seconds for arm tracking
        
        # Analysis parameters
        self.normal_lean_threshold = 15  # degrees
        self.suspicious_lean_threshold = 25  # degrees
        self.critical_lean_threshold = 40  # degrees
        
        # Stability tracking
        self.posture_instability_count = 0
        self.lean_violation_start = None
        self.arm_suspicious_start = None
        
    def extract_pose_landmarks(self, frame):
        """
        Extract pose landmarks from frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Dictionary with landmark coordinates or None if no pose detected
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = {}
        h, w = frame.shape[:2]
        
        for name, idx in self.KEY_LANDMARKS.items():
            if idx < len(results.pose_landmarks.landmark):
                landmark = results.pose_landmarks.landmark[idx]
                landmarks[name] = {
                    'x': landmark.x * w,
                    'y': landmark.y * h,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
        
        return landmarks
    
    def calculate_body_lean(self, landmarks):
        """
        Calculate body lean angle from shoulder and hip positions.
        
        Args:
            landmarks: Pose landmarks dictionary
            
        Returns:
            Dictionary with lean analysis
        """
        if not all(key in landmarks for key in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            return None
        
        # Calculate shoulder line angle
        left_shoulder = landmarks['left_shoulder']
        right_shoulder = landmarks['right_shoulder']
        shoulder_angle = np.arctan2(
            right_shoulder['y'] - left_shoulder['y'],
            right_shoulder['x'] - left_shoulder['x']
        ) * 180 / np.pi
        
        # Calculate hip line angle
        left_hip = landmarks['left_hip']
        right_hip = landmarks['right_hip']
        hip_angle = np.arctan2(
            right_hip['y'] - left_hip['y'],
            right_hip['x'] - left_hip['x']
        ) * 180 / np.pi
        
        # Average lean angle
        avg_lean_angle = abs((shoulder_angle + hip_angle) / 2)
        
        # Determine lean direction
        lean_direction = 'left' if shoulder_angle > 0 else 'right'
        
        # Calculate torso alignment (difference between shoulder and hip angles)
        torso_alignment = abs(shoulder_angle - hip_angle)
        
        return {
            'lean_angle': avg_lean_angle,
            'lean_direction': lean_direction,
            'shoulder_angle': shoulder_angle,
            'hip_angle': hip_angle,
            'torso_alignment': torso_alignment,
            'confidence': self._calculate_lean_confidence(landmarks)
        }
    
    def analyze_arm_positioning(self, landmarks):
        """
        Analyze arm positioning for suspicious behavior.
        
        Args:
            landmarks: Pose landmarks dictionary
            
        Returns:
            Dictionary with arm analysis
        """
        if not all(key in landmarks for key in ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']):
            return None
        
        suspicions = []
        
        # Analyze left arm
        left_analysis = self._analyze_single_arm(
            landmarks['left_shoulder'],
            landmarks['left_elbow'],
            landmarks['left_wrist'],
            'left'
        )
        
        # Analyze right arm
        right_analysis = self._analyze_single_arm(
            landmarks['right_shoulder'],
            landmarks['right_elbow'],
            landmarks['right_wrist'],
            'right'
        )
        
        # Check for covering face/mouth behavior
        nose_pos = landmarks.get('nose')
        if nose_pos:
            face_covering = self._detect_face_covering(landmarks, nose_pos)
            if face_covering:
                suspicions.append(face_covering)
        
        # Check for arms behind body (potential reaching for hidden items)
        behind_body = self._detect_arms_behind_body(landmarks)
        if behind_body:
            suspicions.append(behind_body)
        
        return {
            'left_arm': left_analysis,
            'right_arm': right_analysis,
            'suspicions': suspicions,
            'overall_confidence': (left_analysis['confidence'] + right_analysis['confidence']) / 2
        }
    
    def _analyze_single_arm(self, shoulder, elbow, wrist, side):
        """Analyze positioning of a single arm."""
        # Calculate arm angles
        shoulder_to_elbow = np.array([elbow['x'] - shoulder['x'], elbow['y'] - shoulder['y']])
        elbow_to_wrist = np.array([wrist['x'] - elbow['x'], wrist['y'] - elbow['y']])
        
        # Calculate angles
        upper_arm_angle = np.arctan2(shoulder_to_elbow[1], shoulder_to_elbow[0]) * 180 / np.pi
        forearm_angle = np.arctan2(elbow_to_wrist[1], elbow_to_wrist[0]) * 180 / np.pi
        
        # Calculate arm extension (how far from body)
        arm_extension = np.linalg.norm(shoulder_to_elbow) + np.linalg.norm(elbow_to_wrist)
        
        # Determine arm position category
        position_category = self._categorize_arm_position(upper_arm_angle, forearm_angle, side)
        
        # Calculate confidence based on visibility
        confidence = min(shoulder['visibility'], elbow['visibility'], wrist['visibility'])
        
        return {
            'upper_arm_angle': upper_arm_angle,
            'forearm_angle': forearm_angle,
            'extension': arm_extension,
            'position': position_category,
            'confidence': confidence
        }
    
    def _categorize_arm_position(self, upper_arm_angle, forearm_angle, side):
        """Categorize arm position based on angles."""
        # Normalize angles for left/right side
        if side == 'left':
            upper_arm_angle = -upper_arm_angle
        
        # Categories based on typical exam positions
        if -30 <= upper_arm_angle <= 30 and -45 <= forearm_angle <= 45:
            return 'writing'  # Normal writing position
        elif upper_arm_angle > 60 or upper_arm_angle < -60:
            return 'raised'  # Arm raised (potentially suspicious)
        elif abs(forearm_angle) > 90:
            return 'behind'  # Arm behind body
        elif 30 < upper_arm_angle <= 60:
            return 'extended'  # Arm extended away from body
        else:
            return 'neutral'  # Neutral position
    
    def _detect_face_covering(self, landmarks, nose_pos):
        """Detect if hands are covering face/mouth area."""
        suspicions = []
        
        # Check both wrists
        for side in ['left', 'right']:
            wrist = landmarks.get(f'{side}_wrist')
            if wrist and wrist['visibility'] > 0.5:
                # Calculate distance from wrist to nose
                distance = np.sqrt(
                    (wrist['x'] - nose_pos['x'])**2 + 
                    (wrist['y'] - nose_pos['y'])**2
                )
                
                # If wrist is very close to face
                if distance < 80:  # pixels
                    suspicions.append({
                        'type': 'face_covering',
                        'side': side,
                        'distance': distance,
                        'severity': 'high' if distance < 50 else 'medium'
                    })
        
        return suspicions[0] if suspicions else None
    
    def _detect_arms_behind_body(self, landmarks):
        """Detect if arms are positioned behind the body."""
        suspicious_arms = []
        
        for side in ['left', 'right']:
            shoulder = landmarks.get(f'{side}_shoulder')
            elbow = landmarks.get(f'{side}_elbow')
            wrist = landmarks.get(f'{side}_wrist')
            
            if all(part and part['visibility'] > 0.3 for part in [shoulder, elbow, wrist]):
                # Check if elbow is behind shoulder line (indicating reaching back)
                if side == 'left' and elbow['x'] < shoulder['x'] - 50:
                    suspicious_arms.append(side)
                elif side == 'right' and elbow['x'] > shoulder['x'] + 50:
                    suspicious_arms.append(side)
        
        if suspicious_arms:
            return {
                'type': 'arms_behind_body',
                'arms': suspicious_arms,
                'severity': 'high'
            }
        return None
    
    def analyze_posture_stability(self, landmarks):
        """
        Analyze posture stability over time.
        
        Args:
            landmarks: Current pose landmarks
            
        Returns:
            Dictionary with stability analysis
        """
        current_time = time.time()
        
        # Extract key stability metrics
        if landmarks:
            stability_metrics = {
                'timestamp': current_time,
                'shoulder_height_diff': abs(
                    landmarks['left_shoulder']['y'] - landmarks['right_shoulder']['y']
                ) if 'left_shoulder' in landmarks and 'right_shoulder' in landmarks else 0,
                'body_center_x': (
                    landmarks['left_shoulder']['x'] + landmarks['right_shoulder']['x']
                ) / 2 if 'left_shoulder' in landmarks and 'right_shoulder' in landmarks else 0
            }
            
            self.posture_history.append(stability_metrics)
        
        if len(self.posture_history) < 30:  # Need at least 1 second of data
            return {'status': 'insufficient_data'}
        
        # Calculate stability metrics
        recent_metrics = list(self.posture_history)[-30:]
        
        shoulder_variations = [m['shoulder_height_diff'] for m in recent_metrics]
        center_variations = [m['body_center_x'] for m in recent_metrics]
        
        shoulder_stability = np.std(shoulder_variations)
        center_stability = np.std(center_variations)
        
        # Determine stability level
        is_unstable = (
            shoulder_stability > 15 or  # High shoulder variation
            center_stability > 20      # High center movement
        )
        
        if is_unstable:
            self.posture_instability_count += 1
        else:
            self.posture_instability_count = max(0, self.posture_instability_count - 1)
        
        return {
            'status': 'analyzed',
            'shoulder_stability': shoulder_stability,
            'center_stability': center_stability,
            'is_unstable': is_unstable,
            'instability_count': self.posture_instability_count,
            'stability_score': 1.0 / (1.0 + shoulder_stability + center_stability)
        }
    
    def detect_posture_violations(self, lean_data, arm_data, stability_data):
        """
        Detect posture-related violations.
        
        Args:
            lean_data: Body lean analysis
            arm_data: Arm positioning analysis  
            stability_data: Posture stability analysis
            
        Returns:
            List of detected violations
        """
        violations = []
        current_time = time.time()
        
        # Check body lean violations
        if lean_data and lean_data['confidence'] > 0.6:
            lean_angle = lean_data['lean_angle']
            
            if lean_angle > self.critical_lean_threshold:
                if self.lean_violation_start is None:
                    self.lean_violation_start = current_time
                elif current_time - self.lean_violation_start > 3.0:  # 3 seconds
                    violations.append({
                        'type': 'excessive_lean',
                        'severity': 'high',
                        'angle': lean_angle,
                        'direction': lean_data['lean_direction'],
                        'duration': current_time - self.lean_violation_start,
                        'message': f"Excessive body lean ({lean_angle:.1f}°) for {current_time - self.lean_violation_start:.1f}s"
                    })
            elif lean_angle > self.suspicious_lean_threshold:
                if self.lean_violation_start is None:
                    self.lean_violation_start = current_time
                elif current_time - self.lean_violation_start > 5.0:  # 5 seconds for moderate lean
                    violations.append({
                        'type': 'moderate_lean',
                        'severity': 'medium',
                        'angle': lean_angle,
                        'direction': lean_data['lean_direction'],
                        'duration': current_time - self.lean_violation_start,
                        'message': f"Sustained body lean ({lean_angle:.1f}°) for {current_time - self.lean_violation_start:.1f}s"
                    })
            else:
                self.lean_violation_start = None  # Reset if lean is normal
        
        # Check arm positioning violations
        if arm_data and arm_data['suspicions']:
            for suspicion in arm_data['suspicions']:
                if suspicion['severity'] == 'high':
                    violations.append({
                        'type': 'suspicious_arm_position',
                        'severity': 'high',
                        'details': suspicion,
                        'message': f"Suspicious arm positioning: {suspicion['type']}"
                    })
        
        # Check posture instability violations
        if (stability_data and stability_data.get('status') == 'analyzed' and
            stability_data['instability_count'] > 10):  # Sustained instability
            violations.append({
                'type': 'posture_instability',
                'severity': 'medium',
                'instability_score': 1.0 - stability_data['stability_score'],
                'message': f"Unstable posture detected (score: {stability_data['stability_score']:.2f})"
            })
            # Reset counter after reporting
            self.posture_instability_count = 0
        
        return violations
    
    def _calculate_lean_confidence(self, landmarks):
        """Calculate confidence score for lean measurement."""
        required_landmarks = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        
        if not all(landmark in landmarks for landmark in required_landmarks):
            return 0.0
        
        # Average visibility of required landmarks
        visibility_scores = [landmarks[landmark]['visibility'] for landmark in required_landmarks]
        avg_visibility = np.mean(visibility_scores)
        
        return avg_visibility
    
    def visualize_posture(self, frame, landmarks, lean_data, violations):
        """
        Visualize posture analysis on the frame.
        
        Args:
            frame: Input frame
            landmarks: Pose landmarks
            lean_data: Body lean analysis
            violations: List of detected violations
            
        Returns:
            Frame with posture visualization
        """
        if not landmarks:
            return frame
        
        # Draw skeleton
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('right_shoulder', 'right_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip')
        ]
        
        for start, end in connections:
            if start in landmarks and end in landmarks:
                start_point = (int(landmarks[start]['x']), int(landmarks[start]['y']))
                end_point = (int(landmarks[end]['x']), int(landmarks[end]['y']))
                cv2.line(frame, start_point, end_point, (0, 255, 255), 2)
        
        # Draw key points
        for name, landmark in landmarks.items():
            if landmark['visibility'] > 0.5:
                point = (int(landmark['x']), int(landmark['y']))
                cv2.circle(frame, point, 4, (255, 0, 0), -1)
        
        # Draw lean angle indicator
        if lean_data and lean_data['confidence'] > 0.5:
            # Draw lean direction arrow
            if 'left_shoulder' in landmarks and 'right_shoulder' in landmarks:
                center_x = int((landmarks['left_shoulder']['x'] + landmarks['right_shoulder']['x']) / 2)
                center_y = int((landmarks['left_shoulder']['y'] + landmarks['right_shoulder']['y']) / 2)
                
                # Draw lean angle text
                angle_text = f"Lean: {lean_data['lean_angle']:.1f}° {lean_data['lean_direction']}"
                cv2.putText(frame, angle_text, (center_x - 60, center_y - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw violation indicators
        if violations:
            y_offset = 30
            for violation in violations:
                severity_color = {
                    'low': (0, 255, 255),      # Yellow
                    'medium': (0, 165, 255),   # Orange  
                    'high': (0, 0, 255)        # Red
                }.get(violation['severity'], (255, 255, 255))
                
                cv2.putText(frame, f"POSTURE: {violation['message']}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           severity_color, 2)
                y_offset += 25
        
        return frame
    
    def reset(self):
        """Reset all tracking history and counters."""
        self.posture_history.clear()
        self.lean_history.clear()
        self.arm_position_history.clear()
        self.posture_instability_count = 0
        self.lean_violation_start = None
        self.arm_suspicious_start = None
    
    def get_status_info(self):
        """Get current status information for logging."""
        return {
            'posture_history_length': len(self.posture_history),
            'instability_count': self.posture_instability_count,
            'tracking_active': len(self.posture_history) > 0,
            'has_lean_violation': self.lean_violation_start is not None,
            'has_arm_violation': self.arm_suspicious_start is not None
        }