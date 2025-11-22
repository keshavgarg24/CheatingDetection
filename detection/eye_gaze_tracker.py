"""
Advanced Eye Gaze Tracking for Exam Monitoring

This module provides sophisticated eye gaze analysis to detect suspicious
looking patterns including off-screen gazing, paper switching detection,
and abnormal gaze behavior during exams.
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time


class AdvancedEyeGazeTracker:
    """
    Advanced eye gaze tracking for detecting suspicious looking behavior.
    
    Monitors:
    - Gaze direction and duration
    - Off-screen looking patterns
    - Rapid eye movements (saccades)
    - Sustained gazing at specific areas
    - Paper switching detection
    """
    
    def __init__(self):
        """Initialize eye gaze tracker."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices for MediaPipe Face Mesh
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Pupil estimation indices (iris landmarks)
        self.LEFT_IRIS_INDICES = [474, 475, 476, 477]
        self.RIGHT_IRIS_INDICES = [469, 470, 471, 472]
        
        # Gaze history tracking
        self.gaze_history = deque(maxlen=180)  # 6 seconds at 30 FPS
        self.saccade_history = deque(maxlen=60)  # 2 seconds for saccade detection
        
        # Gaze analysis parameters
        self.screen_regions = {
            'center': (0.3, 0.3, 0.7, 0.7),      # Expected exam paper area
            'left_off': (0.0, 0.0, 0.3, 1.0),     # Left off-screen
            'right_off': (0.7, 0.0, 1.0, 1.0),    # Right off-screen
            'top_off': (0.0, 0.0, 1.0, 0.3),      # Top off-screen
            'bottom_off': (0.0, 0.7, 1.0, 1.0)    # Bottom off-screen
        }
        
        # Suspicious behavior tracking
        self.off_screen_duration = 0
        self.rapid_movement_count = 0
        self.last_gaze_point = None
        self.gaze_fixation_points = []
        
    def extract_eye_landmarks(self, landmarks, indices, frame_width, frame_height):
        """
        Extract eye landmark coordinates.
        
        Args:
            landmarks: MediaPipe face landmarks
            indices: List of landmark indices for the eye
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            Array of (x, y) coordinates for eye landmarks
        """
        points = []
        for idx in indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            points.append([x, y])
        return np.array(points)
    
    def estimate_gaze_direction(self, frame):
        """
        Estimate gaze direction using eye landmarks and iris position.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Dictionary with gaze information or None if no face detected
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Extract eye landmarks
        left_eye = self.extract_eye_landmarks(face_landmarks, self.LEFT_EYE_INDICES, w, h)
        right_eye = self.extract_eye_landmarks(face_landmarks, self.RIGHT_EYE_INDICES, w, h)
        
        # Extract iris landmarks for pupil estimation
        left_iris = self.extract_eye_landmarks(face_landmarks, self.LEFT_IRIS_INDICES, w, h)
        right_iris = self.extract_eye_landmarks(face_landmarks, self.RIGHT_IRIS_INDICES, w, h)
        
        # Calculate eye centers and pupil positions
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        left_pupil = np.mean(left_iris, axis=0) if len(left_iris) > 0 else left_eye_center
        right_pupil = np.mean(right_iris, axis=0) if len(right_iris) > 0 else right_eye_center
        
        # Calculate gaze direction (normalized)
        left_gaze = (left_pupil - left_eye_center) / 20  # Normalize by eye size approximation
        right_gaze = (right_pupil - right_eye_center) / 20
        
        # Average gaze direction
        avg_gaze = (left_gaze + right_gaze) / 2
        
        # Estimate gaze point on screen (simplified projection)
        screen_center = np.array([w/2, h/2])
        gaze_point = screen_center + avg_gaze * 100  # Scale factor for projection
        
        # Normalize to screen coordinates (0-1)
        gaze_normalized = (gaze_point[0] / w, gaze_point[1] / h)
        
        return {
            'left_eye_center': left_eye_center,
            'right_eye_center': right_eye_center,
            'left_pupil': left_pupil,
            'right_pupil': right_pupil,
            'gaze_direction': avg_gaze,
            'gaze_point': gaze_point,
            'gaze_normalized': gaze_normalized,
            'confidence': self._calculate_gaze_confidence(left_eye, right_eye, left_iris, right_iris)
        }
    
    def _calculate_gaze_confidence(self, left_eye, right_eye, left_iris, right_iris):
        """
        Calculate confidence score for gaze estimation.
        
        Args:
            left_eye: Left eye landmarks
            right_eye: Right eye landmarks
            left_iris: Left iris landmarks
            right_iris: Right iris landmarks
            
        Returns:
            Confidence score (0-1)
        """
        # Check if eyes are well-formed (not too small or occluded)
        left_eye_area = cv2.contourArea(left_eye) if len(left_eye) > 3 else 0
        right_eye_area = cv2.contourArea(right_eye) if len(right_eye) > 3 else 0
        
        # Minimum eye area for reliable detection
        min_eye_area = 100  # pixels
        
        if left_eye_area < min_eye_area or right_eye_area < min_eye_area:
            return 0.3  # Low confidence for small eyes
        
        # Check iris detection quality
        iris_detected = len(left_iris) > 0 and len(right_iris) > 0
        
        if iris_detected:
            return 0.9  # High confidence with iris detection
        else:
            return 0.6  # Medium confidence without iris
    
    def analyze_gaze_patterns(self, gaze_data):
        """
        Analyze gaze patterns for suspicious behavior.
        
        Args:
            gaze_data: Gaze information from estimate_gaze_direction
            
        Returns:
            Dictionary with analysis results
        """
        if gaze_data is None:
            return {'status': 'no_data', 'violations': []}
        
        # Add to gaze history
        current_time = time.time()
        gaze_entry = {
            'time': current_time,
            'point': gaze_data['gaze_normalized'],
            'confidence': gaze_data['confidence']
        }
        self.gaze_history.append(gaze_entry)
        
        violations = []
        
        # Analyze off-screen gazing
        off_screen_violation = self._analyze_off_screen_gazing(gaze_entry)
        if off_screen_violation:
            violations.append(off_screen_violation)
        
        # Analyze rapid eye movements (saccades)
        saccade_violation = self._analyze_saccades(gaze_entry)
        if saccade_violation:
            violations.append(saccade_violation)
        
        # Analyze sustained gazing patterns
        fixation_violation = self._analyze_fixation_patterns(gaze_entry)
        if fixation_violation:
            violations.append(fixation_violation)
        
        return {
            'status': 'analyzed',
            'current_region': self._get_gaze_region(gaze_data['gaze_normalized']),
            'off_screen_duration': self.off_screen_duration,
            'confidence': gaze_data['confidence'],
            'violations': violations,
            'gaze_point': gaze_data['gaze_point']
        }
    
    def _analyze_off_screen_gazing(self, gaze_entry):
        """Analyze if student is looking off-screen suspiciously."""
        gaze_point = gaze_entry['point']
        current_region = self._get_gaze_region(gaze_point)
        
        # Check if looking off-screen
        if current_region != 'center':
            self.off_screen_duration += 1/30  # Assuming 30 FPS
            
            # Violation if looking off-screen for more than 3 seconds
            if self.off_screen_duration > 3.0:
                return {
                    'type': 'off_screen_gazing',
                    'severity': 'medium',
                    'duration': self.off_screen_duration,
                    'region': current_region,
                    'message': f"Looking {current_region.replace('_', ' ')} for {self.off_screen_duration:.1f}s"
                }
        else:
            self.off_screen_duration = 0  # Reset when looking back at center
        
        return None
    
    def _analyze_saccades(self, gaze_entry):
        """Analyze rapid eye movements for suspicious patterns."""
        self.saccade_history.append(gaze_entry)
        
        if len(self.saccade_history) < 10:
            return None
        
        # Calculate gaze movement velocity
        recent_gazes = list(self.saccade_history)[-10:]
        movements = []
        
        for i in range(1, len(recent_gazes)):
            prev = recent_gazes[i-1]['point']
            curr = recent_gazes[i]['point']
            distance = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            time_diff = recent_gazes[i]['time'] - recent_gazes[i-1]['time']
            
            if time_diff > 0:
                velocity = distance / time_diff
                movements.append(velocity)
        
        if movements:
            avg_velocity = np.mean(movements)
            # Detect unusually rapid movements (potential scanning/searching)
            if avg_velocity > 2.0:  # Threshold for rapid movements
                self.rapid_movement_count += 1
                
                if self.rapid_movement_count > 5:  # Multiple rapid movements
                    self.rapid_movement_count = 0  # Reset counter
                    return {
                        'type': 'rapid_eye_movements',
                        'severity': 'low',
                        'velocity': avg_velocity,
                        'message': f"Rapid eye movements detected (velocity: {avg_velocity:.2f})"
                    }
        
        return None
    
    def _analyze_fixation_patterns(self, gaze_entry):
        """Analyze fixation patterns for unusual behavior."""
        # Track fixation points (areas where gaze stays for extended periods)
        gaze_point = gaze_entry['point']
        
        # Check if this is a new fixation point
        is_new_fixation = True
        for fixation in self.gaze_fixation_points:
            distance = np.sqrt((gaze_point[0] - fixation['center'][0])**2 + 
                             (gaze_point[1] - fixation['center'][1])**2)
            if distance < 0.1:  # Within 10% of screen size
                fixation['duration'] += 1/30  # Assuming 30 FPS
                fixation['center'] = gaze_point  # Update center
                is_new_fixation = False
                
                # Check for suspicious long fixation outside exam area
                if (fixation['duration'] > 5.0 and 
                    self._get_gaze_region(fixation['center']) != 'center'):
                    return {
                        'type': 'suspicious_fixation',
                        'severity': 'medium',
                        'duration': fixation['duration'],
                        'location': self._get_gaze_region(fixation['center']),
                        'message': f"Prolonged staring at {self._get_gaze_region(fixation['center']).replace('_', ' ')}"
                    }
                break
        
        if is_new_fixation:
            self.gaze_fixation_points.append({
                'center': gaze_point,
                'duration': 1/30,
                'start_time': gaze_entry['time']
            })
            
            # Limit number of tracked fixations
            if len(self.gaze_fixation_points) > 10:
                self.gaze_fixation_points.pop(0)
        
        return None
    
    def _get_gaze_region(self, gaze_point):
        """Determine which screen region the gaze point falls into."""
        x, y = gaze_point
        
        for region_name, (x1, y1, x2, y2) in self.screen_regions.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return region_name
        
        return 'unknown'
    
    def visualize_gaze(self, frame, gaze_data):
        """
        Visualize gaze tracking on the frame.
        
        Args:
            frame: Input frame
            gaze_data: Gaze data from estimate_gaze_direction
            
        Returns:
            Frame with gaze visualization
        """
        if gaze_data is None:
            return frame
        
        # Draw eye centers
        left_center = tuple(map(int, gaze_data['left_eye_center']))
        right_center = tuple(map(int, gaze_data['right_eye_center']))
        cv2.circle(frame, left_center, 3, (0, 255, 0), -1)
        cv2.circle(frame, right_center, 3, (0, 255, 0), -1)
        
        # Draw pupils
        left_pupil = tuple(map(int, gaze_data['left_pupil']))
        right_pupil = tuple(map(int, gaze_data['right_pupil']))
        cv2.circle(frame, left_pupil, 2, (255, 0, 0), -1)
        cv2.circle(frame, right_pupil, 2, (255, 0, 0), -1)
        
        # Draw gaze point
        gaze_point = tuple(map(int, gaze_data['gaze_point']))
        cv2.circle(frame, gaze_point, 5, (0, 0, 255), 2)
        
        # Draw gaze trail
        if len(self.gaze_history) > 1:
            h, w = frame.shape[:2]
            trail_points = []
            for entry in list(self.gaze_history)[-10:]:  # Last 10 points
                screen_point = (int(entry['point'][0] * w), int(entry['point'][1] * h))
                trail_points.append(screen_point)
            
            # Draw trail lines
            for i in range(1, len(trail_points)):
                alpha = i / len(trail_points)  # Fade trail
                color = (int(255 * alpha), int(100 * alpha), 0)
                cv2.line(frame, trail_points[i-1], trail_points[i], color, 2)
        
        return frame
    
    def reset(self):
        """Reset all tracking history and counters."""
        self.gaze_history.clear()
        self.saccade_history.clear()
        self.gaze_fixation_points.clear()
        self.off_screen_duration = 0
        self.rapid_movement_count = 0
        self.last_gaze_point = None
    
    def get_status_info(self):
        """Get current status information for logging."""
        return {
            'gaze_history_length': len(self.gaze_history),
            'off_screen_duration': self.off_screen_duration,
            'rapid_movements': self.rapid_movement_count,
            'fixation_points': len(self.gaze_fixation_points),
            'tracking_active': len(self.gaze_history) > 0
        }