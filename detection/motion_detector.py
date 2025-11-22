"""
Advanced Motion Detection for Exam Monitoring

This module provides sophisticated motion analysis to detect sudden movements,
suspicious patterns, and unusual activity during exams while filtering out
normal exam-related movements.
"""

import cv2
import numpy as np
from collections import deque
import time


class AdvancedMotionDetector:
    """
    Advanced motion detection with pattern analysis and false positive reduction.
    
    Features:
    - Background subtraction for motion detection
    - Sudden movement detection
    - Movement pattern analysis
    - Normal exam movement filtering
    - Directional movement analysis
    """
    
    def __init__(self):
        """Initialize advanced motion detector."""
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=50,  # Threshold for shadow detection
            history=500       # Number of frames for background learning
        )
        
        # Motion analysis parameters
        self.motion_threshold = 25  # Minimum pixel difference for motion
        self.sudden_motion_threshold = 80  # Threshold for sudden movements
        self.min_contour_area = 500  # Minimum area for motion contours
        
        # Motion history tracking
        self.motion_history = deque(maxlen=90)  # 3 seconds at 30 FPS
        self.movement_vectors = deque(maxlen=60)  # 2 seconds of movement vectors
        self.sudden_movements = deque(maxlen=30)  # 1 second of sudden movements
        
        # Motion analysis state
        self.background_learned = False
        self.frame_count = 0
        self.last_motion_time = 0
        self.motion_intensity = 0
        
        # Pattern detection
        self.repetitive_pattern_count = 0
        self.last_movement_direction = None
        
    def process_frame(self, frame):
        """
        Process frame for motion detection and analysis.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Dictionary with motion analysis results
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(blurred)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        significant_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
        
        # Calculate motion metrics
        motion_analysis = self._analyze_motion(significant_contours, frame.shape[:2])
        
        # Add to motion history
        motion_entry = {
            'timestamp': current_time,
            'motion_intensity': motion_analysis['motion_intensity'],
            'contour_count': len(significant_contours),
            'motion_areas': motion_analysis['motion_areas'],
            'movement_vector': motion_analysis.get('movement_vector'),
            'frame_number': self.frame_count
        }
        self.motion_history.append(motion_entry)
        
        # Update background learning status
        if self.frame_count > 100:  # Consider background learned after 100 frames
            self.background_learned = True
        
        # Analyze patterns and detect violations
        violations = self._detect_motion_violations(motion_analysis)
        
        return {
            'motion_detected': motion_analysis['motion_intensity'] > self.motion_threshold,
            'motion_intensity': motion_analysis['motion_intensity'],
            'sudden_movement': motion_analysis.get('sudden_movement', False),
            'movement_vector': motion_analysis.get('movement_vector'),
            'violations': violations,
            'foreground_mask': fg_mask,
            'contours': significant_contours,
            'background_learned': self.background_learned
        }
    
    def _analyze_motion(self, contours, frame_shape):
        """
        Analyze motion characteristics from contours.
        
        Args:
            contours: List of motion contours
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            Dictionary with motion analysis
        """
        h, w = frame_shape
        
        if not contours:
            return {
                'motion_intensity': 0,
                'motion_areas': [],
                'movement_vector': None,
                'sudden_movement': False
            }
        
        # Calculate total motion area
        total_motion_area = sum(cv2.contourArea(contour) for contour in contours)
        motion_intensity = (total_motion_area / (h * w)) * 100  # Percentage of frame
        
        # Calculate motion centers for movement vector analysis
        motion_centers = []
        motion_areas = []
        
        for contour in contours:
            # Calculate contour center
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                motion_centers.append((cx, cy))
                
                # Get bounding rectangle
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                motion_areas.append({
                    'center': (cx, cy),
                    'bbox': (x, y, w_rect, h_rect),
                    'area': cv2.contourArea(contour)
                })
        
        # Calculate overall movement vector
        movement_vector = self._calculate_movement_vector(motion_centers)
        
        # Detect sudden movement
        sudden_movement = self._detect_sudden_movement(motion_intensity)
        
        return {
            'motion_intensity': motion_intensity,
            'motion_areas': motion_areas,
            'movement_vector': movement_vector,
            'sudden_movement': sudden_movement,
            'motion_centers': motion_centers
        }
    
    def _calculate_movement_vector(self, motion_centers):
        """Calculate overall movement vector from motion centers."""
        if len(motion_centers) < 1 or len(self.movement_vectors) < 1:
            return None
        
        # Get previous motion centers if available
        if len(self.motion_history) > 1:
            prev_entry = self.motion_history[-2]
            prev_centers = prev_entry.get('motion_areas', [])
            
            if prev_centers:
                # Calculate average movement between frames
                prev_center_coords = [area['center'] for area in prev_centers]
                
                # Find best matching centers and calculate movement
                total_dx = 0
                total_dy = 0
                matches = 0
                
                for center in motion_centers:
                    # Find closest previous center
                    distances = [np.sqrt((center[0] - prev[0])**2 + (center[1] - prev[1])**2) 
                               for prev in prev_center_coords]
                    
                    if distances:
                        min_idx = np.argmin(distances)
                        if distances[min_idx] < 100:  # Reasonable movement distance
                            prev_center = prev_center_coords[min_idx]
                            total_dx += center[0] - prev_center[0]
                            total_dy += center[1] - prev_center[1]
                            matches += 1
                
                if matches > 0:
                    avg_dx = total_dx / matches
                    avg_dy = total_dy / matches
                    
                    # Calculate magnitude and direction
                    magnitude = np.sqrt(avg_dx**2 + avg_dy**2)
                    direction = np.arctan2(avg_dy, avg_dx) * 180 / np.pi
                    
                    movement_vector = {
                        'dx': avg_dx,
                        'dy': avg_dy,
                        'magnitude': magnitude,
                        'direction': direction
                    }
                    
                    self.movement_vectors.append(movement_vector)
                    return movement_vector
        
        return None
    
    def _detect_sudden_movement(self, current_intensity):
        """Detect sudden movements based on intensity changes."""
        if len(self.motion_history) < 5:
            return False
        
        # Calculate recent motion intensities
        recent_intensities = [entry['motion_intensity'] for entry in list(self.motion_history)[-5:]]
        avg_recent_intensity = np.mean(recent_intensities[:-1])  # Exclude current frame
        
        # Check for sudden increase in motion
        intensity_increase = current_intensity - avg_recent_intensity
        
        is_sudden = intensity_increase > self.sudden_motion_threshold
        
        if is_sudden:
            self.sudden_movements.append({
                'timestamp': time.time(),
                'intensity_increase': intensity_increase,
                'current_intensity': current_intensity,
                'avg_intensity': avg_recent_intensity
            })
        
        return is_sudden
    
    def _detect_motion_violations(self, motion_analysis):
        """Detect motion-related violations."""
        violations = []
        current_time = time.time()
        
        # Sudden movement violation
        if motion_analysis.get('sudden_movement'):
            violations.append({
                'type': 'sudden_movement',
                'severity': 'medium',
                'intensity': motion_analysis['motion_intensity'],
                'message': f"Sudden movement detected (intensity: {motion_analysis['motion_intensity']:.1f}%)"
            })
        
        # Excessive continuous motion
        if (motion_analysis['motion_intensity'] > 15 and  # High motion threshold
            len(self.motion_history) >= 60):  # 2 seconds of data
            
            recent_motion = [entry['motion_intensity'] for entry in list(self.motion_history)[-60:]]
            avg_motion = np.mean(recent_motion)
            
            if avg_motion > 10:  # Sustained high motion
                violations.append({
                    'type': 'excessive_motion',
                    'severity': 'low',
                    'avg_intensity': avg_motion,
                    'duration': 2.0,
                    'message': f"Excessive motion for 2+ seconds (avg: {avg_motion:.1f}%)"
                })
        
        # Repetitive movement patterns (potential signaling)
        pattern_violation = self._detect_repetitive_patterns()
        if pattern_violation:
            violations.append(pattern_violation)
        
        # Large area motion (standing up, major position change)
        if motion_analysis['motion_intensity'] > 25:
            violations.append({
                'type': 'large_movement',
                'severity': 'medium',
                'intensity': motion_analysis['motion_intensity'],
                'message': f"Large area movement detected ({motion_analysis['motion_intensity']:.1f}% of frame)"
            })
        
        return violations
    
    def _detect_repetitive_patterns(self):
        """Detect repetitive movement patterns that might indicate signaling."""
        if len(self.movement_vectors) < 20:
            return None
        
        # Analyze recent movement directions
        recent_vectors = list(self.movement_vectors)[-20:]
        directions = [v['direction'] for v in recent_vectors if v['magnitude'] > 5]
        
        if len(directions) < 10:
            return None
        
        # Look for oscillating patterns (back and forth movement)
        direction_changes = 0
        for i in range(1, len(directions)):
            # Check for significant direction change (> 90 degrees)
            angle_diff = abs(directions[i] - directions[i-1])
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff > 90:
                direction_changes += 1
        
        # If many direction changes, it might be signaling
        if direction_changes > 6:  # More than 6 direction changes in recent history
            self.repetitive_pattern_count += 1
            
            if self.repetitive_pattern_count > 2:  # Multiple detections
                self.repetitive_pattern_count = 0  # Reset counter
                return {
                    'type': 'repetitive_movement',
                    'severity': 'medium',
                    'pattern_count': direction_changes,
                    'message': f"Repetitive movement pattern detected ({direction_changes} direction changes)"
                }
        
        return None
    
    def visualize_motion(self, frame, motion_result):
        """
        Visualize motion detection results on the frame.
        
        Args:
            frame: Input frame
            motion_result: Motion analysis results
            
        Returns:
            Frame with motion visualization
        """
        # Draw motion contours
        if motion_result['contours']:
            cv2.drawContours(frame, motion_result['contours'], -1, (0, 255, 0), 2)
        
        # Draw motion areas with bounding boxes
        for area in motion_result.get('motion_areas', []):
            bbox = area['bbox']
            center = area['center']
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), 
                         (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
            
            # Draw center point
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
            # Draw area info
            cv2.putText(frame, f"Area: {area['area']:.0f}", 
                       (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
        
        # Draw movement vector
        movement_vector = motion_result.get('movement_vector')
        if movement_vector and movement_vector['magnitude'] > 5:
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # Calculate arrow end point
            scale = 5  # Scale factor for visualization
            end_x = int(center_x + movement_vector['dx'] * scale)
            end_y = int(center_y + movement_vector['dy'] * scale)
            
            # Draw movement arrow
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                           (0, 255, 255), 3)
            
            # Draw movement info
            cv2.putText(frame, f"Movement: {movement_vector['magnitude']:.1f}px", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw motion intensity
        intensity_color = (0, 255, 0) if motion_result['motion_intensity'] < 10 else (0, 165, 255)
        cv2.putText(frame, f"Motion: {motion_result['motion_intensity']:.1f}%", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, intensity_color, 2)
        
        # Draw violation indicators
        for violation in motion_result.get('violations', []):
            severity_color = {
                'low': (0, 255, 255),      # Yellow
                'medium': (0, 165, 255),   # Orange
                'high': (0, 0, 255)        # Red
            }.get(violation['severity'], (255, 255, 255))
            
            cv2.putText(frame, f"MOTION: {violation['message'][:40]}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       severity_color, 2)
        
        return frame
    
    def reset(self):
        """Reset motion detection history and state."""
        self.motion_history.clear()
        self.movement_vectors.clear()
        self.sudden_movements.clear()
        self.background_learned = False
        self.frame_count = 0
        self.repetitive_pattern_count = 0
        self.last_movement_direction = None
        
        # Reset background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=50,
            history=500
        )
    
    def get_status_info(self):
        """Get current status information for logging."""
        recent_intensity = 0
        if self.motion_history:
            recent_intensity = self.motion_history[-1]['motion_intensity']
        
        return {
            'background_learned': self.background_learned,
            'frame_count': self.frame_count,
            'motion_history_length': len(self.motion_history),
            'recent_motion_intensity': recent_intensity,
            'sudden_movements_recent': len(self.sudden_movements),
            'tracking_active': len(self.motion_history) > 0
        }