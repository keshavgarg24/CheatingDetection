"""
Enhanced Head Pose Estimation for Exam Monitoring

This module provides accurate 3D head pose estimation using MediaPipe Face Mesh
and PnP (Perspective-n-Point) algorithm for better head turn detection.

Replaces simple keypoint offset calculation with full 3D pose estimation
including yaw, pitch, and roll angles.
"""

import cv2
import numpy as np
import mediapipe as mp


class AccurateHeadPoseEstimator:
    """
    More accurate head pose estimation using 3D model and PnP algorithm.
    
    Uses MediaPipe Face Mesh (468 landmarks) and 6 key facial points
    to calculate precise head orientation angles:
    - Yaw: Left/right head turn
    - Pitch: Up/down head tilt
    - Roll: Head rotation (tilting to side)
    
    This provides more accurate detection than simple 2D keypoint offsets.
    """
    
    def __init__(self):
        """Initialize Face Mesh and 3D model points."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # Use refined landmarks for better accuracy
            min_detection_confidence=0.7
        )
        
        # 3D model points for standard face (in mm)
        # These represent the 3D positions of key facial landmarks
        self.model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye left corner
            (225.0, 170.0, -135.0),    # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ], dtype=np.float32)
        
    def estimate_head_pose(self, frame):
        """
        Estimate head pose angles (yaw, pitch, roll) from frame.
        
        Algorithm:
        1. Process frame with MediaPipe Face Mesh
        2. Extract 6 key facial landmarks
        3. Create camera matrix (approximate for laptop cameras)
        4. Solve PnP to get rotation and translation vectors
        5. Convert rotation vector to Euler angles
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Dictionary with yaw, pitch, roll angles and vectors, or None if no face
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Extract 2D coordinates of key landmarks
        landmarks_2d = []
        
        # MediaPipe Face Mesh landmark indices for key points
        # 1: Nose tip, 152: Chin, 33: Left eye left corner, 263: Right eye right corner
        # 61: Left mouth corner, 291: Right mouth corner
        key_indices = [1, 152, 33, 263, 61, 291]
        
        for idx in key_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_2d.append([x, y])
            
        landmarks_2d = np.array(landmarks_2d, dtype=np.float32)
        
        # Camera calibration (approximate values for laptop cameras)
        # Focal length approximation: frame width
        focal_length = w
        camera_center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, camera_center[0]],
            [0, focal_length, camera_center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Assume no lens distortion for simplicity
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP to get rotation and translation vectors
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            landmarks_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
            
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Calculate Euler angles from rotation matrix
        angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
        
        return {
            'yaw': angles[1],           # Left/right head turn (degrees)
            'pitch': angles[0],         # Up/down head tilt (degrees)
            'roll': angles[2],          # Head roll/rotation (degrees)
            'translation': translation_vector,
            'rotation_vector': rotation_vector
        }
        
    def _rotation_matrix_to_euler_angles(self, R):
        """
        Convert rotation matrix to Euler angles (XYZ convention).
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            numpy array: [pitch, yaw, roll] in degrees
        """
        # Calculate if gimbal lock occurs
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])  # Pitch
            y = np.arctan2(-R[2, 0], sy)      # Yaw
            z = np.arctan2(R[1, 0], R[0, 0])  # Roll
        else:
            # Gimbal lock case
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
            
        # Convert from radians to degrees
        return np.array([x, y, z]) * 180 / np.pi
        
    def is_looking_away(self, pose_angles, threshold_yaw=30, threshold_pitch=20):
        """
        Determine if student is looking away based on pose angles.
        
        Args:
            pose_angles: Dictionary with yaw, pitch, roll angles
            threshold_yaw: Maximum yaw angle before considered "looking away" (degrees)
            threshold_pitch: Maximum pitch angle before considered "looking away" (degrees)
            
        Returns:
            Tuple: (is_away: bool, reason: str)
        """
        if pose_angles is None:
            return False, "No pose detected"
            
        yaw = abs(pose_angles['yaw'])
        pitch = abs(pose_angles['pitch'])
        
        if yaw > threshold_yaw:
            return True, f"Head turned {yaw:.1f}° (threshold: {threshold_yaw}°)"
        elif pitch > threshold_pitch:
            return True, f"Head tilted {pitch:.1f}° (threshold: {threshold_pitch}°)"
        else:
            return False, "Normal head position"
            
    def draw_pose_axes(self, frame, pose_result, camera_matrix, dist_coeffs):
        """
        Draw 3D coordinate axes on frame to visualize head pose.
        
        Useful for debugging and visualization.
        
        Args:
            frame: Input frame to draw on
            pose_result: Result from estimate_head_pose
            camera_matrix: Camera calibration matrix
            dist_coeffs: Distortion coefficients
        """
        if pose_result is None:
            return frame
            
        # Define 3D axes points (in mm)
        axis_points = np.array([
            [0, 0, 0],      # Origin
            [100, 0, 0],    # X axis (red)
            [0, 100, 0],    # Y axis (green)
            [0, 0, 100]     # Z axis (blue)
        ], dtype=np.float32)
        
        # Project 3D points to 2D
        image_points, _ = cv2.projectPoints(
            axis_points,
            pose_result['rotation_vector'],
            pose_result['translation'],
            camera_matrix,
            dist_coeffs
        )
        
        # Convert to integer coordinates
        image_points = image_points.astype(int)
        origin = tuple(image_points[0].ravel())
        
        # Draw axes
        frame = cv2.line(frame, origin, tuple(image_points[1].ravel()), (0, 0, 255), 3)  # X: Red
        frame = cv2.line(frame, origin, tuple(image_points[2].ravel()), (0, 255, 0), 3)  # Y: Green
        frame = cv2.line(frame, origin, tuple(image_points[3].ravel()), (255, 0, 0), 3)  # Z: Blue
        
        return frame
