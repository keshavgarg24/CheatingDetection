"""
Lighting Enhancement Module for Classroom Exam Monitoring

This module provides automatic lighting compensation to ensure consistent
detection accuracy across varying classroom lighting conditions.
"""

import cv2
import numpy as np


class ClassroomLightingCompensator:
    """
    Optimized lighting compensation for classroom environments.
    
    Handles varying lighting conditions including:
    - Dark areas (back of classroom)
    - Bright areas (window glare)
    - Natural light fluctuations
    
    Uses LAB color space and CLAHE for adaptive enhancement.
    """
    
    def __init__(self):
        """Initialize the lighting compensator with CLAHE processor."""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # clipLimit: Threshold for contrast limiting (prevents over-amplification)
        # tileGridSize: Size of grid for histogram equalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.target_brightness = 128  # Target mean brightness (0-255 scale)
        
    def enhance_frame(self, frame):
        """
        Apply automatic lighting compensation to frame.
        
        Optimized algorithm using numpy vectorized operations:
        1. Convert BGR to LAB color space
        2. Extract L (lightness) channel
        3. Calculate mean brightness (vectorized)
        4. Apply conditional brightness adjustment (vectorized)
        5. Apply CLAHE for local contrast enhancement
        6. Reconstruct and convert back to BGR
        
        Args:
            frame: Input BGR frame from camera
            
        Returns:
            Enhanced BGR frame with improved lighting
        """
        # Convert to LAB color space (L=lightness, A=green-red, B=blue-yellow)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate current brightness (mean of L channel) - vectorized operation
        current_brightness = np.mean(l_channel)
        
        # Apply conditional brightness adjustment based on lighting conditions
        if current_brightness < 80:
            # Too dark (back of classroom, poor lighting)
            alpha = 1.4  # Increase brightness
            beta = 25    # Add offset
        elif current_brightness > 200:
            # Too bright (window glare, direct sunlight)
            alpha = 0.7  # Decrease brightness
            beta = -15   # Subtract offset
        else:
            # Normal lighting conditions - skip adjustment for performance
            alpha = 1.0
            beta = 0
        
        # Apply brightness correction using vectorized numpy operations
        # This is faster than cv2.convertScaleAbs for simple linear transforms
        if alpha != 1.0 or beta != 0:
            # Vectorized operation: output = alpha * input + beta
            l_channel = np.clip(l_channel * alpha + beta, 0, 255).astype(np.uint8)
        
        # Apply CLAHE to enhance local contrast
        # This helps with shadows and uneven lighting within the frame
        l_channel = self.clahe.apply(l_channel)
        
        # Reconstruct LAB image with enhanced L channel (in-place for performance)
        lab[:, :, 0] = l_channel
        
        # Convert back to BGR color space
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
