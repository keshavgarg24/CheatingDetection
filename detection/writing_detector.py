"""
Writing Pattern Recognition for Exam Monitoring

This module distinguishes legitimate writing hand movements from suspicious
gestures to reduce false positives during normal exam activities.

Analyzes wrist movement patterns including:
- Position variance (small movements = writing)
- Movement frequency (2-8 movements/sec = natural writing rhythm)
- Vertical consistency (stable height = writing posture)
"""

import numpy as np
from collections import deque


class WritingPatternDetector:
    """
    Distinguish legitimate writing from suspicious hand movements.
    
    Writing characteristics:
    - Small horizontal variance (< 15 pixels)
    - Small vertical variance (< 10 pixels)
    - Consistent frequency (2-8 movements per second)
    - Stable vertical position (consistent writing height)
    
    Suspicious movements:
    - Large position changes
    - Erratic frequency
    - Vertical instability
    """
    
    def __init__(self):
        """Initialize writing pattern detector."""
        # Track last 90 wrist positions (3 seconds at 30 FPS)
        self.wrist_history = deque(maxlen=90)
        
        # Writing detection thresholds
        self.writing_threshold = {
            'x_variance_max': 15,    # Maximum horizontal variance (pixels)
            'y_variance_max': 10,    # Maximum vertical variance (pixels)
            'frequency_min': 2,      # Minimum movements per second
            'frequency_max': 8       # Maximum movements per second
        }
        
    def analyze_hand_movement(self, wrist_position):
        """
        Analyze if hand movement indicates writing.
        
        Args:
            wrist_position: Tuple (x, y) of wrist coordinates, or None
            
        Returns:
            Dictionary with:
                - is_writing: Boolean indicating if movement is writing
                - confidence: Float 0-1 confidence score
                - stats: Dictionary with analysis statistics
        """
        if wrist_position is None:
            return {'is_writing': False, 'confidence': 0.0}
            
        # Add position to history
        self.wrist_history.append(wrist_position)
        
        # Need at least 1 second of data (30 frames)
        if len(self.wrist_history) < 30:
            return {'is_writing': False, 'confidence': 0.0}
            
        # Calculate movement statistics
        positions = list(self.wrist_history)
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Variance analysis
        x_variance = np.var(x_coords)
        y_variance = np.var(y_coords)
        
        # Movement frequency analysis
        movements = self._count_direction_changes(positions)
        frequency = movements / (len(positions) / 30.0)  # movements per second
        
        # Vertical pattern analysis (writing has consistent height)
        vertical_pattern = self._analyze_vertical_pattern(y_coords)
        
        # Determine if this looks like writing
        is_writing = (
            x_variance < self.writing_threshold['x_variance_max'] and
            y_variance < self.writing_threshold['y_variance_max'] and
            self.writing_threshold['frequency_min'] <= frequency <= self.writing_threshold['frequency_max'] and
            vertical_pattern < 0.3  # Low vertical variation
        )
        
        # Calculate confidence score
        confidence = self._calculate_writing_confidence(
            x_variance, y_variance, frequency, vertical_pattern
        )
        
        return {
            'is_writing': is_writing,
            'confidence': confidence,
            'stats': {
                'x_variance': x_variance,
                'y_variance': y_variance,
                'frequency': frequency,
                'vertical_pattern': vertical_pattern
            }
        }
        
    def _count_direction_changes(self, positions):
        """
        Count direction changes in hand movement.
        
        Writing typically has regular back-and-forth motion.
        
        Args:
            positions: List of (x, y) tuples
            
        Returns:
            Integer count of direction changes
        """
        if len(positions) < 3:
            return 0
            
        changes = 0
        for i in range(2, len(positions)):
            # Calculate x-direction movement
            prev_x = positions[i-2][0] - positions[i-1][0]
            curr_x = positions[i-1][0] - positions[i][0]
            
            # Sign change indicates direction change
            if prev_x * curr_x < 0:
                changes += 1
                
        return changes
        
    def _analyze_vertical_pattern(self, y_coords):
        """
        Analyze vertical movement pattern.
        
        Writing typically maintains consistent vertical position.
        
        Args:
            y_coords: List of y coordinates
            
        Returns:
            Float representing vertical variation (lower = more consistent)
        """
        if len(y_coords) < 10:
            return 1.0
            
        # Calculate coefficient of variation (std / mean)
        # Writing has low vertical variation
        mean_y = np.mean(y_coords)
        std_y = np.std(y_coords)
        
        # Avoid division by zero
        if mean_y < 1:
            return 1.0
            
        return std_y / mean_y
        
    def _calculate_writing_confidence(self, x_var, y_var, freq, vertical):
        """
        Calculate confidence that movement is writing.
        
        Uses weighted scoring of multiple factors.
        
        Args:
            x_var: X-axis variance
            y_var: Y-axis variance
            freq: Movement frequency
            vertical: Vertical pattern score
            
        Returns:
            Float 0-1 confidence score
        """
        # Normalize each factor to 0-1 score
        
        # X variance score (lower is better)
        x_score = max(0, 1 - (x_var / self.writing_threshold['x_variance_max']))
        
        # Y variance score (lower is better)
        y_score = max(0, 1 - (y_var / self.writing_threshold['y_variance_max']))
        
        # Frequency score (within range is best)
        if self.writing_threshold['frequency_min'] <= freq <= self.writing_threshold['frequency_max']:
            freq_score = 1.0
        else:
            # Penalize frequencies outside range
            if freq < self.writing_threshold['frequency_min']:
                freq_score = freq / self.writing_threshold['frequency_min']
            else:
                freq_score = self.writing_threshold['frequency_max'] / freq
            freq_score = max(0, min(1, freq_score))
        
        # Vertical consistency score (lower variation is better)
        vertical_score = max(0, 1 - (vertical / 0.5))
        
        # Weighted average (all factors equally important)
        confidence = (x_score + y_score + freq_score + vertical_score) / 4.0
        
        return confidence
        
    def reset(self):
        """Clear history buffer."""
        self.wrist_history.clear()
        
    def get_buffer_size(self):
        """Get current number of samples in buffer."""
        return len(self.wrist_history)
