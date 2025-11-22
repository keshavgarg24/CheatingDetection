"""
Adaptive Threshold Learning System for Exam Monitoring

This module learns individual student behavior patterns during an initial
learning period and dynamically adjusts violation thresholds to reduce
false positives from legitimate exam activities.
"""

import time
import numpy as np
from collections import defaultdict, deque


class AdaptiveThresholdManager:
    """
    Learn student behavior patterns and adapt violation thresholds.
    
    During a learning period (default 5 minutes), the system observes
    normal student behavior and calculates baseline patterns. After the
    learning period, thresholds are adjusted based on mean + 2*std of
    observed behaviors.
    
    This reduces false positives from:
    - Individual writing styles
    - Natural head movement patterns
    - Desk setup variations
    """
    
    def __init__(self, learning_period=300):
        """
        Initialize adaptive threshold manager.
        
        Args:
            learning_period: Duration in seconds to learn baseline (default 300 = 5 minutes)
        """
        self.learning_period = learning_period
        self.student_baselines = defaultdict(dict)
        self.start_time = time.time()
        self.is_learning_phase = True
        
        # Behavior tracking with rolling buffers
        # Each deque stores recent behavior samples for statistical analysis
        self.behavior_history = {
            'head_movements': deque(maxlen=100),      # Head turn frequency/magnitude
            'hand_positions': deque(maxlen=100),      # Hand position variations
            'writing_frequency': deque(maxlen=50)     # Writing activity frequency
        }
        
    def update_behavior(self, behavior_type, value):
        """
        Add a behavior sample to history.
        
        Args:
            behavior_type: Type of behavior ('head_movements', 'hand_positions', 'writing_frequency')
            value: Numerical value representing the behavior measurement
        """
        if behavior_type in self.behavior_history:
            self.behavior_history[behavior_type].append(value)
            
            # Print learning progress for debugging
            if len(self.behavior_history[behavior_type]) % 10 == 0:
                print(f"[Adaptive] {behavior_type}: {len(self.behavior_history[behavior_type])} samples collected")
            
        # Check if learning phase should end
        current_time = time.time()
        if current_time - self.start_time > self.learning_period:
            if self.is_learning_phase:
                print(f"[Adaptive] Learning phase complete after {self.learning_period}s")
                print(f"[Adaptive] Collected samples: {[(k, len(v)) for k, v in self.behavior_history.items()]}")
                
                # Calculate initial baselines
                for behavior_type_key, history in self.behavior_history.items():
                    if len(history) >= 5:  # Minimum samples for baseline
                        mean_val = np.mean(list(history))
                        std_val = np.std(list(history))
                        self.student_baselines[behavior_type_key] = {
                            'mean': mean_val,
                            'std': std_val,
                            'samples': len(history)
                        }
                        print(f"[Adaptive] {behavior_type_key} baseline: mean={mean_val:.3f}, std={std_val:.3f}")
            self.is_learning_phase = False
            
    def get_adaptive_threshold(self, behavior_type, default_threshold):
        """
        Calculate adaptive threshold based on learned behavior.
        
        Uses statistical approach: threshold = mean + 2*std
        This captures ~95% of normal behavior while flagging outliers.
        
        Args:
            behavior_type: Type of behavior to get threshold for
            default_threshold: Fallback threshold if insufficient data
            
        Returns:
            Adaptive threshold value (bounded between 0.5x and 2.0x default)
        """
        # Use default during learning or if no data
        if self.is_learning_phase or behavior_type not in self.behavior_history:
            return default_threshold
            
        history = list(self.behavior_history[behavior_type])
        if len(history) < 10:  # Need minimum samples for statistics
            return default_threshold
            
        # Calculate dynamic threshold using statistical approach
        mean_behavior = np.mean(history)
        std_behavior = np.std(history)
        adaptive_threshold = mean_behavior + 2 * std_behavior
        
        # Ensure reasonable bounds to prevent extreme values
        min_threshold = default_threshold * 0.5
        max_threshold = default_threshold * 2.0
        
        bounded_threshold = max(min_threshold, min(adaptive_threshold, max_threshold))
        
        return bounded_threshold
        
    def is_violation(self, behavior_type, current_value, default_threshold):
        """
        Check if current behavior exceeds adaptive threshold.
        
        During learning phase, applies 1.5x leniency to reduce false positives
        while the system is still learning normal patterns.
        
        Args:
            behavior_type: Type of behavior being checked
            current_value: Current measurement value
            default_threshold: Default threshold value
            
        Returns:
            True if behavior is a violation, False otherwise
        """
        threshold = self.get_adaptive_threshold(behavior_type, default_threshold)
        
        if self.is_learning_phase:
            # During learning, be more lenient (1.5x multiplier)
            # This prevents flagging normal behavior during baseline establishment
            return current_value > (threshold * 1.5)
        else:
            # After learning, use calculated adaptive threshold
            return current_value > threshold
            
    def get_learning_progress(self):
        """
        Get current learning progress as percentage.
        
        Returns:
            Float between 0.0 and 1.0 representing learning progress
        """
        elapsed = time.time() - self.start_time
        progress = min(elapsed / self.learning_period, 1.0)
        return progress
        
    def get_status_info(self):
        """
        Get status information for display/logging.
        
        Returns:
            Dictionary with learning phase status and sample counts
        """
        return {
            'is_learning': self.is_learning_phase,
            'progress': self.get_learning_progress(),
            'samples': {k: len(v) for k, v in self.behavior_history.items()}
        }
