"""
Context-Aware Exam Phase Detection

This module identifies the current phase of an exam and adjusts detection
sensitivity accordingly to reduce false positives during normal exam activities.

Exam Phases:
- Settling (0-5 min): Students getting comfortable, organizing materials
- Reading (5-20 min): Reading questions, planning answers
- Writing (20-80 min): Main exam period, active writing
- Finishing (last 10 min): Reviewing, final checks
"""

import time


class ExamContextDetector:
    """
    Detect exam phases and adjust sensitivity accordingly.
    
    Different exam phases have different normal behaviors:
    - Settling: More movement is normal (organizing, adjusting)
    - Reading: Looking down at paper is normal
    - Writing: Hand movements are normal
    - Finishing: Some movement is normal (reviewing, checking)
    
    The detector applies phase-specific leniency multipliers and
    ignore rules to reduce false positives.
    """
    
    def __init__(self, exam_duration=5400):
        """
        Initialize exam context detector.
        
        Args:
            exam_duration: Total exam duration in seconds (default 5400 = 90 minutes)
        """
        self.exam_start_time = time.time()
        self.exam_duration = exam_duration
        self.current_phase = "settling"
        
        # Phase definitions with duration and leniency multipliers
        self.phase_thresholds = {
            'settling': {
                'duration': 300,      # First 5 minutes
                'leniency': 2.0,      # Very lenient (2x threshold)
                'description': 'Students settling in, organizing materials'
            },
            'reading': {
                'duration': 900,      # Next 15 minutes (5-20 min mark)
                'leniency': 1.5,      # Moderately lenient (1.5x threshold)
                'description': 'Reading questions, planning answers'
            },
            'writing': {
                'duration': None,     # Main exam period (variable)
                'leniency': 1.0,      # Standard detection (1x threshold)
                'description': 'Active writing and problem solving'
            },
            'finishing': {
                'duration': 600,      # Last 10 minutes
                'leniency': 1.3,      # Slightly lenient (1.3x threshold)
                'description': 'Reviewing answers, final checks'
            }
        }
        
    def get_current_phase(self):
        """
        Determine current exam phase based on elapsed time.
        
        Returns:
            String: One of ['settling', 'reading', 'writing', 'finishing']
        """
        elapsed = time.time() - self.exam_start_time
        
        if elapsed < 300:
            # First 5 minutes
            phase = "settling"
        elif elapsed < 1200:
            # Next 15 minutes (5-20 min mark)
            phase = "reading"
        elif elapsed > (self.exam_duration - 600):
            # Last 10 minutes
            phase = "finishing"
        else:
            # Main exam period
            phase = "writing"
            
        # Update current phase if changed
        if phase != self.current_phase:
            self.current_phase = phase
            print(f"[Info] Exam phase changed to: {phase} ({self.phase_thresholds[phase]['description']})")
            
        return phase
        
    def get_leniency_multiplier(self):
        """
        Get leniency multiplier for current exam phase.
        
        Leniency multiplier is applied to violation thresholds:
        - Higher multiplier = more lenient (harder to trigger violation)
        - Lower multiplier = stricter (easier to trigger violation)
        
        Returns:
            Float: Leniency multiplier for current phase
        """
        phase = self.get_current_phase()
        return self.phase_thresholds[phase]['leniency']
        
    def should_ignore_violation(self, violation_type):
        """
        Check if violation should be ignored in current context.
        
        Certain violations are normal during specific exam phases:
        - Settling: Head turns and hand movements (organizing)
        - Reading: Looking down (reading questions)
        - Writing: Hand movements (writing)
        
        Args:
            violation_type: Type of violation being checked
            
        Returns:
            Boolean: True if violation should be ignored, False otherwise
        """
        phase = self.get_current_phase()
        
        # Phase-specific ignore rules
        if phase == "settling":
            # Allow movement during settling period
            if violation_type in ["Head turned", "Hand suspicious"]:
                return True
                
        elif phase == "reading":
            # Allow looking down at paper during reading
            if violation_type == "Looking down":
                return True
                
        elif phase == "writing":
            # Allow hand movements during active writing
            if violation_type == "Hand suspicious":
                return True
                
        # No special rules for finishing phase (use standard detection)
        
        return False
        
    def get_elapsed_time(self):
        """
        Get elapsed time since exam start.
        
        Returns:
            Float: Elapsed time in seconds
        """
        return time.time() - self.exam_start_time
        
    def get_remaining_time(self):
        """
        Get remaining time until exam end.
        
        Returns:
            Float: Remaining time in seconds (0 if exam is over)
        """
        elapsed = self.get_elapsed_time()
        remaining = max(0, self.exam_duration - elapsed)
        return remaining
        
    def get_status_info(self):
        """
        Get status information for display/logging.
        
        Returns:
            Dictionary with phase, leniency, elapsed time, and remaining time
        """
        phase = self.get_current_phase()
        return {
            'phase': phase,
            'description': self.phase_thresholds[phase]['description'],
            'leniency': self.get_leniency_multiplier(),
            'elapsed': self.get_elapsed_time(),
            'remaining': self.get_remaining_time()
        }
