# ==== Exam Monitoring Configuration ====
# Strict exam rules and violation detection
EXAM_MODE = True              # Enable strict exam monitoring mode
MAX_STRIKES = 3               # Maximum strikes before debarring
LOOKING_AWAY_TIMEOUT = 3.0    # Seconds of looking away for strike
BOUNDARY_MARGIN = 0.25        # ±25% boundary window around student face
SOFT_SCORE_TO_STRIKE = 3      # Soft violations needed to convert to strike
SOFT_VIOLATION_INTERVAL = 10.0 # Seconds between soft score increments
VIOLATION_COOLDOWN = 2.0      # Seconds cooldown between same violations

# Detection confidence thresholds for exam mode
EXAM_FACE_CONFIDENCE = 0.7    # Higher confidence for reliable detection
EXAM_PHONE_CONFIDENCE = 0.6   # Phone detection confidence threshold
EXAM_GAZE_SENSITIVITY = 0.8   # Eye gaze detection sensitivity

# Behavioral monitoring in exam mode
FACE_STABILITY_FRAMES = 10    # Frames to confirm stable face detection
BOUNDARY_UPDATE_THRESHOLD = 5 # Frames before updating boundary position
FALSE_POSITIVE_FILTER = True  # Enable filtering of false positive detections

# Violation types and severity for exam mode (ABSENCE CRITERIA REMOVED)
EXAM_VIOLATIONS = {
    "phone_detected": "instant_flag",            # INSTANT FLAG - immediate debar  
    "materials_detected": "instant_flag",        # INSTANT FLAG - immediate debar
    "electronic_device": "instant_flag",         # INSTANT FLAG - immediate debar
    "second_face_inside_boundary": "hard",       # Immediate strike
    "face_proximity": "hard",                    # Immediate strike for faces too close
    "looking_away_sustained": "hard",            # Immediate strike after timeout
    "head_turned": "hard",                       # Immediate strike after timeout
    "looking_down": "soft",                      # Soft violation - normal for reading/writing
    "hand_suspicious": "soft",                   # Soft violation
    "excessive_motion": "soft",                  # Soft violation
    "frequent_side_glances": "soft",             # Soft violation
    "face_partially_out_of_frame": "soft",      # Soft violation
    "prolonged_eye_closure": "soft"              # Soft violation
}

# ==== Camera & Display ====
CAMERA_INDEX = 0           # 0 = default webcam
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
SHOW_WINDOW = True            # Show OpenCV window for demo

# ==== Monitoring scope ====
STUDENT_ID = "222"             # Seat/student identifier for this camera (matches registered student)
ALLOWED_FACES_PER_BENCH = 1   # Only 1 face allowed in exam mode
EXAM_ROOM_ID = "ROOM_A"       # Exam room identifier
STRICT_MONITORING = True      # Enable strict monitoring for exams

# ==== Strike logic - Graduated penalty system ====
STRIKE_THRESHOLD = 5          # At or above this -> FLAGGED (more forgiving)
WARNING_THRESHOLD = 2         # At or above this -> WARNING status
CRITICAL_THRESHOLD = 8        # At or above this -> CRITICAL (immediate attention)

# Violation severity levels (strikes awarded per violation)
VIOLATION_SEVERITY = {
    "minor": 0.5,      # Minor violations (looking around briefly)
    "moderate": 1.0,   # Moderate violations (sustained suspicious behavior) 
    "major": 2.0,      # Major violations (phone detected, identity mismatch)
    "critical": 3.0    # Critical violations (multiple people, clear cheating)
}

# Alternative name for compatibility
VIOLATION_SEVERITY_THRESHOLDS = VIOLATION_SEVERITY

# Head turn duration threshold for enhanced detection
HEAD_TURN_DURATION_THRESHOLD = 5.0  # 5 seconds for realistic exam behavior

# Time thresholds (seconds) - Enhanced detection parameters
LOOK_AWAY_SECONDS = 1.8       # Enhanced sensitivity for looking away
LOOK_DOWN_SECONDS = 15.0      # Longer threshold for reading/writing
EYE_TRACKING_SECONDS = 2.0    # More sensitive eye movement detection
BODY_POSTURE_SECONDS = 8.0    # unusual body posture duration
REPEATED_GESTURE_SECONDS = 15.0 # repetitive suspicious gestures

# Geometry thresholds (ratios relative to face bbox) - Enhanced detection
HEAD_TURN_RATIO = 0.22        # More sensitive head turn detection
LOOK_DOWN_RATIO = 0.18        # More sensitive looking down detection
HAND_NEAR_FACE_RATIO = 0.40   # distance of wrist to face center vs bbox diag (thinking pose)
LAP_ZONE_BOTTOM_RATIO = 0.75  # y > this*height considered lap area (normal lap zone)
EYE_GAZE_DEVIATION_RATIO = 0.5 # eye gaze deviation from expected direction
POSTURE_LEAN_RATIO = 0.35     # body lean angle threshold (30 degrees)
SUSPICIOUS_MOVEMENT_RATIO = 0.6 # threshold for suspicious movement patterns

# Hand gesture heuristics - Realistic detection
HAND_EVENT_REPEAT = 4         # repeated near-face/lap/gesture events -> 1 strike (less sensitive)
HAND_GESTURE_WINDOW = 45      # frames window to evaluate back-and-forth gestures (longer window)
HAND_GESTURE_X_PIXELS = 100   # min horizontal swing to consider as signaling (more motion needed)
WRITING_CONFIDENCE_THRESHOLD = 0.6 # threshold above which hand movement is considered writing (lower)
SUSPICIOUS_HAND_SPEED = 80    # pixels per frame for suspicious hand movement
HAND_COVERING_THRESHOLD = 0.7 # ratio of face area covered by hands (more coverage needed)

# YOLO device detection (specific to cheating devices only)
USE_YOLO = True               # Set False to disable device detection
YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.45        # Higher confidence to reduce false positives
YOLO_CHEATING_DEVICES = ["phone", "laptop", "tablet"]  # Only these devices trigger violations

# Audio detection - DISABLED
USE_AUDIO = False              # Audio detection disabled by user request

# Audio feedback for violations (NEW)
USE_AUDIO_ALERTS = True        # Enable audio alerts for strikes
AUDIO_ALERTS = {
    "first_strike": "You have received your first strike. Please focus on your exam.",
    "second_strike": "You have received your second strike. One more strike will result in disbarment.",
    "third_strike": "You have been disbarred from the examination due to violations.",
    "instant_flag": "You have been immediately disbarred due to unauthorized material detection.",
    "phone_detected": "Electronic device detected. You are being disbarred immediately.",
    "final_warning": "This is your final warning. Any further violations will result in disbarment."
}
AUDIO_VOICE_SPEED = 1.0        # Speech speed (1.0 = normal)
AUDIO_VOLUME = 0.8             # Volume level (0.0 to 1.0)

# Visual alerts configuration
SHOW_STRIKE_ALERTS = True      # Show strike alerts on monitoring screen
SHOW_INSTANT_FLAG_ALERT = True # Show big alert for instant flagging
ALERT_DISPLAY_DURATION = 3.0   # Seconds to show alerts
STRIKE_ALERT_COLOR = (0, 255, 255)     # Yellow for strikes
INSTANT_FLAG_COLOR = (0, 0, 255)       # Red for instant flagging
ALERT_FONT_SIZE = 1.5          # Font size for alerts

# ==== Database ====
DB_PATH = "monitoring.sqlite"

# Detection modules setup 
EYE_GAZE_ENABLED = True
POSTURE_DETECTION_ENABLED = True
MOTION_DETECTION_ENABLED = True
PHONE_DETECTION_ENABLED = True

# Frame processing requirements
MIN_FRAME_QUALITY = 0.6
FRAME_QUALITY_THRESHOLD = 0.7
MIN_FACE_SIZE = 80

# Adaptive monitoring (per-student calibration)
ADAPTIVE_LEARNING_ENABLED = True
LEARNING_PERIOD = 600              # 10 minutes to learn student baseline
ADAPTIVE_CONFIDENCE_THRESHOLD = 0.7
BEHAVIOR_UPDATE_FREQUENCY = 3      # Update behavioral model every 3 frames
ADAPTIVE_SMOOTHING_FACTOR = 0.8    # How much to smooth threshold adaptations
MIN_SAMPLES_FOR_ADAPTATION = 20    # Minimum samples before adapting thresholds
ADAPTIVE_BOUNDS_MULTIPLIER = 2.5   # How much to expand thresholds during adaptation

# Context-aware violations (exam phase based adjustments)
CONTEXT_AWARE_ENABLED = True
EXAM_DURATION = 5400              # 90 minutes total exam time
CONTEXT_SENSITIVITY_MULTIPLIER = 1.5  # Multiplier during exam phase sensitivity adjustment
PHASE_TRANSITION_BUFFER = 60      # 60 second buffer between phase transitions

# Violation cooldowns - prevent repeated violations
VIOLATION_COOLDOWNS = {
    "Phone detected": 120,        # 2 minutes (serious offense)
    "Hand suspicious": 20,        # 20 seconds (frequent during exams)
    "Multiple faces": 60,         # 1 minute (collaboration)
    "Head turned": 15,            # 15 seconds (brief head turns)
    "Looking down": 25,           # 25 seconds (reading/writing)
    "Eye gaze suspicious": 30,    # 30 seconds (eye movement patterns)
    "Motion excessive": 25,       # 25 seconds (body movement)
    "Face not in frame": 20,      # 20 seconds (repositioning)
    "Writing detected": 35,       # 35 seconds (writing patterns)
    "Gesture suspicious": 45,     # 45 seconds (hand gestures)
}