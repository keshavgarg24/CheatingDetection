# ==== Camera & Display ====
CAMERA_INDEX = 0           # 0 = default webcam
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
SHOW_WINDOW = True            # Show OpenCV window for demo

# ==== Monitoring scope ====
STUDENT_ID = "S1"             # Seat/student identifier for this camera
ALLOWED_FACES_PER_BENCH = 1   # If 2 students share a bench, set to 2

# ==== Strike logic ====
STRIKE_THRESHOLD = 3          # At or above this -> FLAGGED

# Time thresholds (seconds)
LOOK_AWAY_SECONDS = 3.0       # head turned away continuously
LOOK_DOWN_SECONDS = 10.0      # looking down continuously
NO_FACE_SECONDS = 5.0         # no face seen for this long -> strike

# Geometry thresholds (ratios relative to face bbox)
HEAD_TURN_RATIO = 0.25        # ~45 degrees yaw approximation
LOOK_DOWN_RATIO = 0.18        # nose below bbox center by this ratio
HAND_NEAR_FACE_RATIO = 0.35   # distance of wrist to face center vs bbox diag
LAP_ZONE_BOTTOM_RATIO = 0.75  # y > this*height considered lap area

# Hand gesture heuristics
HAND_EVENT_REPEAT = 3         # repeated near-face/lap/gesture events -> 1 strike
HAND_GESTURE_WINDOW = 30      # frames window to evaluate back-and-forth gestures
HAND_GESTURE_X_PIXELS = 80    # min horizontal swing to consider as signaling

# YOLO phone detection (optional)
USE_YOLO = False               # Set False to disable phone detection
YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.35
YOLO_CLASS_NAME = "cell phone"  # COCO label used by ultralytics

# Audio (optional, not recommended for crowded rooms)
USE_AUDIO = False              # Set True to experiment (classroom false positives likely)
AUDIO_SECONDS = 2.0            # analyze every N seconds
AUDIO_THRESHOLD = 0.02         # naive RMS threshold (toy demo)

# ==== Database ====
DB_PATH = "monitoring.sqlite"

# Dashboard toggle (run separately in dashboard/)
ENABLE_DASHBOARD = True
