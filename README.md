# Offline Exam Monitoring System (MediaPipe + OpenCV + SQLite)

This project demonstrates **offline exam behavior monitoring** for a single seat/bench using a laptop webcam.

## Features
- Face presence (no face > configurable seconds → strike)
- Multiple faces (beyond allowed bench count → strike)
- Head turned away (continuous seconds → strike)
- Looking down (continuous seconds → strike)
- Hand suspicious: near face / in lap / signaling gestures (repeated events → strike)
- Optional phone detection (YOLOv8)
- Optional naive audio energy (not recommended for crowded rooms)
- SQLite database for strikes + violation logs
- Minimal dashboard (Flask) to view status and recent violations

## Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
# Optional (heavy): phone detection
# pip install ultralytics torch torchvision
```

## Run monitoring
```bash
python monitoring.py
# Press 'q' to quit
```

## Run dashboard (separate terminal)
```bash
cd dashboard
# If venv not active, activate it
python app.py
# visit http://127.0.0.1:5500
```

## Config
Edit `config.py`:
- `ALLOWED_FACES_PER_BENCH` (1 for single-seater; 2 if bench has two students)
- Time/geometry thresholds
- Strike threshold
- `USE_YOLO` / `USE_AUDIO`

## Database
- File: `monitoring.sqlite`
- Tables:
  - `students(student_id, name, seat_no)`
  - `monitoring_status(student_id, strikes, status, last_update)`
  - `violations(student_id, violation_type, detail, ts)`

Use any SQLite viewer or:
```bash
python -c "import sqlite3; import pprint; con=sqlite3.connect('monitoring.sqlite'); print('\nSTATUS:'); print(con.execute('select student_id,strikes,status,last_update from monitoring_status').fetchall()); print('\nVIOLATIONS:'); print(con.execute('select student_id,violation_type,detail,ts from violations order by id desc limit 20').fetchall()); con.close()"
```

## Notes
- This is a **behavior flagger**, not proof of cheating. Tune thresholds to reduce false positives.
- Keep good lighting and stable camera.
- For multi-seat halls, run one instance per camera/seat (set different `STUDENT_ID`).
