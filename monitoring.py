import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, SHOW_WINDOW,
    STUDENT_ID, ALLOWED_FACES_PER_BENCH,
    STRIKE_THRESHOLD,
    LOOK_AWAY_SECONDS, LOOK_DOWN_SECONDS, NO_FACE_SECONDS,
    HEAD_TURN_RATIO, LOOK_DOWN_RATIO,
    HAND_NEAR_FACE_RATIO, LAP_ZONE_BOTTOM_RATIO,
    HAND_EVENT_REPEAT, HAND_GESTURE_WINDOW, HAND_GESTURE_X_PIXELS,
    USE_YOLO, USE_AUDIO
)
from db import init_db, ensure_student, log_violation, set_strikes, get_status
from context import now_iso


yolo_model = None
if USE_YOLO:
    from yolo_phone import load_model, detect_phones
    yolo_model = load_model()

vad = None
if USE_AUDIO:
    from audio_vad import SimpleVAD
    vad = SimpleVAD()

mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands

def bbox_from_detection(det, w, h):
    rel = det.location_data.relative_bounding_box
    x1 = int(rel.xmin * w); y1 = int(rel.ymin * h)
    x2 = int((rel.xmin + rel.width) * w); y2 = int((rel.ymin + rel.height) * h)
    return [max(0,x1), max(0,y1), min(w,x2), min(h,y2)]

def face_center_and_keypoints(det, w, h):
    bbox = bbox_from_detection(det, w, h)
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    kps = {}
    for idx, kp in enumerate(det.location_data.relative_keypoints):
        kps[idx] = (int(kp.x * w), int(kp.y * h))
    return (cx, cy), bbox, kps

def main():
    # DB prep
    init_db()
    ensure_student(STUDENT_ID, name=None, seat_no=None)
    set_strikes(STUDENT_ID, 0)

    # for openeing camera
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print("[Error] Could not open camera")
        return

    if vad:
        vad.start()

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_det, \
         mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0,
                        min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:

        last_face_seen_at = time.monotonic()
        look_away_since = None
        look_down_since = None

        # Gesture tracking
        hand_events = 0
        last_hand_near_face = False
        last_hand_in_lap = False
        # For gesture back-and-forth detection
        wrist_x_history = deque(maxlen=HAND_GESTURE_WINDOW)

        print("[Info] Monitoring started. Press 'q' to quit.")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[Error] Camera read failure")
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Face detection
            fr = face_det.process(rgb)
            detections = fr.detections if fr and fr.detections else []

            # Multiple/none face logic
            if len(detections) >= 1:
                last_face_seen_at = time.monotonic()
                if len(detections) > ALLOWED_FACES_PER_BENCH:
                    log_violation(STUDENT_ID, "Multiple faces", f"count={len(detections)}")
            else:
                if time.monotonic() - last_face_seen_at >= NO_FACE_SECONDS:
                    log_violation(STUDENT_ID, "No face", f">= {NO_FACE_SECONDS}s")
                    last_face_seen_at = time.monotonic()  # avoid repeated spam

            # Pick main face (largest area)
            main_det = None; main_bbox = None; main_center = None; keypoints = None
            if detections:
                det_areas = []
                for det in detections:
                    box = bbox_from_detection(det, w, h)
                    area = (box[2]-box[0])*(box[3]-box[1])
                    det_areas.append((area, det))
                main_det = sorted(det_areas, key=lambda x: x[0], reverse=True)[0][1]
                main_center, main_bbox, keypoints = face_center_and_keypoints(main_det, w, h)

            # Hands
            hr = hands.process(rgb)
            hand_landmarks = hr.multi_hand_landmarks or []

            # Rules
            # 1) Head turned away
            if main_det is not None:
                (cx, cy) = main_center
                x1, y1, x2, y2 = main_bbox
                bbox_w = max(1, x2 - x1); bbox_h = max(1, y2 - y1)
                diag = (bbox_w**2 + bbox_h**2) ** 0.5

                RIGHT_EYE, LEFT_EYE, NOSE_TIP, MOUTH_CENTER = 0, 1, 2, 3
                if all(k in keypoints for k in [RIGHT_EYE, LEFT_EYE, NOSE_TIP]):
                    ex_r, ey_r = keypoints[RIGHT_EYE]
                    ex_l, ey_l = keypoints[LEFT_EYE]
                    nx, ny = keypoints[NOSE_TIP]
                    mid_eye_x = (ex_r + ex_l) / 2.0
                    horiz_offset = abs(nx - mid_eye_x) / float(bbox_w)
                    head_turned = horiz_offset > HEAD_TURN_RATIO

                    now = time.monotonic()
                    if head_turned:
                        if look_away_since is None:
                            look_away_since = now
                        elif (now - look_away_since) >= LOOK_AWAY_SECONDS:
                            log_violation(STUDENT_ID, "Head turned", f">= {LOOK_AWAY_SECONDS}s")
                            look_away_since = None
                    else:
                        look_away_since = None

                    # 2) Looking down
                    bbox_cy = (y1 + y2) / 2.0
                    down_condition = (ny - bbox_cy) / float(bbox_h) > LOOK_DOWN_RATIO
                    if down_condition:
                        if look_down_since is None:
                            look_down_since = now
                        elif (now - look_down_since) >= LOOK_DOWN_SECONDS:
                            log_violation(STUDENT_ID, "Looking down", f">= {LOOK_DOWN_SECONDS}s")
                            look_down_since = None
                    else:
                        look_down_since = None

            # 3) Hands near face / lap & gesture signals
            hand_near_face = False
            hand_in_lap = False
            gesture_signal = False
            wrist_x = None

            if hand_landmarks:
                # compute face center/diag if available for distances
                face_center = None; diag = None
                if main_bbox is not None:
                    x1, y1, x2, y2 = main_bbox
                    bbox_w = max(1, x2 - x1); bbox_h = max(1, y2 - y1)
                    face_center = ((x1 + x2)//2, (y1 + y2)//2)
                    diag = (bbox_w**2 + bbox_h**2) ** 0.5

                for hand in hand_landmarks:
                    wrist = hand.landmark[0]
                    px = int(wrist.x * w); py = int(wrist.y * h)
                    if face_center and diag:
                        d = np.hypot(px - face_center[0], py - face_center[1])
                        if (d / diag) < HAND_NEAR_FACE_RATIO:
                            hand_near_face = True
                    if py > int(LAP_ZONE_BOTTOM_RATIO * h):
                        hand_in_lap = True
                    # gesture heuristic: track wrist x movement
                    wrist_x = px
                    cv2.circle(frame, (px, py), 5, (255,255,255), -1)

            # Rising-edge counters (avoid per-frame spam)
            if hand_near_face and not last_hand_near_face:
                hand_events += 1
            if hand_in_lap and not last_hand_in_lap:
                hand_events += 1
            last_hand_near_face = hand_near_face
            last_hand_in_lap = hand_in_lap

            # gesture detection: back-and-forth within window
            if wrist_x is not None:
                wrist_x_history.append(wrist_x)
                if len(wrist_x_history) >= wrist_x_history.maxlen:
                    span = max(wrist_x_history) - min(wrist_x_history)
                    if span >= HAND_GESTURE_X_PIXELS:
                        hand_events += 1
                        wrist_x_history.clear()

            if hand_events >= HAND_EVENT_REPEAT:
                log_violation(STUDENT_ID, "Hand suspicious", f"events={hand_events}")
                hand_events = 0

            # 4) Phone detection
            if USE_YOLO and yolo_model is not None:
                # Run YOLO every 5 frames to save compute
                if int(time.time() * 5) % 5 == 0:
                    try:
                        from yolo_phone import detect_phones
                        phones = detect_phones(yolo_model, frame)
                        for ph in phones:
                            x1,y1,x2,y2 = ph['bbox']
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                            cv2.putText(frame, 'PHONE', (x1, y1-6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                            log_violation(STUDENT_ID, "Phone detected", f"conf={ph['conf']:.2f}")
                    except Exception as e:
                        print(f"[YOLO] detection error: {e}")

            # 5) Audio (optional, naive)
            if vad and int(time.time()) % max(1,int(vad.rate*0+1)) == 0:  # periodic check
                try:
                    if vad.speech_present():
                        log_violation(STUDENT_ID, "Speech/whisper", "RMS above threshold")
                except Exception as e:
                    print(f"[Audio] {e}")

            # Overlay status
            rec = get_status(STUDENT_ID)
            status_text = "normal"; strikes = 0
            if rec:
                _, sid, strikes, st, _ = rec
                status_text = st

            color = (0,200,0) if status_text == 'normal' else (0,215,255) if status_text == 'warning' else (0,0,255)
            cv2.putText(frame, f"{STUDENT_ID} | {status_text.upper()} | strikes={strikes}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            # Draw lap line
            lap_y = int(LAP_ZONE_BOTTOM_RATIO * h)
            cv2.line(frame, (0, lap_y), (w, lap_y), (128,128,128), 1)
            cv2.putText(frame, "LAP ZONE", (10, lap_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 1, cv2.LINE_AA)

            if SHOW_WINDOW:
                cv2.imshow("Exam Monitoring (q=quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    if vad:
        vad.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("[Info] Monitoring stopped.")

if __name__ == "__main__":
    main()
