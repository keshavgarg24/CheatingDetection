# Optional phone detection via YOLOv8 (Ultralytics).
# This module is only used if USE_YOLO=True in config.py.

from typing import List
from config import YOLO_MODEL, YOLO_CONFIDENCE, YOLO_CLASS_NAME

def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL)
        return model
    except Exception as e:
        print(f"[YOLO] Not available: {e}")
        return None

def detect_phones(model, frame) -> List[dict]:
    """Return list of detections for phones in the frame.
    Each item: { 'bbox': (x1,y1,x2,y2), 'conf': float }
    """
    if model is None:
        return []
    try:
        results = model(frame, conf=YOLO_CONFIDENCE, verbose=False)
    except Exception as e:
        print(f"[YOLO] Inference error: {e}")
        return []
    out = []
    for r in results:
        names = r.names if hasattr(r, 'names') else {}
        boxes = getattr(r, 'boxes', None)
        if boxes is None or boxes.data is None:
            continue
        for det in boxes.data:
            x1, y1, x2, y2, conf, cls_id = det.tolist()
            label = names.get(int(cls_id), str(int(cls_id)))
            if label == YOLO_CLASS_NAME:
                out.append({'bbox': (int(x1), int(y1), int(x2), int(y2)), 'conf': float(conf)})
    return out
