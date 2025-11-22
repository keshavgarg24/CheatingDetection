# Enhanced device detection for exam monitoring - only cheating-relevant devices
# This module detects phones, laptops, and tablets only (not wires, chargers, etc.)

from typing import List
from config import YOLO_MODEL, YOLO_CONFIDENCE

# COCO class names for cheating-relevant devices only
CHEATING_DEVICE_CLASSES = {
    67: 'cell phone',     # Mobile phones
    63: 'laptop',         # Laptops 
    # Note: tablets are often detected as 'cell phone' or sometimes as custom classes
}

def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL)
        print("[YOLO] Loaded model for cheating device detection (phones, laptops, tablets only)")
        return model
    except Exception as e:
        print(f"[YOLO] Not available: {e}")
        return None

def detect_phones(model, frame, confidence=None, debug=False) -> List[dict]:
    """Detect only cheating-relevant devices (phones, laptops, tablets)
    Returns list of detections: [{ 'bbox': (x1,y1,x2,y2), 'conf': float, 'class_name': str }]
    """
    if model is None:
        return []
        
    # Use provided confidence or default from config
    conf_threshold = confidence if confidence is not None else YOLO_CONFIDENCE
    
    try:
        results = model(frame, conf=conf_threshold, verbose=False)
    except Exception as e:
        print(f"[YOLO] Inference error: {e}")
        return []
    
    detected_devices = []
    
    for r in results:
        names = r.names if hasattr(r, 'names') else {}
        boxes = getattr(r, 'boxes', None)
        if boxes is None or boxes.data is None:
            continue
            
        for det in boxes.data:
            x1, y1, x2, y2, conf, cls_id = det.tolist()
            class_id = int(cls_id)
            confidence_score = float(conf)
            label = names.get(class_id, str(class_id))
            
            # Only detect specific cheating-relevant devices
            is_cheating_device = False
            device_name = label.lower()
            
            # Check for phones, laptops, tablets specifically
            if (class_id == 67 or  # Official 'cell phone' class
                'phone' in device_name or 
                'mobile' in device_name):
                is_cheating_device = True
                final_name = 'cell phone'
                
            elif (class_id == 63 or  # Official 'laptop' class
                  'laptop' in device_name or 
                  'notebook' in device_name):
                is_cheating_device = True
                final_name = 'laptop'
                
            elif ('tablet' in device_name or 
                  'ipad' in device_name):
                is_cheating_device = True
                final_name = 'tablet'
            
            # Only add if it's a cheating-relevant device
            if is_cheating_device:
                detected_devices.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)), 
                    'conf': confidence_score,
                    'class_name': final_name,
                    'class_id': class_id
                })
                
                if debug:
                    print(f"[CHEATING DEVICE] Detected {final_name}: conf={confidence_score:.2f}")
    
    return detected_devices
