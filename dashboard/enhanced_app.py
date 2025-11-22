"""
Enhanced Dashboard Backend for Exam Monitoring System

Provides REST API and WebSocket streaming for real-time monitoring dashboard
with face recognition integration, violation analytics, and live camera feed.
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import sqlite3
import base64
import cv2
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
from collections import defaultdict
import threading


class DashboardBackend:
    """
    Backend service for enhanced exam monitoring dashboard.
    
    Provides real-time student status, violation analytics, camera streaming,
    and management actions through REST API and WebSocket connections.
    """
    
    def __init__(self, monitoring_db_path="monitoring.sqlite", faces_db_path="student_faces.sqlite"):
        """
        Initialize the DashboardBackend.
        
        Args:
            monitoring_db_path: Path to monitoring.sqlite database
            faces_db_path: Path to student_faces.sqlite database
        """
        self.monitoring_db = monitoring_db_path
        self.faces_db = faces_db_path
        self.camera_feed_active = False
        self.camera_lock = threading.Lock()
        
        # Cache for performance optimization
        self.status_cache = None
        self.status_cache_time = None
        self.analytics_cache = None
        self.analytics_cache_time = None
        
        print(f"[INFO] DashboardBackend initialized")
        print(f"[INFO] Monitoring DB: {self.monitoring_db}")
        print(f"[INFO] Faces DB: {self.faces_db}")
    
    def get_enhanced_status(self):
        """
        Get enhanced status for currently active student only.
        
        Returns the student that is currently being monitored (has recent last_update).
        Returns None if no student is currently being monitored.
        
        Returns:
            dict or None: Current student status if monitoring, None otherwise
        """
        # Check cache (1 second TTL for performance)
        now = time.time()
        if self.status_cache and self.status_cache_time and (now - self.status_cache_time) < 1.0:
            return self.status_cache
        
        try:
            # Connect to monitoring database
            mon_conn = sqlite3.connect(self.monitoring_db)
            mon_conn.row_factory = sqlite3.Row
            mon_cur = mon_conn.cursor()
            
            # Get the most recently updated student (currently being monitored)
            # Only consider updates within last 5 minutes (300 seconds)
            mon_cur.execute("""
                SELECT ms.student_id, ms.strikes, ms.status, ms.last_update,
                       s.name, s.seat_no
                FROM monitoring_status ms
                LEFT JOIN students s ON ms.student_id = s.student_id
                WHERE datetime(ms.last_update) > datetime('now', '-5 minutes')
                ORDER BY ms.last_update DESC
                LIMIT 1
            """)
            
            row = mon_cur.fetchone()
            
            # If no active monitoring, return None
            if not row:
                mon_conn.close()
                self.status_cache = None
                self.status_cache_time = now
                return None
            
            student_id = row['student_id']
            
            # Get recent violation count for this student
            mon_cur.execute("""
                SELECT COUNT(*) as count
                FROM violations
                WHERE student_id = ? AND ts > datetime('now', '-1 hour')
            """, (student_id,))
            violation_count_row = mon_cur.fetchone()
            violation_count = violation_count_row[0] if violation_count_row else 0
            
            mon_conn.close()
            
            # Connect to faces database to get photo and roll number
            try:
                faces_conn = sqlite3.connect(self.faces_db)
                faces_conn.row_factory = sqlite3.Row
                faces_cur = faces_conn.cursor()
                
                faces_cur.execute("""
                    SELECT roll_number, registration_photo
                    FROM students
                    WHERE student_id = ?
                """, (student_id,))
                
                face_row = faces_cur.fetchone()
                roll_number = face_row['roll_number'] if face_row else None
                photo_blob = face_row['registration_photo'] if face_row else None
                
                # Convert photo BLOB to base64
                photo = None
                if photo_blob:
                    photo_base64 = base64.b64encode(photo_blob).decode('utf-8')
                    photo = f"data:image/jpeg;base64,{photo_base64}"
                
                faces_conn.close()
            except sqlite3.Error as e:
                print(f"[WARNING] Could not access faces database: {e}")
                roll_number = None
                photo = None
            
            # Build status response
            status_dict = {
                'student_id': student_id,
                'name': row['name'] or 'Unknown',
                'roll_number': roll_number or 'N/A',
                'seat_number': row['seat_no'] or 'N/A',
                'strikes': row['strikes'],
                'status': row['status'],
                'last_update': row['last_update'],
                'photo': photo,
                'recent_violations': violation_count
            }
            
            # Update cache
            self.status_cache = status_dict
            self.status_cache_time = now
            
            return status_dict
            
        except sqlite3.Error as e:
            print(f"[ERROR] Database error in get_enhanced_status: {e}")
            return None
    
    def get_violation_analytics(self, student_id=None):
        """
        Get violation analytics for current student only.
        
        Returns violation distribution and timeline for the active student.
        Returns empty analytics if no student is being monitored.
        
        Args:
            student_id: Student ID to get analytics for (if None, uses current active student)
        
        Returns:
            dict: Analytics data with type_distribution and timeline
        """
        # If no student_id provided, get current active student
        if not student_id:
            current_status = self.get_enhanced_status()
            if not current_status:
                return {
                    'type_distribution': {},
                    'timeline': []
                }
            student_id = current_status['student_id']
        
        # Check cache (5 second TTL for performance)
        now = time.time()
        cache_key = f"{student_id}_{now // 5}"  # Cache key changes every 5 seconds
        if self.analytics_cache and hasattr(self, 'analytics_cache_key') and self.analytics_cache_key == cache_key:
            return self.analytics_cache
        
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cur = conn.cursor()
            
            # Violation type distribution for this student (last 1 hour)
            cur.execute("""
                SELECT violation_type, COUNT(*) as count
                FROM violations
                WHERE student_id = ? AND ts > datetime('now', '-1 hour')
                GROUP BY violation_type
                ORDER BY count DESC
            """, (student_id,))
            
            type_distribution = {}
            for row in cur.fetchall():
                type_distribution[row[0]] = row[1]
            
            # Violation timeline for this student (last 2 hours, 1-minute buckets)
            cur.execute("""
                SELECT 
                    strftime('%Y-%m-%d %H:%M', ts) as minute,
                    COUNT(*) as count
                FROM violations
                WHERE student_id = ? AND ts > datetime('now', '-2 hours')
                GROUP BY minute
                ORDER BY minute
            """, (student_id,))
            
            timeline = []
            for row in cur.fetchall():
                timeline.append({
                    'time': row[0],
                    'count': row[1]
                })
            
            conn.close()
            
            analytics = {
                'type_distribution': type_distribution,
                'timeline': timeline
            }
            
            # Update cache
            self.analytics_cache = analytics
            self.analytics_cache_key = cache_key
            self.analytics_cache_time = now
            
            return analytics
            
        except sqlite3.Error as e:
            print(f"[ERROR] Database error in get_violation_analytics: {e}")
            return {
                'type_distribution': {},
                'timeline': []
            }


# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'exam-monitoring-secret-key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize backend
dashboard_backend = DashboardBackend(
    monitoring_db_path=str(Path(__file__).resolve().parents[1] / 'monitoring.sqlite'),
    faces_db_path=str(Path(__file__).resolve().parents[1] / 'student_faces.sqlite')
)


# REST API Endpoints

@app.route('/')
def index():
    """Serve the enhanced dashboard HTML."""
    return render_template('enhanced_dashboard.html')


@app.route('/api/enhanced-status')
def api_enhanced_status():
    """Get enhanced student status with photos and identity info."""
    status = dashboard_backend.get_enhanced_status()
    return jsonify(status)


@app.route('/api/analytics')
def api_analytics():
    """Get violation analytics with type distribution and timeline."""
    analytics = dashboard_backend.get_violation_analytics()
    return jsonify(analytics)


@app.route('/api/live-status')
def api_live_status():
    """Get real-time live monitoring status from live_monitoring.json."""
    try:
        live_json_path = Path(__file__).resolve().parents[1] / 'live_monitoring.json'
        
        if not live_json_path.exists():
            return jsonify({
                'error': 'Live monitoring data not available',
                'active': False
            })
        
        with open(live_json_path, 'r') as f:
            live_data = json.load(f)
        
        # Check if data is recent (within last 10 seconds)
        if 'timestamp' in live_data:
            try:
                timestamp = datetime.fromisoformat(live_data['timestamp'].replace('Z', '+00:00'))
                time_diff = (datetime.now(timezone.utc) - timestamp.replace(tzinfo=timezone.utc)).total_seconds()
                live_data['active'] = time_diff < 10
                live_data['last_update_ago'] = f"{int(time_diff)}s ago"
            except:
                live_data['active'] = True
        else:
            live_data['active'] = True
        
        return jsonify(live_data)
        
    except Exception as e:
        print(f"[ERROR] Failed to read live monitoring data: {e}")
        return jsonify({
            'error': str(e),
            'active': False
        })


@app.route('/api/violations')
def api_violations():
    """Get recent violations for current student only."""
    try:
        limit = request.args.get('limit', 20, type=int)
        
        # Get current active student
        current_status = dashboard_backend.get_enhanced_status()
        if not current_status:
            return jsonify([])  # No active monitoring
        
        student_id = current_status['student_id']
        
        conn = sqlite3.connect(dashboard_backend.monitoring_db)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Get violations for this student only
        cur.execute("""
            SELECT v.violation_type, v.detail, v.ts
            FROM violations v
            WHERE v.student_id = ?
            ORDER BY v.id DESC
            LIMIT ?
        """, (student_id, limit,))
        
        # Build violations list
        violations = []
        for row in cur.fetchall():
            violations.append({
                'type': row['violation_type'],
                'detail': row['detail'],
                'ts': row['ts'],
                'severity': 'high' if 'phone' in row['violation_type'].lower() or 'debarred' in row['violation_type'].lower() else 'medium'
            })
        
        conn.close()
        return jsonify(violations)
        
    except sqlite3.Error as e:
        print(f"[ERROR] Database error in api_violations: {e}")
        return jsonify([])


@app.route('/api/camera-feed')
def api_camera_feed():
    """Start or stop camera feed streaming."""
    action = request.args.get('action', 'status')
    
    if action == 'start':
        dashboard_backend.camera_feed_active = True
        # Start streaming in background thread
        socketio.start_background_task(stream_camera_feed)
        return jsonify({'status': 'started', 'message': 'Camera feed started'})
    
    elif action == 'stop':
        dashboard_backend.camera_feed_active = False
        return jsonify({'status': 'stopped', 'message': 'Camera feed stopped'})
    
    else:
        return jsonify({
            'status': 'active' if dashboard_backend.camera_feed_active else 'inactive'
        })


@app.route('/api/reset-strikes/<student_id>', methods=['POST'])
def api_reset_strikes(student_id):
    """
    Reset strike count for a student.
    
    Args:
        student_id: Student ID to reset strikes for
        
    Returns:
        JSON response with success/error status
    """
    try:
        conn = sqlite3.connect(dashboard_backend.monitoring_db)
        cur = conn.cursor()
        
        # Check if student exists
        cur.execute("SELECT 1 FROM monitoring_status WHERE student_id = ?", (student_id,))
        if not cur.fetchone():
            conn.close()
            return jsonify({
                'success': False,
                'error': f'Student {student_id} not found'
            }), 404
        
        # Reset strikes to 0 and status to normal
        cur.execute("""
            UPDATE monitoring_status 
            SET strikes = 0, status = 'normal', last_update = datetime('now')
            WHERE student_id = ?
        """, (student_id,))
        
        conn.commit()
        conn.close()
        
        # Clear cache
        dashboard_backend.status_cache = None
        
        print(f"[INFO] Strikes reset for student {student_id}")
        
        return jsonify({
            'success': True,
            'message': f'Strikes reset for student {student_id}'
        })
        
    except sqlite3.Error as e:
        print(f"[ERROR] Database error in reset_strikes: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/flag-student/<student_id>', methods=['POST'])
def api_flag_student(student_id):
    """
    Manually flag a student for review.
    
    Args:
        student_id: Student ID to flag
        
    Returns:
        JSON response with success/error status
    """
    try:
        conn = sqlite3.connect(dashboard_backend.monitoring_db)
        cur = conn.cursor()
        
        # Check if student exists
        cur.execute("SELECT strikes FROM monitoring_status WHERE student_id = ?", (student_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return jsonify({
                'success': False,
                'error': f'Student {student_id} not found'
            }), 404
        
        # Set status to flagged (keep current strikes)
        cur.execute("""
            UPDATE monitoring_status 
            SET status = 'flagged', last_update = datetime('now')
            WHERE student_id = ?
        """, (student_id,))
        
        conn.commit()
        conn.close()
        
        # Clear cache
        dashboard_backend.status_cache = None
        
        print(f"[INFO] Student {student_id} manually flagged")
        
        return jsonify({
            'success': True,
            'message': f'Student {student_id} flagged for review'
        })
        
    except sqlite3.Error as e:
        print(f"[ERROR] Database error in flag_student: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# WebSocket Event Handlers

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('[INFO] Client connected to WebSocket')
    emit('connection_status', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('[INFO] Client disconnected from WebSocket')


@socketio.on('request_status_update')
def handle_status_request():
    """Handle client request for status update."""
    status = dashboard_backend.get_enhanced_status()
    emit('status_update', status)


def stream_camera_feed():
    """
    Stream camera feed via WebSocket with error handling and reconnection logic.
    
    Captures frames from camera, encodes as JPEG with quality=70,
    converts to base64, and emits at 10 FPS (0.1 second intervals).
    """
    cap = None
    consecutive_failures = 0
    max_consecutive_failures = 10
    
    try:
        # Try to open camera with multiple backends
        camera_backends = [0, cv2.CAP_ANY]
        camera_opened = False
        
        for backend in camera_backends:
            try:
                print(f"[INFO] Attempting to open camera with backend {backend}...")
                cap = cv2.VideoCapture(backend)
                if cap.isOpened():
                    print(f"[INFO] Camera opened successfully with backend {backend}")
                    camera_opened = True
                    break
                else:
                    if cap:
                        cap.release()
            except Exception as e:
                print(f"[ERROR] Failed to open camera with backend {backend}: {e}")
                if cap:
                    cap.release()
        
        if not camera_opened:
            print("[ERROR] Could not open camera for streaming")
            socketio.emit('camera_error', {'error': 'Could not open camera'})
            return
        
        print("[INFO] Camera streaming started")
        
        while dashboard_backend.camera_feed_active:
            try:
                with dashboard_backend.camera_lock:
                    ret, frame = cap.read()
                
                if not ret:
                    consecutive_failures += 1
                    print(f"[WARNING] Failed to read frame from camera (failure {consecutive_failures}/{max_consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        print("[ERROR] Too many consecutive failures, stopping stream")
                        socketio.emit('camera_error', {'error': 'Camera disconnected'})
                        break
                    
                    time.sleep(0.1)
                    continue
                
                # Reset failure counter on successful read
                consecutive_failures = 0
                
                # Encode frame as JPEG with quality=70
                try:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                    success, buffer = cv2.imencode('.jpg', frame, encode_param)
                    
                    if not success:
                        print("[WARNING] Failed to encode frame")
                        continue
                    
                    # Convert to base64
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Emit frame via WebSocket
                    socketio.emit('camera_frame', {
                        'frame': f"data:image/jpeg;base64,{frame_base64}",
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"[ERROR] Error encoding/emitting frame: {e}")
                    continue
                
                # Maintain 10 FPS (0.1 second sleep) for performance optimization
                # This limits bandwidth usage and client-side processing load
                time.sleep(0.1)
                
            except Exception as e:
                print(f"[ERROR] Error in streaming loop: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    break
                time.sleep(0.1)
        
        print("[INFO] Camera streaming stopped")
        
    except Exception as e:
        print(f"[ERROR] Fatal error in camera streaming: {e}")
        socketio.emit('camera_error', {'error': str(e)})
    
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception as e:
                print(f"[ERROR] Error releasing camera: {e}")
        dashboard_backend.camera_feed_active = False


if __name__ == '__main__':
    print("\n=== Enhanced Exam Monitoring Dashboard ===")
    print("Starting server on http://127.0.0.1:5500")
    print("Press Ctrl+C to stop\n")
    
    socketio.run(app, host='127.0.0.1', port=5500, debug=True, allow_unsafe_werkzeug=True)
