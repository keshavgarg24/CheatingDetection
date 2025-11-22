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
from datetime import datetime, timedelta
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
        Get enhanced student status with identity info and photos.
        
        Optimized with batched queries and 1-second cache.
        Joins monitoring_status with students table from both databases,
        includes student photos as base64, and recent violation counts.
        
        Returns:
            list: List of dicts containing enhanced student status
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
            
            # Optimized: Get monitoring status with JOIN (single query instead of multiple)
            mon_cur.execute("""
                SELECT ms.student_id, ms.strikes, ms.status, ms.last_update,
                       s.name, s.seat_no
                FROM monitoring_status ms
                LEFT JOIN students s ON ms.student_id = s.student_id
                ORDER BY s.seat_no
            """)
            
            status_rows = mon_cur.fetchall()
            
            # Optimized: Get recent violation counts per student (uses index on ts and student_id)
            violation_counts = {}
            mon_cur.execute("""
                SELECT student_id, COUNT(*) as count
                FROM violations
                WHERE ts > datetime('now', '-1 hour')
                GROUP BY student_id
            """)
            for row in mon_cur.fetchall():
                violation_counts[row[0]] = row[1]
            
            mon_conn.close()
            
            # Connect to faces database
            try:
                faces_conn = sqlite3.connect(self.faces_db)
                faces_conn.row_factory = sqlite3.Row
                faces_cur = faces_conn.cursor()
                
                # Get student photos and additional info
                faces_cur.execute("""
                    SELECT student_id, roll_number, registration_photo
                    FROM students
                """)
                
                student_photos = {}
                student_rolls = {}
                for row in faces_cur.fetchall():
                    student_id = row['student_id']
                    student_rolls[student_id] = row['roll_number']
                    
                    # Convert photo BLOB to base64
                    if row['registration_photo']:
                        photo_base64 = base64.b64encode(row['registration_photo']).decode('utf-8')
                        student_photos[student_id] = f"data:image/jpeg;base64,{photo_base64}"
                    else:
                        student_photos[student_id] = None
                
                faces_conn.close()
            except sqlite3.Error as e:
                print(f"[WARNING] Could not access faces database: {e}")
                student_photos = {}
                student_rolls = {}
            
            # Build enhanced status response
            enhanced_status = []
            for row in status_rows:
                student_id = row['student_id']
                
                status_dict = {
                    'student_id': student_id,
                    'name': row['name'] or 'Unknown',
                    'roll_number': student_rolls.get(student_id, 'N/A'),
                    'seat_number': row['seat_no'] or 'N/A',
                    'strikes': row['strikes'],
                    'status': row['status'],
                    'last_update': row['last_update'],
                    'photo': student_photos.get(student_id),
                    'recent_violations': violation_counts.get(student_id, 0)
                }
                
                enhanced_status.append(status_dict)
            
            # Update cache
            self.status_cache = enhanced_status
            self.status_cache_time = now
            
            return enhanced_status
            
        except sqlite3.Error as e:
            print(f"[ERROR] Database error in get_enhanced_status: {e}")
            return []
    
    def get_violation_analytics(self):
        """
        Get violation analytics with type distribution and timeline.
        
        Optimized with batched queries and 5-second cache.
        Calculates:
        - Violation distribution by type (last 1 hour)
        - Violation timeline (last 2 hours, 1-minute granularity)
        - Risk distribution (high/medium/low/none based on strikes)
        
        Returns:
            dict: Analytics data with type_distribution, timeline, risk_distribution
        """
        # Check cache (5 second TTL for performance)
        now = time.time()
        if self.analytics_cache and self.analytics_cache_time and (now - self.analytics_cache_time) < 5.0:
            return self.analytics_cache
        
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cur = conn.cursor()
            
            # 1. Violation type distribution (last 1 hour)
            cur.execute("""
                SELECT violation_type, COUNT(*) as count
                FROM violations
                WHERE ts > datetime('now', '-1 hour')
                GROUP BY violation_type
                ORDER BY count DESC
            """)
            
            type_distribution = {}
            for row in cur.fetchall():
                type_distribution[row[0]] = row[1]
            
            # 2. Violation timeline (last 2 hours, 1-minute buckets)
            cur.execute("""
                SELECT 
                    strftime('%Y-%m-%d %H:%M', ts) as minute,
                    COUNT(*) as count
                FROM violations
                WHERE ts > datetime('now', '-2 hours')
                GROUP BY minute
                ORDER BY minute
            """)
            
            timeline = []
            for row in cur.fetchall():
                timeline.append({
                    'time': row[0],
                    'count': row[1]
                })
            
            # 3. Risk distribution (based on current strikes)
            cur.execute("""
                SELECT strikes FROM monitoring_status
            """)
            
            risk_distribution = {
                'high': 0,    # strikes >= 3
                'medium': 0,  # strikes >= 2
                'low': 0,     # strikes >= 1
                'none': 0     # strikes = 0
            }
            
            for row in cur.fetchall():
                strikes = row[0]
                if strikes >= 3:
                    risk_distribution['high'] += 1
                elif strikes >= 2:
                    risk_distribution['medium'] += 1
                elif strikes >= 1:
                    risk_distribution['low'] += 1
                else:
                    risk_distribution['none'] += 1
            
            conn.close()
            
            analytics = {
                'type_distribution': type_distribution,
                'timeline': timeline,
                'risk_distribution': risk_distribution
            }
            
            # Update cache
            self.analytics_cache = analytics
            self.analytics_cache_time = now
            
            return analytics
            
        except sqlite3.Error as e:
            print(f"[ERROR] Database error in get_violation_analytics: {e}")
            return {
                'type_distribution': {},
                'timeline': [],
                'risk_distribution': {'high': 0, 'medium': 0, 'low': 0, 'none': 0}
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


@app.route('/api/violations')
def api_violations():
    """Get recent violations with optimized query."""
    try:
        limit = request.args.get('limit', 20, type=int)
        
        conn = sqlite3.connect(dashboard_backend.monitoring_db)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Optimized: Single JOIN query instead of multiple lookups
        # Uses index on violations.id for fast DESC ordering
        cur.execute("""
            SELECT v.student_id, v.violation_type, v.detail, v.ts, s.name
            FROM violations v
            LEFT JOIN students s ON v.student_id = s.student_id
            ORDER BY v.id DESC
            LIMIT ?
        """, (limit,))
        
        # Batch process all rows at once
        violations = []
        for row in cur.fetchall():
            violations.append({
                'student_id': row['student_id'],
                'student_name': row['name'] or 'Unknown',
                'type': row['violation_type'],
                'detail': row['detail'],
                'ts': row['ts'],
                'severity': 'high' if 'phone' in row['violation_type'].lower() else 'medium'
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
