/**
 * Enhanced Dashboard for Exam Monitoring System
 * Single student view with minimal colors
 */

class EnhancedDashboard {
    constructor() {
        this.socket = null;
        this.charts = {
            violationTimeline: null
        };
        this.cameraActive = false;
        this.examStartTime = Date.now();
        this.refreshInterval = null;
        this.timerInterval = null;
        this.currentStudentId = null;
    }

    /**
     * Initialize the dashboard
     */
    init() {
        console.log('Initializing Enhanced Dashboard...');
        
        this.setupWebSocket();
        this.setupCharts();
        this.startExamTimer();
        this.refreshData();
        
        // Set up periodic data refresh (every 3 seconds)
        this.refreshInterval = setInterval(() => {
            this.refreshData();
        }, 3000);
        
        // Set up event listeners
        this.setupEventListeners();
        
        console.log('Dashboard initialized successfully');
    }

    /**
     * Set up WebSocket connection
     */
    setupWebSocket() {
        console.log('Setting up WebSocket connection...');
        
        this.socket = io({
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            reconnectionAttempts: Infinity
        });
        
        this.socket.on('connect', () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus('Active', 'success');
        });
        
        this.socket.on('disconnect', () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus('Disconnected', 'danger');
        });
        
        this.socket.on('camera_frame', (data) => {
            try {
                if (this.cameraActive && data.frame) {
                    const cameraFeed = document.getElementById('camera-feed');
                    if (cameraFeed) {
                        cameraFeed.src = data.frame;
                    cameraFeed.style.display = 'block';
                    document.getElementById('camera-placeholder').style.display = 'none';
                    }
                }
            } catch (error) {
                console.error('Error displaying camera frame:', error);
            }
        });
        
        this.socket.on('camera_error', (data) => {
            console.error('Camera error:', data.error);
            this.cameraActive = false;
            const btn = document.getElementById('toggle-camera-btn');
            if (btn) {
                btn.textContent = 'Start';
            }
            if (document.getElementById('camera-feed')) {
            document.getElementById('camera-feed').style.display = 'none';
            document.getElementById('camera-placeholder').style.display = 'block';
            }
        });
    }

    /**
     * Update connection status
     */
    updateConnectionStatus(text, type) {
        const statusEl = document.getElementById('connection-status');
        if (statusEl) {
            statusEl.textContent = text;
            statusEl.className = `status-badge ${type}`;
        }
    }

    /**
     * Set up Chart.js visualization
     */
    setupCharts() {
        console.log('Setting up charts...');
        
        const timelineCtx = document.getElementById('violation-timeline-chart');
        if (!timelineCtx) return;
        
        this.charts.violationTimeline = new Chart(timelineCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Violations',
                    data: [],
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#6c757d'
                        },
                        grid: {
                            color: 'rgba(108, 117, 125, 0.1)'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: '#6c757d',
                            stepSize: 1
                        },
                        grid: {
                            color: 'rgba(108, 117, 125, 0.1)'
                        }
                    }
                }
            }
        });
        
        console.log('Charts initialized');
    }

    /**
     * Refresh data from API endpoints
     */
    async refreshData() {
        try {
            // Fetch live monitoring status (real-time data)
            const liveResponse = await fetch('/api/live-status');
            let liveData = null;
            
            if (liveResponse.ok) {
                liveData = await liveResponse.json();
                
                if (liveData && liveData.active && liveData.student_name) {
                    // Active monitoring session
                    this.showMonitoringContent(true);
                    this.updateLiveStatus(liveData);
                    
                    // Also fetch student status from database
                    const statusResponse = await fetch('/api/enhanced-status');
                if (statusResponse.ok) {
                    const statusData = await statusResponse.json();
                        if (statusData && statusData.student_id) {
                            this.currentStudentId = statusData.student_id;
                    this.updateStudentStatus(statusData);
                            await this.refreshAnalytics();
                            await this.refreshViolations();
                        }
                    }
                } else {
                    // No active monitoring
                    this.currentStudentId = null;
                    this.showMonitoringContent(false);
                }
            } else {
                // Fallback to database status
                const statusResponse = await fetch('/api/enhanced-status');
                if (statusResponse.ok) {
                    const statusData = await statusResponse.json();
                    if (statusData && statusData.student_id) {
                        this.currentStudentId = statusData.student_id;
                        this.showMonitoringContent(true);
                        this.updateStudentStatus(statusData);
                        await this.refreshAnalytics();
                        await this.refreshViolations();
                    } else {
                        this.showMonitoringContent(false);
                    }
                } else {
                    this.showMonitoringContent(false);
                }
            }
        } catch (error) {
            console.error('Error refreshing data:', error);
            this.showMonitoringContent(false);
        }
    }

    /**
     * Update live monitoring status from live_monitoring.json
     */
    updateLiveStatus(liveData) {
        if (!liveData) return;
        
        // Update real-time status fields
        this.updateStatusField('live-student-present', liveData.student_present || '-', liveData.student_present === 'yes');
        this.updateStatusField('live-eye-gaze', liveData.eye_gaze_status || '-', liveData.eye_gaze_status === 'normal');
        this.updateStatusField('live-head-pose', liveData.head_pose_status || '-', liveData.head_pose_status === 'normal');
        this.updateStatusField('live-motion', liveData.motion_status || '-', liveData.motion_status === 'normal');
        this.updateStatusField('live-object', liveData.object_detected || '-', liveData.object_detected === 'none');
        this.updateStatusField('live-boundary', liveData.boundary_violation || '-', liveData.boundary_violation === 'no');
        
        // Update strikes and soft score
        const strikesEl = document.getElementById('stat-strikes');
        if (strikesEl) {
            const strikes = liveData.strike_count || 0;
            strikesEl.textContent = strikes;
            strikesEl.className = `stat-value ${strikes >= 3 ? 'text-danger' : strikes > 0 ? 'text-warning' : ''}`;
        }
        
        const softScoreEl = document.getElementById('stat-soft-score');
        if (softScoreEl) {
            softScoreEl.textContent = liveData.soft_score || 0;
        }
        
        // Update decision badge
        const decisionEl = document.getElementById('live-decision');
        if (decisionEl) {
            const decision = liveData.decision || 'monitoring';
            decisionEl.textContent = decision;
            decisionEl.className = `status-value-badge ${decision === 'debarred' ? 'danger' : decision === 'strike_issued' ? 'warning' : 'normal'}`;
        }
        
        // Update timestamp
        if (liveData.timestamp) {
            try {
                const timestamp = new Date(liveData.timestamp);
                const lastUpdateEl = document.getElementById('stat-last-update');
                if (lastUpdateEl) {
                    lastUpdateEl.textContent = timestamp.toLocaleTimeString();
                }
            } catch (e) {
                console.error('Error parsing timestamp:', e);
            }
        }
    }
    
    /**
     * Update a status field with value and highlight
     */
    updateStatusField(elementId, value, isNormal) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
            element.className = `status-value ${isNormal ? 'normal' : 'warning'}`;
        }
    }

    /**
     * Show or hide monitoring content
     */
    showMonitoringContent(show) {
        const noMonitoring = document.getElementById('no-monitoring-message');
        const monitoringContent = document.getElementById('monitoring-content');
        
        if (noMonitoring) {
            noMonitoring.style.display = show ? 'none' : 'block';
        }
        if (monitoringContent) {
            monitoringContent.style.display = show ? 'block' : 'none';
        }
    }

    /**
     * Update student status display
     */
    updateStudentStatus(student) {
        if (!student) return;
        
        // Update student info
        const nameEl = document.getElementById('student-name');
        if (nameEl) nameEl.textContent = student.name || 'Unknown';
        
        const rollEl = document.getElementById('student-roll');
        if (rollEl) rollEl.textContent = student.roll_number || 'N/A';
        
        const seatEl = document.getElementById('student-seat');
        if (seatEl) seatEl.textContent = student.seat_number || 'N/A';
        
        const idEl = document.getElementById('student-id');
        if (idEl) idEl.textContent = student.student_id || 'N/A';
        
        // Update photo
        const photoEl = document.getElementById('student-photo');
        if (photoEl) {
            if (student.photo) {
                photoEl.src = student.photo;
                photoEl.style.display = 'block';
        } else {
                photoEl.style.display = 'none';
            }
        }
        
        // Update status badge
        const statusBadgeEl = document.getElementById('status-badge');
        if (statusBadgeEl) {
            statusBadgeEl.textContent = student.status || 'normal';
            statusBadgeEl.className = `badge status-${student.status || 'normal'}`;
        }
        
        // Update stats
        const strikesEl = document.getElementById('stat-strikes');
        if (strikesEl) {
            strikesEl.textContent = student.strikes || 0;
            strikesEl.className = `stat-value ${student.strikes >= 3 ? 'text-danger' : student.strikes > 0 ? 'text-warning' : ''}`;
        }
        
        const violationsEl = document.getElementById('stat-violations');
        if (violationsEl) violationsEl.textContent = student.recent_violations || 0;
        
        const lastUpdateEl = document.getElementById('stat-last-update');
        if (lastUpdateEl && student.last_update) {
            const updateTime = new Date(student.last_update);
            lastUpdateEl.textContent = updateTime.toLocaleTimeString();
        }
    }

    /**
     * Refresh analytics data
     */
    async refreshAnalytics() {
        if (!this.currentStudentId) return;
        
        try {
            const analyticsResponse = await fetch('/api/analytics');
            if (analyticsResponse.ok) {
                const analyticsData = await analyticsResponse.json();
                this.updateAnalytics(analyticsData);
            }
        } catch (error) {
            console.error('Error fetching analytics:', error);
        }
    }

    /**
     * Update analytics charts
     */
    updateAnalytics(analytics) {
        if (!analytics || !this.charts.violationTimeline) return;
        
        try {
            // Update violation timeline chart
            if (analytics.timeline && analytics.timeline.length > 0) {
                const labels = analytics.timeline.map(item => {
                    const date = new Date(item.time);
                    return date.toLocaleTimeString();
                });
                const data = analytics.timeline.map(item => item.count);
                
                this.charts.violationTimeline.data.labels = labels;
                this.charts.violationTimeline.data.datasets[0].data = data;
                this.charts.violationTimeline.update();
            }
            
            // Update violation types count
            if (analytics.type_distribution) {
                const typesCount = Object.keys(analytics.type_distribution).length;
                const typesEl = document.getElementById('stat-types');
                if (typesEl) typesEl.textContent = typesCount;
            }
        } catch (error) {
            console.error('Error updating analytics:', error);
        }
    }

    /**
     * Refresh violations data
     */
    async refreshViolations() {
        if (!this.currentStudentId) return;
        
        try {
            const violationsResponse = await fetch('/api/violations?limit=20');
            if (violationsResponse.ok) {
                const violationsData = await violationsResponse.json();
                this.updateRecentViolations(violationsData);
            }
        } catch (error) {
            console.error('Error fetching violations:', error);
        }
    }

    /**
     * Update recent violations table
     */
    updateRecentViolations(violations) {
        const tbody = document.getElementById('recent-violations-tbody');
        if (!tbody) return;
        
        if (!violations || violations.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No violations recorded</td></tr>';
            return;
        }
        
        tbody.innerHTML = violations.map(violation => {
            const time = new Date(violation.ts).toLocaleTimeString();
            const severityClass = violation.severity === 'high' ? 'severity-high' : 'severity-medium';
            
            return `
                <tr>
                    <td>${time}</td>
                    <td>${violation.type || 'N/A'}</td>
                    <td>${violation.detail || ''}</td>
                    <td><span class="badge ${severityClass}">${violation.severity || 'medium'}</span></td>
                </tr>
            `;
        }).join('');
    }

    /**
     * Toggle camera feed on/off
     */
    async toggleCamera() {
        const btn = document.getElementById('toggle-camera-btn');
        const action = this.cameraActive ? 'stop' : 'start';
        
        try {
            const response = await fetch(`/api/camera-feed?action=${action}`);
            if (response.ok) {
                this.cameraActive = !this.cameraActive;
                if (btn) {
                    btn.textContent = this.cameraActive ? 'Stop' : 'Start';
                }
                
                if (!this.cameraActive) {
                    const feedEl = document.getElementById('camera-feed');
                    const placeholderEl = document.getElementById('camera-placeholder');
                    if (feedEl) feedEl.style.display = 'none';
                    if (placeholderEl) placeholderEl.style.display = 'block';
                }
            }
        } catch (error) {
            console.error('Error toggling camera:', error);
        }
    }

    /**
     * Start exam timer
     */
    startExamTimer() {
        this.timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.examStartTime) / 1000);
            const hours = Math.floor(elapsed / 3600);
            const minutes = Math.floor((elapsed % 3600) / 60);
            const seconds = elapsed % 60;
            
            const timeString = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
            const timerEl = document.getElementById('exam-timer');
            if (timerEl) timerEl.textContent = timeString;
        }, 1000);
    }

    /**
     * Reset strikes for current student
     */
    async resetStrikes() {
        if (!this.currentStudentId) return;
        
        if (!confirm('Are you sure you want to reset strikes for this student?')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/reset-strikes/${this.currentStudentId}`, {
                method: 'POST'
            });
            
            if (response.ok) {
                console.log('Strikes reset successfully');
                this.refreshData();
            } else {
                console.error('Failed to reset strikes');
            }
        } catch (error) {
            console.error('Error resetting strikes:', error);
        }
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Camera toggle button
        const cameraBtn = document.getElementById('toggle-camera-btn');
        if (cameraBtn) {
            cameraBtn.addEventListener('click', () => {
            this.toggleCamera();
        });
        }
    }
}

// Make dashboard instance globally accessible
let dashboard = null;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    dashboard = new EnhancedDashboard();
    dashboard.init();
});