/**
 * Enhanced Dashboard for Exam Monitoring System
 * Provides real-time monitoring, analytics, and control interface
 */

class EnhancedDashboard {
    constructor() {
        this.socket = null;
        this.charts = {
            violationTypes: null,
            violationTimeline: null
        };
        this.cameraActive = false;
        this.examStartTime = Date.now();
        this.refreshInterval = null;
        this.timerInterval = null;
        // Cached data for fallback when API fails
        this.cachedStatusData = null;
        this.cachedAnalyticsData = null;
        this.cachedViolationsData = null;
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
        
        // Set up periodic data refresh (every 5 seconds)
        this.refreshInterval = setInterval(() => {
            this.refreshData();
        }, 5000);
        
        // Set up event listeners
        this.setupEventListeners();
        
        console.log('Dashboard initialized successfully');
    }

    /**
     * Set up WebSocket connection using Socket.IO with auto-reconnect
     */
    setupWebSocket() {
        console.log('Setting up WebSocket connection...');
        
        // Configure Socket.IO with reconnection options
        this.socket = io({
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            reconnectionAttempts: Infinity
        });
        
        // Connection event handlers
        this.socket.on('connect', () => {
            console.log('WebSocket connected');
            document.getElementById('connection-status').textContent = 'Active';
            document.getElementById('connection-status').className = 'badge bg-success';
            this.showNotification('Connected to monitoring system', 'success');
        });
        
        this.socket.on('disconnect', () => {
            console.log('WebSocket disconnected');
            document.getElementById('connection-status').textContent = 'Disconnected';
            document.getElementById('connection-status').className = 'badge bg-danger';
            this.showNotification('Disconnected from monitoring system', 'warning');
        });
        
        this.socket.on('reconnect_attempt', (attemptNumber) => {
            console.log(`WebSocket reconnection attempt ${attemptNumber}...`);
            document.getElementById('connection-status').textContent = 'Reconnecting...';
            document.getElementById('connection-status').className = 'badge bg-warning';
        });
        
        this.socket.on('reconnect', (attemptNumber) => {
            console.log(`WebSocket reconnected after ${attemptNumber} attempts`);
            this.showNotification('Reconnected to monitoring system', 'success');
            // Refresh data after reconnection
            this.refreshData();
        });
        
        this.socket.on('reconnect_error', (error) => {
            console.error('WebSocket reconnection error:', error);
        });
        
        this.socket.on('reconnect_failed', () => {
            console.error('WebSocket reconnection failed');
            this.showNotification('Failed to reconnect to monitoring system', 'danger');
        });
        
        // Camera frame handler with error handling
        this.socket.on('camera_frame', (data) => {
            try {
                if (this.cameraActive) {
                    const cameraFeed = document.getElementById('camera-feed');
                    cameraFeed.src = 'data:image/jpeg;base64,' + data.frame;
                    cameraFeed.style.display = 'block';
                    document.getElementById('camera-placeholder').style.display = 'none';
                }
            } catch (error) {
                console.error('Error displaying camera frame:', error);
            }
        });
        
        // Camera error handler
        this.socket.on('camera_error', (data) => {
            console.error('Camera error:', data.error);
            this.showNotification(`Camera error: ${data.error}`, 'danger');
            this.cameraActive = false;
            const btn = document.getElementById('toggle-camera-btn');
            if (btn) {
                btn.textContent = 'Start Feed';
                btn.className = 'btn btn-sm btn-primary';
            }
            document.getElementById('camera-feed').style.display = 'none';
            document.getElementById('camera-placeholder').style.display = 'block';
        });
        
        // Violation alert handler with error handling
        this.socket.on('violation_alert', (data) => {
            try {
                console.log('Violation alert received:', data);
                this.showViolationAlert(data);
                this.playAlertSound();
            } catch (error) {
                console.error('Error handling violation alert:', error);
            }
        });
        
        // Status update handler with error handling
        this.socket.on('status_update', (data) => {
            try {
                console.log('Status update received');
                this.updateStudentStatus(data.students);
            } catch (error) {
                console.error('Error handling status update:', error);
            }
        });
    }

    /**
     * Set up Chart.js visualizations
     */
    setupCharts() {
        console.log('Setting up charts...');
        
        // Violation Types Doughnut Chart
        const typesCtx = document.getElementById('violation-types-chart').getContext('2d');
        this.charts.violationTypes = new Chart(typesCtx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#FF6384',
                        '#36A2EB',
                        '#FFCE56',
                        '#4BC0C0',
                        '#9966FF',
                        '#FF9F40'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#ffffff'
                        }
                    }
                }
            }
        });
        
        // Violation Timeline Line Chart
        const timelineCtx = document.getElementById('violation-timeline-chart').getContext('2d');
        this.charts.violationTimeline = new Chart(timelineCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Violations',
                    data: [],
                    borderColor: '#FF6384',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    x: {
                        ticks: {
                            color: '#ffffff'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: '#ffffff',
                            stepSize: 1
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                }
            }
        });
        
        console.log('Charts initialized');
    }

    /**
     * Refresh data from API endpoints with error handling and cached fallback
     */
    async refreshData() {
        try {
            // Fetch enhanced status with error handling
            try {
                const statusResponse = await fetch('/api/enhanced-status', {
                    timeout: 5000
                });
                if (statusResponse.ok) {
                    const statusData = await statusResponse.json();
                    this.updateStudentStatus(statusData);
                    // Cache successful response
                    this.cachedStatusData = statusData;
                } else {
                    console.error('Status API returned error:', statusResponse.status);
                    // Use cached data if available
                    if (this.cachedStatusData) {
                        console.log('Using cached status data');
                        this.updateStudentStatus(this.cachedStatusData);
                    }
                }
            } catch (error) {
                console.error('Error fetching status:', error);
                // Use cached data if available
                if (this.cachedStatusData) {
                    console.log('Using cached status data due to error');
                    this.updateStudentStatus(this.cachedStatusData);
                }
            }
            
            // Fetch analytics with error handling
            try {
                const analyticsResponse = await fetch('/api/analytics', {
                    timeout: 5000
                });
                if (analyticsResponse.ok) {
                    const analyticsData = await analyticsResponse.json();
                    this.updateAnalytics(analyticsData);
                    // Cache successful response
                    this.cachedAnalyticsData = analyticsData;
                } else {
                    console.error('Analytics API returned error:', analyticsResponse.status);
                    // Use cached data if available
                    if (this.cachedAnalyticsData) {
                        console.log('Using cached analytics data');
                        this.updateAnalytics(this.cachedAnalyticsData);
                    }
                }
            } catch (error) {
                console.error('Error fetching analytics:', error);
                // Use cached data if available
                if (this.cachedAnalyticsData) {
                    console.log('Using cached analytics data due to error');
                    this.updateAnalytics(this.cachedAnalyticsData);
                }
            }
            
            // Fetch recent violations with error handling
            try {
                const violationsResponse = await fetch('/api/violations', {
                    timeout: 5000
                });
                if (violationsResponse.ok) {
                    const violationsData = await violationsResponse.json();
                    this.updateRecentViolations(violationsData);
                    // Cache successful response
                    this.cachedViolationsData = violationsData;
                } else {
                    console.error('Violations API returned error:', violationsResponse.status);
                    // Use cached data if available
                    if (this.cachedViolationsData) {
                        console.log('Using cached violations data');
                        this.updateRecentViolations(this.cachedViolationsData);
                    }
                }
            } catch (error) {
                console.error('Error fetching violations:', error);
                // Use cached data if available
                if (this.cachedViolationsData) {
                    console.log('Using cached violations data due to error');
                    this.updateRecentViolations(this.cachedViolationsData);
                }
            }
            
        } catch (error) {
            console.error('Fatal error refreshing data:', error);
            this.showNotification('Error refreshing dashboard data', 'danger');
        }
    }

    /**
     * Update student status table
     */
    updateStudentStatus(students) {
        const tbody = document.getElementById('student-status-tbody');
        
        if (!students || students.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No student data available</td></tr>';
            return;
        }
        
        // Update quick stats
        let normalCount = 0;
        let warningCount = 0;
        let flaggedCount = 0;
        let totalViolations = 0;
        
        students.forEach(student => {
            if (student.status === 'normal') normalCount++;
            else if (student.status === 'warning') warningCount++;
            else if (student.status === 'flagged') flaggedCount++;
            totalViolations += student.strikes || 0;
        });
        
        document.getElementById('stat-normal').textContent = normalCount;
        document.getElementById('stat-warning').textContent = warningCount;
        document.getElementById('stat-flagged').textContent = flaggedCount;
        document.getElementById('stat-violations').textContent = totalViolations;
        
        // Update table
        tbody.innerHTML = students.map(student => {
            const photoSrc = student.photo 
                ? `data:image/jpeg;base64,${student.photo}` 
                : 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="40" height="40"%3E%3Crect fill="%23666" width="40" height="40"/%3E%3C/svg%3E';
            
            const statusBadge = this.getStatusBadge(student.status);
            const strikesBadge = this.getStrikesBadge(student.strikes || 0);
            
            return `
                <tr>
                    <td><img src="${photoSrc}" alt="${student.name}" class="student-photo"></td>
                    <td>${student.name || 'Unknown'}</td>
                    <td>${student.roll_number || 'N/A'}</td>
                    <td>${student.seat_number || student.seat_id || 'N/A'}</td>
                    <td>${statusBadge}</td>
                    <td>${strikesBadge}</td>
                    <td>
                        <button class="btn btn-sm btn-warning me-1" onclick="dashboard.resetStrikes('${student.student_id}')">Reset</button>
                        <button class="btn btn-sm btn-danger" onclick="dashboard.flagStudent('${student.student_id}')">Flag</button>
                    </td>
                </tr>
            `;
        }).join('');
    }

    /**
     * Get status badge HTML
     */
    getStatusBadge(status) {
        const badges = {
            'normal': '<span class="badge bg-success">Normal</span>',
            'warning': '<span class="badge bg-warning text-dark">Warning</span>',
            'flagged': '<span class="badge bg-danger">Flagged</span>'
        };
        return badges[status] || '<span class="badge bg-secondary">Unknown</span>';
    }

    /**
     * Get strikes badge HTML
     */
    getStrikesBadge(strikes) {
        if (strikes === 0) {
            return '<span class="badge bg-success">0</span>';
        } else if (strikes < 3) {
            return `<span class="badge bg-warning text-dark">${strikes}</span>`;
        } else {
            return `<span class="badge bg-danger">${strikes}</span>`;
        }
    }

    /**
     * Update analytics charts with error handling
     */
    updateAnalytics(analytics) {
        if (!analytics) return;
        
        try {
            // Update violation types chart
            if (analytics.type_distribution) {
                const types = Object.keys(analytics.type_distribution);
                const counts = Object.values(analytics.type_distribution);
                
                this.charts.violationTypes.data.labels = types;
                this.charts.violationTypes.data.datasets[0].data = counts;
                this.charts.violationTypes.update();
            }
        } catch (error) {
            console.error('Error updating violation types chart:', error);
        }
        
        try {
            // Update violation timeline chart
            if (analytics.timeline) {
                const labels = analytics.timeline.map(item => item.time);
                const data = analytics.timeline.map(item => item.count);
                
                this.charts.violationTimeline.data.labels = labels;
                this.charts.violationTimeline.data.datasets[0].data = data;
                this.charts.violationTimeline.update();
            }
        } catch (error) {
            console.error('Error updating violation timeline chart:', error);
        }
    }

    /**
     * Update recent violations table
     */
    updateRecentViolations(violations) {
        const tbody = document.getElementById('recent-violations-tbody');
        
        if (!violations || violations.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No recent violations</td></tr>';
            return;
        }
        
        tbody.innerHTML = violations.slice(0, 20).map(violation => {
            const severityBadge = this.getSeverityBadge(violation.severity || 'medium');
            const time = new Date(violation.ts).toLocaleTimeString();
            
            return `
                <tr>
                    <td>${time}</td>
                    <td>${violation.student_name || violation.seat_id || 'Unknown'}</td>
                    <td>${violation.violation_type || 'N/A'}</td>
                    <td>${violation.detail || ''}</td>
                    <td>${severityBadge}</td>
                </tr>
            `;
        }).join('');
    }

    /**
     * Get severity badge HTML
     */
    getSeverityBadge(severity) {
        const badges = {
            'low': '<span class="badge bg-info">Low</span>',
            'medium': '<span class="badge bg-warning text-dark">Medium</span>',
            'high': '<span class="badge bg-danger">High</span>'
        };
        return badges[severity] || '<span class="badge bg-secondary">Unknown</span>';
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
                btn.textContent = this.cameraActive ? 'Stop Feed' : 'Start Feed';
                btn.className = this.cameraActive ? 'btn btn-sm btn-danger' : 'btn btn-sm btn-primary';
                
                if (!this.cameraActive) {
                    document.getElementById('camera-feed').style.display = 'none';
                    document.getElementById('camera-placeholder').style.display = 'block';
                }
            }
        } catch (error) {
            console.error('Error toggling camera:', error);
        }
    }

    /**
     * Show notification banner
     */
    showNotification(message, type = 'info') {
        const container = document.getElementById('alert-container');
        if (!container) return;
        
        const alertClass = {
            'success': 'alert-success',
            'danger': 'alert-danger',
            'warning': 'alert-warning',
            'info': 'alert-info'
        }[type] || 'alert-info';
        
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert ${alertClass} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        container.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }

    /**
     * Show violation alert banner
     */
    showViolationAlert(violation) {
        const container = document.getElementById('alert-container');
        if (!container) return;
        
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show';
        alertDiv.innerHTML = `
            <strong>Violation Alert!</strong> 
            ${violation.student_name || violation.seat_id} - ${violation.violation_type}: ${violation.detail || ''}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        container.appendChild(alertDiv);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 10000);
    }

    /**
     * Play alert sound using Web Audio API
     */
    playAlertSound() {
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.frequency.value = 800; // 800Hz sine wave
            oscillator.type = 'sine';
            
            gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
            
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.5); // 0.5 second duration
        } catch (error) {
            console.error('Error playing alert sound:', error);
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
            document.getElementById('exam-timer').textContent = timeString;
        }, 1000);
    }

    /**
     * Reset strikes for a student
     */
    async resetStrikes(studentId) {
        if (!confirm('Are you sure you want to reset strikes for this student?')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/reset-strikes/${studentId}`, {
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
     * Flag a student for review
     */
    async flagStudent(studentId) {
        if (!confirm('Are you sure you want to flag this student?')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/flag-student/${studentId}`, {
                method: 'POST'
            });
            
            if (response.ok) {
                console.log('Student flagged successfully');
                this.refreshData();
            } else {
                console.error('Failed to flag student');
            }
        } catch (error) {
            console.error('Error flagging student:', error);
        }
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Camera toggle button
        document.getElementById('toggle-camera-btn').addEventListener('click', () => {
            this.toggleCamera();
        });
    }
}

// Make dashboard instance globally accessible
let dashboard = null;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    dashboard = new EnhancedDashboard();
    dashboard.init();
});
