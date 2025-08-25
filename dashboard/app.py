from flask import Flask, render_template, jsonify
import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).resolve().parents[1] / 'monitoring.sqlite')

app = Flask(__name__)

def get_status():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    rows = cur.execute("SELECT student_id, strikes, status, last_update FROM monitoring_status").fetchall()
    con.close()
    return [
        { 'student_id': r[0], 'strikes': r[1], 'status': r[2], 'last_update': r[3] }
        for r in rows
    ]

def get_violations(limit=50):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    rows = cur.execute("SELECT student_id, violation_type, detail, ts FROM violations ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [
        { 'student_id': r[0], 'type': r[1], 'detail': r[2], 'ts': r[3] }
        for r in rows
    ]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    return jsonify(get_status())

@app.route('/api/violations')
def api_violations():
    return jsonify(get_violations())

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5500, debug=True)
