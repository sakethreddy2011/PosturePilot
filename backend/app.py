import threading
import time
import os
import cv2
import mediapipe as mp
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime
from fpdf import FPDF

from Final import PostureMonitor, PostureState

app = Flask(__name__)
CORS(app)

# ── Shared state ──────────────────────────────────────────────────────────────
state_lock = threading.Lock()
stop_event = threading.Event()
bg_thread = None
monitor = None

shared_state = {
    "running": False,
    "calibrated": False,
    "pose_detected": False,
    "state": "GOOD",
    "score": None,
    "defects": [],
    "defect_times": {"forward_head": 0, "hunching": 0, "lateral_lean": 0, "hip_sliding": 0},
    "bad_posture_duration": 0,
    "in_microbreak": False,
    "microbreak_elapsed": 0,
    "microbreak_countdown": 60,
    "microbreak_exercise": "",
    "next_break_in": 60,
    "show_nodes": True,
    "landmarks": None,
    "_raw_landmarks": None,   # NOT sent to client
    "latest_frame": None,     # NOT sent to client
    "defect_emas": {"forward_head": 0.0, "hunching": 0.0, "lateral_lean": 0.0, "hip_sliding": 0.0},
}



# ── Background webcam thread ──────────────────────────────────────────────────
def webcam_loop():
    global monitor

    mp_pose_mod = mp.solutions.pose
    pose_local = mp_pose_mod.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)

    bad_event_start = None

    try:
        while not stop_event.is_set() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.033)
                continue

            frame = cv2.flip(frame, 1)
            current_time = time.time()

            # Encode raw frame (no drawings)
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpeg = buf.tobytes()

            # MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose_local.process(rgb)
            rgb.flags.writeable = True

            if monitor is None:
                with state_lock:
                    shared_state["latest_frame"] = jpeg
                continue

            # Microbreak auto-trigger
            with state_lock:
                cal = shared_state["calibrated"]
                mb = shared_state["in_microbreak"]
                mb_interval = monitor.microbreak_interval
                last_mb = monitor.last_microbreak

            if cal and not mb and (current_time - last_mb >= mb_interval):
                monitor.in_microbreak = True  # type: ignore[union-attr]
                monitor.microbreak_start = current_time  # type: ignore[union-attr]
                with state_lock:
                    shared_state["in_microbreak"] = True

            # Re-read microbreak state after possible trigger
            mb = monitor.in_microbreak

            landmark_data = None
            raw_lm = None
            pose_detected = False
            score = None
            defects = []
            bad_duration = 0
            mb_elapsed = 0
            mb_countdown = 60
            mb_exercise = ""

            if results.pose_landmarks:
                raw_lm = results.pose_landmarks.landmark
                landmark_data = monitor.extract_landmarks(raw_lm)
                pose_detected = bool(landmark_data)

            if pose_detected and not mb:
                score, raw_defects = monitor.analyze_posture_biomechanics(landmark_data)
                
                # Apply EMA on severities to prevent UI flickering and confusion without touching Final.py
                with state_lock:
                    _e = shared_state.get("defect_emas")
                    if isinstance(_e, dict):
                        emas = dict(_e)
                    else:
                        emas = {"forward_head": 0.0, "hunching": 0.0, "lateral_lean": 0.0, "hip_sliding": 0.0}
                
                raw_dict = {k: v for k, v in raw_defects}
                alpha = 0.15
                smoothed_defects = []
                for k in ["forward_head", "hunching", "lateral_lean", "hip_sliding"]:
                    val = float(raw_dict.get(k, 0.0))
                    prev = float(emas.get(k, 0.0))
                    new_val = (alpha * val) + ((1.0 - alpha) * prev)
                    emas[k] = new_val
                    if new_val > 4.0:
                        smoothed_defects.append((k, new_val))
                
                # Prevent Hunching and Hip Sliding confusion (mutual exclusivity)
                hunch_idx = next((i for i, d in enumerate(smoothed_defects) if d[0] == "hunching"), -1)
                hip_idx = next((i for i, d in enumerate(smoothed_defects) if d[0] == "hip_sliding"), -1)
                if hunch_idx != -1 and hip_idx != -1:
                    if smoothed_defects[hunch_idx][1] >= smoothed_defects[hip_idx][1]:
                        smoothed_defects.pop(hip_idx)
                    else:
                        smoothed_defects.pop(hunch_idx)
                
                with state_lock:
                    shared_state["defect_emas"] = emas
                
                defects = smoothed_defects
                # Pass None like the original code — any defect triggers BAD state
                state_val, state_duration = monitor.update_state_machine(None, defects)

                # Bad-event tracking
                if state_val == PostureState.BAD:
                    if bad_event_start is None:
                        bad_event_start = current_time
                    bad_duration = int(current_time - bad_event_start)  # type: ignore[operator]
                    if bad_duration >= 3:
                        monitor.capture_bad_posture_event(frame, defects, bad_duration)
                else:
                    bad_event_start = None
                    bad_duration = 0

            # Microbreak timing
            if mb and monitor.microbreak_start:
                mb_elapsed = int(current_time - monitor.microbreak_start)
                mb_countdown = max(0, 60 - mb_elapsed)
                ex_idx = (mb_elapsed // 20) % len(monitor.microbreak_exercises)
                mb_exercise = monitor.microbreak_exercises[ex_idx]
                if mb_elapsed >= 60:
                    monitor.in_microbreak = False  # type: ignore[union-attr]
                    monitor.last_microbreak = current_time  # type: ignore[union-attr]
                    mb = False

            next_break_in = max(0, int(monitor.microbreak_interval - (current_time - monitor.last_microbreak)))

            # Serialize landmarks
            lm_serial = None
            if landmark_data:
                lm_serial = {
                    k: list(v) if v is not None else None
                    for k, v in landmark_data.items()
                }

            # Serialize defects
            defects_serial = [{"key": k, "severity": float(int(float(s) * 100)) / 100} for k, s in defects]

            _state = monitor.current_state if monitor else PostureState.GOOD
            current_state_str = _state.value

            if stop_event.is_set():
                break

            with state_lock:
                shared_state.update({  # type: ignore[call-overload]
                    "latest_frame": jpeg,
                    "calibrated": monitor.calibrated,
                    "pose_detected": pose_detected,
                    "state": current_state_str,
                    "score": round(score, 2) if score is not None else None,
                    "defects": defects_serial,
                    "defect_times": {k: int(v) for k, v in monitor.defect_time.items()},
                    "bad_posture_duration": bad_duration,
                    "in_microbreak": monitor.in_microbreak,
                    "microbreak_elapsed": mb_elapsed,
                    "microbreak_countdown": mb_countdown,
                    "microbreak_exercise": mb_exercise,
                    "next_break_in": next_break_in,
                    "landmarks": lm_serial,
                    "_raw_landmarks": raw_lm,
                })

    finally:
        cap.release()
        pose_local.close()
        with state_lock:
            shared_state["running"] = False
            shared_state["latest_frame"] = None
            shared_state["_raw_landmarks"] = None


# ── MJPEG stream generator ────────────────────────────────────────────────────
def generate_frames():
    while True:
        with state_lock:
            frame = shared_state.get("latest_frame")
        if frame:
            _f: bytes = frame if isinstance(frame, bytes) else b''
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                _f +
                b'\r\n'
            )
        else:
            time.sleep(0.033)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/status')
def api_status():
    with state_lock:
        snap = {k: v for k, v in shared_state.items()
                if k not in ("latest_frame", "_raw_landmarks")}
    return jsonify(snap)


@app.route('/api/start', methods=['POST'])
def api_start():
    global bg_thread, monitor

    with state_lock:
        if shared_state["running"]:
            return jsonify({"ok": False, "message": "Already running"})

    stop_event.clear()
    monitor = PostureMonitor()

    with state_lock:
        shared_state.update({
            "running": True,
            "calibrated": False,
            "pose_detected": False,
            "state": "GOOD",
            "score": None,
            "defects": [],
            "defect_times": {"forward_head": 0, "hunching": 0, "lateral_lean": 0, "hip_sliding": 0},
            "bad_posture_duration": 0,
            "in_microbreak": False,
            "microbreak_elapsed": 0,
            "microbreak_countdown": 60,
            "microbreak_exercise": "",
            "next_break_in": 60,
            "landmarks": None,
            "_raw_landmarks": None,
            "latest_frame": None,
            "defect_emas": {"forward_head": 0.0, "hunching": 0.0, "lateral_lean": 0.0, "hip_sliding": 0.0},
        })

    bg_thread = threading.Thread(target=webcam_loop, daemon=True)
    bg_thread.start()
    return jsonify({"ok": True, "message": "Camera started"})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    global bg_thread, monitor

    with state_lock:
        if not shared_state["running"]:
            return jsonify({"ok": False, "message": "Not running"})

    stop_event.set()
    if bg_thread:
        bg_thread.join(timeout=3)

    with state_lock:
        shared_state.update({
            "running": False,
            "calibrated": False,
            "pose_detected": False,
            "state": "GOOD",
            "score": None,
            "defects": [],
            "defect_times": {"forward_head": 0, "hunching": 0, "lateral_lean": 0, "hip_sliding": 0},
            "bad_posture_duration": 0,
            "in_microbreak": False,
            "microbreak_elapsed": 0,
            "microbreak_countdown": 60,
            "microbreak_exercise": "",
            "next_break_in": 60,
            "landmarks": None,
            "_raw_landmarks": None,
            "latest_frame": None,
            "defect_emas": {"forward_head": 0.0, "hunching": 0.0, "lateral_lean": 0.0, "hip_sliding": 0.0},
        })

    report_text = ""
    screenshots = []
    pdf_base64 = None
    if monitor:
        report_text = monitor.generate_medical_report()
        
        try:
            import io
            import base64
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Courier", size=10)
            clean_text = report_text.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(w=0, align='L', text=clean_text)
            
            for event in monitor.bad_posture_events:
                dt = event.get('defect_type', 'unknown')
                
                ret, buf = cv2.imencode('.jpg', event['frame'], [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    img_bytes = buf.tobytes()
                    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                    img_data_url = f"data:image/jpeg;base64,{img_b64}"
                    
                    screenshots.append({
                        "label": dt.replace('_', ' ').title(),
                        "data": img_data_url
                    })
                    
                    pdf.add_page()
                    pdf.set_font("Courier", size=12)
                    pdf.multi_cell(w=0, h=10, align='L', text=f"Captured Maximum Severity: {dt.replace('_', ' ').title()}")
                    pdf.ln(10)
                    
                    import tempfile
                    import os
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        tmp.write(img_bytes)
                        tmp_name = tmp.name
                        
                    # Let FPDF automatically handle the Y coordinate and center it nicely
                    pdf.image(tmp_name, x="C", w=150)
                    os.remove(tmp_name)

            out = pdf.output()
            b = bytes(out) if not isinstance(out, str) else out.encode('latin-1')
            pdf_b64 = base64.b64encode(b).decode('utf-8')
            pdf_base64 = f"data:application/pdf;base64,{pdf_b64}"
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print("PDF GENERATION FAILED:\n" + err_msg)

    return jsonify({
        "ok": True, 
        "message": "Camera stopped", 
        "report": report_text, 
        "screenshots": screenshots,
        "pdf_report": pdf_base64
    })


@app.route('/api/calibrate', methods=['POST'])
def api_calibrate():
    with state_lock:
        if not shared_state["running"]:
            return jsonify({"ok": False, "message": "Camera not running"})
        raw_lm = shared_state.get("_raw_landmarks")

    if not raw_lm:
        return jsonify({"ok": False, "message": "No pose detected"})

    if monitor and monitor.calibrate_baseline(raw_lm):
        with state_lock:
            shared_state["calibrated"] = True
        return jsonify({"ok": True, "message": "Calibrated"})

    return jsonify({"ok": False, "message": "Calibration failed"})


@app.route('/api/toggle_nodes', methods=['POST'])
def api_toggle_nodes():
    with state_lock:
        shared_state["show_nodes"] = not shared_state["show_nodes"]
        val = shared_state["show_nodes"]
    return jsonify({"ok": True, "show_nodes": val})


@app.route('/api/end_microbreak', methods=['POST'])
def api_end_microbreak():
    with state_lock:
        if not shared_state["in_microbreak"]:
            return jsonify({"ok": False, "message": "Not in microbreak"})

    if monitor:
        monitor.in_microbreak = False  # type: ignore[union-attr]
        monitor.last_microbreak = time.time()  # type: ignore[union-attr]
    with state_lock:
        shared_state["in_microbreak"] = False

    return jsonify({"ok": True, "message": "Microbreak ended"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
