'''
import cv2
import mediapipe as mp  
import numpy as np  
import time
import math
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any
try:
    import winsound
except Exception:
    winsound = None

# MediaPipe pose module alias (landmark indices + model).
mp_pose = mp.solutions.pose
# MediaPipe drawing utils for skeleton overlay.
mp_drawing = mp.solutions.drawing_utils
# Global pose detector configured for stable tracking.
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Finite posture states used by the session state machine.
class PostureState(Enum):
    GOOD = "GOOD"
    TOLERANCE = "ACCEPTABLE"
    BAD = "BAD"

# Central state container and analytics engine for posture monitoring.
# Owns calibration, scoring, episode counting, and reporting state.
class PostureMonitor:
    def __init__(self) -> None:
        self.calibrated: bool = False
        self.baseline_landmarks: Optional[Dict[str, Any]] = None
        self.ideal_landmarks: Optional[Dict[str, Any]] = None
        
        # Session timing
        self.session_start: float = time.time()
        self.last_check_time: float = time.time()
        
        # MODULE B: Episode-based state machine
        self.current_state = PostureState.GOOD
        self.state_start_time: float = time.time()
        self.episode_counts: Dict[str, int] = {
            "forward_head": 0,
            "hunching": 0,
            "lateral_lean": 0,
            "hip_sliding": 0
        }
        
        # Time tracking
        self.good_posture_time: float = 0.0
        self.tolerance_posture_time: float = 0.0
        self.bad_posture_time: float = 0.0

        # Per-defect bad-posture time (seconds accumulated while each defect is dominant)
        self.defect_time: Dict[str, float] = {
            "forward_head": 0.0,
            "hunching": 0.0,
            "lateral_lean": 0.0,
            "hip_sliding": 0.0
        }
        
        # Analytics
        self.max_forward_head_angle: float = 0.0
        self.max_hunch_angle: float = 0.0
        self.max_lean_angle: float = 0.0
        self.max_hip_slide: float = 0.0

        # Forward head hybrid metric state (angle + z)
        self.base_head_angle: Optional[float] = None
        self.base_head_z_rel: Optional[float] = None
        self.fh_angle_ema: float = 0.0
        self.fh_z_ema: float = 0.0
        self.fh_ema_alpha: float = 0.3

        # Lateral lean baseline offsets relative to vertical axis (eye midpoint)
        self.base_shoulder_axis_offset: Optional[float] = None
        self.base_hip_axis_offset: Optional[float] = None
        
        # Bad posture event capture
        self.bad_posture_events: List[Dict[str, Any]] = []
        self.current_bad_event: Optional[Dict[str, Any]] = None
        
        # Microbreak timer
        self.last_microbreak: float = time.time()
        self.microbreak_interval: int = 60  # 60 seconds as required
        self.in_microbreak: bool = False
        self.microbreak_start: Optional[float] = None
        self.microbreak_exercises: List[str] = [
            "Neck Rolls: Slowly rotate your head clockwise, then counter-clockwise",
            "Shoulder Retractions: Pull shoulder blades together, hold 5 seconds",
            "Spinal Extensions: Place hands on lower back, gently arch backward"
        ]

        # Episode cooldown: track last time each defect episode was counted
        # to avoid rapid re-counting when score oscillates near the BAD boundary.
        self.last_episode_time: Dict[str, float] = {
            "forward_head": 0.0,
            "hunching": 0.0,
            "lateral_lean": 0.0,
            "hip_sliding": 0.0
        }
        self.episode_cooldown: float = 5.0   # seconds minimum between episodes
        
    # Compute the angle (degrees) formed by three 2D points at point2.
    # Returns 0 when vectors are degenerate to avoid divide-by-zero.
    def calculate_angle(self, point1, point2, point3):
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return math.degrees(angle)
    
    # Extract required pose landmarks into a plain dict of tuples.
    # Returns None if any critical landmark is missing.
    def extract_landmarks(self, landmarks):
        try:
            nose           = landmarks[mp_pose.PoseLandmark.NOSE.value]           # type: ignore[index]
            left_shoulder  = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]  # type: ignore[index]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] # type: ignore[index]
            left_hip       = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]       # type: ignore[index]
            right_hip      = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]      # type: ignore[index]
            left_ear       = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]       # type: ignore[index]
            right_ear      = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]      # type: ignore[index]
            left_eye       = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]       # type: ignore[index]
            right_eye      = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]      # type: ignore[index]
            
            # Try to get knee landmarks (may not be visible)
            try:
                left_knee  = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]   # type: ignore[index]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]  # type: ignore[index]
            except:
                left_knee = None
                right_knee = None
            
            return {
                'nose': (nose.x, nose.y, nose.z),
                'left_shoulder': (left_shoulder.x, left_shoulder.y, left_shoulder.z),
                'right_shoulder': (right_shoulder.x, right_shoulder.y, right_shoulder.z),
                'left_hip': (left_hip.x, left_hip.y, left_hip.z),
                'right_hip': (right_hip.x, right_hip.y, right_hip.z),
                'left_ear': (left_ear.x, left_ear.y, left_ear.z),
                'right_ear': (right_ear.x, right_ear.y, right_ear.z),
                'left_eye': (left_eye.x, left_eye.y, left_eye.z),
                'right_eye': (right_eye.x, right_eye.y, right_eye.z),
                'left_knee': (left_knee.x, left_knee.y, left_knee.z) if left_knee else None,
                'right_knee': (right_knee.x, right_knee.y, right_knee.z) if right_knee else None,
            }
        except Exception as e:
            return None
    
    # Calibrate the baseline (ideal) posture from a single frame.
    # Stores reference landmarks and midpoints for overlay.
    def calibrate_baseline(self, landmarks):
        landmark_data = self.extract_landmarks(landmarks)
        if landmark_data:
            self.baseline_landmarks = landmark_data
            
            # Store ideal skeleton for overlay
            shoulder_mid = (
                (landmark_data['left_shoulder'][0] + landmark_data['right_shoulder'][0]) / 2,  # type: ignore[operator]
                (landmark_data['left_shoulder'][1] + landmark_data['right_shoulder'][1]) / 2   # type: ignore[operator]
            )
            hip_mid = (
                (landmark_data['left_hip'][0] + landmark_data['right_hip'][0]) / 2,  # type: ignore[operator]
                (landmark_data['left_hip'][1] + landmark_data['right_hip'][1]) / 2   # type: ignore[operator]
            )
            eye_mid = (
                (landmark_data['left_eye'][0] + landmark_data['right_eye'][0]) / 2,  # type: ignore[operator]
                (landmark_data['left_eye'][1] + landmark_data['right_eye'][1]) / 2   # type: ignore[operator]
            )

            self.ideal_landmarks = {
                'eye_mid': eye_mid,
                'shoulder_mid': shoulder_mid,
                'hip_mid': hip_mid
            }

            # Baseline values for forward-head hybrid metric
            ear_mid = (
                (landmark_data['left_ear'][0] + landmark_data['right_ear'][0]) / 2,  # type: ignore[operator]
                (landmark_data['left_ear'][1] + landmark_data['right_ear'][1]) / 2,  # type: ignore[operator]
                (landmark_data['left_ear'][2] + landmark_data['right_ear'][2]) / 2   # type: ignore[operator]
            )
            shoulder_mid3 = (
                (landmark_data['left_shoulder'][0] + landmark_data['right_shoulder'][0]) / 2,  # type: ignore[operator]
                (landmark_data['left_shoulder'][1] + landmark_data['right_shoulder'][1]) / 2,  # type: ignore[operator]
                (landmark_data['left_shoulder'][2] + landmark_data['right_shoulder'][2]) / 2   # type: ignore[operator]
            )
            hip_mid3 = (
                (landmark_data['left_hip'][0] + landmark_data['right_hip'][0]) / 2,  # type: ignore[operator]
                (landmark_data['left_hip'][1] + landmark_data['right_hip'][1]) / 2,  # type: ignore[operator]
                (landmark_data['left_hip'][2] + landmark_data['right_hip'][2]) / 2   # type: ignore[operator]
            )
            v1 = np.array([ear_mid[0] - shoulder_mid3[0], ear_mid[1] - shoulder_mid3[1]])
            v2 = np.array([hip_mid3[0] - shoulder_mid3[0], hip_mid3[1] - shoulder_mid3[1]])
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1.0
            cosang = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
            self.base_head_angle = math.degrees(math.acos(cosang))
            self.base_head_z_rel = float(ear_mid[2]) - float(shoulder_mid3[2])
            self.fh_angle_ema = 0.0
            self.fh_z_ema = 0.0

            # Baseline lateral offsets from vertical axis (eye midpoint)
            self.base_shoulder_axis_offset = float(shoulder_mid[0]) - float(eye_mid[0])
            self.base_hip_axis_offset = float(hip_mid[0]) - float(eye_mid[0])

            # Reset microbreak timer on calibration
            self.last_microbreak = time.time()
            self.in_microbreak = False
            self.microbreak_start = None
            
            self.calibrated = True
            return True
        return False
    
    # Score posture from current landmarks using isolated biomechanical detectors.
    # Returns (score, defects[]) where defects are (defect_key, severity).
    # 
    # MediaPipe coords (normalized 0-1):
    # x: left->right, y: top->bottom (larger=lower), z: negative=closer to cam
    # 
    # Detector isolation rules:
    # - Forward head: nose Z relative to shoulder midpoint moves closer.
    # - Hunching: shoulders move closer (z decreases) and drop (y increases).
    # - Lateral lean: shoulder tilt in X-Y plane only (no Z).
    # - Hip sliding: shoulders move away (z increases) and sink (y increases).
    # - Hunching vs hip sliding are mutually exclusive by Z direction.
    def analyze_posture_biomechanics(self, current_landmarks):
        """
        Strictly isolated biomechanical detectors.

        MediaPipe coords (normalised 0-1):
          x: left→right,  y: top→bottom (larger=lower),  z: negative=closer to cam

        DETECTOR ISOLATION RULES:
          FWD HEAD   : nose Z relative to shoulders becomes more negative (head closer)
          HUNCHING   : shoulder Z more negative (closer) AND shoulder Y increases (drops)
          LAT LEAN   : shoulder tilt angle deviates left/right (purely X-axis tilt)
          HIP SLIDE  : shoulder Z more positive (farther) AND shoulder Y increases (sinks)
          → Hunching and Hip Sliding are mutually exclusive by Z direction.
          → Forward Head is always relative to shoulder midpoint – body-movement cancelled.
        """
        if not self.calibrated or not current_landmarks:
            return None, []

        baseline = self.baseline_landmarks
        current  = current_landmarks
        defects  = []

        # ── shared midpoints ─────────────────────────────────────────────────────────
        def mid3(a, b):
            return np.array([(a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2])

        sh_cur  = mid3(current['left_shoulder'],  current['right_shoulder'])   # type: ignore[arg-type]
        sh_base = mid3(baseline['left_shoulder'], baseline['right_shoulder'])   # type: ignore[arg-type]
        nose_c  = np.array(current['nose'])   # type: ignore[arg-type]
        nose_b  = np.array(baseline['nose'])  # type: ignore[arg-type]

        # Raw Z change for shoulders (used by hunching and hip sliding)
        sh_z_change = float(sh_cur[2]) - float(sh_base[2])   # negative=closer, positive=farther
        sh_y_change = float(sh_cur[1]) - float(sh_base[1])   # positive=dropped in frame

        # -- 1. FORWARD HEAD (HYBRID: ANGLE + Z) --------------------------------------------
        # Uses ear-midpoint angle vs baseline + relative Z depth to shoulder midpoint.
        ANGLE_DEADBAND = 2.0
        Z_DEADBAND = 0.01
        angle_weight = 1.0
        z_weight = 1.0
        angle_scale = 3.0
        z_scale = 250.0

        if self.base_head_angle is not None and self.base_head_z_rel is not None:
            shoulder_span = abs(float(current['left_shoulder'][0]) - float(current['right_shoulder'][0]))  # type: ignore[index]
            if shoulder_span > 1e-3:
                ear_mid = np.array([
                    (float(current['left_ear'][0]) + float(current['right_ear'][0])) / 2,  # type: ignore[index]
                    (float(current['left_ear'][1]) + float(current['right_ear'][1])) / 2,  # type: ignore[index]
                    (float(current['left_ear'][2]) + float(current['right_ear'][2])) / 2   # type: ignore[index]
                ])

                # Current head angle in X-Y plane
                v1 = np.array([ear_mid[0] - sh_cur[0], ear_mid[1] - sh_cur[1]])
                v2 = np.array([((current['left_hip'][0] + current['right_hip'][0]) / 2) - sh_cur[0],  # type: ignore[index]
                               ((current['left_hip'][1] + current['right_hip'][1]) / 2) - sh_cur[1]])  # type: ignore[index]
                denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1.0
                cosang = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
                head_angle = math.degrees(math.acos(cosang))

                # Relative Z depth (ear-mid to shoulder-mid)
                head_z_rel_cur = float(ear_mid[2]) - float(sh_cur[2])

                angle_delta = max(0.0, head_angle - self.base_head_angle - ANGLE_DEADBAND)
                z_delta = max(0.0, self.base_head_z_rel - head_z_rel_cur - Z_DEADBAND)

                # EMA smoothing
                a = self.fh_ema_alpha
                self.fh_angle_ema = a * angle_delta + (1.0 - a) * self.fh_angle_ema
                self.fh_z_ema = a * z_delta + (1.0 - a) * self.fh_z_ema

                if self.fh_angle_ema > 0.0 or self.fh_z_ema > 0.0:
                    fwd_severity = min(30.0, (angle_weight * self.fh_angle_ema * angle_scale) +
                                       (z_weight * self.fh_z_ema * z_scale))
                else:
                    fwd_severity = 0.0

                if fwd_severity > 5.0:
                    defects.append(("forward_head", fwd_severity))
                    self.max_forward_head_angle = max(self.max_forward_head_angle,
                                                     round(self.fh_angle_ema, 1))  # type: ignore[call-overload]

        # ── 2. HUNCHBACK (KYPHOSIS) ────────────────────────────────────────────
        # Shoulders move CLOSER to camera (z decreases = sh_z_change < 0)
        # AND shoulders drop vertically (y increases = sh_y_change > 0)
        # GUARD: sh_z_change must be negative (closer). If positive, it is hip sliding.
        sh_z_towards = -sh_z_change  # positive when shoulders move toward cam
        hunch_z = max(0.0, sh_z_towards - 0.02) * 130.0
        hunch_y = max(0.0, sh_y_change  - 0.02) * 180.0
        hunch_severity = max(0.0, min(30.0, hunch_z + hunch_y)) if sh_z_towards > 0.02 else 0.0
        if hunch_severity > 8.0:
            defects.append(("hunching", hunch_severity))
            self.max_hunch_angle = max(self.max_hunch_angle,
                                      round(max(sh_z_towards, sh_y_change) * 100, 1))  # type: ignore[call-overload]

        # ?????? 3. LATERAL LEAN (TORSO ANGLE THRESHOLD) ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        # Trigger when torso angle from horizontal is outside [78?, 120?].
        sh_mid = (
            (float(current['left_shoulder'][0]) + float(current['right_shoulder'][0])) / 2,  # type: ignore[index]
            (float(current['left_shoulder'][1]) + float(current['right_shoulder'][1])) / 2   # type: ignore[index]
        )
        hip_mid = (
            (float(current['left_hip'][0]) + float(current['right_hip'][0])) / 2,  # type: ignore[index]
            (float(current['left_hip'][1]) + float(current['right_hip'][1])) / 2   # type: ignore[index]
        )
        v_x = hip_mid[0] - sh_mid[0]
        v_y = hip_mid[1] - sh_mid[1]
        if abs(v_x) + abs(v_y) > 1e-6:
            torso_angle = abs(math.degrees(math.atan2(v_y, v_x)))
            if torso_angle > 180.0:
                torso_angle = 360.0 - torso_angle

            LOW_ANGLE = 91.0
            HIGH_ANGLE = 94.0
            if torso_angle < LOW_ANGLE:
                dev = LOW_ANGLE - torso_angle
            elif torso_angle > HIGH_ANGLE:
                dev = torso_angle - HIGH_ANGLE
            else:
                dev = 0.0

            lean_severity = max(0.0, min(30.0, dev * 1.5))
            if lean_severity > 0.0:
                defects.append(("lateral_lean", lean_severity))
                self.max_lean_angle = max(self.max_lean_angle, round(torso_angle, 1))  # type: ignore[call-overload]

        # ── 4. HIP SLIDING ──────────────────────────────────────────────────
        # Shoulders move AWAY from camera (z increases = sh_z_change > 0)
        # AND body sinks (y increases = sh_y_change > 0)
        # GUARD: sh_z_change must be positive (away). If negative, it is hunching.
        sh_z_away = sh_z_change   # positive when shoulders move away from cam
        hip_z = max(0.0, sh_z_away  - 0.03) * 130.0
        hip_y = max(0.0, sh_y_change - 0.03) * 180.0
        hip_severity = max(0.0, min(30.0, hip_z + hip_y)) if sh_z_away > 0.03 and sh_y_change > 0.03 else 0.0
        if hip_severity > 3.5:
            defects.append(("hip_sliding", hip_severity))
            self.max_hip_slide = max(self.max_hip_slide,
                                    round((sh_z_away + sh_y_change) * 100, 1))  # type: ignore[call-overload]

        # ── Final score ───────────────────────────────────────────────────────────
        total_deduction = sum(d[1] for d in defects)
        score = max(0.0, 100.0 - total_deduction)
        return score, defects
    
    # Update posture state machine with hysteresis and episode counting.
    # GOOD >= 82, ACCEPTABLE >= 65, BAD < 65.
    # Episodes are counted only on sustained BAD + cooldown per defect.
    def update_state_machine(self, score, defects):
        """MODULE B: State machine with hysteresis to prevent rapid oscillation.
        States:
          GOOD       : score >= 82
          ACCEPTABLE : score >= 65
          BAD        : score <  65
        A transition to BAD is only counted as an episode if the score has been
        below 65 for at least 2 continuous seconds (prevents brief dips from
        inflating episode counts).
        """
        current_time = time.time()
        elapsed = current_time - self.last_check_time

        # Determine new state based on defect counts (4 metrics)
        if score is None:
            defect_keys = {d[0] for d in defects} if defects else set()
            bad_count = len(defect_keys)
            if bad_count == 0:
                new_state = PostureState.GOOD
            else:
                new_state = PostureState.BAD
        elif score >= 82:
            new_state = PostureState.GOOD
        elif score >= 65:
            new_state = PostureState.TOLERANCE
        else:
            new_state = PostureState.BAD

        # Accumulate time in current state
        if self.current_state == PostureState.GOOD:
            self.good_posture_time += elapsed
        elif self.current_state == PostureState.TOLERANCE:
            self.tolerance_posture_time += elapsed
        elif self.current_state == PostureState.BAD:
            self.bad_posture_time += elapsed
        # Accumulate per-defect time for any active defects (independent of state)
        if defects:
            for key, _sev in defects:
                self.defect_time[key] = self.defect_time.get(key, 0.0) + elapsed

        state_duration = current_time - self.state_start_time

        # Only count an episode when transitioning INTO BAD AND:
        #   - we've been in GOOD/ACCEPTABLE for > 5s (hysteresis: bad posture sustained)
        #   - enough cooldown since last episode of this defect type
        if self.current_state in [PostureState.GOOD, PostureState.TOLERANCE] and new_state == PostureState.BAD:
            if state_duration > 5.0 and defects:
                dominant_defect = max(defects, key=lambda x: x[1])
                defect_key = dominant_defect[0]
                time_since_last = current_time - self.last_episode_time.get(defect_key, 0.0)
                if time_since_last >= self.episode_cooldown:
                    self.episode_counts[defect_key] += 1
                    self.last_episode_time[defect_key] = current_time
                    print(f"[EPISODE] {defect_key} detected (Episode #{self.episode_counts[defect_key]})")

        if new_state != self.current_state:
            self.current_state = new_state
            self.state_start_time = current_time

        self.last_check_time = current_time
        return self.current_state, state_duration
    
    # Track a single representative screenshot per defect type.
    # Keeps the frame with the longest continuous BAD duration.
    def capture_bad_posture_event(self, frame, defects, duration):
        """Keep ONE best-frame per defect type, selected by longest continuous duration.
        Uses index-based replacement to avoid ValueError from numpy array equality in list.remove().
        """
        if not defects:
            return

        dominant_defect = max(defects, key=lambda x: x[1])
        defect_key = dominant_defect[0]

        new_event = {
            'frame': frame.copy(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'duration': duration,
            'defect_type': defect_key,
            'severity': float(dominant_defect[1])
        }

        # Find the existing entry for this defect type by index (identity-safe)
        existing_idx = None
        existing_dur = 0
        for i, e in enumerate(self.bad_posture_events):
            if e['defect_type'] == defect_key:
                existing_idx = i
                existing_dur = e['duration']
                break

        if existing_idx is None:
            self.bad_posture_events.append(new_event)
        elif duration > existing_dur:
            assert existing_idx is not None  # already checked above; helps Pyre
            self.bad_posture_events[existing_idx] = new_event  # type: ignore[index]

        # Sort by longest duration first
        self.bad_posture_events.sort(key=lambda x: x['duration'], reverse=True)

    # Produce a text summary of the session: time in states, maxima,
    # and ergonomic recommendations based on the dominant defect.
    def generate_medical_report(self):
        """MODULE E: Generate medical report with episode counts"""
        total_time = time.time() - self.session_start
        
        report: List[str] = []  # explicit List[str] silences LiteralString mismatch
        report.append("=" * 80)
        report.append("BIOMECHANICAL POSTURE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Session Time: {int(total_time // 60)} minutes {int(total_time % 60)} seconds")
        report.append("")
        
        # Episode Counts removed per user request
        # Only time-in-bad-posture is shown


        # Maximum Deviations — real physical measurements
        report.append("MAXIMUM BIOMECHANICAL DEVIATIONS:")
        report.append("-" * 80)
        report.append(f"  Forward Head:     {self.max_forward_head_angle:.1f}%  (head depth toward camera vs baseline)")
        report.append(f"  Hunching:         {self.max_hunch_angle:.1f}%  (shoulder drop/forward shift vs baseline)")
        report.append(f"  Lateral Lean:     {self.max_lean_angle:.1f} deg (shoulder tilt angle vs baseline)")
        report.append(f"  Hip Sliding:      {self.max_hip_slide:.1f}%  (combined depth+vertical shift vs baseline)")
        report.append("")
        
        # Time Distribution
        good_pct = (self.good_posture_time / total_time * 100) if total_time > 0 else 0
        tolerance_pct = (self.tolerance_posture_time / total_time * 100) if total_time > 0 else 0
        bad_pct = (self.bad_posture_time / total_time * 100) if total_time > 0 else 0
        
        report.append("TIME DISTRIBUTION:")
        report.append("-" * 80)
        report.append(f"  Good Posture:       {good_pct:.1f}% ({int(self.good_posture_time)}s)")
        report.append(f"  Acceptable Zone:    {tolerance_pct:.1f}% ({int(self.tolerance_posture_time)}s)")
        report.append(f"  Bad Posture:        {bad_pct:.1f}% ({int(self.bad_posture_time)}s)")
        report.append("")
        
        # Per-defect time in bad posture
        report.append("TIME SPENT IN EACH BAD POSTURE:")
        report.append("-" * 80)
        total_bad = self.bad_posture_time if self.bad_posture_time > 0 else 1
        for key, label in [("forward_head", "Forward Head Posture"),
                           ("hunching",     "Hunching (Kyphosis) "),
                           ("lateral_lean", "Lateral Lean        "),
                           ("hip_sliding",  "Hip Sliding         ")]:
            secs = int(self.defect_time.get(key, 0.0))
            pct  = self.defect_time.get(key, 0.0) / total_bad * 100
            report.append(f"  {label}: {secs}s  ({pct:.1f}% of bad-posture time)")
        report.append("")

        # Dominant issue = most seconds in bad posture
        dominant_key = max(self.defect_time, key=lambda k: self.defect_time[k])
        dominant_secs = int(self.defect_time[dominant_key])

        # Recommendations
        report.append("CLINICAL RECOMMENDATIONS:")
        report.append("-" * 80)
        if dominant_secs == 0:
            report.append("  Excellent posture maintenance throughout session.")
        elif dominant_key == "forward_head":
            report.append("  - Raise monitor to eye level")
            report.append("  - Practice chin tuck exercises")
            report.append("  - Strengthen deep neck flexors")
        elif dominant_key == "hunching":
            report.append("  - Strengthen thoracic extensors")
            report.append("  - Perform scapular retraction exercises")
            report.append("  - Consider ergonomic chair adjustment")
        elif dominant_key == "lateral_lean":
            report.append("  - Check workstation symmetry")
            report.append("  - Strengthen oblique muscles")
            report.append("  - Alternate phone ear usage")
        elif dominant_key == "hip_sliding":
            report.append("  - Sit on sit bones, not tailbone")
            report.append("  - Adjust chair depth and lumbar support")
            report.append("  - Strengthen core muscles")

        report.append("=" * 80)

        return "\n".join(report)

"""
Draw a semi-transparent "ideal" skeleton based on baseline midpoints."""
def draw_ideal_skeleton_overlay(image, monitor, w, h, current_landmarks=None):
    if not monitor.calibrated or monitor.ideal_landmarks is None:
        return
    
    try:
        if current_landmarks and current_landmarks.get('left_eye') and current_landmarks.get('right_eye'):
            eye_mid = (
                (current_landmarks['left_eye'][0] + current_landmarks['right_eye'][0]) / 2,
                (current_landmarks['left_eye'][1] + current_landmarks['right_eye'][1]) / 2
            )
        else:
            eye_mid = monitor.ideal_landmarks['eye_mid']

        if current_landmarks and current_landmarks.get('left_shoulder') and current_landmarks.get('right_shoulder'):
            shoulder_mid = (
                (current_landmarks['left_shoulder'][0] + current_landmarks['right_shoulder'][0]) / 2,
                (current_landmarks['left_shoulder'][1] + current_landmarks['right_shoulder'][1]) / 2
            )
        else:
            shoulder_mid = monitor.ideal_landmarks['shoulder_mid']

        if current_landmarks and current_landmarks.get('left_hip') and current_landmarks.get('right_hip'):
            hip_mid = (
                (current_landmarks['left_hip'][0] + current_landmarks['right_hip'][0]) / 2,
                (current_landmarks['left_hip'][1] + current_landmarks['right_hip'][1]) / 2
            )
        else:
            hip_mid = monitor.ideal_landmarks['hip_mid']
        
        eye_px = (int(eye_mid[0] * w), int(eye_mid[1] * h))
        shoulder_px = (int(shoulder_mid[0] * w), int(shoulder_mid[1] * h))
        hip_px = (int(hip_mid[0] * w), int(hip_mid[1] * h))
        
        overlay = image.copy()
        color = (0, 255, 0)
        
        cv2.line(overlay, eye_px, shoulder_px, color, 3)
        cv2.line(overlay, shoulder_px, hip_px, color, 3)
        cv2.circle(overlay, eye_px, 8, color, -1)
        cv2.circle(overlay, shoulder_px, 10, color, -1)
        cv2.circle(overlay, hip_px, 10, color, -1)
        
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
        
    except:
        pass

def draw_selected_landmarks(image, landmarks, w, h):
    """Draw only the keypoints used by this app."""
    if landmarks is None:
        return
    used = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EYE,
        mp_pose.PoseLandmark.RIGHT_EYE,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
    ]
    pts = {}
    for lm in used:
        p = landmarks[lm.value]
        pts[lm] = (int(p.x * w), int(p.y * h))
        cv2.circle(image, pts[lm], 3, (80, 110, 255), -1)

    # Connect shoulders and hips, and midline
    if mp_pose.PoseLandmark.LEFT_SHOULDER in pts and mp_pose.PoseLandmark.RIGHT_SHOULDER in pts:
        cv2.line(image, pts[mp_pose.PoseLandmark.LEFT_SHOULDER], pts[mp_pose.PoseLandmark.RIGHT_SHOULDER], (80, 200, 120), 2)
    if mp_pose.PoseLandmark.LEFT_HIP in pts and mp_pose.PoseLandmark.RIGHT_HIP in pts:
        cv2.line(image, pts[mp_pose.PoseLandmark.LEFT_HIP], pts[mp_pose.PoseLandmark.RIGHT_HIP], (80, 200, 120), 2)
    # Midline: eye-mid to shoulder-mid to hip-mid
    if (mp_pose.PoseLandmark.LEFT_EYE in pts and mp_pose.PoseLandmark.RIGHT_EYE in pts and
        mp_pose.PoseLandmark.LEFT_SHOULDER in pts and mp_pose.PoseLandmark.RIGHT_SHOULDER in pts and
        mp_pose.PoseLandmark.LEFT_HIP in pts and mp_pose.PoseLandmark.RIGHT_HIP in pts):
        eye_mid = ((pts[mp_pose.PoseLandmark.LEFT_EYE][0] + pts[mp_pose.PoseLandmark.RIGHT_EYE][0]) // 2,
                   (pts[mp_pose.PoseLandmark.LEFT_EYE][1] + pts[mp_pose.PoseLandmark.RIGHT_EYE][1]) // 2)
        sh_mid = ((pts[mp_pose.PoseLandmark.LEFT_SHOULDER][0] + pts[mp_pose.PoseLandmark.RIGHT_SHOULDER][0]) // 2,
                  (pts[mp_pose.PoseLandmark.LEFT_SHOULDER][1] + pts[mp_pose.PoseLandmark.RIGHT_SHOULDER][1]) // 2)
        hip_mid = ((pts[mp_pose.PoseLandmark.LEFT_HIP][0] + pts[mp_pose.PoseLandmark.RIGHT_HIP][0]) // 2,
                   (pts[mp_pose.PoseLandmark.LEFT_HIP][1] + pts[mp_pose.PoseLandmark.RIGHT_HIP][1]) // 2)
        cv2.line(image, eye_mid, sh_mid, (80, 200, 120), 2)
        cv2.line(image, sh_mid, hip_mid, (80, 200, 120), 2)

# Render the compact dashboard panel (score, state, defect, timers).
def draw_translucent_dashboard(image, monitor, score, defects, state_duration, show_nodes):
    """MODULE C: Compact translucent dashboard - bottom-right corner."""
    h, w = image.shape[:2]

    panel_w = 200
    panel_h = 210
    panel_x = w - panel_w - 10
    panel_y = h - panel_h - 10

    # Translucent background (40% opaque)
    overlay = image.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (160, 160, 160), 1)

    fs   = 0.35   # standard font scale
    fs_t = 0.42   # title
    yo   = panel_y + 16
    xo   = panel_x + 8

    # Title
    cv2.putText(image, "POSTURE MONITOR", (xo, yo), cv2.FONT_HERSHEY_SIMPLEX, fs_t, (255, 255, 255), 1, cv2.LINE_AA)
    yo += 17

    # Calibration status
    if monitor.calibrated:
        cv2.putText(image, "Calibrated", (xo, yo), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 230, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(image, "Press c to calibrate", (xo, yo), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 180, 0), 1, cv2.LINE_AA)
    yo += 18

    # Nodes toggle status
    nodes_label = "Nodes: ON (n)" if show_nodes else "Nodes: OFF (n)"
    cv2.putText(image, nodes_label, (xo, yo), cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 200, 200), 1, cv2.LINE_AA)
    yo += 16

    if monitor.calibrated:
        state_colors = {
            PostureState.GOOD: (0, 230, 0),
            PostureState.BAD: (50, 50, 255)
        }
        sc = state_colors.get(monitor.current_state, (220, 220, 220))

        # State
        cv2.putText(image, f"State: {monitor.current_state.value}",
                    (xo, yo), cv2.FONT_HERSHEY_SIMPLEX, fs, sc, 1, cv2.LINE_AA)
        yo += 16

        # Active defect
        if defects:
            dominant    = max(defects, key=lambda x: x[1])
            defect_name = dominant[0].replace('_', ' ').title()
            cv2.putText(image, f"! {defect_name}", (xo, yo),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 180, 60), 1, cv2.LINE_AA)
        yo += 16
    else:
        cv2.putText(image, "Metrics hidden", (xo, yo),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (140, 140, 140), 1, cv2.LINE_AA)
        yo += 20

    # Divider
    cv2.line(image, (xo, yo - 3), (panel_x + panel_w - 8, yo - 3), (100, 100, 100), 1)

    if monitor.calibrated:
        # Bad posture time per defect
        fc  = (190, 190, 190)
        def _fmt(key: str, label: str) -> str:
            sec = int(monitor.defect_time.get(key, 0.0))
            return f"{label}: {sec}s"
        cv2.putText(image, _fmt('forward_head', 'Fwd Head   '),
                    (xo, yo + 12), cv2.FONT_HERSHEY_SIMPLEX, fs, fc, 1, cv2.LINE_AA)
        cv2.putText(image, _fmt('hunching',    'Hunching   '),
                    (xo, yo + 26), cv2.FONT_HERSHEY_SIMPLEX, fs, fc, 1, cv2.LINE_AA)
        cv2.putText(image, _fmt('lateral_lean','Lat Lean   '),
                    (xo, yo + 40), cv2.FONT_HERSHEY_SIMPLEX, fs, fc, 1, cv2.LINE_AA)
        cv2.putText(image, _fmt('hip_sliding', 'Hip Sliding'),
                    (xo, yo + 54), cv2.FONT_HERSHEY_SIMPLEX, fs, fc, 1, cv2.LINE_AA)
        yo += 62
    else:
        yo += 12

    # Next microbreak countdown
    if monitor.calibrated:
        time_to_break = int(monitor.microbreak_interval - (time.time() - monitor.last_microbreak))
        if time_to_break > 0:
            cv2.putText(image, f"Break in {time_to_break}s", (xo, yo + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (100, 200, 255), 1, cv2.LINE_AA)


def draw_microbreak_screen(image, monitor):
    h, w = image.shape[:2]
    
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 60, 20), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    
    panel_w = 750
    panel_h = 450
    panel_x = (w - panel_w) // 2
    panel_y = (h - panel_h) // 2
    
    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (50, 100, 50), -1)
    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (100, 255, 100), 5)
    
    y_pos = panel_y + 70
    
    cv2.putText(image, "MICROBREAK TIME", (panel_x + 200, y_pos),
               cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    y_pos += 70
    
    elapsed = int(time.time() - monitor.microbreak_start)
    cv2.putText(image, f"Time: {elapsed}s / 60s", (panel_x + 280, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2, cv2.LINE_AA)
    y_pos += 60
    
    # Cycle exercises every 20 seconds
    exercise_index = (elapsed // 20) % len(monitor.microbreak_exercises)
    current_exercise = monitor.microbreak_exercises[exercise_index]
    
    cv2.putText(image, "Current Exercise:", (panel_x + 60, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    y_pos += 60
    
    # Word wrap
    words = current_exercise.split()
    line = ""
    for word in words:
        test_line = line + word + " "
        if len(test_line) > 50:
            cv2.putText(image, line, (panel_x + 80, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            y_pos += 40
            line = word + " "
        else:
            line = test_line
    if line:
        cv2.putText(image, line, (panel_x + 80, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    y_pos += 70
    cv2.putText(image, "Press 'b' to end break early", (panel_x + 220, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 230, 255), 1, cv2.LINE_AA)

# Show saved bad-posture screenshots in separate windows.
def display_captured_screenshots(monitor):
    if not monitor.bad_posture_events:
        print("No bad posture events captured.")
        return
    
    print(f"\nDisplaying {len(monitor.bad_posture_events)} captured events...")
    
    for idx, event in enumerate(monitor.bad_posture_events):
        frame = event['frame']
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f"Bad Posture Event #{idx + 1}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {event['timestamp']}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Type: {event['defect_type']} | Severity: {event['severity']:.1f}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Duration: {event['duration']}s", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1)
        
        cv2.imshow(f"Event {idx + 1}/{len(monitor.bad_posture_events)}", frame)
    
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# App entrypoint: webcam loop, pose detection, scoring, UI overlays,
# microbreak handling, and final report generation.
def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    monitor = PostureMonitor()
    
    print("\n" + "=" * 80)
    print("ADVANCED BIOMECHANICAL POSTURE ANALYSIS SYSTEM")
    print("=" * 80)
    print("\nFEATURES:")
    print("  - Vector-based biomechanical analysis")
    print("  - Episode counting (not frame counting)")
    print("  - Translucent dashboard (60% alpha blending)")
    print("  - Microbreak protocol (every 60 seconds)")
    print("  - Medical reporting with episode counts")
    print("\nCONTROLS:")
    print("  'c' - Calibrate baseline posture")
    print("  'b' - End microbreak early")
    print("  'q' - Quit and generate report")
    print("\nCALIBRATE FIRST: Sit in your BEST posture and press 'c'")
    print("=" * 80 + "\n")
    
    bad_event_start = None
    current_defects = []
    show_nodes = True
    last_bad_sound_time = 0.0
    BAD_SOUND_COOLDOWN = 3.0
    # Cache latest landmarks for keyboard calibration
    latest_landmarks = None

    # ── Display cache: refresh dashboard every 2 seconds ──────────────────────
    DISPLAY_REFRESH_S: float = 2.5
    last_display_update: float = 0.0
    disp_score: Optional[float]  = None
    disp_defects: list           = []
    disp_state_dur: float        = 0.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        h, w, c = image.shape

        
        # Microbreak check
        current_time = time.time()
        if monitor.calibrated and not monitor.in_microbreak and (current_time - monitor.last_microbreak >= monitor.microbreak_interval):
            monitor.in_microbreak = True
            monitor.microbreak_start = current_time
            print("[MICROBREAK] Started!")
        
        landmark_data = None

        if monitor.in_microbreak:
            draw_microbreak_screen(image, monitor)
            
            elapsed_break = current_time - (monitor.microbreak_start or current_time)
            if elapsed_break >= 60:
                monitor.in_microbreak = False
                monitor.last_microbreak = current_time
                print("[MICROBREAK] Completed!")
        else:
            if results.pose_landmarks:
                if show_nodes:
                    draw_selected_landmarks(image, results.pose_landmarks.landmark, w, h)

                latest_landmarks = results.pose_landmarks.landmark
                landmark_data = monitor.extract_landmarks(latest_landmarks)

                if show_nodes:
                    draw_ideal_skeleton_overlay(image, monitor, w, h, landmark_data)
                _score, defects = monitor.analyze_posture_biomechanics(landmark_data)
                state, state_duration = monitor.update_state_machine(None, defects)

                # Debug: show hip slide inputs
                try:
                    cv2.putText(image, f"Hip Z away: {sh_z_away:.3f}  Hip Y drop: {sh_y_change:.3f}  Hip sev: {hip_severity:.1f}",
                                (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
                except:
                    pass
                
                # ── Bad-posture event tracking ────────────────────────────
                if state == PostureState.BAD:
                    if bad_event_start is None:
                        bad_event_start = time.time()
                        current_defects = defects

                    _t: float = time.time()
                    bad_event_start_f: float = bad_event_start if bad_event_start is not None else _t  # type: ignore[assignment]
                    duration = int(time.time() - bad_event_start_f)

                    # Continuously update/capture so the stored frame
                    # always reflects the longest running episode.
                    if duration >= 3:
                        monitor.capture_bad_posture_event(image, defects, duration)

                    if duration >= 5:
                        alert_w = 450
                        alert_h = 90
                        alert_x = (w - alert_w) // 2
                        alert_y = 60

                        cv2.rectangle(image, (alert_x, alert_y),
                                      (alert_x + alert_w, alert_y + alert_h), (0, 0, 200), -1)
                        cv2.rectangle(image, (alert_x, alert_y),
                                      (alert_x + alert_w, alert_y + alert_h), (255, 255, 255), 4)
                        cv2.putText(image, "CORRECT YOUR POSTURE!",
                                    (alert_x + 60, alert_y + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(image, f"Bad posture: {duration}s",
                                    (alert_x + 140, alert_y + 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 1, cv2.LINE_AA)
                        # Soft notification sound (debounced)
                        now = time.time()
                        if winsound and (now - last_bad_sound_time) >= BAD_SOUND_COOLDOWN:
                            try:
                                winsound.Beep(1000, 150)
                            except Exception:
                                pass
                            last_bad_sound_time = now
                else:
                    bad_event_start = None
                
                # \u2500\u2500 Refresh display cache every 2 seconds \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
                now_disp = time.time()
                if now_disp - last_display_update >= DISPLAY_REFRESH_S:
                    disp_score     = None
                    disp_defects   = defects
                    disp_state_dur = state_duration
                    last_display_update = now_disp

                draw_translucent_dashboard(image, monitor, disp_score, disp_defects, disp_state_dur, show_nodes)
            else:
                cv2.putText(image, "Position yourself in frame", (50, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                draw_translucent_dashboard(image, monitor, None, [], 0, show_nodes)
        
        cv2.imshow('Biomechanical Posture Analysis', image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Try current frame landmarks first, fallback to last known
            current_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
            use_landmarks = current_landmarks or latest_landmarks
            if use_landmarks:
                if monitor.calibrate_baseline(use_landmarks):
                    print("[SUCCESS] Baseline calibrated!")
            else:
                print("[ERROR] No pose detected")
        elif key == ord('b'):
            if monitor.in_microbreak:
                monitor.in_microbreak = False
                monitor.last_microbreak = time.time()
                print("[MICROBREAK] Ended early")
        elif key == ord('n'):
            show_nodes = not show_nodes
            print(f"[NODES] {'ON' if show_nodes else 'OFF'}")
    
    # Generate report
    print("\n" + "=" * 80)
    print("GENERATING MEDICAL REPORT...")
    print("=" * 80 + "\n")
    
    report = monitor.generate_medical_report()
    print(report)
    
    filename = f"posture_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved: {filename}")
    
    display_captured_screenshots(monitor)
    
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == '__main__':
    main()
'''

import cv2
import mediapipe as mp  
import numpy as np  
import time
import math
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any
try:
    import winsound
except Exception:
    winsound = None

# MediaPipe pose module alias (landmark indices + model).
mp_pose = mp.solutions.pose
# MediaPipe drawing utils for skeleton overlay.
mp_drawing = mp.solutions.drawing_utils
# Global pose detector configured for stable tracking.
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Finite posture states used by the session state machine.
class PostureState(Enum):
    GOOD = "GOOD"
    BAD = "BAD"

# Central state container and analytics engine for posture monitoring.
# Owns calibration, scoring, episode counting, and reporting state.
class PostureMonitor:
    def __init__(self) -> None:
        self.calibrated: bool = False
        self.baseline_landmarks: Optional[Dict[str, Any]] = None
        self.ideal_landmarks: Optional[Dict[str, Any]] = None
        
        # Session timing
        self.session_start: float = time.time()
        self.last_check_time: float = time.time()
        
        # MODULE B: Episode-based state machine
        self.current_state = PostureState.GOOD
        self.state_start_time: float = time.time()
        self.episode_counts: Dict[str, int] = {
            "hunching": 0,
            "lateral_lean": 0,
            "hip_sliding": 0
        }
        
        # Time tracking
        self.good_posture_time: float = 0.0
        self.bad_posture_time: float = 0.0

        # Per-defect bad-posture time (seconds accumulated while each defect is dominant)
        self.defect_time: Dict[str, float] = {
            "hunching": 0.0,
            "lateral_lean": 0.0,
            "hip_sliding": 0.0
        }
        
        # Analytics
        self.max_hunch_angle: float = 0.0
        self.max_lean_angle: float = 0.0
        self.max_hip_slide: float = 0.0

        # Forward head hybrid metric state (angle + z)
        self.base_head_angle: Optional[float] = None
        self.base_head_z_rel: Optional[float] = None
        self.fh_angle_ema: float = 0.0
        self.fh_z_ema: float = 0.0
        self.fh_ema_alpha: float = 0.3

        # EMA for hunch inputs (smooths noisy Z/Y depth from MediaPipe)
        self.hunch_z_ema: float = 0.0
        self.hunch_y_ema: float = 0.0
        self.hunch_ema_alpha: float = 0.2  # lower = more damping
        
        # New final-severity EMAs to stop GOOD/BAD flickering at the border
        self.hunch_severity_ema: float = 0.0
        self.lean_severity_ema: float = 0.0
        
        # Track raw (un-smoothed) severities exactly for screenshot selection
        self.current_raw_severities: Dict[str, float] = {}

        # Lateral lean: shoulder-tilt EMA and calibrated baseline tilt
        self.lean_tilt_ema: float = 0.0
        self.base_shoulder_tilt: Optional[float] = None  # calibrated shoulder tilt (degrees)

        # Lateral lean baseline offsets relative to vertical axis (eye midpoint)
        self.base_shoulder_axis_offset: Optional[float] = None
        self.base_hip_axis_offset: Optional[float] = None
        self.base_torso_angle: Optional[float] = None   # calibrated torso angle (degrees)
        
        # Bad posture event capture
        self.bad_posture_events: List[Dict[str, Any]] = []
        self.current_bad_event: Optional[Dict[str, Any]] = None
        
        # Microbreak timer
        self.last_microbreak: float = time.time()
        self.microbreak_interval: int = 60  # 60 seconds as required
        self.in_microbreak: bool = False
        self.microbreak_start: Optional[float] = None
        self.microbreak_exercises: List[str] = [
            "Neck Rolls: Slowly rotate your head clockwise, then counter-clockwise",
            "Shoulder Retractions: Pull shoulder blades together, hold 5 seconds",
            "Spinal Extensions: Place hands on lower back, gently arch backward"
        ]

        # Episode cooldown: track last time each defect episode was counted
        # to avoid rapid re-counting when score oscillates near the BAD boundary.
        self.last_episode_time: Dict[str, float] = {
            "hunching": 0.0,
            "lateral_lean": 0.0,
            "hip_sliding": 0.0
        }
        self.episode_cooldown: float = 5.0   # seconds minimum between episodes
        
    # Compute the angle (degrees) formed by three 2D points at point2.
    # Returns 0 when vectors are degenerate to avoid divide-by-zero.
    def calculate_angle(self, point1, point2, point3):
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return math.degrees(angle)
    
    # Extract required pose landmarks into a plain dict of tuples.
    # Returns None if any critical landmark is missing.
    def extract_landmarks(self, landmarks):
        try:
            nose           = landmarks[mp_pose.PoseLandmark.NOSE.value]           # type: ignore[index]
            left_shoulder  = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]  # type: ignore[index]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] # type: ignore[index]
            left_hip       = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]       # type: ignore[index]
            right_hip      = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]      # type: ignore[index]
            left_ear       = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]       # type: ignore[index]
            right_ear      = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]      # type: ignore[index]
            left_eye       = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]       # type: ignore[index]
            right_eye      = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]      # type: ignore[index]
            
            # Try to get knee landmarks (may not be visible)
            try:
                left_knee  = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]   # type: ignore[index]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]  # type: ignore[index]
            except:
                left_knee = None
                right_knee = None
            
            return {
                'nose': (nose.x, nose.y, nose.z),
                'left_shoulder': (left_shoulder.x, left_shoulder.y, left_shoulder.z),
                'right_shoulder': (right_shoulder.x, right_shoulder.y, right_shoulder.z),
                'left_hip': (left_hip.x, left_hip.y, left_hip.z),
                'right_hip': (right_hip.x, right_hip.y, right_hip.z),
                'left_ear': (left_ear.x, left_ear.y, left_ear.z),
                'right_ear': (right_ear.x, right_ear.y, right_ear.z),
                'left_eye': (left_eye.x, left_eye.y, left_eye.z),
                'right_eye': (right_eye.x, right_eye.y, right_eye.z),
                'left_knee': (left_knee.x, left_knee.y, left_knee.z) if left_knee else None,
                'right_knee': (right_knee.x, right_knee.y, right_knee.z) if right_knee else None,
            }
        except Exception as e:
            return None
    
    # Calibrate the baseline (ideal) posture from a single frame.
    # Stores reference landmarks and midpoints for overlay.
    def calibrate_baseline(self, landmarks):
        landmark_data = self.extract_landmarks(landmarks)
        if landmark_data:
            self.baseline_landmarks = landmark_data
            
            # Store ideal skeleton for overlay
            shoulder_mid = (
                (landmark_data['left_shoulder'][0] + landmark_data['right_shoulder'][0]) / 2,  # type: ignore[operator]
                (landmark_data['left_shoulder'][1] + landmark_data['right_shoulder'][1]) / 2   # type: ignore[operator]
            )
            hip_mid = (
                (landmark_data['left_hip'][0] + landmark_data['right_hip'][0]) / 2,  # type: ignore[operator]
                (landmark_data['left_hip'][1] + landmark_data['right_hip'][1]) / 2   # type: ignore[operator]
            )
            eye_mid = (
                (landmark_data['left_eye'][0] + landmark_data['right_eye'][0]) / 2,  # type: ignore[operator]
                (landmark_data['left_eye'][1] + landmark_data['right_eye'][1]) / 2   # type: ignore[operator]
            )

            self.ideal_landmarks = {
                'eye_mid': eye_mid,
                'shoulder_mid': shoulder_mid,
                'hip_mid': hip_mid
            }

            # Baseline values for forward-head hybrid metric
            ear_mid = (
                (landmark_data['left_ear'][0] + landmark_data['right_ear'][0]) / 2,  # type: ignore[operator]
                (landmark_data['left_ear'][1] + landmark_data['right_ear'][1]) / 2,  # type: ignore[operator]
                (landmark_data['left_ear'][2] + landmark_data['right_ear'][2]) / 2   # type: ignore[operator]
            )
            shoulder_mid3 = (
                (landmark_data['left_shoulder'][0] + landmark_data['right_shoulder'][0]) / 2,  # type: ignore[operator]
                (landmark_data['left_shoulder'][1] + landmark_data['right_shoulder'][1]) / 2,  # type: ignore[operator]
                (landmark_data['left_shoulder'][2] + landmark_data['right_shoulder'][2]) / 2   # type: ignore[operator]
            )
            hip_mid3 = (
                (landmark_data['left_hip'][0] + landmark_data['right_hip'][0]) / 2,  # type: ignore[operator]
                (landmark_data['left_hip'][1] + landmark_data['right_hip'][1]) / 2,  # type: ignore[operator]
                (landmark_data['left_hip'][2] + landmark_data['right_hip'][2]) / 2   # type: ignore[operator]
            )
            v1 = np.array([ear_mid[0] - shoulder_mid3[0], ear_mid[1] - shoulder_mid3[1]])
            v2 = np.array([hip_mid3[0] - shoulder_mid3[0], hip_mid3[1] - shoulder_mid3[1]])
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1.0
            cosang = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
            self.base_head_angle = math.degrees(math.acos(cosang))
            self.base_head_z_rel = float(ear_mid[2]) - float(shoulder_mid3[2])
            self.fh_angle_ema = 0.0
            self.fh_z_ema = 0.0

            # Baseline lateral offsets from vertical axis (eye midpoint)
            self.base_shoulder_axis_offset = float(shoulder_mid[0]) - float(eye_mid[0])
            self.base_hip_axis_offset = float(hip_mid[0]) - float(eye_mid[0])

            # Baseline torso angle (kept for reference) and shoulder tilt for lateral lean
            sh_x = float(shoulder_mid[0])
            sh_y_val = float(shoulder_mid[1])
            hip_x = float(hip_mid[0])
            hip_y_val = float(hip_mid[1])
            _vx = hip_x - sh_x
            _vy = hip_y_val - sh_y_val
            if abs(_vx) + abs(_vy) > 1e-6:
                _ta = abs(math.degrees(math.atan2(_vy, _vx)))
                self.base_torso_angle = 360.0 - _ta if _ta > 180.0 else _ta
            else:
                self.base_torso_angle = 90.0

            # Baseline shoulder tilt (left→right shoulder line from horizontal)
            ls_cal = landmark_data['left_shoulder']
            rs_cal = landmark_data['right_shoulder']
            _sdx = float(rs_cal[0]) - float(ls_cal[0])  # type: ignore[index]
            _sdy = float(rs_cal[1]) - float(ls_cal[1])  # type: ignore[index]
            if abs(_sdx) > 1e-4:
                _bst = float(math.degrees(math.atan2(_sdy, _sdx)))
                self.base_shoulder_tilt = _bst
                self.lean_tilt_ema = _bst  # seed EMA at baseline
            else:
                self.base_shoulder_tilt = 0.0
                self.lean_tilt_ema = 0.0

            # Reset microbreak timer on calibration
            self.last_microbreak = time.time()
            self.in_microbreak = False
            self.microbreak_start = None
            
            self.calibrated = True
            return True
        return False
    
    # Score posture from current landmarks using isolated biomechanical detectors.
    # Returns (score, defects[]) where defects are (defect_key, severity).
    # 
    # MediaPipe coords (normalized 0-1):
    # x: left->right, y: top->bottom (larger=lower), z: negative=closer to cam
    # 
    # Detector isolation rules:
    # - Forward head: nose Z relative to shoulder midpoint moves closer.
    # - Hunching: shoulders move closer (z decreases) and drop (y increases).
    # - Lateral lean: shoulder tilt in X-Y plane only (no Z).
    # - Hip sliding: shoulders move away (z increases) and sink (y increases).
    # - Hunching vs hip sliding are mutually exclusive by Z direction.
    def analyze_posture_biomechanics(self, current_landmarks):
        """
        Strictly isolated biomechanical detectors.

        MediaPipe coords (normalised 0-1):
          x: left→right,  y: top→bottom (larger=lower),  z: negative=closer to cam

        DETECTOR ISOLATION RULES:
          FWD HEAD   : nose Z relative to shoulders becomes more negative (head closer)
          HUNCHING   : shoulder Z more negative (closer) AND shoulder Y increases (drops)
          LAT LEAN   : shoulder tilt angle deviates left/right (purely X-axis tilt)
          HIP SLIDE  : shoulder Z more positive (farther) AND shoulder Y increases (sinks)
          → Hunching and Hip Sliding are mutually exclusive by Z direction.
          → Forward Head is always relative to shoulder midpoint – body-movement cancelled.
        """
        if not self.calibrated or not current_landmarks:
            return None, []

        baseline = self.baseline_landmarks
        current  = current_landmarks
        defects  = []

        # ── shared midpoints ─────────────────────────────────────────────────────────
        def mid3(a, b):
            return np.array([(a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2])

        sh_cur  = mid3(current['left_shoulder'],  current['right_shoulder'])   # type: ignore[arg-type]
        sh_base = mid3(baseline['left_shoulder'], baseline['right_shoulder'])   # type: ignore[arg-type]
        nose_c  = np.array(current['nose'])   # type: ignore[arg-type]
        nose_b  = np.array(baseline['nose'])  # type: ignore[arg-type]

        # Raw Z change for shoulders (used by hunching and hip sliding)
        sh_z_change = float(sh_cur[2]) - float(sh_base[2])   # negative=closer, positive=farther
        sh_y_change = float(sh_cur[1]) - float(sh_base[1])   # positive=dropped in frame

        # -- 1. FORWARD HEAD (HYBRID: ANGLE + Z) --------------------------------------------
        # Uses ear-midpoint angle vs baseline + relative Z depth to shoulder midpoint.
        ANGLE_DEADBAND = 2.0
        Z_DEADBAND = 0.01
        angle_weight = 1.0
        z_weight = 1.0
        angle_scale = 3.0
        z_scale = 250.0

        # ── 1+2. FORWARD HEAD + HUNCHBACK (KYPHOSIS) — combined as "hunching" ──────
        # Forward head severity (hybrid angle+Z) is folded into the hunching
        # severity so both head-forward and shoulder-rounding are captured
        # under a single label that the user sees.

        # Forward-head component (hybrid angle + Z)
        fwd_component = 0.0
        if self.base_head_angle is not None and self.base_head_z_rel is not None:
            shoulder_span = abs(float(current['left_shoulder'][0]) - float(current['right_shoulder'][0]))  # type: ignore[index]
            if shoulder_span > 1e-3:
                ear_mid = np.array([
                    (float(current['left_ear'][0]) + float(current['right_ear'][0])) / 2,  # type: ignore[index]
                    (float(current['left_ear'][1]) + float(current['right_ear'][1])) / 2,  # type: ignore[index]
                    (float(current['left_ear'][2]) + float(current['right_ear'][2])) / 2   # type: ignore[index]
                ])
                v1 = np.array([ear_mid[0] - sh_cur[0], ear_mid[1] - sh_cur[1]])
                v2 = np.array([((current['left_hip'][0] + current['right_hip'][0]) / 2) - sh_cur[0],  # type: ignore[index]
                               ((current['left_hip'][1] + current['right_hip'][1]) / 2) - sh_cur[1]])  # type: ignore[index]
                denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1.0
                cosang = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
                head_angle = math.degrees(math.acos(cosang))
                head_z_rel_cur = float(ear_mid[2]) - float(sh_cur[2])
                angle_delta = max(0.0, head_angle - self.base_head_angle - 2.0)  # type: ignore[operator]
                z_delta     = max(0.0, self.base_head_z_rel - head_z_rel_cur - 0.01)  # type: ignore[operator]
                a = self.fh_ema_alpha
                self.fh_angle_ema = a * angle_delta + (1.0 - a) * self.fh_angle_ema
                self.fh_z_ema     = a * z_delta     + (1.0 - a) * self.fh_z_ema
                fwd_component = self.fh_angle_ema * 3.0 + self.fh_z_ema * 250.0  # unbound

        # Shoulder-rounding (kyphosis) component — EMA-smoothed to kill Z noise
        sh_z_towards = -sh_z_change
        ha = self.hunch_ema_alpha
        self.hunch_z_ema = ha * sh_z_towards + (1.0 - ha) * self.hunch_z_ema
        self.hunch_y_ema = ha * sh_y_change  + (1.0 - ha) * self.hunch_y_ema
        hunch_z = max(0.0, self.hunch_z_ema - 0.02) * 130.0
        hunch_y = max(0.0, self.hunch_y_ema - 0.02) * 180.0
        hunch_component = max(0.0, hunch_z + hunch_y) if self.hunch_z_ema > 0.02 else 0.0  # unbound

        # Combined hunching severity — heavily smoothed to prevent boundary flickering
        raw_hunch_severity = fwd_component + hunch_component  # unbound to capture true peak
        self.current_raw_severities["hunching"] = raw_hunch_severity
        h_alpha = 0.1  # 10% new value per frame, very smooth
        self.hunch_severity_ema = (1.0 - h_alpha) * self.hunch_severity_ema + h_alpha * raw_hunch_severity
        
        if self.hunch_severity_ema > 8.5:  # lowered threshold from 10.0 to 8.5
            clamped_hunch = min(30.0, self.hunch_severity_ema)
            defects.append(("hunching", clamped_hunch))
            self.max_hunch_angle = max(self.max_hunch_angle,
                                      round(clamped_hunch, 1))  # type: ignore[call-overload]

        # ── 3. LATERAL LEAN ───────────────────────────────────────────────────
        # User requirement: detect lateral lean ONLY when BOTH ears cross one side of the 
        # vertical y-axis (calibrated eye midpoint).
        
        lean_severity = 0.0
                
        # B) Ears crossing vertical axis check
        # The true calibrated axis is at self.ideal_landmarks['eye_mid'][0]
        ideal = self.ideal_landmarks
        raw_lean_severity = 0.0
        if ideal is not None and 'eye_mid' in ideal:
            base_axis_x = float(ideal['eye_mid'][0])
            left_ear_x = float(current['left_ear'][0])   # type: ignore[index]
            right_ear_x = float(current['right_ear'][0]) # type: ignore[index]
            
            # Both ears on the same side of the axis (from camera POV)
            crossed_left = left_ear_x < base_axis_x and right_ear_x < base_axis_x
            crossed_right = left_ear_x > base_axis_x and right_ear_x > base_axis_x
            
            if crossed_left or crossed_right:
                # Add severity based on how far past the axis the closest ear went
                closest_ear = max(left_ear_x, right_ear_x) if crossed_left else min(left_ear_x, right_ear_x)
                axis_dev = abs(closest_ear - base_axis_x)
                # scale up the normalized coord diff
                raw_lean_severity = max(0.0, axis_dev * 500.0)  # unbound
                
        self.current_raw_severities["lateral_lean"] = raw_lean_severity
        
        # Smooth lateral lean to prevent fast flickering
        l_alpha = 0.15
        self.lean_severity_ema = (1.0 - l_alpha) * self.lean_severity_ema + l_alpha * raw_lean_severity

        if self.lean_severity_ema > 5.0:  # slight buffer before triggering
            clamped_lean = min(30.0, self.lean_severity_ema)
            defects.append(("lateral_lean", clamped_lean))
            self.max_lean_angle = max(self.max_lean_angle,
                                     round(clamped_lean, 1))  # type: ignore[call-overload]

        # ── 4. HIP SLIDING ──────────────────────────────────────────────────
        # Shoulders move AWAY from camera (z increases = sh_z_change > 0)
        # AND body sinks (y increases = sh_y_change > 0)
        # GUARD: sh_z_change must be positive (away). If negative, it is hunching.
        sh_z_away = sh_z_change   # positive when shoulders move away from cam
        hip_z = max(0.0, sh_z_away  - 0.03) * 130.0
        hip_y = max(0.0, sh_y_change - 0.03) * 180.0
        hip_severity = max(0.0, hip_z + hip_y) if sh_z_away > 0.03 and sh_y_change > 0.03 else 0.0  # unbound
        
        # GUARD: Suppress hip sliding if a lateral lean OR hunching is actively detected,
        # because leaning or slouching naturally drops the shoulder midpoint and 
        # can falsely trigger the hip sliding detector.
        if self.lean_severity_ema > 5.0 or self.hunch_severity_ema > 8.5:
            hip_severity = 0.0
            
        self.current_raw_severities["hip_sliding"] = hip_severity
        
        if hip_severity > 3.5:
            clamped_hip = min(30.0, hip_severity)
            defects.append(("hip_sliding", clamped_hip))
            self.max_hip_slide = max(self.max_hip_slide,
                                    round(clamped_hip, 1))  # type: ignore[call-overload]

        # ── Final score ───────────────────────────────────────────────────────────
        total_deduction = sum(d[1] for d in defects)
        score = max(0.0, 100.0 - total_deduction)
        return score, defects
    
    # Update posture state machine with hysteresis and episode counting.
    # GOOD >= 82, ACCEPTABLE >= 65, BAD < 65.
    # Episodes are counted only on sustained BAD + cooldown per defect.
    def update_state_machine(self, score, defects):
        """MODULE B: State machine with hysteresis to prevent rapid oscillation.
        States:
          GOOD       : score >= 82
          ACCEPTABLE : score >= 65
          BAD        : score <  65
        A transition to BAD is only counted as an episode if the score has been
        below 65 for at least 2 continuous seconds (prevents brief dips from
        inflating episode counts).
        """
        current_time = time.time()
        elapsed = current_time - self.last_check_time

        # Determine new state based on defect counts (4 metrics)
        if score is None:
            defect_keys = {d[0] for d in defects} if defects else set()
            bad_count = len(defect_keys)
            if bad_count == 0:
                new_state = PostureState.GOOD
            else:
                new_state = PostureState.BAD
        elif score >= 80:
            new_state = PostureState.GOOD
        else:
            new_state = PostureState.BAD

        # Accumulate time in current state
        if self.current_state == PostureState.GOOD:
            self.good_posture_time += elapsed
        elif self.current_state == PostureState.BAD:
            self.bad_posture_time += elapsed
        # Accumulate per-defect time for any active defects (independent of state)
        if defects:
            for key, _sev in defects:
                self.defect_time[key] = self.defect_time.get(key, 0.0) + elapsed

        state_duration = current_time - self.state_start_time

        # Only count an episode when transitioning INTO BAD AND:
        #   - we've been in GOOD for > 5s (hysteresis: bad posture sustained)
        #   - enough cooldown since last episode of this defect type
        if self.current_state == PostureState.GOOD and new_state == PostureState.BAD:
            if state_duration > 5.0 and defects:
                dominant_defect = max(defects, key=lambda x: x[1])
                defect_key = dominant_defect[0]
                time_since_last = current_time - self.last_episode_time.get(defect_key, 0.0)
                if time_since_last >= self.episode_cooldown:
                    self.episode_counts[defect_key] += 1
                    self.last_episode_time[defect_key] = current_time
                    print(f"[EPISODE] {defect_key} detected (Episode #{self.episode_counts[defect_key]})")

        if new_state != self.current_state:
            self.current_state = new_state
            self.state_start_time = current_time

        self.last_check_time = current_time
        return self.current_state, state_duration
    
    # Track a single representative screenshot per defect type.
    # Keeps the frame with the longest continuous BAD duration.
    def capture_bad_posture_event(self, frame, defects, duration):
        """Keep ONE best-frame per defect type, selected by highest RAW physical severity.
        This captures the moment the posture was at its absolute worst, rather than the
        end of an episode or an averaged-out peak.
        """
        if not defects:
            return

        dominant_defect = max(defects, key=lambda x: x[1])
        defect_key = dominant_defect[0]
        raw_sev = self.current_raw_severities.get(defect_key, 0.0)

        new_event = {
            'frame': frame.copy(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'duration': duration,
            'defect_type': defect_key,
            'severity': float(dominant_defect[1]),
            'raw_severity': raw_sev
        }

        # Find the existing entry for this defect type
        existing_idx = None
        existing_raw_severity = 0.0
        for i, e in enumerate(self.bad_posture_events):
            if e['defect_type'] == defect_key:
                existing_idx = i
                existing_raw_severity = e.get('raw_severity', 0.0)
                break

        if existing_idx is None:
            self.bad_posture_events.append(new_event)
        elif raw_sev > existing_raw_severity:
            assert existing_idx is not None  # already checked above; helps Pyre
            self.bad_posture_events[existing_idx] = new_event  # type: ignore[index]

        # Sort by highest severity first
        self.bad_posture_events.sort(key=lambda x: x['severity'], reverse=True)

    # Produce a text summary of the session: time in states, maxima,
    # and ergonomic recommendations based on the dominant defect.
    def generate_medical_report(self):
        """MODULE E: Generate medical report with episode counts"""
        total_time = time.time() - self.session_start
        
        report: List[str] = []  # explicit List[str] silences LiteralString mismatch
        report.append("=" * 80)
        report.append("BIOMECHANICAL POSTURE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Session Time: {int(total_time // 60)} minutes {int(total_time % 60)} seconds")
        report.append("")
        
        # Episode Counts removed per user request
        # Only time-in-bad-posture is shown


        # Maximum Deviations — real physical measurements
        report.append("MAXIMUM BIOMECHANICAL DEVIATIONS:")
        report.append("-" * 80)
        report.append(f"  Hunching:         {self.max_hunch_angle:.1f}  (combined head/shoulder deviation)")
        report.append(f"  Lateral Lean:     {self.max_lean_angle:.1f} deg (torso tilt from calibrated angle)")
        report.append(f"  Hip Sliding:      {self.max_hip_slide:.1f}%  (combined depth+vertical shift vs baseline)")
        report.append("")
        
        # Time Distribution
        good_pct = (self.good_posture_time / total_time * 100) if total_time > 0 else 0
        bad_pct = (self.bad_posture_time / total_time * 100) if total_time > 0 else 0
        
        report.append("TIME DISTRIBUTION:")
        report.append("-" * 80)
        report.append(f"  Good Posture:       {good_pct:.1f}% ({int(self.good_posture_time)}s)")
        report.append(f"  Bad Posture:        {bad_pct:.1f}% ({int(self.bad_posture_time)}s)")
        report.append("")
        
        # Per-defect time in bad posture
        report.append("TIME SPENT IN EACH BAD POSTURE:")
        report.append("-" * 80)
        total_bad = self.bad_posture_time if self.bad_posture_time > 0 else 1
        for key, label in [("hunching",     "Hunching    "),
                           ("lateral_lean", "Lateral Lean"),
                           ("hip_sliding",  "Hip Sliding ")]:
            secs = int(self.defect_time.get(key, 0.0))
            pct  = self.defect_time.get(key, 0.0) / total_bad * 100
            report.append(f"  {label}: {secs}s  ({pct:.1f}% of bad-posture time)")
        report.append("")

        # Dominant issue = most seconds in bad posture
        dominant_key = max(self.defect_time, key=lambda k: self.defect_time[k])
        dominant_secs = int(self.defect_time[dominant_key])

        # Recommendations
        report.append("CLINICAL RECOMMENDATIONS:")
        report.append("-" * 80)
        if dominant_secs == 0:
            report.append("  Excellent posture maintenance throughout session.")
        elif dominant_key == "hunching":
            report.append("  - Raise monitor to eye level and strengthen deep neck flexors")
            report.append("  - Strengthen thoracic extensors")
            report.append("  - Perform scapular retraction exercises")
            report.append("  - Consider ergonomic chair adjustment")
        elif dominant_key == "lateral_lean":
            report.append("  - Check workstation symmetry")
            report.append("  - Strengthen oblique muscles")
            report.append("  - Alternate phone ear usage")
        elif dominant_key == "hip_sliding":
            report.append("  - Sit on sit bones, not tailbone")
            report.append("  - Adjust chair depth and lumbar support")
            report.append("  - Strengthen core muscles")

        report.append("=" * 80)

        return "\n".join(report)

"""
Draw a semi-transparent "ideal" skeleton based on baseline midpoints."""
def draw_ideal_skeleton_overlay(image, monitor, w, h, current_landmarks=None):
    if not monitor.calibrated or monitor.ideal_landmarks is None:
        return
    
    try:
        if current_landmarks and current_landmarks.get('left_eye') and current_landmarks.get('right_eye'):
            eye_mid = (
                (current_landmarks['left_eye'][0] + current_landmarks['right_eye'][0]) / 2,
                (current_landmarks['left_eye'][1] + current_landmarks['right_eye'][1]) / 2
            )
        else:
            eye_mid = monitor.ideal_landmarks['eye_mid']

        if current_landmarks and current_landmarks.get('left_shoulder') and current_landmarks.get('right_shoulder'):
            shoulder_mid = (
                (current_landmarks['left_shoulder'][0] + current_landmarks['right_shoulder'][0]) / 2,
                (current_landmarks['left_shoulder'][1] + current_landmarks['right_shoulder'][1]) / 2
            )
        else:
            shoulder_mid = monitor.ideal_landmarks['shoulder_mid']

        if current_landmarks and current_landmarks.get('left_hip') and current_landmarks.get('right_hip'):
            hip_mid = (
                (current_landmarks['left_hip'][0] + current_landmarks['right_hip'][0]) / 2,
                (current_landmarks['left_hip'][1] + current_landmarks['right_hip'][1]) / 2
            )
        else:
            hip_mid = monitor.ideal_landmarks['hip_mid']
        
        eye_px = (int(eye_mid[0] * w), int(eye_mid[1] * h))
        shoulder_px = (int(shoulder_mid[0] * w), int(shoulder_mid[1] * h))
        hip_px = (int(hip_mid[0] * w), int(hip_mid[1] * h))
        
        overlay = image.copy()
        color = (0, 255, 0)
        
        cv2.line(overlay, eye_px, shoulder_px, color, 3)
        cv2.line(overlay, shoulder_px, hip_px, color, 3)
        cv2.circle(overlay, eye_px, 8, color, -1)
        cv2.circle(overlay, shoulder_px, 10, color, -1)
        cv2.circle(overlay, hip_px, 10, color, -1)
        
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
        
    except:
        pass

def draw_selected_landmarks(image, landmarks, w, h):
    """Draw only the keypoints used by this app."""
    if landmarks is None:
        return
    used = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EYE,
        mp_pose.PoseLandmark.RIGHT_EYE,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
    ]
    pts = {}
    for lm in used:
        p = landmarks[lm.value]
        pts[lm] = (int(p.x * w), int(p.y * h))
        cv2.circle(image, pts[lm], 3, (80, 110, 255), -1)

    # Connect shoulders and hips, and midline
    if mp_pose.PoseLandmark.LEFT_SHOULDER in pts and mp_pose.PoseLandmark.RIGHT_SHOULDER in pts:
        cv2.line(image, pts[mp_pose.PoseLandmark.LEFT_SHOULDER], pts[mp_pose.PoseLandmark.RIGHT_SHOULDER], (80, 200, 120), 2)
    if mp_pose.PoseLandmark.LEFT_HIP in pts and mp_pose.PoseLandmark.RIGHT_HIP in pts:
        cv2.line(image, pts[mp_pose.PoseLandmark.LEFT_HIP], pts[mp_pose.PoseLandmark.RIGHT_HIP], (80, 200, 120), 2)
    # Midline: eye-mid to shoulder-mid to hip-mid
    if (mp_pose.PoseLandmark.LEFT_EYE in pts and mp_pose.PoseLandmark.RIGHT_EYE in pts and
        mp_pose.PoseLandmark.LEFT_SHOULDER in pts and mp_pose.PoseLandmark.RIGHT_SHOULDER in pts and
        mp_pose.PoseLandmark.LEFT_HIP in pts and mp_pose.PoseLandmark.RIGHT_HIP in pts):
        eye_mid = ((pts[mp_pose.PoseLandmark.LEFT_EYE][0] + pts[mp_pose.PoseLandmark.RIGHT_EYE][0]) // 2,
                   (pts[mp_pose.PoseLandmark.LEFT_EYE][1] + pts[mp_pose.PoseLandmark.RIGHT_EYE][1]) // 2)
        sh_mid = ((pts[mp_pose.PoseLandmark.LEFT_SHOULDER][0] + pts[mp_pose.PoseLandmark.RIGHT_SHOULDER][0]) // 2,
                  (pts[mp_pose.PoseLandmark.LEFT_SHOULDER][1] + pts[mp_pose.PoseLandmark.RIGHT_SHOULDER][1]) // 2)
        hip_mid = ((pts[mp_pose.PoseLandmark.LEFT_HIP][0] + pts[mp_pose.PoseLandmark.RIGHT_HIP][0]) // 2,
                   (pts[mp_pose.PoseLandmark.LEFT_HIP][1] + pts[mp_pose.PoseLandmark.RIGHT_HIP][1]) // 2)
        cv2.line(image, eye_mid, sh_mid, (80, 200, 120), 2)
        cv2.line(image, sh_mid, hip_mid, (80, 200, 120), 2)

# Render the compact dashboard panel (score, state, defect, timers).
def draw_translucent_dashboard(image, monitor, score, defects, state_duration, show_nodes):
    """MODULE C: Compact translucent dashboard - bottom-right corner."""
    h, w = image.shape[:2]

    panel_w = 245
    panel_h = 265
    panel_x = w - panel_w - 10
    panel_y = h - panel_h - 10

    # Translucent background (40% opaque)
    overlay = image.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (160, 160, 160), 1)

    fs   = 0.35   # standard font scale
    fs_t = 0.42   # title
    yo   = panel_y + 16
    xo   = panel_x + 8

    # Title
    cv2.putText(image, "POSTURE MONITOR", (xo, yo), cv2.FONT_HERSHEY_SIMPLEX, fs_t, (255, 255, 255), 1, cv2.LINE_AA)
    yo += 17

    # Calibration status
    if monitor.calibrated:
        cv2.putText(image, "Calibrated", (xo, yo), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 230, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(image, "Press c to calibrate", (xo, yo), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 180, 0), 1, cv2.LINE_AA)
    yo += 18

    # Nodes toggle status
    nodes_label = "Nodes: ON (n)" if show_nodes else "Nodes: OFF (n)"
    cv2.putText(image, nodes_label, (xo, yo), cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 200, 200), 1, cv2.LINE_AA)
    yo += 16

    if monitor.calibrated:
        state_colors = {
            PostureState.GOOD: (0, 230, 0),
            PostureState.BAD: (50, 50, 255)
        }
        sc = state_colors.get(monitor.current_state, (220, 220, 220))

        # State
        cv2.putText(image, f"State: {monitor.current_state.value}",
                    (xo, yo), cv2.FONT_HERSHEY_SIMPLEX, fs, sc, 1, cv2.LINE_AA)
        yo += 16

        # Overall score (colour-coded)
        if score is not None:
            score_val = float(score)
            score_color = (0, 230, 0) if score_val >= 82 else ((0, 200, 255) if score_val >= 65 else (50, 50, 255))
            cv2.putText(image, f"Score: {score_val:.1f}/100",
                        (xo, yo), cv2.FONT_HERSHEY_SIMPLEX, fs, score_color, 1, cv2.LINE_AA)
        yo += 16

        # Active defect name
        if defects:
            dominant    = max(defects, key=lambda x: x[1])
            defect_name = dominant[0].replace('_', ' ').title()
            cv2.putText(image, f"! {defect_name}", (xo, yo),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 180, 60), 1, cv2.LINE_AA)
        yo += 14
    else:
        cv2.putText(image, "Metrics hidden", (xo, yo),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (140, 140, 140), 1, cv2.LINE_AA)
        yo += 20

    # Divider
    cv2.line(image, (xo, yo - 3), (panel_x + panel_w - 8, yo - 3), (100, 100, 100), 1)

    if monitor.calibrated:
        # Build severity lookup from live defects
        sev_map: Dict[str, float] = {str(k): float(s) for k, s in defects} if defects else {}

        # Per-defect rows: elapsed time + live severity
        rows = [
            ('hunching',     'Hunch   '),
            ('lateral_lean', 'LatLean '),
            ('hip_sliding',  'HipSlide'),
        ]
        row_y = yo + 13
        for key, label in rows:
            sec = int(monitor.defect_time.get(key, 0.0))
            sev = sev_map.get(key)
            if sev is not None:
                txt   = f"{label}: {sec}s  sev:{sev:.1f}"
                color = (50, 200, 255)   # cyan when active
            else:
                txt   = f"{label}: {sec}s"
                color = (170, 170, 170)  # grey when inactive
            cv2.putText(image, txt, (xo, row_y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)
            row_y += 16
        yo = row_y
    else:
        yo += 12

    # Next microbreak countdown
    if monitor.calibrated:
        time_to_break = int(monitor.microbreak_interval - (time.time() - monitor.last_microbreak))
        if time_to_break > 0:
            cv2.putText(image, f"Break in {time_to_break}s", (xo, yo + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (100, 200, 255), 1, cv2.LINE_AA)


def draw_microbreak_screen(image, monitor):
    h, w = image.shape[:2]
    
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 60, 20), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    
    panel_w = 750
    panel_h = 450
    panel_x = (w - panel_w) // 2
    panel_y = (h - panel_h) // 2
    
    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (50, 100, 50), -1)
    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (100, 255, 100), 5)
    
    y_pos = panel_y + 70
    
    cv2.putText(image, "MICROBREAK TIME", (panel_x + 200, y_pos),
               cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    y_pos += 70
    
    elapsed = int(time.time() - monitor.microbreak_start)
    cv2.putText(image, f"Time: {elapsed}s / 60s", (panel_x + 280, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2, cv2.LINE_AA)
    y_pos += 60
    
    # Cycle exercises every 20 seconds
    exercise_index = (elapsed // 20) % len(monitor.microbreak_exercises)
    current_exercise = monitor.microbreak_exercises[exercise_index]
    
    cv2.putText(image, "Current Exercise:", (panel_x + 60, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    y_pos += 60
    
    # Word wrap
    words = current_exercise.split()
    line = ""
    for word in words:
        test_line = line + word + " "
        if len(test_line) > 50:
            cv2.putText(image, line, (panel_x + 80, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            y_pos += 40
            line = word + " "
        else:
            line = test_line
    if line:
        cv2.putText(image, line, (panel_x + 80, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    y_pos += 70
    cv2.putText(image, "Press 'b' to end break early", (panel_x + 220, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 230, 255), 1, cv2.LINE_AA)

# Show saved bad-posture screenshots in separate windows.
def display_captured_screenshots(monitor):
    if not monitor.bad_posture_events:
        print("No bad posture events captured.")
        return
    
    print(f"\nDisplaying {len(monitor.bad_posture_events)} captured events...")
    
    for idx, event in enumerate(monitor.bad_posture_events):
        frame = event['frame']
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f"Bad Posture Event #{idx + 1}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {event['timestamp']}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Type: {event['defect_type']} | Severity: {event['severity']:.1f}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        total_defect_time = int(monitor.defect_time.get(event['defect_type'], 0))
        cv2.putText(frame, f"Total Time in Posture: {total_defect_time}s", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1)
        
        cv2.imshow(f"Event {idx + 1}/{len(monitor.bad_posture_events)}", frame)
    
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# App entrypoint: webcam loop, pose detection, scoring, UI overlays,
# microbreak handling, and final report generation.
def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    monitor: PostureMonitor = PostureMonitor()
    
    print("\n" + "=" * 80)
    print("ADVANCED BIOMECHANICAL POSTURE ANALYSIS SYSTEM")
    print("=" * 80)
    print("\nFEATURES:")
    print("  - Vector-based biomechanical analysis")
    print("  - Episode counting (not frame counting)")
    print("  - Translucent dashboard (60% alpha blending)")
    print("  - Microbreak protocol (every 60 seconds)")
    print("  - Medical reporting with episode counts")
    print("\nCONTROLS:")
    print("  'c' - Calibrate baseline posture")
    print("  'b' - End microbreak early")
    print("  'q' - Quit and generate report")
    print("\nCALIBRATE FIRST: Sit in your BEST posture and press 'c'")
    print("=" * 80 + "\n")
    
    bad_event_start = None
    current_defects = []
    show_nodes = True
    last_bad_sound_time = 0.0
    BAD_SOUND_COOLDOWN = 3.0
    # Cache latest landmarks for keyboard calibration
    latest_landmarks = None

    # ── Display cache: refresh dashboard every 2 seconds ──────────────────────
    DISPLAY_REFRESH_S: float = 2.5
    last_display_update: float = 0.0
    disp_score: Optional[float]  = None
    disp_defects: list           = []
    disp_state_dur: float        = 0.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        h, w, c = image.shape

        
        # Microbreak check
        current_time = time.time()
        if monitor.calibrated and not monitor.in_microbreak and (current_time - monitor.last_microbreak >= monitor.microbreak_interval):  # type: ignore[attr-defined]
            monitor.in_microbreak = True  # type: ignore[attr-defined]
            monitor.microbreak_start = current_time  # type: ignore[attr-defined]
            print("[MICROBREAK] Started!")
        
        landmark_data = None

        if monitor.in_microbreak:  # type: ignore[attr-defined]
            draw_microbreak_screen(image, monitor)
            
            elapsed_break = current_time - (monitor.microbreak_start or current_time)  # type: ignore[attr-defined]
            if elapsed_break >= 60:
                monitor.in_microbreak = False  # type: ignore[attr-defined]
                monitor.last_microbreak = current_time  # type: ignore[attr-defined]
                print("[MICROBREAK] Completed!")
        else:
            if results.pose_landmarks:
                if show_nodes:
                    draw_selected_landmarks(image, results.pose_landmarks.landmark, w, h)

                latest_landmarks = results.pose_landmarks.landmark
                landmark_data = monitor.extract_landmarks(latest_landmarks)  # type: ignore[attr-defined]

                if show_nodes:
                    draw_ideal_skeleton_overlay(image, monitor, w, h, landmark_data)
                _score, defects = monitor.analyze_posture_biomechanics(landmark_data)  # type: ignore[attr-defined]
                state, state_duration = monitor.update_state_machine(None, defects)  # type: ignore[attr-defined]

                # Debug line removed — sh_z_away / sh_y_change / hip_severity are
                # local to analyze_posture_biomechanics and not accessible here.
                
                # ── Bad-posture event tracking ────────────────────────────
                if state == PostureState.BAD:
                    if bad_event_start is None:
                        bad_event_start = time.time()
                        current_defects = defects

                    _t: float = time.time()
                    bad_event_start_f: float = bad_event_start if bad_event_start is not None else _t  # type: ignore[assignment]
                    duration = int(time.time() - bad_event_start_f)

                    # Continuously update/capture so the stored frame
                    # always reflects the longest running episode.
                    if duration >= 3:
                        monitor.capture_bad_posture_event(image, defects, duration)

                    if duration >= 5:
                        alert_w = 450
                        alert_h = 90
                        alert_x = (w - alert_w) // 2
                        alert_y = 60

                        cv2.rectangle(image, (alert_x, alert_y),
                                      (alert_x + alert_w, alert_y + alert_h), (0, 0, 200), -1)
                        cv2.rectangle(image, (alert_x, alert_y),
                                      (alert_x + alert_w, alert_y + alert_h), (255, 255, 255), 4)
                        cv2.putText(image, "CORRECT YOUR POSTURE!",
                                    (alert_x + 60, alert_y + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(image, f"Bad posture: {duration}s",
                                    (alert_x + 140, alert_y + 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 1, cv2.LINE_AA)
                        # Soft notification sound (debounced)
                        now = time.time()
                        if winsound and (now - last_bad_sound_time) >= BAD_SOUND_COOLDOWN:
                            try:
                                winsound.Beep(1000, 150)  # type: ignore[union-attr]
                            except Exception:
                                pass
                            last_bad_sound_time = now
                else:
                    bad_event_start = None
                
                # \u2500\u2500 Refresh display cache every 2 seconds \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
                now_disp = time.time()
                if now_disp - last_display_update >= DISPLAY_REFRESH_S:
                    disp_score     = None
                    disp_defects   = defects
                    disp_state_dur = state_duration
                    last_display_update = now_disp

                draw_translucent_dashboard(image, monitor, disp_score, disp_defects, disp_state_dur, show_nodes)
            else:
                cv2.putText(image, "Position yourself in frame", (50, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                draw_translucent_dashboard(image, monitor, None, [], 0, show_nodes)
        
        cv2.imshow('Biomechanical Posture Analysis', image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Try current frame landmarks first, fallback to last known
            current_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
            use_landmarks = current_landmarks or latest_landmarks
            if use_landmarks:
                if monitor.calibrate_baseline(use_landmarks):  # type: ignore[attr-defined]
                    print("[SUCCESS] Baseline calibrated!")
            else:
                print("[ERROR] No pose detected")
        elif key == ord('b'):
            if monitor.in_microbreak:  # type: ignore[attr-defined]
                monitor.in_microbreak = False  # type: ignore[attr-defined]
                monitor.last_microbreak = time.time()  # type: ignore[attr-defined]
                print("[MICROBREAK] Ended early")
        elif key == ord('n'):
            show_nodes = not show_nodes
            print(f"[NODES] {'ON' if show_nodes else 'OFF'}")
    
    # Generate report
    print("\n" + "=" * 80)
    print("GENERATING MEDICAL REPORT...")
    print("=" * 80 + "\n")
    
    report = monitor.generate_medical_report()  # type: ignore[attr-defined]
    print(report)
    
    filename = f"posture_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved: {filename}")
    
    display_captured_screenshots(monitor)
    
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == '__main__':
    main()