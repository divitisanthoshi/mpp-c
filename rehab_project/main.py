"""
Real-Time Rehabilitation Exercise Monitor
==========================================
Main application — runs webcam inference with:
  • MediaPipe BlazePose skeleton overlay
  • ST-GCN++ quality scoring (model + DTW fallback)
  • Peak-detection repetition counting
  • Joint error detection with colour highlights
  • Exercise selection menu
  • Progress bar & session summary

Usage:
    python main.py
"""

import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import json
import time

# Fix Python path
sys.path.insert(0, os.path.dirname(__file__))

from src.utils.preprocessing import (
    landmarks_to_array, normalize_skeleton,
    get_exercise_angles, detect_errors,
    FrameBuffer, ScoreSmoother,
    JOINTS, SEQUENCE_LENGTH, INPUT_DIM,
)
from src.utils.repetition_counter import RepetitionCounter
from src.models.st_gcn import dtw_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ── Config ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/rehab_model.keras"
META_PATH  = "models/meta.json"
REF_DIR    = "data/reference"

EXERCISES = [
    ("squat",     "Squat",           "1"),
    ("deep_squat","Deep Squat",      "2"),
    ("lunge",     "Lunge",           "3"),
    ("arm_raise", "Arm Raise",       "4"),
    ("bird_dog",  "Bird Dog",        "5"),
    ("bridge",    "Hip Bridge",      "6"),
]
TARGET_REPS = 10

# Colours (BGR)
CLR_GREEN  = (50,  220,  50)
CLR_YELLOW = (50,  220, 220)
CLR_RED    = (50,   50, 220)
CLR_WHITE  = (240, 240, 240)
CLR_DARK   = (20,   20,  20)
CLR_PANEL  = (30,   30,  30)
CLR_ACCENT = (220, 150,  50)


# ── Model loader ───────────────────────────────────────────────────────────────
def load_model_safe():
    try:
        import tensorflow as tf
        if os.path.isfile(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print(f"[✓] Loaded model from {MODEL_PATH}")
            return model
    except Exception as e:
        print(f"[!] Model load failed: {e}")
    print("[!] Running in DTW-only mode")
    return None


def load_exercises_meta():
    if os.path.isfile(META_PATH):
        with open(META_PATH) as f:
            return json.load(f).get("exercises", [e[0] for e in EXERCISES])
    return [e[0] for e in EXERCISES]


def load_reference(exercise: str) -> np.ndarray | None:
    path = os.path.join(REF_DIR, f"{exercise}_reference.npy")
    if os.path.isfile(path):
        ref = np.load(path)
        if ref.ndim == 3:
            ref = ref.reshape(ref.shape[0], INPUT_DIM)
        return ref
    return None


# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_panel(frame, x, y, w, h, alpha=0.6):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), CLR_PANEL, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def put_text(frame, text, pos, scale=0.6, color=CLR_WHITE, thick=1, bold=False):
    if bold:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, CLR_DARK, thick + 3)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, thick)


def draw_score_bar(frame, x, y, w, h, score, label="Score"):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill = int(score / 100 * w)
    color = CLR_GREEN if score >= 80 else CLR_YELLOW if score >= 55 else CLR_RED
    cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), CLR_WHITE, 1)
    put_text(frame, f"{label}: {score:.0f}%", (x + 5, y + h - 4), 0.45, CLR_WHITE)


def draw_progress_bar(frame, x, y, w, h, reps, target):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill = int(min(reps / max(target, 1), 1.0) * w)
    cv2.rectangle(frame, (x, y), (x + fill, y + h), CLR_ACCENT, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), CLR_WHITE, 1)
    put_text(frame, f"Reps: {reps}/{target}", (x + 5, y + h - 4), 0.45, CLR_WHITE)


def draw_skeleton_errors(frame, landmarks, errors, frame_shape):
    """Highlight error joints in red."""
    ERROR_JOINTS_MAP = {
        "knee": ["left_knee", "right_knee"],
        "back": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        "arm":  ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"],
        "hip":  ["left_hip", "right_hip"],
        "torso":["left_shoulder", "left_hip"],
    }
    error_joints = set()
    for err in errors:
        for key, jnames in ERROR_JOINTS_MAP.items():
            if key in err.lower():
                error_joints.update(jnames)

    h, w = frame_shape[:2]
    if landmarks:
        for jname in error_joints:
            idx = JOINTS.get(jname)
            if idx is None:
                continue
            lm = landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 12, CLR_RED, -1)
            cv2.circle(frame, (cx, cy), 14, (255, 255, 255), 2)


# ── Session summary screen ─────────────────────────────────────────────────────
def show_summary(frame, exercise_name, total_reps, avg_score):
    overlay = frame.copy()
    cv2.rectangle(overlay, (80, 80), (frame.shape[1] - 80, frame.shape[0] - 80),
                  (20, 20, 40), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cx = frame.shape[1] // 2
    put_text(frame, "SESSION COMPLETE", (cx - 180, 150), 1.1, CLR_ACCENT, 2, bold=True)
    put_text(frame, f"Exercise : {exercise_name}", (cx - 180, 230), 0.75, CLR_WHITE)
    put_text(frame, f"Total Reps: {total_reps}", (cx - 180, 275), 0.75, CLR_GREEN)
    avg_col = CLR_GREEN if avg_score >= 80 else CLR_YELLOW if avg_score >= 55 else CLR_RED
    put_text(frame, f"Avg Score : {avg_score:.0f}%", (cx - 180, 320), 0.75, avg_col)
    put_text(frame, "Press R to restart  |  Q to quit",
             (cx - 200, frame.shape[0] - 110), 0.65, CLR_WHITE)
    return frame


# ── Main application ───────────────────────────────────────────────────────────
def run():
    model        = load_model_safe()
    ex_meta      = load_exercises_meta()
    mp_pose      = mp.solutions.pose
    mp_drawing   = mp.solutions.drawing_utils
    mp_styles    = mp.solutions.drawing_styles

    # State
    current_ex_idx = 0
    exercise_id, exercise_label, _ = EXERCISES[current_ex_idx]

    buf         = FrameBuffer(SEQUENCE_LENGTH)
    smoother    = ScoreSmoother(window=12)
    rep_counter = RepetitionCounter(exercise_id)
    ref_seq     = load_reference(exercise_id)

    score        = 0.0
    errors       = []
    show_menu    = True
    session_done = False
    score_history = []

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    FW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with mp_pose.Pose(
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
        model_complexity=1,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # ── Menu overlay ────────────────────────────────────────────────
            if show_menu:
                draw_panel(frame, 0, 0, FW, FH, alpha=0.7)
                put_text(frame, "REHABILITATION EXERCISE MONITOR",
                         (FW // 2 - 330, 80), 1.0, CLR_ACCENT, 2, bold=True)
                put_text(frame, "Select Exercise:", (FW // 2 - 150, 140), 0.75, CLR_WHITE)
                for i, (eid, elabel, key) in enumerate(EXERCISES):
                    marker = ">>" if i == current_ex_idx else "  "
                    col = CLR_ACCENT if i == current_ex_idx else CLR_WHITE
                    put_text(frame, f"  [{key}] {marker} {elabel}",
                             (FW // 2 - 160, 190 + i * 45), 0.72, col)
                put_text(frame, "Press number key to select. ENTER to start.",
                         (FW // 2 - 280, FH - 60), 0.65, CLR_WHITE)
                cv2.imshow("Rehabilitation Monitor", frame)
                key = cv2.waitKey(1) & 0xFF
                for i, (eid, elabel, kchar) in enumerate(EXERCISES):
                    if key == ord(kchar):
                        if current_ex_idx == i:
                            show_menu = False
                            exercise_id, exercise_label, _ = EXERCISES[i]
                            buf.reset(); smoother.reset()
                            rep_counter.set_exercise(exercise_id)
                            ref_seq = load_reference(exercise_id)
                            score = 0.0; errors = []; score_history = []; session_done = False
                        else:
                            current_ex_idx = i
                            exercise_id, exercise_label, _ = EXERCISES[i]
                if key == 13:   # Enter
                    show_menu = False
                    buf.reset()
                    smoother.reset()
                    rep_counter.set_exercise(exercise_id)
                    ref_seq = load_reference(exercise_id)
                    score = 0.0
                    errors = []
                    score_history = []
                    session_done = False
                elif key == ord('q'):
                    break
                continue

            # ── Session summary ──────────────────────────────────────────────
            if session_done:
                avg = float(np.mean(score_history)) if score_history else 0.0
                frame = show_summary(frame, exercise_label,
                                     rep_counter._reps, avg)
                cv2.imshow("Rehabilitation Monitor", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    session_done = False
                    buf.reset()
                    smoother.reset()
                    rep_counter.reset()
                    score_history = []
                    score = 0.0
                    errors = []
                elif key == ord('q'):
                    break
                elif key == ord('m'):
                    show_menu = True
                continue

            # ── Pose estimation ──────────────────────────────────────────────
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

                raw_arr = landmarks_to_array(result.pose_landmarks.landmark)
                norm    = normalize_skeleton(raw_arr)

                # Buffer & repetition
                buf.add(raw_arr)
                reps = rep_counter.update(norm)

                # Angle-based errors
                angles = get_exercise_angles(norm, exercise_id)
                errors = detect_errors(angles, exercise_id)

                # Draw red circles on bad joints
                draw_skeleton_errors(frame, result.pose_landmarks.landmark,
                                     errors, frame.shape)

                # Score calculation
                if buf.ready():
                    seq = buf.get_sequence()   # (1, T, 99)

                    if model is not None:
                        try:
                            preds = model.predict(seq, verbose=0)
                            raw_score = float(preds[0][0][0]) * 100
                        except Exception:
                            raw_score = 50.0
                    else:
                        raw_score = 50.0

                    # Blend with DTW if reference exists
                    if ref_seq is not None:
                        dtw = dtw_score(seq[0], ref_seq)
                        raw_score = 0.5 * raw_score + 0.5 * dtw

                    # Apply error penalty
                    penalty = len(errors) * 5
                    raw_score = max(0, raw_score - penalty)
                    score = smoother.update(raw_score)
                    score_history.append(score)

                # Session done?
                if reps >= TARGET_REPS:
                    session_done = True
                    continue

            # ── UI panels ────────────────────────────────────────────────────

            # Top panel
            draw_panel(frame, 0, 0, FW, 55)
            put_text(frame, f"Exercise: {exercise_label}", (15, 37),
                     0.8, CLR_ACCENT, 2, bold=True)
            phase = rep_counter.get_phase().upper()
            put_text(frame, f"Phase: {phase}", (FW - 200, 37), 0.65, CLR_WHITE)

            # Right panel
            PW, PH = 260, FH
            px = FW - PW
            draw_panel(frame, px, 55, PW, PH - 55)

            score_color = CLR_GREEN if score >= 80 else CLR_YELLOW if score >= 55 else CLR_RED
            put_text(frame, "QUALITY SCORE", (px + 10, 90), 0.6, CLR_WHITE, bold=True)
            put_text(frame, f"{score:.0f}%", (px + 60, 150), 1.6, score_color, 3, bold=True)
            draw_score_bar(frame, px + 10, 160, PW - 20, 20, score)

            put_text(frame, "REPETITIONS", (px + 10, 210), 0.6, CLR_WHITE, bold=True)
            put_text(frame, str(rep_counter._reps), (px + 90, 265), 1.6, CLR_ACCENT, 3, bold=True)
            draw_progress_bar(frame, px + 10, 275, PW - 20, 20,
                              rep_counter._reps, TARGET_REPS)

            put_text(frame, "FEEDBACK", (px + 10, 320), 0.6, CLR_WHITE, bold=True)
            if errors:
                for i, err in enumerate(errors[:4]):
                    put_text(frame, f"• {err}", (px + 10, 348 + i * 28),
                             0.48, CLR_RED)
            else:
                put_text(frame, "✓ Good form!", (px + 10, 348), 0.55, CLR_GREEN)

            # Bottom controls
            draw_panel(frame, 0, FH - 40, FW, 40)
            put_text(frame, "M=Menu   Q=Quit   R=Reset",
                     (15, FH - 12), 0.55, CLR_WHITE)
            put_text(frame, "RED circles = incorrect joints",
                     (FW - 320, FH - 12), 0.55, CLR_RED)

            cv2.imshow("Rehabilitation Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                show_menu = True
            elif key == ord('r'):
                buf.reset()
                smoother.reset()
                rep_counter.reset()
                score_history = []
                score = 0.0
                errors = []

    # Final summary to terminal
    if score_history:
        avg = float(np.mean(score_history))
        print(f"\n{'='*50}")
        print(f"  Exercise : {exercise_label}")
        print(f"  Reps     : {rep_counter._reps}")
        print(f"  Avg Score: {avg:.1f}%")
        print(f"{'='*50}\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
