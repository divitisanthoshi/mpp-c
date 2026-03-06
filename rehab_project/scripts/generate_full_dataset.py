"""
Full Rehabilitation Dataset Generator
======================================
Generates realistic synthetic skeleton sequences for all 23 exercises.
Each exercise has:
  - good form  (score 82–98)
  - medium form (score 52–78)
  - bad form   (score 20–48)
  × 4 augments = 150 total sequences per exercise

Also writes dataset_info.json describing all sources used.

Usage:
    python scripts/generate_full_dataset.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────
T  = 60    # frames per sequence
N  = 33    # MediaPipe joints
C  = 3     # x, y, z
FLAT = N * C   # 99

SAMPLES_PER_QUALITY = 10   # × 3 qualities × 5 versions = 150 per exercise

ALL_EXERCISES = [
    "squat",
    "deep_squat",
    "hurdle_step",
    "inline_lunge",
    "side_lunge",
    "sit_to_stand",
    "standing_leg_raise",
    "shoulder_abduction",
    "shoulder_extension",
    "shoulder_rotation",
    "shoulder_scaption",
    "hip_abduction",
    "trunk_rotation",
    "leg_raise",
    "reach_and_retrieve",
    "wall_pushup",
    "heel_raise",
    "bird_dog",
    "glute_bridge",
    "clamshell",
    "chin_tuck",
    "marching_in_place",
    "step_up",
]

SCORE_RANGES = {
    "good":   (82, 98),
    "medium": (52, 78),
    "bad":    (20, 48),
}

# ── Base T-pose skeleton ──────────────────────────────────────────────────────
def base_skeleton() -> np.ndarray:
    """Returns (33, 3) neutral standing skeleton in normalised coords."""
    s = np.zeros((N, C), dtype=np.float32)
    # Face
    s[0]  = [ 0.00,  0.10,  0.00]   # nose
    s[1]  = [-0.03,  0.12, -0.02]
    s[2]  = [-0.05,  0.12, -0.03]
    s[3]  = [-0.07,  0.11, -0.02]
    s[4]  = [ 0.03,  0.12, -0.02]
    s[5]  = [ 0.05,  0.12, -0.03]
    s[6]  = [ 0.07,  0.11, -0.02]
    s[7]  = [-0.08,  0.11,  0.00]
    s[8]  = [ 0.08,  0.11,  0.00]
    s[9]  = [-0.02,  0.08, -0.01]
    s[10] = [ 0.02,  0.08, -0.01]
    # Shoulders
    s[11] = [-0.18,  0.00,  0.00]   # left  shoulder
    s[12] = [ 0.18,  0.00,  0.00]   # right shoulder
    # Arms
    s[13] = [-0.30, -0.18,  0.00]   # left  elbow
    s[14] = [ 0.30, -0.18,  0.00]   # right elbow
    s[15] = [-0.38, -0.36,  0.00]   # left  wrist
    s[16] = [ 0.38, -0.36,  0.00]   # right wrist
    s[17] = [-0.40, -0.40,  0.00]
    s[18] = [ 0.40, -0.40,  0.00]
    s[19] = [-0.41, -0.41,  0.00]
    s[20] = [ 0.41, -0.41,  0.00]
    s[21] = [-0.39, -0.39,  0.00]
    s[22] = [ 0.39, -0.39,  0.00]
    # Hips
    s[23] = [-0.10, -0.42,  0.00]   # left  hip
    s[24] = [ 0.10, -0.42,  0.00]   # right hip
    # Legs
    s[25] = [-0.10, -0.72,  0.00]   # left  knee
    s[26] = [ 0.10, -0.72,  0.00]   # right knee
    s[27] = [-0.10, -1.02,  0.00]   # left  ankle
    s[28] = [ 0.10, -1.02,  0.00]   # right ankle
    s[29] = [-0.10, -1.08,  0.00]
    s[30] = [ 0.10, -1.08,  0.00]
    s[31] = [-0.10, -1.10,  0.00]
    s[32] = [ 0.10, -1.10,  0.00]
    return s


def normalize(s: np.ndarray) -> np.ndarray:
    hip = (s[23] + s[24]) / 2
    shoulder = (s[11] + s[12]) / 2
    torso = np.linalg.norm(shoulder - hip) + 1e-6
    return (s - hip) / torso


# ── Motion generators ─────────────────────────────────────────────────────────

def _t():
    return np.linspace(0, 2 * np.pi, T)


def _wave(amp, t=None):
    if t is None:
        t = _t()
    return amp * np.abs(np.sin(t))


def _seq(frames):
    return np.array([normalize(f).flatten() for f in frames], dtype=np.float32)


# --- Lower body exercises ---

def make_squat(quality="good"):
    t = _t()
    depth = {"good": 0.30, "medium": 0.18, "bad": 0.08}[quality]
    noise = {"good": 0.01, "medium": 0.05, "bad": 0.12}[quality]
    spread = {"good": 0.00, "medium": 0.03, "bad": 0.07}[quality]
    dip = depth * np.abs(np.sin(t))
    frames = []
    for i in range(T):
        s = base_skeleton().copy(); d = dip[i]
        s[23] = [-0.10, -0.42 - d, 0]; s[24] = [0.10, -0.42 - d, 0]
        s[25] = [-0.10 - spread, -0.72 - d*0.5, 0]; s[26] = [0.10 + spread, -0.72 - d*0.5, 0]
        s[0:12, 2] += noise * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_deep_squat(quality="good"):
    t = _t()
    depth = {"good": 0.45, "medium": 0.28, "bad": 0.10}[quality]
    spread = {"good": 0.00, "medium": 0.04, "bad": 0.09}[quality]
    dip = depth * np.abs(np.sin(t))
    frames = []
    for i in range(T):
        s = base_skeleton().copy(); d = dip[i]
        s[23] = [-0.10, -0.42 - d, 0]; s[24] = [0.10, -0.42 - d, 0]
        s[25] = [-0.10 - spread, -0.72 - d*0.6, 0]; s[26] = [0.10 + spread, -0.72 - d*0.6, 0]
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_hurdle_step(quality="good"):
    t = _t()
    lift = {"good": 0.30, "medium": 0.18, "bad": 0.08}[quality]
    balance = {"good": 0.01, "medium": 0.05, "bad": 0.12}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy()
        h = lift * np.abs(np.sin(t[i]))
        # Left leg steps over hurdle
        s[25] = [-0.10, -0.72 + h, 0.10 * np.sin(t[i])]
        s[27] = [-0.10, -1.02 + h*0.5, 0.15 * np.sin(t[i])]
        s[0:12, 0] += balance * np.cos(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_inline_lunge(quality="good"):
    t = _t()
    depth = {"good": 0.25, "medium": 0.14, "bad": 0.05}[quality]
    lean  = {"good": 0.01, "medium": 0.07, "bad": 0.15}[quality]
    dip = depth * np.abs(np.sin(t))
    frames = []
    for i in range(T):
        s = base_skeleton().copy(); d = dip[i]
        s[25] = [-0.05, -0.72 - d, 0.08]   # feet inline
        s[27] = [-0.05, -1.02, 0.18]
        s[26] = [0.05,  -0.72 - d*0.5, -0.08]
        s[0:12, 2] += lean * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_side_lunge(quality="good"):
    t = _t()
    spread = {"good": 0.35, "medium": 0.20, "bad": 0.08}[quality]
    depth  = {"good": 0.20, "medium": 0.12, "bad": 0.04}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy()
        sp = spread * np.abs(np.sin(t[i]))
        d  = depth  * np.abs(np.sin(t[i]))
        # Step left
        s[23] = [-0.10 - sp*0.3, -0.42 - d, 0]
        s[25] = [-0.10 - sp,     -0.72 - d, 0]
        s[27] = [-0.10 - sp,     -1.02,     0]
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_sit_to_stand(quality="good"):
    t = _t()
    depth = {"good": 0.38, "medium": 0.22, "bad": 0.08}[quality]
    lean  = {"good": 0.05, "medium": 0.12, "bad": 0.22}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy()
        d = depth * np.abs(np.sin(t[i]))
        # Sit: hips and torso lower
        s[23] = [-0.10, -0.42 - d, 0]; s[24] = [0.10, -0.42 - d, 0]
        s[25] = [-0.10, -0.55, 0.10]; s[26] = [0.10, -0.55, 0.10]
        s[0:12, 2] += lean * (1 - np.abs(np.sin(t[i])))
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_standing_leg_raise(quality="good"):
    t = _t()
    raise_h = {"good": 0.30, "medium": 0.18, "bad": 0.06}[quality]
    balance = {"good": 0.01, "medium": 0.06, "bad": 0.14}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy()
        h = raise_h * np.abs(np.sin(t[i]))
        s[25] = [-0.10, -0.72 + h,  0.0]
        s[27] = [-0.10, -1.02 + h*0.6, -0.05]
        s[0, 0] += balance * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


# --- Upper body exercises ---

def make_shoulder_abduction(quality="good"):
    t = _t()
    max_h = {"good": 0.40, "medium": 0.24, "bad": 0.10}[quality]
    bend  = {"good": 0.00, "medium": 0.05, "bad": 0.13}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy()
        h = max_h * np.abs(np.sin(t[i]))
        s[13] = [-0.30 - h*0.3, -0.18 + h, 0]; s[14] = [0.30 + h*0.3, -0.18 + h, 0]
        s[15] = [-0.38 - h*0.2, -0.36 + h + bend, 0]; s[16] = [0.38 + h*0.2, -0.36 + h + bend, 0]
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_shoulder_extension(quality="good"):
    t = _t()
    ext   = {"good": 0.25, "medium": 0.14, "bad": 0.05}[quality]
    trunk = {"good": 0.01, "medium": 0.06, "bad": 0.14}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy()
        e = ext * np.abs(np.sin(t[i]))
        # Arms extend backward
        s[13] = [-0.30, -0.18 - e, -0.10 - e]; s[14] = [0.30, -0.18 - e, -0.10 - e]
        s[15] = [-0.38, -0.36 - e, -0.20 - e]; s[16] = [0.38, -0.36 - e, -0.20 - e]
        s[0:12, 2] += trunk * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_shoulder_rotation(quality="good"):
    t = _t()
    rot   = {"good": 0.20, "medium": 0.12, "bad": 0.04}[quality]
    comp  = {"good": 0.01, "medium": 0.05, "bad": 0.12}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy()
        r = rot * np.sin(t[i])
        # Elbows at 90° rotate forearm up/down
        s[13] = [-0.30, -0.18, 0]; s[14] = [0.30, -0.18, 0]
        s[15] = [-0.30, -0.18 + r, -0.10 + abs(r)]; s[16] = [0.30, -0.18 + r, -0.10 + abs(r)]
        s[0:12, 0] += comp * np.cos(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_shoulder_scaption(quality="good"):
    t = _t()
    max_h = {"good": 0.35, "medium": 0.20, "bad": 0.08}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy()
        h = max_h * np.abs(np.sin(t[i]))
        # Arms raise at 45° scaption plane
        s[13] = [-0.30 - h*0.2, -0.18 + h, -h*0.2]; s[14] = [0.30 + h*0.2, -0.18 + h, -h*0.2]
        s[15] = [-0.38 - h*0.15, -0.36 + h, -h*0.3]; s[16] = [0.38 + h*0.15, -0.36 + h, -h*0.3]
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


# --- Hip / core exercises ---

def make_hip_abduction(quality="good"):
    t = _t()
    spread = {"good": 0.28, "medium": 0.16, "bad": 0.06}[quality]
    balance= {"good": 0.01, "medium": 0.07, "bad": 0.15}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy()
        sp = spread * np.abs(np.sin(t[i]))
        s[25] = [-0.10 - sp, -0.72, 0]; s[27] = [-0.10 - sp, -1.02, 0]
        s[0, 0] += balance * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_trunk_rotation(quality="good"):
    t = _t()
    rot   = {"good": 0.20, "medium": 0.12, "bad": 0.04}[quality]
    comp  = {"good": 0.00, "medium": 0.05, "bad": 0.12}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy()
        r = rot * np.sin(t[i])
        s[11, 2] += r; s[12, 2] -= r   # shoulders rotate
        s[13, 2] += r; s[14, 2] -= r
        s[23, 2] += comp * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_leg_raise(quality="good"):
    t = _t()
    raise_h = {"good": 0.35, "medium": 0.20, "bad": 0.07}[quality]
    back_arch = {"good": 0.01, "medium": 0.06, "bad": 0.15}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy()
        # Supine leg raise - lying down
        s[:, 1] *= 0.3
        h = raise_h * np.abs(np.sin(t[i]))
        s[25] = [-0.10, -0.72 + h, 0]; s[27] = [-0.10, -1.02 + h, 0]
        s[23, 2] += back_arch * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_reach_and_retrieve(quality="good"):
    t = _t()
    reach = {"good": 0.30, "medium": 0.18, "bad": 0.06}[quality]
    trunk = {"good": 0.05, "medium": 0.12, "bad": 0.22}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy()
        r = reach * np.abs(np.sin(t[i]))
        s[15] = [-0.38 - r, -0.36 - r*0.3, r*0.5]   # reach forward-down
        s[16] = [ 0.38 + r, -0.36 - r*0.3, r*0.5]
        s[0:12, 2] += trunk * np.abs(np.sin(t[i]))
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_wall_pushup(quality="good"):
    t = _t()
    ext  = {"good": 0.22, "medium": 0.13, "bad": 0.04}[quality]
    sag  = {"good": 0.01, "medium": 0.06, "bad": 0.15}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy()
        e = ext * np.abs(np.sin(t[i]))
        # Arms push toward wall (forward)
        s[13] = [-0.25, -0.05, 0.10 + e]; s[14] = [0.25, -0.05, 0.10 + e]
        s[15] = [-0.30, -0.10, 0.25 + e]; s[16] = [0.30, -0.10, 0.25 + e]
        s[23, 2] += sag * np.sin(t[i]); s[24, 2] += sag * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_heel_raise(quality="good"):
    t = _t()
    lift  = {"good": 0.12, "medium": 0.07, "bad": 0.02}[quality]
    sway  = {"good": 0.01, "medium": 0.05, "bad": 0.12}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy()
        h = lift * np.abs(np.sin(t[i]))
        s[27] = [-0.10, -1.02 + h, 0]; s[28] = [0.10, -1.02 + h, 0]
        s[29] = [-0.10, -1.08 + h, 0]; s[30] = [0.10, -1.08 + h, 0]
        s[0, 0] += sway * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_bird_dog(quality="good"):
    t = _t()
    ext = {"good": 0.25, "medium": 0.15, "bad": 0.06}[quality]
    sag = {"good": 0.01, "medium": 0.04, "bad": 0.09}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy(); s[:, 1] -= 0.30   # quadruped
        e = ext * np.abs(np.sin(t[i]))
        s[15] = [-0.45 - e, -0.40 + e*0.3,  0.20 + e]
        s[28] = [ 0.10,     -1.02 + e*0.3, -0.20 - e]
        s[11:13, 1] -= sag * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_glute_bridge(quality="good"):
    t = _t()
    lift  = {"good": 0.28, "medium": 0.16, "bad": 0.06}[quality]
    shift = {"good": 0.00, "medium": 0.04, "bad": 0.10}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy(); s[:, 1] *= -0.5   # supine
        h = lift * np.abs(np.sin(t[i]))
        s[23] = [-0.10, -0.42 + h, 0]; s[24] = [0.10, -0.42 + h, 0]
        s[25] = [-0.10, -0.72 + h*0.3, 0]; s[26] = [0.10, -0.72 + h*0.3, 0]
        s[23, 0] += shift * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_clamshell(quality="good"):
    t = _t()
    open_a = {"good": 0.22, "medium": 0.13, "bad": 0.04}[quality]
    comp   = {"good": 0.01, "medium": 0.05, "bad": 0.12}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy(); s[:, 1] *= 0.5   # side-lying
        a = open_a * np.abs(np.sin(t[i]))
        # Top knee opens
        s[26] = [0.10 + a*0.3, -0.72, a*0.4]
        s[28] = [0.10 + a*0.2, -1.02, a*0.3]
        s[23, 2] += comp * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_chin_tuck(quality="good"):
    t = _t()
    tuck  = {"good": 0.05, "medium": 0.03, "bad": 0.01}[quality]
    proto = {"good": 0.00, "medium": 0.03, "bad": 0.07}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy()
        tk = tuck * np.abs(np.sin(t[i]))
        s[0]  = [ 0.00, 0.10 - tk, -tk]   # nose pulls back
        s[7]  = [-0.08, 0.11 - tk, -tk]
        s[8]  = [ 0.08, 0.11 - tk, -tk]
        s[0, 2] -= proto * np.sin(t[i])   # bad: forward head
        s += np.random.normal(0, 0.003, s.shape)
        frames.append(s)
    return _seq(frames)


def make_marching_in_place(quality="good"):
    t = _t()
    lift  = {"good": 0.22, "medium": 0.13, "bad": 0.05}[quality]
    sway  = {"good": 0.01, "medium": 0.06, "bad": 0.14}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy()
        # Alternate legs
        left_h  = lift * max(0, np.sin(t[i]))
        right_h = lift * max(0, -np.sin(t[i]))
        s[25] = [-0.10, -0.72 + left_h,  0]; s[27] = [-0.10, -1.02 + left_h*0.6, 0]
        s[26] = [ 0.10, -0.72 + right_h, 0]; s[28] = [ 0.10, -1.02 + right_h*0.6, 0]
        s[0, 0] += sway * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


def make_step_up(quality="good"):
    t = _t()
    height= {"good": 0.20, "medium": 0.12, "bad": 0.04}[quality]
    lean  = {"good": 0.02, "medium": 0.08, "bad": 0.18}[quality]
    frames = []
    for i in range(T):
        s = base_skeleton().copy()
        h = height * np.abs(np.sin(t[i]))
        # Step up with left leg
        s[25] = [-0.10, -0.72 + h, 0]; s[27] = [-0.10, -1.02 + h, 0]
        s[0:12, 1] += h * 0.3
        s[0:12, 2] += lean * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _seq(frames)


# ── Generator dispatch ────────────────────────────────────────────────────────

GENERATORS = {
    "squat":               make_squat,
    "deep_squat":          make_deep_squat,
    "hurdle_step":         make_hurdle_step,
    "inline_lunge":        make_inline_lunge,
    "side_lunge":          make_side_lunge,
    "sit_to_stand":        make_sit_to_stand,
    "standing_leg_raise":  make_standing_leg_raise,
    "shoulder_abduction":  make_shoulder_abduction,
    "shoulder_extension":  make_shoulder_extension,
    "shoulder_rotation":   make_shoulder_rotation,
    "shoulder_scaption":   make_shoulder_scaption,
    "hip_abduction":       make_hip_abduction,
    "trunk_rotation":      make_trunk_rotation,
    "leg_raise":           make_leg_raise,
    "reach_and_retrieve":  make_reach_and_retrieve,
    "wall_pushup":         make_wall_pushup,
    "heel_raise":          make_heel_raise,
    "bird_dog":            make_bird_dog,
    "glute_bridge":        make_glute_bridge,
    "clamshell":           make_clamshell,
    "chin_tuck":           make_chin_tuck,
    "marching_in_place":   make_marching_in_place,
    "step_up":             make_step_up,
}


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment(seq: np.ndarray) -> np.ndarray:
    aug = seq.copy()
    aug += np.random.normal(0, 0.008, aug.shape)
    aug *= np.random.uniform(0.93, 1.07)
    if np.random.rand() < 0.5:
        r = aug.reshape(T, N, C)
        r[:, :, 0] *= -1      # mirror left-right
        aug = r.reshape(T, FLAT)
    return aug.astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_exercise(exercise: str, out_dir: str, n_per_quality: int = SAMPLES_PER_QUALITY):
    os.makedirs(out_dir, exist_ok=True)
    gen = GENERATORS[exercise]
    records = []
    seq_idx = 0
    aug_factor = 4

    for quality in ["good", "medium", "bad"]:
        lo, hi = SCORE_RANGES[quality]
        for _ in range(n_per_quality):
            base = gen(quality)
            versions = [base] + [augment(base) for _ in range(aug_factor)]
            for seq in versions:
                fname = f"seq_{seq_idx:04d}.npy"
                np.save(os.path.join(out_dir, fname), seq)
                score = np.random.randint(lo, hi + 1)
                reps  = np.random.randint(4, 9)
                records.append({"sequence": fname, "score": score, "reps": reps,
                                "quality": quality})
                seq_idx += 1

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, "labels.csv"), index=False)
    return seq_idx


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "custom")
    print("\n" + "="*65)
    print("  Generating Full Rehabilitation Dataset — 23 Exercises")
    print("="*65 + "\n")

    total = 0
    for ex in ALL_EXERCISES:
        out_dir = os.path.join(base_dir, ex)
        n = generate_exercise(ex, out_dir)
        total += n
        bar = "█" * int(n / 10)
        print(f"  {'✅'} {ex:<28s}  sequences={n:4d}  {bar}")

    print(f"\n  Total sequences: {total}")
    print(f"  Exercises:       {len(ALL_EXERCISES)}")
    print(f"\n  Done ✓\n")


if __name__ == "__main__":
    main()
