"""
build_dataset.py — Full Multi-Source Rehabilitation Dataset Builder
====================================================================

Generates / merges data from four sources:

  ┌──────────────┬────────────────────────────────────────────────────┐
  │ Source       │ Notes                                              │
  ├──────────────┼────────────────────────────────────────────────────┤
  │ Custom       │ Webcam (MediaPipe BlazePose) — 33 joints, xyz     │
  │ NTU RGB+D    │ Kinect skeleton — 25 joints → mapped to MP 33     │
  │ UI-PRMD      │ Vicon mocap — 20 joints → mapped to MP 33         │
  │ KIMORE       │ OpenPose — 18 joints → mapped to MP 33            │
  └──────────────┴────────────────────────────────────────────────────┘

If real dataset files are present in data/ntu/, data/uiprmd/, data/kimore/
they are parsed and converted.  Otherwise, high-quality biomechanically
realistic synthetic stand-ins are generated for every source, each with
the noise characteristics of that capture modality.

Output:   data/unified/<exercise>/seq_XXXXX.npy  +  labels.csv
          data/dataset_info.json

Usage:
    python scripts/build_dataset.py [--samples N]
"""

import os, sys, glob, json, argparse
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
T_LEN = 60      # frames
N_JNT = 33      # MediaPipe joints
N_CRD = 3       # x, y, z
FLAT  = N_JNT * N_CRD   # 99

ALL_EXERCISES = [
    "squat", "deep_squat", "hurdle_step", "inline_lunge", "side_lunge",
    "sit_to_stand", "standing_leg_raise", "shoulder_abduction",
    "shoulder_extension", "shoulder_rotation", "shoulder_scaption",
    "hip_abduction", "trunk_rotation", "leg_raise", "reach_and_retrieve",
    "wall_pushup", "heel_raise", "bird_dog", "glute_bridge", "clamshell",
    "chin_tuck", "marching_in_place", "step_up",
]

SCORE_RANGES = {
    "good":   (82, 98),
    "medium": (52, 78),
    "bad":    (20, 48),
}

# Which real-dataset exercises map to which exercise IDs
NTU_MAP = {
    "squat":               "A007",
    "sit_to_stand":        "A002",
    "standing_leg_raise":  "A023",
    "shoulder_abduction":  "A030",
    "trunk_rotation":      "A015",
    "marching_in_place":   "A012",
    "step_up":             "A049",
    "wall_pushup":         "A057",
}

UIPRMD_MAP = {
    "deep_squat":          "P001",
    "hurdle_step":         "P002",
    "inline_lunge":        "P003",
    "side_lunge":          "P004",
    "sit_to_stand":        "P005",
    "standing_leg_raise":  "P006",
    "shoulder_abduction":  "P007",
    "shoulder_extension":  "P008",
    "shoulder_scaption":   "P009",
    "trunk_rotation":      "P010",
}

KIMORE_MAP = {
    "squat":               "E1",
    "side_lunge":          "E2",
    "trunk_rotation":      "E3",
    "shoulder_abduction":  "E4",
    "reach_and_retrieve":  "E5",
}

# ─────────────────────────────────────────────────────────────────────────────
# Base skeleton (MediaPipe 33-joint T-pose)
# ─────────────────────────────────────────────────────────────────────────────
def base_skel():
    s = np.zeros((N_JNT, N_CRD), dtype=np.float32)
    s[0]  = [ 0.00,  0.10,  0.00]   # nose
    s[1]  = [-0.03,  0.12, -0.02]; s[2]  = [-0.05,  0.12, -0.03]
    s[3]  = [-0.07,  0.11, -0.02]; s[4]  = [ 0.03,  0.12, -0.02]
    s[5]  = [ 0.05,  0.12, -0.03]; s[6]  = [ 0.07,  0.11, -0.02]
    s[7]  = [-0.08,  0.11,  0.00]; s[8]  = [ 0.08,  0.11,  0.00]
    s[9]  = [-0.02,  0.08, -0.01]; s[10] = [ 0.02,  0.08, -0.01]
    s[11] = [-0.18,  0.00,  0.00]   # L shoulder
    s[12] = [ 0.18,  0.00,  0.00]   # R shoulder
    s[13] = [-0.30, -0.18,  0.00]   # L elbow
    s[14] = [ 0.30, -0.18,  0.00]   # R elbow
    s[15] = [-0.38, -0.36,  0.00]   # L wrist
    s[16] = [ 0.38, -0.36,  0.00]   # R wrist
    s[17] = [-0.40, -0.40,  0.00]; s[18] = [ 0.40, -0.40,  0.00]
    s[19] = [-0.41, -0.41,  0.00]; s[20] = [ 0.41, -0.41,  0.00]
    s[21] = [-0.39, -0.39,  0.00]; s[22] = [ 0.39, -0.39,  0.00]
    s[23] = [-0.10, -0.42,  0.00]   # L hip
    s[24] = [ 0.10, -0.42,  0.00]   # R hip
    s[25] = [-0.10, -0.72,  0.00]   # L knee
    s[26] = [ 0.10, -0.72,  0.00]   # R knee
    s[27] = [-0.10, -1.02,  0.00]   # L ankle
    s[28] = [ 0.10, -1.02,  0.00]   # R ankle
    s[29] = [-0.10, -1.08,  0.00]; s[30] = [ 0.10, -1.08,  0.00]
    s[31] = [-0.10, -1.10,  0.00]; s[32] = [ 0.10, -1.10,  0.00]
    return s

def normalize(s):
    hip   = (s[23] + s[24]) / 2
    torso = np.linalg.norm((s[11] + s[12]) / 2 - hip) + 1e-6
    return (s - hip) / torso

def norm_seq(seq):
    """seq: (T, FLAT)  → normalise every frame"""
    out = seq.reshape(T_LEN, N_JNT, N_CRD).copy()
    for i, f in enumerate(out):
        hip   = (f[23] + f[24]) / 2
        torso = np.linalg.norm((f[11] + f[12]) / 2 - hip) + 1e-6
        out[i] = (f - hip) / torso
    return out.reshape(T_LEN, FLAT)

def pad_trim(seq):
    """Ensure (T_LEN, FLAT)"""
    if seq.ndim == 3:
        seq = seq.reshape(seq.shape[0], FLAT)
    n = seq.shape[0]
    if n < T_LEN:
        seq = np.vstack([seq, np.zeros((T_LEN - n, FLAT), dtype=np.float32)])
    return seq[:T_LEN].astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────
def augment(seq, noise=0.008):
    a = seq.copy().astype(np.float32)
    a += np.random.normal(0, noise, a.shape).astype(np.float32)
    a *= np.random.uniform(0.93, 1.07)
    if np.random.rand() < 0.5:
        r = a.reshape(T_LEN, N_JNT, N_CRD); r[:, :, 0] *= -1
        a = r.reshape(T_LEN, FLAT)
    return a

# ─────────────────────────────────────────────────────────────────────────────
# Biomechanical motion generators  (one per exercise, all three quality levels)
# ─────────────────────────────────────────────────────────────────────────────
def _frames_to_seq(frames):
    return np.array([normalize(f).flatten() for f in frames], dtype=np.float32)

def _t():
    return np.linspace(0, 2 * np.pi, T_LEN)

# ── Lower body ────────────────────────────────────────────────────────────────
def gen_squat(quality):
    t = _t()
    depth  = {"good": 0.30, "medium": 0.18, "bad": 0.07}[quality]
    spread = {"good": 0.00, "medium": 0.03, "bad": 0.08}[quality]
    lean   = {"good": 0.01, "medium": 0.05, "bad": 0.13}[quality]
    dip    = depth * np.abs(np.sin(t))
    frames = []
    for i in range(T_LEN):
        s = base_skel(); d = dip[i]
        s[23] = [-0.10, -0.42 - d, 0]; s[24] = [0.10, -0.42 - d, 0]
        s[25] = [-0.10 - spread, -0.72 - d * 0.5, 0]
        s[26] = [ 0.10 + spread, -0.72 - d * 0.5, 0]
        s[:12, 2] += lean * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_deep_squat(quality):
    t = _t()
    depth  = {"good": 0.45, "medium": 0.27, "bad": 0.09}[quality]
    spread = {"good": 0.00, "medium": 0.04, "bad": 0.10}[quality]
    dip    = depth * np.abs(np.sin(t))
    frames = []
    for i in range(T_LEN):
        s = base_skel(); d = dip[i]
        s[23] = [-0.10, -0.42 - d, 0]; s[24] = [0.10, -0.42 - d, 0]
        s[25] = [-0.10 - spread, -0.72 - d * 0.6, 0]
        s[26] = [ 0.10 + spread, -0.72 - d * 0.6, 0]
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_hurdle_step(quality):
    t = _t()
    lift    = {"good": 0.30, "medium": 0.17, "bad": 0.06}[quality]
    balance = {"good": 0.01, "medium": 0.06, "bad": 0.14}[quality]
    frames  = []
    for i in range(T_LEN):
        s = base_skel()
        h = lift * np.abs(np.sin(t[i]))
        s[25] = [-0.10, -0.72 + h, 0.10 * np.sin(t[i])]
        s[27] = [-0.10, -1.02 + h * 0.5, 0.15 * np.sin(t[i])]
        s[:12, 0] += balance * np.cos(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_inline_lunge(quality):
    t = _t()
    depth = {"good": 0.25, "medium": 0.14, "bad": 0.04}[quality]
    lean  = {"good": 0.01, "medium": 0.07, "bad": 0.16}[quality]
    dip   = depth * np.abs(np.sin(t))
    frames = []
    for i in range(T_LEN):
        s = base_skel(); d = dip[i]
        s[25] = [-0.05, -0.72 - d, 0.08]
        s[27] = [-0.05, -1.02,     0.18]
        s[26] = [ 0.05, -0.72 - d * 0.5, -0.08]
        s[:12, 2] += lean * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_side_lunge(quality):
    t = _t()
    spread = {"good": 0.36, "medium": 0.20, "bad": 0.07}[quality]
    depth  = {"good": 0.20, "medium": 0.11, "bad": 0.03}[quality]
    frames = []
    for i in range(T_LEN):
        s = base_skel()
        sp = spread * np.abs(np.sin(t[i]))
        d  = depth  * np.abs(np.sin(t[i]))
        s[23] = [-0.10 - sp * 0.3, -0.42 - d, 0]
        s[25] = [-0.10 - sp,       -0.72 - d, 0]
        s[27] = [-0.10 - sp,       -1.02,     0]
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_sit_to_stand(quality):
    t = _t()
    depth = {"good": 0.38, "medium": 0.22, "bad": 0.07}[quality]
    lean  = {"good": 0.05, "medium": 0.13, "bad": 0.24}[quality]
    frames = []
    for i in range(T_LEN):
        s = base_skel()
        d = depth * np.abs(np.sin(t[i]))
        s[23] = [-0.10, -0.42 - d, 0]; s[24] = [0.10, -0.42 - d, 0]
        s[25] = [-0.10, -0.55, 0.10]; s[26] = [0.10, -0.55, 0.10]
        s[:12, 2] += lean * (1 - np.abs(np.sin(t[i])))
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_standing_leg_raise(quality):
    t = _t()
    raise_h = {"good": 0.30, "medium": 0.17, "bad": 0.05}[quality]
    balance = {"good": 0.01, "medium": 0.07, "bad": 0.15}[quality]
    frames  = []
    for i in range(T_LEN):
        s = base_skel()
        h = raise_h * np.abs(np.sin(t[i]))
        s[25] = [-0.10, -0.72 + h, 0.0]
        s[27] = [-0.10, -1.02 + h * 0.6, -0.05]
        s[0, 0] += balance * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_hip_abduction(quality):
    t = _t()
    spread  = {"good": 0.28, "medium": 0.15, "bad": 0.05}[quality]
    balance = {"good": 0.01, "medium": 0.07, "bad": 0.16}[quality]
    frames  = []
    for i in range(T_LEN):
        s = base_skel()
        sp = spread * np.abs(np.sin(t[i]))
        s[25] = [-0.10 - sp, -0.72, 0]; s[27] = [-0.10 - sp, -1.02, 0]
        s[0, 0] += balance * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_leg_raise(quality):
    t = _t()
    raise_h   = {"good": 0.35, "medium": 0.20, "bad": 0.06}[quality]
    back_arch = {"good": 0.01, "medium": 0.06, "bad": 0.16}[quality]
    frames    = []
    for i in range(T_LEN):
        s = base_skel()
        s[:, 1] *= 0.3   # supine
        h = raise_h * np.abs(np.sin(t[i]))
        s[25] = [-0.10, -0.72 + h, 0]; s[27] = [-0.10, -1.02 + h, 0]
        s[23, 2] += back_arch * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_heel_raise(quality):
    t = _t()
    lift = {"good": 0.12, "medium": 0.07, "bad": 0.02}[quality]
    sway = {"good": 0.01, "medium": 0.05, "bad": 0.13}[quality]
    frames = []
    for i in range(T_LEN):
        s = base_skel()
        h = lift * np.abs(np.sin(t[i]))
        s[27] = [-0.10, -1.02 + h, 0]; s[28] = [0.10, -1.02 + h, 0]
        s[29] = [-0.10, -1.08 + h, 0]; s[30] = [0.10, -1.08 + h, 0]
        s[0, 0] += sway * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_marching_in_place(quality):
    t = _t()
    lift  = {"good": 0.22, "medium": 0.12, "bad": 0.04}[quality]
    sway  = {"good": 0.01, "medium": 0.06, "bad": 0.15}[quality]
    frames = []
    for i in range(T_LEN):
        s = base_skel()
        lh = lift * max(0.0, float(np.sin(t[i])))
        rh = lift * max(0.0, float(-np.sin(t[i])))
        s[25] = [-0.10, -0.72 + lh, 0]; s[27] = [-0.10, -1.02 + lh * 0.6, 0]
        s[26] = [ 0.10, -0.72 + rh, 0]; s[28] = [ 0.10, -1.02 + rh * 0.6, 0]
        s[0, 0] += sway * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_step_up(quality):
    t = _t()
    height = {"good": 0.20, "medium": 0.11, "bad": 0.03}[quality]
    lean   = {"good": 0.02, "medium": 0.09, "bad": 0.19}[quality]
    frames = []
    for i in range(T_LEN):
        s = base_skel()
        h = height * np.abs(np.sin(t[i]))
        s[25] = [-0.10, -0.72 + h, 0]; s[27] = [-0.10, -1.02 + h, 0]
        s[:12, 1] += h * 0.3
        s[:12, 2] += lean * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_sit_to_stand_alt(quality):  # alias
    return gen_sit_to_stand(quality)

# ── Upper body ────────────────────────────────────────────────────────────────
def gen_shoulder_abduction(quality):
    t = _t()
    max_h = {"good": 0.40, "medium": 0.23, "bad": 0.08}[quality]
    bend  = {"good": 0.00, "medium": 0.05, "bad": 0.14}[quality]
    frames = []
    for i in range(T_LEN):
        s = base_skel()
        h = max_h * np.abs(np.sin(t[i]))
        s[13] = [-0.30 - h * 0.3, -0.18 + h, 0]
        s[14] = [ 0.30 + h * 0.3, -0.18 + h, 0]
        s[15] = [-0.38 - h * 0.2, -0.36 + h + bend, 0]
        s[16] = [ 0.38 + h * 0.2, -0.36 + h + bend, 0]
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_shoulder_extension(quality):
    t = _t()
    ext   = {"good": 0.25, "medium": 0.13, "bad": 0.04}[quality]
    trunk = {"good": 0.01, "medium": 0.06, "bad": 0.15}[quality]
    frames = []
    for i in range(T_LEN):
        s = base_skel()
        e = ext * np.abs(np.sin(t[i]))
        s[13] = [-0.30, -0.18 - e, -0.10 - e]
        s[14] = [ 0.30, -0.18 - e, -0.10 - e]
        s[15] = [-0.38, -0.36 - e, -0.20 - e]
        s[16] = [ 0.38, -0.36 - e, -0.20 - e]
        s[:12, 2] += trunk * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_shoulder_rotation(quality):
    t = _t()
    rot  = {"good": 0.20, "medium": 0.11, "bad": 0.03}[quality]
    comp = {"good": 0.01, "medium": 0.05, "bad": 0.13}[quality]
    frames = []
    for i in range(T_LEN):
        s = base_skel()
        r = rot * np.sin(t[i])
        s[13] = [-0.30, -0.18, 0]; s[14] = [0.30, -0.18, 0]
        s[15] = [-0.30, -0.18 + r, -0.10 + abs(r)]
        s[16] = [ 0.30, -0.18 + r, -0.10 + abs(r)]
        s[:12, 0] += comp * np.cos(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_shoulder_scaption(quality):
    t = _t()
    max_h = {"good": 0.35, "medium": 0.19, "bad": 0.07}[quality]
    frames = []
    for i in range(T_LEN):
        s = base_skel()
        h = max_h * np.abs(np.sin(t[i]))
        s[13] = [-0.30 - h * 0.2, -0.18 + h, -h * 0.2]
        s[14] = [ 0.30 + h * 0.2, -0.18 + h, -h * 0.2]
        s[15] = [-0.38 - h * 0.15,-0.36 + h, -h * 0.3]
        s[16] = [ 0.38 + h * 0.15,-0.36 + h, -h * 0.3]
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_trunk_rotation(quality):
    t = _t()
    rot  = {"good": 0.20, "medium": 0.11, "bad": 0.03}[quality]
    comp = {"good": 0.00, "medium": 0.05, "bad": 0.13}[quality]
    frames = []
    for i in range(T_LEN):
        s = base_skel()
        r = rot * np.sin(t[i])
        s[11, 2] += r; s[12, 2] -= r
        s[13, 2] += r; s[14, 2] -= r
        s[23, 2] += comp * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_reach_and_retrieve(quality):
    t = _t()
    reach = {"good": 0.30, "medium": 0.17, "bad": 0.05}[quality]
    trunk = {"good": 0.05, "medium": 0.13, "bad": 0.23}[quality]
    frames = []
    for i in range(T_LEN):
        s = base_skel()
        r = reach * np.abs(np.sin(t[i]))
        s[15] = [-0.38 - r, -0.36 - r * 0.3,  r * 0.5]
        s[16] = [ 0.38 + r, -0.36 - r * 0.3,  r * 0.5]
        s[:12, 2] += trunk * np.abs(np.sin(t[i]))
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_wall_pushup(quality):
    t = _t()
    ext = {"good": 0.22, "medium": 0.12, "bad": 0.03}[quality]
    sag = {"good": 0.01, "medium": 0.06, "bad": 0.16}[quality]
    frames = []
    for i in range(T_LEN):
        s = base_skel()
        e = ext * np.abs(np.sin(t[i]))
        s[13] = [-0.25, -0.05, 0.10 + e]; s[14] = [0.25, -0.05, 0.10 + e]
        s[15] = [-0.30, -0.10, 0.25 + e]; s[16] = [0.30, -0.10, 0.25 + e]
        s[23, 2] += sag * np.sin(t[i]); s[24, 2] += sag * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

# ── Floor / core exercises ────────────────────────────────────────────────────
def gen_bird_dog(quality):
    t = _t()
    ext = {"good": 0.25, "medium": 0.14, "bad": 0.05}[quality]
    sag = {"good": 0.01, "medium": 0.04, "bad": 0.10}[quality]
    frames = []
    for i in range(T_LEN):
        s = base_skel(); s[:, 1] -= 0.30   # quadruped
        e = ext * np.abs(np.sin(t[i]))
        s[15] = [-0.45 - e, -0.40 + e * 0.3,  0.20 + e]
        s[28] = [ 0.10,     -1.02 + e * 0.3, -0.20 - e]
        s[11:13, 1] -= sag * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_glute_bridge(quality):
    t = _t()
    lift  = {"good": 0.28, "medium": 0.15, "bad": 0.05}[quality]
    shift = {"good": 0.00, "medium": 0.04, "bad": 0.11}[quality]
    frames = []
    for i in range(T_LEN):
        s = base_skel(); s[:, 1] *= -0.5   # supine
        h = lift * np.abs(np.sin(t[i]))
        s[23] = [-0.10, -0.42 + h, 0]; s[24] = [0.10, -0.42 + h, 0]
        s[25] = [-0.10, -0.72 + h * 0.3, 0]; s[26] = [0.10, -0.72 + h * 0.3, 0]
        s[23, 0] += shift * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_clamshell(quality):
    t = _t()
    open_a = {"good": 0.22, "medium": 0.12, "bad": 0.03}[quality]
    comp   = {"good": 0.01, "medium": 0.05, "bad": 0.13}[quality]
    frames = []
    for i in range(T_LEN):
        s = base_skel(); s[:, 1] *= 0.5   # side-lying
        a = open_a * np.abs(np.sin(t[i]))
        s[26] = [0.10 + a * 0.3, -0.72, a * 0.4]
        s[28] = [0.10 + a * 0.2, -1.02, a * 0.3]
        s[23, 2] += comp * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

def gen_chin_tuck(quality):
    t = _t()
    tuck  = {"good": 0.05, "medium": 0.03, "bad": 0.01}[quality]
    proto = {"good": 0.00, "medium": 0.03, "bad": 0.08}[quality]
    frames = []
    for i in range(T_LEN):
        s = base_skel()
        tk = tuck * np.abs(np.sin(t[i]))
        s[0] = [0.00, 0.10 - tk, -tk]
        s[7] = [-0.08, 0.11 - tk, -tk]; s[8] = [0.08, 0.11 - tk, -tk]
        s[0, 2] -= proto * np.sin(t[i])
        s += np.random.normal(0, 0.003, s.shape)
        frames.append(s)
    return _frames_to_seq(frames)

# ── Dispatch table ────────────────────────────────────────────────────────────
GENERATORS = {
    "squat":               gen_squat,
    "deep_squat":          gen_deep_squat,
    "hurdle_step":         gen_hurdle_step,
    "inline_lunge":        gen_inline_lunge,
    "side_lunge":          gen_side_lunge,
    "sit_to_stand":        gen_sit_to_stand,
    "standing_leg_raise":  gen_standing_leg_raise,
    "shoulder_abduction":  gen_shoulder_abduction,
    "shoulder_extension":  gen_shoulder_extension,
    "shoulder_rotation":   gen_shoulder_rotation,
    "shoulder_scaption":   gen_shoulder_scaption,
    "hip_abduction":       gen_hip_abduction,
    "trunk_rotation":      gen_trunk_rotation,
    "leg_raise":           gen_leg_raise,
    "reach_and_retrieve":  gen_reach_and_retrieve,
    "wall_pushup":         gen_wall_pushup,
    "heel_raise":          gen_heel_raise,
    "bird_dog":            gen_bird_dog,
    "glute_bridge":        gen_glute_bridge,
    "clamshell":           gen_clamshell,
    "chin_tuck":           gen_chin_tuck,
    "marching_in_place":   gen_marching_in_place,
    "step_up":             gen_step_up,
}

# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — Custom webcam data
# ─────────────────────────────────────────────────────────────────────────────
def load_custom(exercise, custom_dir):
    """Load existing custom .npy files + labels.csv."""
    d = os.path.join(custom_dir, exercise)
    if not os.path.isdir(d):
        return []
    results = []
    label_map = {}
    csv = os.path.join(d, "labels.csv")
    if os.path.isfile(csv):
        df = pd.read_csv(csv)
        for _, row in df.iterrows():
            fname = str(row.iloc[0]).strip()
            try:    score = float(row["score"])
            except: score = 75.0
            try:    reps  = int(row["reps"])
            except: reps  = 5
            label_map[fname] = (score, reps)
    for npy in sorted(glob.glob(os.path.join(d, "*.npy"))):
        try:
            seq   = pad_trim(np.load(npy))
            fname = os.path.basename(npy)
            score, reps = label_map.get(fname, (75.0, 5))
            results.append(dict(seq=seq, score=score, reps=reps, source="custom"))
        except Exception:
            pass
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — NTU RGB+D  (Shahroudy et al., CVPR 2016)
# ─────────────────────────────────────────────────────────────────────────────
# 25-joint Kinect → MediaPipe 33-joint approximate mapping
NTU_TO_MP = {
    0:23, 1:24, 2:0,  3:0,   # spine/head → nose/hips
    4:11, 5:13, 6:15, 7:19,  # L arm
    8:12, 9:14, 10:16,11:20, # R arm
    12:23,13:25,14:27,15:31, # L leg
    16:24,17:26,18:28,19:32, # R leg
    20:0, 21:22,22:17,23:21, 24:18,
}

def _parse_ntu_file(path):
    try:
        with open(path) as f:
            lines = f.read().split()
        idx  = 0
        nf   = int(lines[idx]); idx += 1
        frames = []
        for _ in range(nf):
            nb = int(lines[idx]); idx += 1
            body = None
            for _ in range(nb):
                idx += 10
                nj = int(lines[idx]); idx += 1
                joints = []
                for _ in range(nj):
                    x, y, z = float(lines[idx]), float(lines[idx+1]), float(lines[idx+2])
                    idx += 12
                    joints.append([x, y, z])
                if body is None:
                    body = np.array(joints, dtype=np.float32)
            frames.append(body if body is not None else np.zeros((25,3), np.float32))
        return np.array(frames, dtype=np.float32)
    except Exception:
        return None

def _ntu_to_mp(raw):
    T_ = len(raw)
    mp = np.zeros((T_, N_JNT, N_CRD), np.float32)
    for t, f in enumerate(raw):
        for nj, mj in NTU_TO_MP.items():
            if nj < len(f):
                mp[t, mj] = f[nj]
    return mp

def load_ntu(exercise, ntu_dir, n_synth, gen):
    action = NTU_MAP.get(exercise)
    results = []

    # Try real files first
    if action:
        adir = os.path.join(ntu_dir, action)
        if os.path.isdir(adir):
            for fp in sorted(glob.glob(os.path.join(adir, "*.skeleton")))[:60]:
                raw = _parse_ntu_file(fp)
                if raw is not None:
                    mp  = _ntu_to_mp(raw)
                    seq = pad_trim(norm_seq(mp.reshape(len(mp), FLAT)))
                    results.append(dict(seq=seq, score=float(np.random.randint(55,96)),
                                        reps=5, source="ntu"))
            if results:
                return results

    # Synthetic NTU stand-ins: noisier + slight scale variation (Kinect depth noise)
    if gen is None:
        return []
    for quality in ["good", "medium", "bad"]:
        lo, hi = {"good":(75,95),"medium":(50,74),"bad":(20,49)}[quality]
        for _ in range(n_synth):
            seq = gen(quality).copy()
            seq += np.random.normal(0, 0.020, seq.shape).astype(np.float32)   # Kinect noise
            seq *= float(np.random.uniform(0.85, 1.15))                        # depth scale
            results.append(dict(seq=seq.astype(np.float32),
                                score=float(np.random.randint(lo, hi+1)),
                                reps=5, source="ntu_syn"))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Source 3 — UI-PRMD  (Vakanski et al., IEEE Access 2018)
# ─────────────────────────────────────────────────────────────────────────────
# 20-joint Vicon → MediaPipe approximate mapping
UIPRMD_TO_MP = {
    0:0, 1:11, 2:12, 3:13, 4:14, 5:15, 6:16,
    7:23, 8:24, 9:25, 10:26, 11:27, 12:28,
    13:31, 14:32, 15:0, 16:0, 17:23, 18:11, 19:12,
}

def _parse_uiprmd_csv(path):
    try:
        df = pd.read_csv(path, header=None)
        data = df.values.astype(np.float32)
        if data.shape[1] < 60:  # 20 joints × 3
            return None
        return data[:, :60].reshape(len(data), 20, 3)
    except Exception:
        return None

def _uiprmd_to_mp(raw):
    T_ = len(raw)
    mp = np.zeros((T_, N_JNT, N_CRD), np.float32)
    for t, f in enumerate(raw):
        for uj, mj in UIPRMD_TO_MP.items():
            if uj < len(f):
                mp[t, mj] = f[uj]
    return mp

def load_uiprmd(exercise, uiprmd_dir, n_synth, gen):
    ex_id = UIPRMD_MAP.get(exercise)
    results = []

    if ex_id:
        eDir = os.path.join(uiprmd_dir, ex_id)
        if os.path.isdir(eDir):
            for fp in sorted(glob.glob(os.path.join(eDir, "*.csv")))[:50]:
                raw = _parse_uiprmd_csv(fp)
                if raw is not None:
                    mp  = _uiprmd_to_mp(raw)
                    seq = pad_trim(norm_seq(mp.reshape(len(mp), FLAT)))
                    results.append(dict(seq=seq,
                                        score=float(np.random.randint(60,97)),
                                        reps=5, source="uiprmd"))
            if results:
                return results

    # Synthetic UI-PRMD stand-ins: Vicon is very clean (low noise) but clinical range
    if gen is None:
        return []
    for quality in ["good", "medium", "bad"]:
        lo, hi = {"good":(78,98),"medium":(55,77),"bad":(25,54)}[quality]
        for _ in range(n_synth):
            seq = gen(quality).copy()
            seq += np.random.normal(0, 0.004, seq.shape).astype(np.float32)   # Vicon: very clean
            results.append(dict(seq=seq.astype(np.float32),
                                score=float(np.random.randint(lo, hi+1)),
                                reps=5, source="uiprmd_syn"))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Source 4 — KIMORE  (Capecci et al., IEEE TNSRE 2019)
# ─────────────────────────────────────────────────────────────────────────────
# 18-joint OpenPose → MediaPipe approximate mapping
KIMORE_TO_MP = {
    0:0, 1:12, 2:12, 3:14, 4:16,
    5:11, 6:13, 7:15, 8:24, 9:26,
    10:28, 11:23, 12:25, 13:27,
    14:2, 15:5, 16:8, 17:7,
}

def _parse_kimore_json(path):
    try:
        with open(path) as f:
            data = json.load(f)
        frames = []
        for fd in data:
            people = fd.get("people", [])
            if not people:
                continue
            kp = people[0].get("pose_keypoints_2d", [])
            joints = []
            for j in range(18):
                x = float(kp[j*3])     / 1920.0 if len(kp) > j*3   else 0.0
                y = float(kp[j*3+1])   / 1080.0 if len(kp) > j*3+1 else 0.0
                c = float(kp[j*3+2])            if len(kp) > j*3+2 else 0.0
                joints.append([x, y, c])
            frames.append(joints)
        return np.array(frames, dtype=np.float32)
    except Exception:
        return None

def _kimore_to_mp(raw):
    T_ = len(raw)
    mp = np.zeros((T_, N_JNT, N_CRD), np.float32)
    for t, f in enumerate(raw):
        for kj, mj in KIMORE_TO_MP.items():
            if kj < len(f):
                mp[t, mj] = f[kj]
    return mp

def load_kimore(exercise, kimore_dir, n_synth, gen):
    ex_id = KIMORE_MAP.get(exercise)
    results = []

    if ex_id:
        eDir = os.path.join(kimore_dir, ex_id)
        if os.path.isdir(eDir):
            for fp in sorted(glob.glob(os.path.join(eDir,"**","*.json"),
                                        recursive=True))[:40]:
                raw = _parse_kimore_json(fp)
                if raw is not None and len(raw) > 5:
                    mp  = _kimore_to_mp(raw)
                    seq = pad_trim(norm_seq(mp.reshape(len(mp), FLAT)))
                    results.append(dict(seq=seq,
                                        score=float(np.random.randint(50,96)),
                                        reps=5, source="kimore"))
            if results:
                return results

    # Synthetic KIMORE stand-ins: 2D projection noise, medium noise level
    if gen is None:
        return []
    for quality in ["good", "medium", "bad"]:
        lo, hi = {"good":(70,95),"medium":(45,69),"bad":(20,44)}[quality]
        for _ in range(n_synth):
            seq = gen(quality).copy()
            seq += np.random.normal(0, 0.015, seq.shape).astype(np.float32)
            seq[:, ::3] *= float(np.random.uniform(0.92, 1.08))  # x-axis (2D projection)
            results.append(dict(seq=seq.astype(np.float32),
                                score=float(np.random.randint(lo, hi+1)),
                                reps=5, source="kimore_syn"))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Merge all sources → unified/<exercise>/
# ─────────────────────────────────────────────────────────────────────────────
def build(args):
    custom_dir  = os.path.join(ROOT, "data", "custom")
    ntu_dir     = os.path.join(ROOT, "data", "ntu")
    uiprmd_dir  = os.path.join(ROOT, "data", "uiprmd")
    kimore_dir  = os.path.join(ROOT, "data", "kimore")
    unified_dir = os.path.join(ROOT, "data", "unified")
    os.makedirs(unified_dir, exist_ok=True)

    n_synth = args.synth_per_quality   # per quality tier per source

    print("\n" + "═"*68)
    print("  Multi-Source Rehabilitation Dataset Builder")
    print("  Sources: Custom │ NTU RGB+D │ UI-PRMD │ KIMORE")
    print("═"*68)

    grand_total  = 0
    source_tally = {}
    exercise_summary = []

    for ex in ALL_EXERCISES:
        gen = GENERATORS.get(ex)

        samples  = []
        samples += load_custom(ex, custom_dir)
        samples += load_ntu    (ex, ntu_dir,    n_synth, gen)
        samples += load_uiprmd (ex, uiprmd_dir, n_synth, gen)
        samples += load_kimore (ex, kimore_dir, n_synth, gen)

        # Augment each sample once more for diversity
        extra = []
        for s in samples:
            extra.append(dict(seq=augment(s["seq"]).astype(np.float32),
                              score=s["score"], reps=s["reps"],
                              source=s["source"] + "_aug"))
        samples += extra

        # Save
        out_dir = os.path.join(unified_dir, ex)
        os.makedirs(out_dir, exist_ok=True)
        records = []
        for i, s in enumerate(samples):
            fname = f"seq_{i:05d}.npy"
            np.save(os.path.join(out_dir, fname), s["seq"])
            records.append({"sequence": fname,
                             "score":    round(float(np.clip(s["score"], 0, 100)), 1),
                             "reps":     int(s["reps"]),
                             "source":   s["source"]})
            src = s["source"]
            source_tally[src] = source_tally.get(src, 0) + 1

        pd.DataFrame(records).to_csv(os.path.join(out_dir, "labels.csv"), index=False)
        grand_total += len(samples)
        exercise_summary.append({"exercise": ex, "count": len(samples)})

        bar = "█" * min(40, len(samples) // 10)
        print(f"  ✅  {ex:<28s}  {len(samples):4d}  {bar}")

    # Write dataset_info.json
    info = {
        "total_sequences": grand_total,
        "exercises": len(ALL_EXERCISES),
        "sequence_shape": [T_LEN, FLAT],
        "joints": N_JNT,
        "normalization": "hip-centred, torso-length scaled",
        "per_exercise": exercise_summary,
        "sources": {
            "custom": {
                "description": "Webcam recordings via MediaPipe BlazePose (33 joints)",
                "how_to_add":  "Record with: python scripts/record_data.py --exercise <name>"
            },
            "ntu": {
                "description": "NTU RGB+D — Kinect V2 skeleton, 25 joints, mapped to MP-33",
                "paper": "Shahroudy et al., CVPR 2016",
                "download": "https://rose1.ntu.edu.sg/dataset/actionRecognition/",
                "how_to_add": "Place .skeleton files in data/ntu/<ACTION_ID>/",
                "action_ids": NTU_MAP
            },
            "uiprmd": {
                "description": "UI-PRMD — Vicon mocap, 10 rehab exercises, 20 joints, mapped to MP-33",
                "paper": "Vakanski et al., IEEE Access 2018",
                "download": "https://ieeexplore.ieee.org/document/8367416",
                "how_to_add": "Place .csv files in data/uiprmd/<P00X>/",
                "exercise_ids": UIPRMD_MAP
            },
            "kimore": {
                "description": "KIMORE — OpenPose 18-joint + clinical quality scores, 5 exercises",
                "paper": "Capecci et al., IEEE TNSRE 2019",
                "download": "https://vrai.dii.univpm.it/content/kimore-dataset",
                "how_to_add": "Place OpenPose JSON files in data/kimore/<EX>/",
                "exercise_ids": KIMORE_MAP
            }
        },
        "source_tally": source_tally
    }
    info_path = os.path.join(ROOT, "data", "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  {'─'*64}")
    print(f"  Total sequences   : {grand_total:,}")
    print(f"  Exercises         : {len(ALL_EXERCISES)}")
    print(f"\n  Source breakdown:")
    for src in sorted(source_tally):
        print(f"    {src:<30s}: {source_tally[src]:,}")
    print(f"\n  dataset_info.json → {info_path}")
    print(f"  Unified data      → {unified_dir}/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth-per-quality", type=int, default=15,
                        help="Synthetic sequences per quality tier per source (default 15)")
    args = parser.parse_args()
    build(args)
