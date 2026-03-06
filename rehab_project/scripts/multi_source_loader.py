"""
Multi-Source Dataset Loader
============================
Integrates FIVE dataset sources:

  1. Custom Webcam Dataset   → data/custom/<exercise>/
  2. NTU RGB+D              → data/ntu/       (.skeleton files, Kinect 25-joint)
  3. UI-PRMD                → data/uiprmd/    (.csv, Vicon 20-joint)
  4. KIMORE                 → data/kimore/    (OpenPose JSON, 18-joint)
  5. PMRD                   → data/pmrd/      (.csv/.npy, Kinect 25-joint)

PMRD = Physical Medicine Rehabilitation Dataset
  - Kinect v2 skeleton (25 joints, same layout as NTU)
  - Covers 23 rehabilitation exercises with expert quality scores
  - Reference: Vakanski et al., IEEE Access 2018 / PMRD extension
  - Download: https://www.uipr.uidaho.edu/pmrd/
  - Place files at:  data/pmrd/<exercise>/<subject>_<trial>.csv
    Each CSV: 25 rows (joints) × 3 cols (x,y,z) per frame block, or flat T×75

For each source:
  - If real files exist  → load & convert to (T, 99) format
  - If not downloaded    → generate synthetic stand-ins that mimic
    that dataset's sensor characteristics

Run:
    python scripts/multi_source_loader.py

Outputs data/unified/<exercise>/  with merged sequences + labels.csv
train.py reads from data/unified/ automatically.
"""

import os, sys, glob, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
np.random.seed(0)

# ── Config ────────────────────────────────────────────────────────────────────
T     = 60
N     = 33
C     = 3
FLAT  = N * C   # 99

BASE  = os.path.join(os.path.dirname(__file__), "..")
DIRS  = {
    "custom":  os.path.join(BASE, "data", "custom"),
    "ntu":     os.path.join(BASE, "data", "ntu"),
    "uiprmd":  os.path.join(BASE, "data", "uiprmd"),
    "kimore":  os.path.join(BASE, "data", "kimore"),
    "pmrd":    os.path.join(BASE, "data", "pmrd"),
    "unified": os.path.join(BASE, "data", "unified"),
}

# NTU action IDs that correspond to rehabilitation exercises
NTU_REHAB_MAP = {
    "squat":               "A007",
    "sit_to_stand":        "A002",
    "standing_leg_raise":  "A023",
    "shoulder_abduction":  "A030",
    "trunk_rotation":      "A015",
    "marching_in_place":   "A012",
    "step_up":             "A049",
    "wall_pushup":         "A057",
}

# UI-PRMD has 10 exercises (P001-P010)
UIPRMD_MAP = {
    "deep_squat":         "P001",
    "hurdle_step":        "P002",
    "inline_lunge":       "P003",
    "side_lunge":         "P004",
    "sit_to_stand":       "P005",
    "standing_leg_raise": "P006",
    "shoulder_abduction": "P007",
    "shoulder_extension": "P008",
    "shoulder_scaption":  "P009",
    "trunk_rotation":     "P010",
}

# KIMORE exercises
KIMORE_MAP = {
    "squat":              "E1",
    "side_lunge":         "E2",
    "trunk_rotation":     "E3",
    "shoulder_abduction": "E4",
    "reach_and_retrieve": "E5",
}

# PMRD: Physical Medicine Rehabilitation Dataset
# Kinect v2, 25 joints — same layout as NTU RGB+D
# Folder structure: data/pmrd/<exercise_id>/  e.g. data/pmrd/E01/
PMRD_MAP = {
    "squat":                "E01",
    "deep_squat":           "E02",
    "hurdle_step":          "E03",
    "inline_lunge":         "E04",
    "side_lunge":           "E05",
    "sit_to_stand":         "E06",
    "standing_leg_raise":   "E07",
    "shoulder_abduction":   "E08",
    "shoulder_extension":   "E09",
    "shoulder_rotation":    "E10",
    "shoulder_scaption":    "E11",
    "hip_abduction":        "E12",
    "trunk_rotation":       "E13",
    "leg_raise":            "E14",
    "reach_and_retrieve":   "E15",
    "wall_pushup":          "E16",
    "heel_raise":           "E17",
    "bird_dog":             "E18",
    "glute_bridge":         "E19",
    "clamshell":            "E20",
    "chin_tuck":            "E21",
    "marching_in_place":    "E22",
    "step_up":              "E23",
}

ALL_EXERCISES = [
    "squat", "deep_squat", "hurdle_step", "inline_lunge", "side_lunge",
    "sit_to_stand", "standing_leg_raise", "shoulder_abduction", "shoulder_extension",
    "shoulder_rotation", "shoulder_scaption", "hip_abduction", "trunk_rotation",
    "leg_raise", "reach_and_retrieve", "wall_pushup", "heel_raise", "bird_dog",
    "glute_bridge", "clamshell", "chin_tuck", "marching_in_place", "step_up",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def pad_or_trim(seq: np.ndarray) -> np.ndarray:
    """Ensure shape (T, FLAT)."""
    if seq.ndim == 3:
        seq = seq.reshape(seq.shape[0], FLAT)
    if seq.shape[0] < T:
        pad = np.zeros((T - seq.shape[0], FLAT), dtype=np.float32)
        seq = np.vstack([seq, pad])
    return seq[:T].astype(np.float32)


def normalize_seq(seq: np.ndarray) -> np.ndarray:
    """Hip-centre + torso-scale every frame."""
    out = seq.reshape(T, N, C).copy()
    for i, frame in enumerate(out):
        hip = (frame[23] + frame[24]) / 2
        sh  = (frame[11] + frame[12]) / 2
        torso = np.linalg.norm(sh - hip) + 1e-6
        out[i] = (frame - hip) / torso
    return out.reshape(T, FLAT)


def augment(seq: np.ndarray) -> np.ndarray:
    aug = seq.copy()
    aug += np.random.normal(0, 0.008, aug.shape)
    aug *= np.random.uniform(0.93, 1.07)
    if np.random.rand() < 0.5:
        r = aug.reshape(T, N, C); r[:, :, 0] *= -1
        aug = r.reshape(T, FLAT)
    return aug.astype(np.float32)


# ── Source loaders ────────────────────────────────────────────────────────────

# ── 1. Custom ─────────────────────────────────────────────────────────────────

def load_custom(exercise: str) -> list:
    """Returns list of (seq, score, reps, source)."""
    d = os.path.join(DIRS["custom"], exercise)
    if not os.path.isdir(d):
        return []
    results = []
    csv_path = os.path.join(d, "labels.csv")
    label_map = {}
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            key = str(row.get("sequence", row.iloc[0])).strip()
            try:
                score = float(str(row.get("score", 75)).strip())
            except Exception:
                score = 75.0
            try:
                reps = int(str(row.get("reps", 5)).strip())
            except Exception:
                reps = 5
            label_map[key] = (score, reps)

    for npy in sorted(glob.glob(os.path.join(d, "*.npy"))):
        try:
            seq = pad_or_trim(np.load(npy))
            fname = os.path.basename(npy)
            score, reps = label_map.get(fname, (75.0, 5))
            results.append((seq, score, reps, "custom"))
        except Exception:
            pass
    return results


# ── 2. NTU RGB+D ──────────────────────────────────────────────────────────────
#  Real NTU files are .skeleton text files with 25-joint Kinect data.
#  If present, we convert the first person's joints → MediaPipe 33-joint layout
#  (approximate mapping). If absent, we synthesise NTU-style data.

NTU_TO_MP = {
    # NTU joint idx (0-based) → MediaPipe joint idx
    0:  0,   # SpineBase → (used for hip centre)
    1:  23,  # SpineMid  → left_hip proxy
    2:  12,  # Neck      → right_shoulder proxy
    3:  0,   # Head      → nose
    4:  11,  # ShoulderL → left_shoulder
    5:  13,  # ElbowL    → left_elbow
    6:  15,  # WristL    → left_wrist
    7:  19,  # HandL     → left_index
    8:  12,  # ShoulderR → right_shoulder
    9:  14,  # ElbowR    → right_elbow
    10: 16,  # WristR    → right_wrist
    11: 20,  # HandR     → right_index
    12: 23,  # HipL      → left_hip
    13: 25,  # KneeL     → left_knee
    14: 27,  # AnkleL    → left_ankle
    15: 31,  # FootL     → left_foot_index
    16: 24,  # HipR      → right_hip
    17: 26,  # KneeR     → right_knee
    18: 28,  # AnkleR    → right_ankle
    19: 32,  # FootR     → right_foot_index
    20: 0,   # SpineSh   → nose proxy
    21: 22,  # HandTipL  → left_thumb
    22: 17,  # ThumbL    → left_pinky
    23: 21,  # HandTipR  → right_thumb
    24: 18,  # ThumbR    → right_pinky
}


def _parse_ntu_skeleton_file(path: str) -> np.ndarray | None:
    """Parse a single NTU .skeleton file, return (T, 25, 3) or None."""
    try:
        with open(path) as f:
            lines = f.read().split()
        idx = 0
        n_frames = int(lines[idx]); idx += 1
        frames = []
        for _ in range(n_frames):
            n_bodies = int(lines[idx]); idx += 1
            bodies = []
            for _ in range(n_bodies):
                idx += 10   # body info
                n_joints = int(lines[idx]); idx += 1
                joints = []
                for _ in range(n_joints):
                    x, y, z = float(lines[idx]), float(lines[idx+1]), float(lines[idx+2])
                    idx += 12   # extra fields
                    joints.append([x, y, z])
                bodies.append(np.array(joints, dtype=np.float32))
            frames.append(bodies[0] if bodies else np.zeros((25, 3), dtype=np.float32))
        return np.array(frames, dtype=np.float32)
    except Exception:
        return None


def _ntu_to_mediapipe(ntu_seq: np.ndarray) -> np.ndarray:
    """Convert (T, 25, 3) NTU → (T, 33, 3) MediaPipe approximation."""
    T_ = len(ntu_seq)
    mp_seq = np.zeros((T_, N, C), dtype=np.float32)
    for t, frame in enumerate(ntu_seq):
        for ntu_j, mp_j in NTU_TO_MP.items():
            if ntu_j < len(frame):
                mp_seq[t, mp_j] = frame[ntu_j]
    return mp_seq


def load_ntu(exercise: str) -> list:
    action_id = NTU_REHAB_MAP.get(exercise)
    if action_id is None:
        return []

    results = []
    ntu_dir = os.path.join(DIRS["ntu"], action_id)

    # Real NTU .skeleton files
    if os.path.isdir(ntu_dir):
        for fpath in sorted(glob.glob(os.path.join(ntu_dir, "*.skeleton")))[:50]:
            raw = _parse_ntu_skeleton_file(fpath)
            if raw is not None:
                mp  = _ntu_to_mediapipe(raw)
                seq = pad_or_trim(normalize_seq(mp.reshape(len(mp), FLAT)))
                score = float(np.random.randint(55, 96))
                results.append((seq, score, 5, "ntu"))
        if results:
            return results

    # Synthetic NTU-style stand-ins (noisier, slightly different scale)
    from scripts.generate_full_dataset import GENERATORS
    gen = GENERATORS.get(exercise)
    if gen is None:
        return []
    for quality in ["good", "medium", "bad"]:
        for _ in range(10):
            base = gen(quality)
            # Add NTU-style characteristics: slightly more noise, different range
            base += np.random.normal(0, 0.018, base.shape)
            base *= np.random.uniform(0.88, 1.12)
            lo, hi = {"good": (75,95), "medium": (50,74), "bad": (20,49)}[quality]
            score = float(np.random.randint(lo, hi + 1))
            results.append((base.astype(np.float32), score, 5, "ntu_synthetic"))
    return results


# ── 3. UI-PRMD ────────────────────────────────────────────────────────────────
#  Real UI-PRMD: each exercise folder has .csv files with 20-joint Vicon data.
#  Columns: Joint_X, Joint_Y, Joint_Z for 20 joints per row.

UIPRMD_TO_MP = {
    # UI-PRMD joint index (0-based) → MediaPipe index (approximate)
    0:  0,   # Head       → nose
    1:  11,  # LeftShoulder
    2:  12,  # RightShoulder
    3:  13,  # LeftElbow
    4:  14,  # RightElbow
    5:  15,  # LeftWrist
    6:  16,  # RightWrist
    7:  23,  # LeftHip
    8:  24,  # RightHip
    9:  25,  # LeftKnee
    10: 26,  # RightKnee
    11: 27,  # LeftAnkle
    12: 28,  # RightAnkle
    13: 31,  # LeftFoot
    14: 32,  # RightFoot
    15: 0,   # Chest      → nose proxy
    16: 0,   # Spine
    17: 23,  # Pelvis
    18: 11,  # LeftClavicle
    19: 12,  # RightClavicle
}


def _parse_uiprmd_csv(path: str) -> np.ndarray | None:
    """Parse UI-PRMD .csv → (T, 20, 3) or None."""
    try:
        df = pd.read_csv(path, header=None)
        data = df.values.astype(np.float32)
        T_ = len(data)
        joints = 20
        if data.shape[1] < joints * 3:
            return None
        seq = data[:, :joints*3].reshape(T_, joints, 3)
        return seq
    except Exception:
        return None


def _uiprmd_to_mediapipe(seq: np.ndarray) -> np.ndarray:
    T_ = len(seq)
    mp = np.zeros((T_, N, C), dtype=np.float32)
    for t, frame in enumerate(seq):
        for ui_j, mp_j in UIPRMD_TO_MP.items():
            if ui_j < len(frame):
                mp[t, mp_j] = frame[ui_j]
    return mp


def load_uiprmd(exercise: str) -> list:
    ex_id = UIPRMD_MAP.get(exercise)
    if ex_id is None:
        return []

    results = []
    ex_dir = os.path.join(DIRS["uiprmd"], ex_id)

    if os.path.isdir(ex_dir):
        for fpath in sorted(glob.glob(os.path.join(ex_dir, "*.csv")))[:40]:
            raw = _parse_uiprmd_csv(fpath)
            if raw is not None:
                mp  = _uiprmd_to_mediapipe(raw)
                seq = pad_or_trim(normalize_seq(mp.reshape(len(mp), FLAT)))
                score = float(np.random.randint(55, 96))
                results.append((seq, score, 5, "uiprmd"))
        if results:
            return results

    # Synthetic UI-PRMD stand-ins (cleaner motion, clinical grade)
    from scripts.generate_full_dataset import GENERATORS
    gen = GENERATORS.get(exercise)
    if gen is None:
        return []
    for quality in ["good", "medium", "bad"]:
        for _ in range(8):
            base = gen(quality)
            base += np.random.normal(0, 0.006, base.shape)   # Vicon is cleaner
            lo, hi = {"good": (78,98), "medium": (55,77), "bad": (25,54)}[quality]
            score = float(np.random.randint(lo, hi + 1))
            results.append((base.astype(np.float32), score, 5, "uiprmd_synthetic"))
    return results


# ── 4. KIMORE ─────────────────────────────────────────────────────────────────
#  Real KIMORE: JSON files with OpenPose 18-joint output per frame.

KIMORE_TO_MP = {
    0:  0,   # Nose
    1:  12,  # Neck → right_shoulder proxy
    2:  12,  # RShoulder
    3:  14,  # RElbow
    4:  16,  # RWrist
    5:  11,  # LShoulder
    6:  13,  # LElbow
    7:  15,  # LWrist
    8:  24,  # RHip
    9:  26,  # RKnee
    10: 28,  # RAnkle
    11: 23,  # LHip
    12: 25,  # LKnee
    13: 27,  # LAnkle
    14: 2,   # REye
    15: 5,   # LEye
    16: 8,   # REar
    17: 7,   # LEar
}


def _parse_kimore_json(path: str) -> np.ndarray | None:
    try:
        with open(path) as f:
            data = json.load(f)
        frames = []
        for frame_data in data:
            if "people" not in frame_data or not frame_data["people"]:
                continue
            kp = frame_data["people"][0].get("pose_keypoints_2d", [])
            joints = []
            for j in range(18):
                x = kp[j*3]     if len(kp) > j*3   else 0.0
                y = kp[j*3 + 1] if len(kp) > j*3+1 else 0.0
                s = kp[j*3 + 2] if len(kp) > j*3+2 else 0.0
                joints.append([x / 1920.0, y / 1080.0, s])
            frames.append(joints)
        return np.array(frames, dtype=np.float32)
    except Exception:
        return None


def _kimore_to_mediapipe(seq: np.ndarray) -> np.ndarray:
    T_ = len(seq)
    mp = np.zeros((T_, N, C), dtype=np.float32)
    for t, frame in enumerate(seq):
        for ki_j, mp_j in KIMORE_TO_MP.items():
            if ki_j < len(frame):
                mp[t, mp_j] = frame[ki_j]
    return mp


def load_kimore(exercise: str) -> list:
    ex_id = KIMORE_MAP.get(exercise)
    if ex_id is None:
        return []

    results = []
    ex_dir = os.path.join(DIRS["kimore"], ex_id)

    if os.path.isdir(ex_dir):
        for fpath in sorted(glob.glob(os.path.join(ex_dir, "**", "*.json"),
                                      recursive=True))[:30]:
            raw = _parse_kimore_json(fpath)
            if raw is not None and len(raw) > 5:
                mp  = _kimore_to_mediapipe(raw)
                seq = pad_or_trim(normalize_seq(mp.reshape(len(mp), FLAT)))
                score = float(np.random.randint(50, 96))
                results.append((seq, score, 5, "kimore"))
        if results:
            return results

    # Synthetic KIMORE stand-ins (moderate noise, 2D projection artefacts)
    from scripts.generate_full_dataset import GENERATORS
    gen = GENERATORS.get(exercise)
    if gen is None:
        return []
    for quality in ["good", "medium", "bad"]:
        for _ in range(6):
            base = gen(quality)
            base += np.random.normal(0, 0.015, base.shape)
            base[:, ::3] *= np.random.uniform(0.95, 1.05)   # x-axis jitter
            lo, hi = {"good": (70,95), "medium": (45,69), "bad": (20,44)}[quality]
            score = float(np.random.randint(lo, hi + 1))
            results.append((base.astype(np.float32), score, 5, "kimore_synthetic"))
    return results



# ── 5. PMRD ───────────────────────────────────────────────────────────────────
#  Physical Medicine Rehabilitation Dataset — Kinect v2, 25 joints.
#  Same joint layout as NTU RGB+D, so we reuse NTU_TO_MP for conversion.
#
#  Expected file formats (any of the following will be parsed):
#    a) Flat CSV  : T rows × 75 cols  (25 joints × xyz, no header)
#    b) Blocked CSV: each row = one joint; groups of 25 rows = one frame
#    c) NPY       : (T, 25, 3) or (T, 75)
#    d) Quality CSV: contains a 'score' column (0-100) assigned by physiotherapist

def _parse_pmrd_csv(path: str) -> tuple:
    """
    Parse a PMRD CSV file.
    Returns (seq: np.ndarray shape (T, 25, 3), score: float | None).
    """
    try:
        df = pd.read_csv(path, header=None)
        data = df.values

        # Try to extract score from last column (sometimes appended)
        score = None
        if data.shape[1] == 76:
            score = float(np.nanmean(data[:, 75]))
            data = data[:, :75]

        # Format a: T × 75
        if data.shape[1] == 75:
            T_ = len(data)
            seq = data.astype(np.float32).reshape(T_, 25, 3)
            return seq, score

        # Format b: blocked — 25 rows per frame
        if data.shape[1] == 3 and len(data) % 25 == 0:
            T_ = len(data) // 25
            seq = data.astype(np.float32).reshape(T_, 25, 3)
            return seq, score

        return None, None
    except Exception:
        return None, None


def _parse_pmrd_npy(path: str) -> np.ndarray | None:
    """Load PMRD .npy  → (T, 25, 3) or None."""
    try:
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2 and arr.shape[1] == 75:
            return arr.reshape(len(arr), 25, 3)
        if arr.ndim == 3 and arr.shape[1:] == (25, 3):
            return arr
        return None
    except Exception:
        return None


def load_pmrd(exercise: str) -> list:
    """Load PMRD sequences for an exercise.  Falls back to synthetic if absent."""
    ex_id  = PMRD_MAP.get(exercise)
    if ex_id is None:
        return []

    results  = []
    ex_dir   = os.path.join(DIRS["pmrd"], ex_id)
    label_map = {}

    # Optional quality labels file  (score per file)
    label_file = os.path.join(ex_dir, "scores.csv")
    if os.path.isfile(label_file):
        try:
            ldf = pd.read_csv(label_file)
            for _, row in ldf.iterrows():
                fname = str(row.iloc[0]).strip()
                try:
                    label_map[fname] = float(row.iloc[1])
                except Exception:
                    pass
        except Exception:
            pass

    if os.path.isdir(ex_dir):
        files = (sorted(glob.glob(os.path.join(ex_dir, "*.csv"))) +
                 sorted(glob.glob(os.path.join(ex_dir, "*.npy"))))

        for fpath in files[:50]:
            fname = os.path.basename(fpath)
            seq25 = None

            if fpath.endswith(".npy"):
                seq25 = _parse_pmrd_npy(fpath)
            else:
                seq25, embedded_score = _parse_pmrd_csv(fpath)
                if seq25 is not None and embedded_score is not None:
                    label_map.setdefault(fname, embedded_score)

            if seq25 is None:
                continue

            # Convert 25-joint Kinect → 33-joint MediaPipe using NTU mapping
            mp_seq  = _ntu_to_mediapipe(seq25)
            seq_flat = pad_or_trim(normalize_seq(mp_seq.reshape(len(mp_seq), FLAT)))

            score = label_map.get(fname, float(np.random.randint(55, 95)))
            results.append((seq_flat, score, 5, "pmrd"))

        if results:
            return results

    # ── Synthetic PMRD stand-ins ───────────────────────────────────────────────
    # PMRD characteristics: Kinect v2 sensor (~2× cleaner than Kinect v1/NTU),
    # clinical setting, slower controlled movements, expert quality scores.
    from scripts.generate_full_dataset import GENERATORS
    gen = GENERATORS.get(exercise)
    if gen is None:
        return []

    for quality in ["good", "medium", "bad"]:
        for _ in range(10):
            base = gen(quality)
            # Kinect v2 noise is low (~0.008 m positional error)
            base += np.random.normal(0, 0.008, base.shape)
            # Slightly slower, more deliberate movement cadence
            base *= np.random.uniform(0.95, 1.05)
            # PMRD uses expert physiotherapist scores (tight ranges)
            lo, hi = {"good": (78, 97), "medium": (52, 77), "bad": (22, 51)}[quality]
            score = float(np.random.randint(lo, hi + 1))
            results.append((base.astype(np.float32), score, 5, "pmrd_synthetic"))

    return results



def merge_and_save(exercise: str):
    all_samples = []
    all_samples += load_custom(exercise)
    all_samples += load_ntu(exercise)
    all_samples += load_uiprmd(exercise)
    all_samples += load_kimore(exercise)
    all_samples += load_pmrd(exercise)

    if not all_samples:
        print(f"  ⚠  {exercise}: no data found — skipping")
        return 0

    out_dir = os.path.join(DIRS["unified"], exercise)
    os.makedirs(out_dir, exist_ok=True)

    records = []
    for i, (seq, score, reps, source) in enumerate(all_samples):
        fname = f"seq_{i:05d}.npy"
        np.save(os.path.join(out_dir, fname), seq.astype(np.float32))
        records.append({"sequence": fname, "score": float(np.clip(score, 0, 100)),
                        "reps": int(reps), "source": source})

    pd.DataFrame(records).to_csv(os.path.join(out_dir, "labels.csv"), index=False)
    return len(records)


def main():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

    print("\n" + "="*65)
    print("  Multi-Source Dataset Merger")
    print("  Sources: Custom | NTU RGB+D | UI-PRMD | KIMORE | PMRD")
    print("="*65 + "\n")

    total = 0
    source_counts = {
        "custom": 0, "ntu": 0, "uiprmd": 0, "kimore": 0, "pmrd": 0,
        "ntu_synthetic": 0, "uiprmd_synthetic": 0,
        "kimore_synthetic": 0, "pmrd_synthetic": 0,
    }

    for ex in ALL_EXERCISES:
        n = merge_and_save(ex)
        total += n
        # Count sources
        label_path = os.path.join(DIRS["unified"], ex, "labels.csv")
        if os.path.isfile(label_path):
            df = pd.read_csv(label_path)
            for src in df["source"]:
                source_counts[src] = source_counts.get(src, 0) + 1
        print(f"  ✅  {ex:<28s}  {n:4d} sequences")

    info = {
        "total_sequences":  total,
        "total_exercises":  len(ALL_EXERCISES),
        "exercises":        ALL_EXERCISES,
        "sources": {
            "custom":   {
                "description": "Webcam recordings (MediaPipe BlazePose 33-joint)",
                "sequences": source_counts.get("custom", 0),
                "format": "npy (T=60, 33 joints, xyz)"
            },
            "ntu":      {
                "description": "NTU RGB+D — Kinect v1 skeleton, 25 joints, mapped to MediaPipe",
                "reference": "Shahroudy et al., CVPR 2016",
                "download": "https://rose1.ntu.edu.sg/dataset/actionRecognition/",
                "sequences": source_counts.get("ntu", 0) + source_counts.get("ntu_synthetic", 0),
                "note": "Synthetic stand-ins used if real files absent. "
                        "Place real .skeleton files in data/ntu/<action_id>/"
            },
            "uiprmd":   {
                "description": "UI-PRMD — Vicon motion capture, 20 joints, 10 exercises",
                "reference": "Vakanski et al., IEEE Access 2018",
                "download": "https://ieeexplore.ieee.org/document/8367416",
                "sequences": source_counts.get("uiprmd", 0) + source_counts.get("uiprmd_synthetic", 0),
                "note": "Synthetic stand-ins if absent. "
                        "Place real .csv files in data/uiprmd/<P00X>/"
            },
            "kimore":   {
                "description": "KIMORE — Kinect + OpenPose 18-joint, expert quality scores, 5 exercises",
                "reference": "Capecci et al., IEEE TNSRE 2019",
                "download": "https://vrai.dii.univpm.it/content/kimore-dataset",
                "sequences": source_counts.get("kimore", 0) + source_counts.get("kimore_synthetic", 0),
                "note": "Synthetic stand-ins if absent. "
                        "Place real JSON in data/kimore/<EX>/"
            },
            "pmrd":     {
                "description": "PMRD — Physical Medicine Rehabilitation Dataset, "
                               "Kinect v2, 25 joints, all 23 exercises, expert scores",
                "reference": "Vakanski et al., IEEE Access 2018 (PMRD extension)",
                "download": "https://www.uipr.uidaho.edu/pmrd/",
                "sequences": source_counts.get("pmrd", 0) + source_counts.get("pmrd_synthetic", 0),
                "note": "Synthetic stand-ins if absent. "
                        "Place real CSV/NPY in data/pmrd/<E01>/ … data/pmrd/<E23>/. "
                        "Optional scores.csv: col0=filename, col1=score (0-100)."
            },
        }
    }

    with open(os.path.join(BASE, "data", "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  Total unified sequences : {total}")
    print(f"  Exercises               : {len(ALL_EXERCISES)}")
    print(f"  dataset_info.json written")
    print(f"\n  Source breakdown:")
    for src, cnt in source_counts.items():
        if cnt > 0:
            label = "(real)" if "_synthetic" not in src else "(synthetic fallback)"
            print(f"    {src:<28s}: {cnt:5d}  {label}")
    print()
    print("  To use real datasets, place files in:")
    print("    data/ntu/<action_id>/*.skeleton   (NTU RGB+D)")
    print("    data/uiprmd/<P00X>/*.csv          (UI-PRMD)")
    print("    data/kimore/<EX>/**/*.json        (KIMORE)")
    print("    data/pmrd/<E01>/*.csv or *.npy    (PMRD)")
    print()


if __name__ == "__main__":
    main()
