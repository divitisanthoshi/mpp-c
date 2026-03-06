"""
Generate Synthetic Rehabilitation Dataset
==========================================
Creates realistic skeleton sequences for 6 exercises with:
  - Good form (score 80-100)
  - Medium form (score 50-79)
  - Bad form (score 20-49)

Output: data/custom/<exercise>/seq_XXX.npy + labels.csv
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

SEQUENCE_LENGTH = 60
N_JOINTS = 33
N_COORDS = 3

EXERCISES = ["squat", "deep_squat", "lunge", "arm_raise", "bird_dog", "bridge"]
SAMPLES_PER_QUALITY = 30   # 30 good + 30 medium + 30 bad = 90 per exercise

# ── Base T-pose skeleton (MediaPipe coordinate space) ─────────────────────────
def base_skeleton():
    """Returns (33, 3) neutral standing skeleton."""
    skel = np.zeros((N_JOINTS, N_COORDS), dtype=np.float32)

    # Face
    skel[0]  = [0.00,  0.10, 0.00]   # nose
    skel[1]  = [-0.03, 0.12, -0.02]
    skel[2]  = [-0.05, 0.12, -0.03]
    skel[3]  = [-0.07, 0.11, -0.02]
    skel[4]  = [0.03,  0.12, -0.02]
    skel[5]  = [0.05,  0.12, -0.03]
    skel[6]  = [0.07,  0.11, -0.02]
    skel[7]  = [-0.08, 0.11, 0.00]
    skel[8]  = [0.08,  0.11, 0.00]
    skel[9]  = [-0.02, 0.08, -0.01]
    skel[10] = [0.02,  0.08, -0.01]

    # Shoulders
    skel[11] = [-0.18, 0.00, 0.00]   # left shoulder
    skel[12] = [0.18,  0.00, 0.00]   # right shoulder

    # Arms
    skel[13] = [-0.30, -0.18, 0.00]  # left elbow
    skel[14] = [0.30,  -0.18, 0.00]  # right elbow
    skel[15] = [-0.38, -0.36, 0.00]  # left wrist
    skel[16] = [0.38,  -0.36, 0.00]  # right wrist
    skel[17] = [-0.40, -0.40, 0.00]
    skel[18] = [0.40,  -0.40, 0.00]
    skel[19] = [-0.41, -0.41, 0.00]
    skel[20] = [0.41,  -0.41, 0.00]
    skel[21] = [-0.39, -0.39, 0.00]
    skel[22] = [0.39,  -0.39, 0.00]

    # Hips
    skel[23] = [-0.10, -0.42, 0.00]  # left hip
    skel[24] = [0.10,  -0.42, 0.00]  # right hip

    # Legs
    skel[25] = [-0.10, -0.72, 0.00]  # left knee
    skel[26] = [0.10,  -0.72, 0.00]  # right knee
    skel[27] = [-0.10, -1.02, 0.00]  # left ankle
    skel[28] = [0.10,  -1.02, 0.00]  # right ankle
    skel[29] = [-0.10, -1.08, 0.00]
    skel[30] = [0.10,  -1.08, 0.00]
    skel[31] = [-0.10, -1.10, 0.00]
    skel[32] = [0.10,  -1.10, 0.00]

    return skel


# ── Normalisation ──────────────────────────────────────────────────────────────
def normalize(skel):
    hip_c = (skel[23] + skel[24]) / 2
    mid_s = (skel[11] + skel[12]) / 2
    torso = np.linalg.norm(mid_s - hip_c) + 1e-6
    return (skel - hip_c) / torso


# ── Motion generators ──────────────────────────────────────────────────────────

def make_squat(quality="good", deep=False):
    """Squat / deep squat sequence."""
    t = np.linspace(0, 2 * np.pi, SEQUENCE_LENGTH)
    depth_map = {"good": 0.30 if not deep else 0.45,
                 "medium": 0.18 if not deep else 0.28,
                 "bad":  0.08}
    back_noise = {"good": 0.01, "medium": 0.05, "bad": 0.12}
    knee_spread = {"good": 0.00, "medium": 0.03, "bad": 0.07}

    depth = depth_map[quality]
    dip   = depth * np.abs(np.sin(t))

    seq = []
    for i in range(SEQUENCE_LENGTH):
        s = base_skeleton().copy()
        d = dip[i]
        s[23] = [-0.10, -0.42 - d, 0]
        s[24] = [0.10,  -0.42 - d, 0]
        s[25] = [-0.10 - knee_spread[quality], -0.72 - d * 0.5, 0]
        s[26] = [0.10  + knee_spread[quality], -0.72 - d * 0.5, 0]
        s[27] = [-0.10, -1.02, 0]
        s[28] = [0.10,  -1.02, 0]
        # Back lean (bad = forward lean)
        lean = back_noise[quality] * np.sin(t[i])
        s[0:12, 2] += lean
        s += np.random.normal(0, 0.005, s.shape)
        seq.append(normalize(s).flatten())

    return np.array(seq, dtype=np.float32)


def make_lunge(quality="good"):
    t = np.linspace(0, 2 * np.pi, SEQUENCE_LENGTH)
    depth_map = {"good": 0.25, "medium": 0.15, "bad": 0.05}
    torso_lean = {"good": 0.02, "medium": 0.08, "bad": 0.15}

    dip = depth_map[quality] * np.abs(np.sin(t))
    lean = torso_lean[quality]

    seq = []
    for i in range(SEQUENCE_LENGTH):
        s = base_skeleton().copy()
        d = dip[i]
        # Left leg steps forward and bends
        s[25] = [-0.10, -0.72 - d, 0.10]
        s[27] = [-0.10, -1.02,     0.20]
        # Right leg stays, bends at knee
        s[26] = [0.10, -0.72 - d * 0.5, -0.10]
        # Torso slight lean
        s[0:12, 2] += lean * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        seq.append(normalize(s).flatten())

    return np.array(seq, dtype=np.float32)


def make_arm_raise(quality="good"):
    t = np.linspace(0, 2 * np.pi, SEQUENCE_LENGTH)
    max_raise = {"good": 0.38, "medium": 0.22, "bad": 0.10}
    elbow_bend = {"good": 0.00, "medium": 0.05, "bad": 0.12}

    raise_h = max_raise[quality] * np.abs(np.sin(t))
    bend    = elbow_bend[quality]

    seq = []
    for i in range(SEQUENCE_LENGTH):
        s = base_skeleton().copy()
        r = raise_h[i]
        # Raise both arms sideways
        s[13] = [-0.30 - r * 0.3, -0.18 + r, 0]   # left elbow
        s[14] = [0.30  + r * 0.3, -0.18 + r, 0]
        s[15] = [-0.38 - r * 0.2, -0.36 + r + bend, 0]
        s[16] = [0.38  + r * 0.2, -0.36 + r + bend, 0]
        s += np.random.normal(0, 0.005, s.shape)
        seq.append(normalize(s).flatten())

    return np.array(seq, dtype=np.float32)


def make_bird_dog(quality="good"):
    t = np.linspace(0, 2 * np.pi, SEQUENCE_LENGTH)
    ext_map = {"good": 0.25, "medium": 0.15, "bad": 0.06}
    back_sag = {"good": 0.01, "medium": 0.04, "bad": 0.09}

    ext = ext_map[quality] * np.abs(np.sin(t))
    sag = back_sag[quality]

    seq = []
    for i in range(SEQUENCE_LENGTH):
        s = base_skeleton().copy()
        # Quadruped position — lower whole body
        s[:, 1] -= 0.30
        e = ext[i]
        # Extend left arm forward, right leg back
        s[15] = [-0.45 - e, -0.40 + e * 0.3, 0.20 + e]
        s[28] = [0.10,      -1.02 + e * 0.3, -0.20 - e]
        # Back sag
        s[11:13, 1] -= sag * np.sin(t[i])
        s += np.random.normal(0, 0.005, s.shape)
        seq.append(normalize(s).flatten())

    return np.array(seq, dtype=np.float32)


def make_bridge(quality="good"):
    t = np.linspace(0, 2 * np.pi, SEQUENCE_LENGTH)
    lift_map = {"good": 0.28, "medium": 0.16, "bad": 0.06}
    knee_map = {"good": 90, "medium": 115, "bad": 145}

    lift = lift_map[quality] * np.abs(np.sin(t))

    seq = []
    for i in range(SEQUENCE_LENGTH):
        s = base_skeleton().copy()
        # Lying on back — rotate
        s[:, 1] *= -0.5
        l = lift[i]
        s[23] = [-0.10, -0.42 + l, 0]
        s[24] = [0.10,  -0.42 + l, 0]
        s[25] = [-0.10, -0.72 + l * 0.3, 0]
        s[26] = [0.10,  -0.72 + l * 0.3, 0]
        s += np.random.normal(0, 0.005, s.shape)
        seq.append(normalize(s).flatten())

    return np.array(seq, dtype=np.float32)


# ── Augmentation ───────────────────────────────────────────────────────────────
def augment(seq):
    aug = seq.copy()
    aug += np.random.normal(0, 0.008, aug.shape)
    aug *= np.random.uniform(0.93, 1.07)
    if np.random.rand() < 0.5:
        # Mirror left-right (flip x coord of every joint)
        reshaped = aug.reshape(SEQUENCE_LENGTH, N_JOINTS, N_COORDS)
        reshaped[:, :, 0] *= -1
        aug = reshaped.reshape(SEQUENCE_LENGTH, N_JOINTS * N_COORDS)
    return aug.astype(np.float32)


# ── Main generator ─────────────────────────────────────────────────────────────
SCORE_RANGES = {
    "good":   (82, 98),
    "medium": (52, 78),
    "bad":    (20, 48),
}

GENERATORS = {
    "squat":      lambda q: make_squat(q, deep=False),
    "deep_squat": lambda q: make_squat(q, deep=True),
    "lunge":      lambda q: make_lunge(q),
    "arm_raise":  lambda q: make_arm_raise(q),
    "bird_dog":   lambda q: make_bird_dog(q),
    "bridge":     lambda q: make_bridge(q),
}


def generate_dataset(n_per_quality=SAMPLES_PER_QUALITY, augment_factor=4):
    for exercise in EXERCISES:
        out_dir = os.path.join("data", "custom", exercise)
        os.makedirs(out_dir, exist_ok=True)

        records = []
        seq_idx = 0
        gen = GENERATORS[exercise]

        for quality in ["good", "medium", "bad"]:
            lo, hi = SCORE_RANGES[quality]

            for k in range(n_per_quality):
                base_seq = gen(quality)
                seqs = [base_seq] + [augment(base_seq) for _ in range(augment_factor)]

                for seq in seqs:
                    fname = f"seq_{seq_idx:04d}.npy"
                    np.save(os.path.join(out_dir, fname), seq)
                    score = np.random.randint(lo, hi + 1)
                    reps  = np.random.randint(4, 8)
                    records.append({"sequence": fname, "score": score, "reps": reps})
                    seq_idx += 1

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(out_dir, "labels.csv"), index=False)
        print(f"  ✅ {exercise:15s}  sequences={seq_idx}  labels={len(df)}")


if __name__ == "__main__":
    print("Generating synthetic rehabilitation dataset…\n")
    generate_dataset()
    print(f"\nDone. Total exercises: {len(EXERCISES)}")
    print("Each exercise: ~{} sequences (30 good + 30 medium + 30 bad) × 5 augments".format(
        SAMPLES_PER_QUALITY * 5 * 3))
