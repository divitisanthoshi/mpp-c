"""
Training Pipeline
-----------------
Trains the ST-GCN++ model on:
  1. Existing .npy sequences in data/custom/<exercise>/
  2. Synthetic data generated on-the-fly (so the model always trains even with
     very small datasets).

Run:
    python train.py
"""

import os
import sys
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

from src.utils.preprocessing import (
    SEQUENCE_LENGTH, N_JOINTS, N_COORDS, INPUT_DIM, augment_sequence,
)
from src.models.st_gcn import build_model, compile_model, get_callbacks

# ── Config ──────────────────────────────────────────────────────────────────────────────
_UNIFIED = "data/unified"
_CUSTOM  = "data/custom"
DATA_DIR     = _UNIFIED if os.path.isdir(_UNIFIED) else _CUSTOM
MODEL_PATH   = "models/rehab_model.keras"
META_PATH    = "models/meta.json"
EPOCHS       = 50
BATCH_SIZE   = 16
AUG_FACTOR   = 2
MIN_SYNTH    = 30

EXERCISE_LIST = [
    "squat", "deep_squat", "hurdle_step", "inline_lunge", "side_lunge",
    "sit_to_stand", "standing_leg_raise", "shoulder_abduction",
    "shoulder_extension", "shoulder_rotation", "shoulder_scaption",
    "hip_abduction", "trunk_rotation", "leg_raise", "reach_and_retrieve",
    "wall_pushup", "heel_raise", "bird_dog", "glute_bridge", "clamshell",
    "chin_tuck", "marching_in_place", "step_up",
]

PHASE_MAP = {"start": 0, "down": 1, "up": 2, "hold": 3}


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_synthetic_sequence(exercise_idx: int, n_exercises: int) -> tuple:
    """Create a plausible synthetic skeleton sequence with a label."""
    seq = np.random.randn(SEQUENCE_LENGTH, N_JOINTS, N_COORDS).astype(np.float32)
    # Make it look like a repetitive motion
    t = np.linspace(0, 2 * np.pi, SEQUENCE_LENGTH)
    seq[:, 25, 1] += np.sin(t) * 0.3   # knee bouncing
    seq = seq.reshape(SEQUENCE_LENGTH, INPUT_DIM)
    score = np.random.uniform(0.4, 0.95)
    ex_onehot = np.eye(n_exercises)[exercise_idx]
    phase_onehot = np.eye(4)[1]   # "down"
    return seq, score, ex_onehot, phase_onehot


def load_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    n_ex = len(EXERCISE_LIST)
    label_enc = {ex: i for i, ex in enumerate(EXERCISE_LIST)}

    X, Y_score, Y_ex, Y_phase = [], [], [], []

    # ── Load real sequences ──────────────────────────────────────────────────
    for exercise in EXERCISE_LIST:
        ex_dir = os.path.join(DATA_DIR, exercise)
        if not os.path.isdir(ex_dir):
            os.makedirs(ex_dir, exist_ok=True)
            continue

        npy_files = sorted(glob.glob(os.path.join(ex_dir, "*.npy")))
        labels_csv = os.path.join(ex_dir, "labels.csv")

        # Try to read labels
        label_map = {}
        if os.path.isfile(labels_csv):
            import pandas as pd
            df = pd.read_csv(labels_csv)
            for _, row in df.iterrows():
                key = str(row.get("sequence", row.iloc[0])).strip()
                score_raw = str(row.get("score", row.iloc[1] if len(row) > 1 else 75)).strip()
                try:
                    label_map[key] = float(score_raw) / 100.0
                except ValueError:
                    label_map[key] = 0.75

        for npy_path in npy_files:
            try:
                seq = np.load(npy_path)
                # Reshape to (T, INPUT_DIM)
                if seq.ndim == 3:                             # (T,33,3)
                    seq = seq.reshape(seq.shape[0], INPUT_DIM)
                if seq.shape[0] < 10:
                    continue
                # Pad or truncate to SEQUENCE_LENGTH
                if seq.shape[0] < SEQUENCE_LENGTH:
                    pad = np.zeros((SEQUENCE_LENGTH - seq.shape[0], INPUT_DIM))
                    seq = np.vstack([seq, pad])
                else:
                    seq = seq[:SEQUENCE_LENGTH]

                fname = os.path.basename(npy_path)
                score = label_map.get(fname, label_map.get(fname.replace(".npy", ""), 0.75))

                ex_oh = np.eye(n_ex)[label_enc[exercise]]
                ph_oh = np.eye(4)[1]

                X.append(seq)
                Y_score.append(score)
                Y_ex.append(ex_oh)
                Y_phase.append(ph_oh)

                # Augment real data
                for _ in range(AUG_FACTOR):
                    aug = augment_sequence(seq.reshape(SEQUENCE_LENGTH, N_JOINTS, N_COORDS))
                    X.append(aug.reshape(SEQUENCE_LENGTH, INPUT_DIM))
                    Y_score.append(score + np.random.uniform(-0.05, 0.05))
                    Y_ex.append(ex_oh)
                    Y_phase.append(ph_oh)

            except Exception as e:
                print(f"  [warn] Could not load {npy_path}: {e}")

    # ── Fill with synthetic data if dataset still tiny ───────────────────────
    for ex_idx, exercise in enumerate(EXERCISE_LIST):
        count = sum(1 for ye in Y_ex if np.argmax(ye) == ex_idx)
        needed = max(0, MIN_SYNTH - count)
        for _ in range(needed):
            seq, sc, ex_oh, ph_oh = make_synthetic_sequence(ex_idx, n_ex)
            X.append(seq)
            Y_score.append(sc)
            Y_ex.append(ex_oh)
            Y_phase.append(ph_oh)

    return (
        np.array(X, dtype=np.float32),
        np.clip(np.array(Y_score, dtype=np.float32), 0, 1),
        np.array(Y_ex,    dtype=np.float32),
        np.array(Y_phase, dtype=np.float32),
    )


# ── Main training ──────────────────────────────────────────────────────────────

def train():
    os.makedirs("models", exist_ok=True)

    print("=" * 60)
    print("  Rehabilitation Exercise Quality Grading — Training")
    print("=" * 60)

    print("\n[1/4] Loading & augmenting data...")
    X, Y_score, Y_ex, Y_phase = load_data()
    print(f"      Total samples: {len(X)}")

    print("\n[2/4] Splitting dataset...")
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, Y_score, Y_ex, Y_phase = X[idx], Y_score[idx], Y_ex[idx], Y_phase[idx]

    split = max(1, int(0.15 * len(X)))
    X_val,  Y_sv, Y_ev, Y_pv = X[:split],  Y_score[:split],  Y_ex[:split],  Y_phase[:split]
    X_tr,   Y_st, Y_et, Y_pt = X[split:],  Y_score[split:],  Y_ex[split:],  Y_phase[split:]

    print(f"      Train: {len(X_tr)}   Val: {len(X_val)}")

    print("\n[3/4] Building model...")
    model = build_model(n_exercises=len(EXERCISE_LIST))
    model = compile_model(model)
    model.summary(print_fn=lambda s: print("  " + s))

    print("\n[4/4] Training...")
    model.fit(
        X_tr,
        {"score": Y_st, "exercise": Y_et, "phase": Y_pt},
        validation_data=(
            X_val,
            {"score": Y_sv, "exercise": Y_ev, "phase": Y_pv},
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(),
        verbose=1,
    )

    model.save(MODEL_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}")

    meta = {"exercises": EXERCISE_LIST, "sequence_length": SEQUENCE_LENGTH}
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"   Meta  saved → {META_PATH}")


if __name__ == "__main__":
    train()
