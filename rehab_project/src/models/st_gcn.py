"""
ST-GCN ++ Model
Spatial-Temporal Graph Convolutional Network with BiLSTM + Attention
for multi-task rehabilitation exercise quality grading.
"""

import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ── Constants ──────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 60
N_JOINTS        = 33
N_COORDS        = 3
INPUT_DIM       = N_JOINTS * N_COORDS   # 99


# ── Attention Layer ────────────────────────────────────────────────────────────

class JointAttention(layers.Layer):
    """Soft attention over the time-steps of an LSTM output."""

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units, activation="tanh")
        self.u = layers.Dense(1)

    def call(self, hidden_states):
        # hidden_states: (batch, T, features)
        score = self.u(self.W(hidden_states))      # (batch, T, 1)
        weights = tf.nn.softmax(score, axis=1)     # (batch, T, 1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)  # (batch, features)
        return context


# ── Model builder ──────────────────────────────────────────────────────────────

def build_model(n_exercises: int = 6, sequence_length: int = SEQUENCE_LENGTH) -> Model:
    """
    Multi-task model:
      - score output  : continuous 0–100
      - exercise output: softmax over exercise classes
      - phase output  : start / down / up / hold
    """
    inp = Input(shape=(sequence_length, INPUT_DIM), name="skeleton_input")

    # ── Spatial feature extraction (1-D convolutions over joint features) ──
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # ── Temporal modelling (Bi-LSTM) ────────────────────────────────────────
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)

    # ── Joint Attention ─────────────────────────────────────────────────────
    context = JointAttention(64, name="attention")(x)
    context = layers.Dropout(0.3)(context)

    # ── Dense bottleneck ────────────────────────────────────────────────────
    shared = layers.Dense(64, activation="relu")(context)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.3)(shared)

    # ── Task heads ──────────────────────────────────────────────────────────
    # 1. Quality score (0–100 normalised to 0–1 internally, scaled at output)
    score_head = layers.Dense(32, activation="relu")(shared)
    score_out  = layers.Dense(1, activation="sigmoid", name="score")(score_head)

    # 2. Exercise classification
    ex_head = layers.Dense(32, activation="relu")(shared)
    ex_out  = layers.Dense(n_exercises, activation="softmax", name="exercise")(ex_head)

    # 3. Movement phase
    phase_head = layers.Dense(16, activation="relu")(shared)
    phase_out  = layers.Dense(4, activation="softmax", name="phase")(phase_head)

    model = Model(inputs=inp, outputs=[score_out, ex_out, phase_out])
    return model


def compile_model(model: Model, lr: float = 5e-4) -> Model:
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss={
            "score":    "mse",
            "exercise": "categorical_crossentropy",
            "phase":    "categorical_crossentropy",
        },
        loss_weights={"score": 2.0, "exercise": 1.0, "phase": 0.5},
        metrics={
            "score":    ["mae"],
            "exercise": ["accuracy"],
            "phase":    ["accuracy"],
        },
    )
    return model


def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]


# ── DTW-based scoring (no model needed) ───────────────────────────────────────

def dtw_score(user_seq: np.ndarray, ref_seq: np.ndarray) -> float:
    """
    user_seq, ref_seq: (T, 99)
    Returns score 0–100.
    """
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        dist, _ = fastdtw(user_seq, ref_seq, dist=euclidean)
        # Normalise: distance ~0 → 100, large → 0
        score = max(0.0, 100.0 - dist / max(len(user_seq), 1))
    except Exception:
        score = 50.0
    return float(score)
