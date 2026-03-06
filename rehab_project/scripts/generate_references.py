"""
Generate Reference Sequences
-----------------------------
Creates synthetic "perfect" reference skeleton sequences for each exercise.
These are used for DTW-based scoring when no real references exist.

Run:
    python scripts/generate_references.py
"""

import os
import numpy as np

EXERCISES = ["squat", "deep_squat", "lunge", "arm_raise", "bird_dog", "bridge"]
T = 60      # frames
N = 33      # joints
C = 3       # coords


def make_reference(exercise: str) -> np.ndarray:
    """Generate a plausible neutral reference skeleton sequence."""
    seq = np.zeros((T, N, C), dtype=np.float32)
    t   = np.linspace(0, 2 * np.pi, T)

    # Base T-pose normalised skeleton
    # Positions roughly follow MediaPipe coordinate conventions
    base = {
        0:  [0.00,  0.20, 0],    # nose
        11: [-0.20, 0.00, 0],    # left_shoulder
        12: [ 0.20, 0.00, 0],    # right_shoulder
        13: [-0.35,-0.20, 0],    # left_elbow
        14: [ 0.35,-0.20, 0],
        15: [-0.45,-0.40, 0],    # left_wrist
        16: [ 0.45,-0.40, 0],
        23: [-0.12,-0.40, 0],    # left_hip
        24: [ 0.12,-0.40, 0],
        25: [-0.12,-0.70, 0],    # left_knee
        26: [ 0.12,-0.70, 0],
        27: [-0.12,-1.00, 0],    # left_ankle
        28: [ 0.12,-1.00, 0],
    }
    for joint, pos in base.items():
        for frame in range(T):
            seq[frame, joint] = pos

    # Add exercise-specific motion
    if exercise in ("squat", "deep_squat"):
        depth = 0.3 if exercise == "squat" else 0.45
        dip = depth * np.abs(np.sin(t))
        for frame in range(T):
            d = dip[frame]
            seq[frame, 23] = [-0.12, -0.40 - d, 0]
            seq[frame, 24] = [ 0.12, -0.40 - d, 0]
            seq[frame, 25] = [-0.12, -0.70 - d * 0.5, 0]
            seq[frame, 26] = [ 0.12, -0.70 - d * 0.5, 0]

    elif exercise in ("lunge",):
        dip = 0.25 * np.abs(np.sin(t))
        for frame in range(T):
            d = dip[frame]
            seq[frame, 25] = [-0.12, -0.70 - d, 0]
            seq[frame, 27] = [-0.12, -1.00 - d, 0]

    elif exercise in ("arm_raise", "shoulder_abduction"):
        raise_angle = np.abs(np.sin(t))
        for frame in range(T):
            a = raise_angle[frame]
            seq[frame, 13] = [-0.35 - a * 0.25, -0.20 + a * 0.35, 0]
            seq[frame, 14] = [ 0.35 + a * 0.25, -0.20 + a * 0.35, 0]
            seq[frame, 15] = [-0.45 - a * 0.30, -0.40 + a * 0.50, 0]
            seq[frame, 16] = [ 0.45 + a * 0.30, -0.40 + a * 0.50, 0]

    elif exercise in ("bird_dog", "quadruped"):
        ext = 0.2 * np.abs(np.sin(t))
        for frame in range(T):
            seq[frame, 15] = [-0.45 - ext[frame], -0.40 + ext[frame] * 0.5, 0]
            seq[frame, 27] = [-0.12 - ext[frame], -1.00 + ext[frame] * 0.5, 0]

    elif exercise in ("bridge", "hip_bridge"):
        lift = 0.25 * np.abs(np.sin(t))
        for frame in range(T):
            seq[frame, 23] = [-0.12, -0.40 + lift[frame], 0]
            seq[frame, 24] = [ 0.12, -0.40 + lift[frame], 0]

    return seq.reshape(T, N * C)


if __name__ == "__main__":
    for ex in EXERCISES:
        out_dir = os.path.join("data", "reference")
        os.makedirs(out_dir, exist_ok=True)
        ref = make_reference(ex)
        path = os.path.join(out_dir, f"{ex}_reference.npy")
        np.save(path, ref)
        print(f"  ✅ {path}  shape={ref.shape}")
    print("\nDone.")
