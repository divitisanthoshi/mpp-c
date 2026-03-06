"""
Repetition Counter
Uses joint-angle peak detection for stable, accurate rep counting.
"""

import numpy as np
from scipy.signal import find_peaks
from src.utils.preprocessing import calculate_angle, JOINTS


# ── Which joints to track per exercise ────────────────────────────────────────

EXERCISE_JOINTS = {
    "squat":             ("left_hip",      "left_knee",  "left_ankle"),
    "deep_squat":        ("left_hip",      "left_knee",  "left_ankle"),
    "lunge":             ("left_hip",      "left_knee",  "left_ankle"),
    "forward_lunge":     ("left_hip",      "left_knee",  "left_ankle"),
    "arm_raise":         ("left_hip",      "left_shoulder", "left_elbow"),
    "shoulder_abduction":("left_hip",      "left_shoulder", "left_elbow"),
    "bird_dog":          ("left_shoulder", "left_hip",   "left_knee"),
    "quadruped":         ("left_shoulder", "left_hip",   "left_knee"),
    "bridge":            ("left_shoulder", "left_hip",   "left_knee"),
    "hip_bridge":        ("left_shoulder", "left_hip",   "left_knee"),
    "sit_to_stand":      ("left_hip",      "left_knee",  "left_ankle"),
    "step_up":           ("left_hip",      "left_knee",  "left_ankle"),
}

DEFAULT_JOINTS = ("left_hip", "left_knee", "left_ankle")


class RepetitionCounter:
    """
    Counts repetitions by tracking a joint angle over time and
    detecting peaks (top of each rep cycle) using scipy.signal.find_peaks.
    """

    def __init__(self, exercise: str = "squat",
                 peak_distance: int = 15,
                 peak_prominence: float = 8.0):
        self.exercise        = exercise
        self.peak_distance   = peak_distance
        self.peak_prominence = peak_prominence

        joints = EXERCISE_JOINTS.get(exercise, DEFAULT_JOINTS)
        self.joint_a = JOINTS[joints[0]]
        self.joint_b = JOINTS[joints[1]]
        self.joint_c = JOINTS[joints[2]]

        self._angles: list  = []
        self._reps:   int   = 0
        self._phase:  str   = "start"   # start / down / up
        self._last_peaks: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> int:
        """
        frame: (33, 3) normalised skeleton for current frame.
        Returns current rep count.
        """
        angle = calculate_angle(
            frame[self.joint_a],
            frame[self.joint_b],
            frame[self.joint_c],
        )
        self._angles.append(angle)
        self._update_phase(angle)

        # Detect reps once enough data collected
        if len(self._angles) >= self.peak_distance * 2:
            self._detect_reps()

        return self._reps

    def get_phase(self) -> str:
        return self._phase

    def get_angle(self) -> float:
        return self._angles[-1] if self._angles else 0.0

    def reset(self):
        self._angles.clear()
        self._reps       = 0
        self._phase      = "start"
        self._last_peaks = 0

    def set_exercise(self, exercise: str):
        self.exercise = exercise
        joints = EXERCISE_JOINTS.get(exercise, DEFAULT_JOINTS)
        self.joint_a = JOINTS[joints[0]]
        self.joint_b = JOINTS[joints[1]]
        self.joint_c = JOINTS[joints[2]]
        self.reset()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _detect_reps(self):
        arr = np.array(self._angles)
        peaks, _ = find_peaks(
            arr,
            distance=self.peak_distance,
            prominence=self.peak_prominence,
        )
        self._reps = len(peaks)

    def _update_phase(self, angle: float):
        if len(self._angles) < 3:
            return
        recent = self._angles[-3:]
        if recent[-1] < recent[-2] < recent[-3]:
            self._phase = "down"
        elif recent[-1] > recent[-2] > recent[-3]:
            self._phase = "up"
        else:
            self._phase = "hold"
