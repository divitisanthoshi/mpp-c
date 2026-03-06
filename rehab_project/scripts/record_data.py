"""
Record Custom Exercise Data
---------------------------
Records webcam sessions and saves skeleton sequences as .npy files.

Usage:
    python scripts/record_data.py --exercise squat --score 90 --reps 5
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
import time

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

SEQUENCE_LENGTH = 60
EXERCISES = ["squat", "deep_squat", "lunge", "arm_raise", "bird_dog", "bridge"]


def record(exercise: str, score: int, reps: int):
    save_dir = os.path.join("data", "custom", exercise)
    os.makedirs(save_dir, exist_ok=True)

    # Determine next sequence index
    existing = [f for f in os.listdir(save_dir) if f.endswith(".npy")]
    seq_idx = len(existing)

    cap = cv2.VideoCapture(0)
    frames = []
    recording = False
    countdown = 3

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        print(f"\nRecording exercise: {exercise.upper()}")
        print("Press SPACE to start/stop recording. Press Q to quit.\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

                if recording:
                    lm = result.pose_landmarks.landmark
                    arr = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
                    frames.append(arr)

            # UI overlay
            status = "● REC" if recording else "○ READY"
            color  = (0, 0, 255) if recording else (0, 200, 0)
            cv2.putText(frame, f"{exercise.upper()}  {status}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Frames: {len(frames)}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            cv2.putText(frame, "SPACE=Start/Stop  Q=Quit", (20, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            cv2.imshow("Record Exercise Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if not recording:
                    frames.clear()
                    recording = True
                    print("  Recording started…")
                else:
                    recording = False
                    # Save
                    if len(frames) >= 20:
                        arr = np.array(frames, dtype=np.float32)
                        fname = f"seq_{seq_idx:03d}.npy"
                        np.save(os.path.join(save_dir, fname), arr)

                        # Append to labels.csv
                        csv_path = os.path.join(save_dir, "labels.csv")
                        write_header = not os.path.isfile(csv_path)
                        with open(csv_path, "a") as f:
                            if write_header:
                                f.write("sequence,score,reps\n")
                            f.write(f"{fname},{score},{reps}\n")

                        print(f"  ✅ Saved {fname}  (frames={len(frames)}, score={score}, reps={reps})")
                        seq_idx += 1
                    else:
                        print("  ⚠️  Too short — not saved.")

            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", default="squat", choices=EXERCISES)
    parser.add_argument("--score",    type=int, default=80,
                        help="Quality score (0-100) for this recording")
    parser.add_argument("--reps",     type=int, default=5)
    args = parser.parse_args()
    record(args.exercise, args.score, args.reps)
