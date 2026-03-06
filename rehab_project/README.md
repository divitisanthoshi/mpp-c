# Automatic Quality Grading of Rehabilitation Exercise
## Skeleton-Based Deep Learning System

**Project by:**
- Diviti Santhoshi – 160122771003
- Y Bhavanishwarya – 160122771025
- G Vineeth – 160122771042

**Guide:** Ms. Anjum Nabi Sheikh

---

## System Overview

```
Webcam
  │
  ▼
MediaPipe BlazePose  ──►  33 keypoints (x, y, z)
  │
  ▼
Skeleton Normalisation  ──►  Hip-centred, torso-scaled
  │
  ▼
Frame Buffer (60 frames)
  │
  ├──► ST-GCN++ Model  ──►  Quality Score (0–100)
  │        │
  │        └──► Exercise Classification
  │
  ├──► DTW Similarity  ──►  Reference Comparison Score
  │
  ├──► Peak Detection  ──►  Repetition Count
  │
  └──► Angle Rules     ──►  Error Feedback
                                │
                                ▼
                         OpenCV Real-Time UI
```

---

## Installation

### Requirements
- Python 3.8 or newer
- Webcam
- GPU optional (CPU works fine for inference)

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## First-Time Setup (Run Once)

```bash
python setup.py
```

This will:
1. Create all necessary directories
2. Generate reference skeleton sequences
3. Train the ST-GCN++ model
4. Verify the system is working

---

## Running the System

```bash
python main.py
```

### Controls
| Key | Action |
|-----|--------|
| `1`–`6` | Select exercise in menu |
| `Enter` | Start selected exercise |
| `M` | Return to menu |
| `R` | Reset current session |
| `Q` | Quit |

---

## Recording Your Own Data (Improves Accuracy)

```bash
# Record a good-form squat (score=90, 5 reps)
python scripts/record_data.py --exercise squat --score 90 --reps 5

# Record a bad-form squat (score=40, 5 reps)
python scripts/record_data.py --exercise squat --score 40 --reps 5
```

After recording 10–20 sequences per exercise, retrain:

```bash
python train.py
```

---

## Supported Exercises

| Key | Exercise |
|-----|----------|
| 1 | Squat |
| 2 | Deep Squat |
| 3 | Lunge |
| 4 | Arm Raise |
| 5 | Bird Dog |
| 6 | Hip Bridge |

---

## UI Description

```
┌──────────────────────────────────────────┬─────────────────┐
│  Exercise: Squat          Phase: DOWN    │  QUALITY SCORE  │
│                                          │     87%         │
│   [Live webcam + skeleton overlay]       │  ████████░░░░   │
│                                          │  REPETITIONS    │
│   RED circles = incorrect joints         │     6/10        │
│                                          │  ███████░░░░░   │
│                                          │  FEEDBACK       │
│                                          │  ✓ Good form!   │
├──────────────────────────────────────────┴─────────────────┤
│  M=Menu   Q=Quit   R=Reset        RED circles=bad joints   │
└────────────────────────────────────────────────────────────┘
```

---

## Model Architecture

```
Input: (60 frames × 99 features)
  │
Conv1D × 3  (128 → 64 filters)
  │
BatchNorm + Dropout
  │
BiLSTM (64 units) → BiLSTM (32 units)
  │
Joint Attention Layer
  │
Dense (64)
  │
  ├── Score Head    → Dense(1)  sigmoid  → Quality 0–100%
  ├── Exercise Head → Dense(6)  softmax  → Exercise class
  └── Phase Head    → Dense(4)  softmax  → down/up/hold/start
```

---

## Scoring Formula

```
Final Score = 0.5 × (Model Score) + 0.5 × (DTW Similarity)
            − 5 × (number of detected errors)
```

Smoothed over a 12-frame rolling window to prevent flickering.

---

## Project Structure

```
rehab_project/
├── main.py                    ← Run this to start the system
├── train.py                   ← Train the model
├── setup.py                   ← First-time setup
├── requirements.txt
├── src/
│   ├── models/
│   │   └── st_gcn.py          ← ST-GCN++ architecture
│   └── utils/
│       ├── preprocessing.py   ← Normalisation, angles, errors
│       └── repetition_counter.py  ← Peak-detection rep counter
├── scripts/
│   ├── record_data.py         ← Record custom exercise data
│   └── generate_references.py ← Generate reference sequences
├── data/
│   ├── custom/                ← Your recorded sequences (.npy)
│   └── reference/             ← Reference sequences per exercise
└── models/
    ├── rehab_model.keras      ← Trained model
    └── meta.json              ← Exercise labels
```
