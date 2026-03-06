# Train on Kaggle or Google Colab (free GPU)

Training on your PC can take **many hours** (e.g. 50 epochs × 2707 steps × ~170 ms/step ≈ 6+ hours per run) because it’s likely using the **CPU**. On **Kaggle** or **Google Colab** you get a **free GPU**, which can make the same run **roughly 10–20× faster** (often under an hour).

---

## Option 1: Kaggle (recommended)

**Free:** 30 hours of GPU per week (P100 or T4).

### Step 1: Upload your data as a Kaggle dataset

1. Go to [kaggle.com](https://www.kaggle.com) and sign in.
2. **Datasets** → **New Dataset**.
3. Upload your project folder (or at least the `data` folder so that Kaggle has `data/unified/` or `data/custom/` with the exercise subfolders and `.npy` files).
4. Create the dataset (e.g. name: `rehab-exercise-data`).

### Step 2: Create a Notebook with GPU

1. **Code** → **New Notebook**.
2. In the right sidebar: **Settings** → **Accelerator** → **GPU** (P100 or T4).
3. **Add your dataset:** **+ Add Data** → **Your Datasets** → select the dataset you created.

### Step 3: Add project code and run

**Option A – Upload project as a second dataset**

1. Zip your whole `rehab_project` folder (with `train.py`, `run_on_kaggle.py`, `src/`, etc.).
2. Create another Kaggle dataset from that zip (e.g. `rehab-project-code`).
3. In the notebook, add this dataset too.
4. In the notebook, run:

```python
# Copy project into working dir and go there
!cp -r /kaggle/input/rehab-project-code/* /kaggle/working/
%cd /kaggle/working

# Install deps (Kaggle has most; add any missing)
!pip install -q fastdtw pyttsx3

# Run training (uses GPU and your data from the first dataset)
!python run_on_kaggle.py
```

**Option B – Only data on Kaggle, code from GitHub/local**

1. In the notebook, clone or upload your repo (e.g. clone from GitHub or upload `train.py`, `run_on_kaggle.py`, and the `src/` folder).
2. Add your **data** dataset in the notebook (Step 2 above).
3. Run:

```python
%cd /kaggle/working   # or wherever your code is
!pip install -q fastdtw pyttsx3
!python run_on_kaggle.py
```

`run_on_kaggle.py` will use the first input directory under `/kaggle/input/` as the data root (and will look for `unified` or `custom` inside it, or use the folder directly if it contains the exercise subfolders).

### Step 4: Download the trained model

- After training, the model is in `/kaggle/working/models/` (e.g. `rehab_model.keras` and `meta.json`).
- In the notebook: **Output** → **Save Version** (or **Save & Run All**), then download the output from the version.
- Or in the notebook: **File** → **Download** to get the working directory including `models/`.

---

## Option 2: Google Colab

**Free:** GPU runtime (T4) with usage limits; good for a single long run.

### Step 1: Open Colab and turn on GPU

1. Go to [colab.research.google.com](https://colab.research.google.com).
2. **File** → **New notebook**.
3. **Runtime** → **Change runtime type** → **Hardware accelerator** → **GPU** → **Save**.

### Step 2: Get your code and data into Colab

**Option A – From Google Drive**

1. Upload your full `rehab_project` folder to Google Drive (e.g. `MyDrive/rehab_project`).
2. In the first cell:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/rehab_project
```

3. Put your data under `rehab_project/data/unified/` (or `custom/`) on Drive, or at `MyDrive/rehab_project_data/` and adjust `run_on_kaggle.py` if needed.

**Option B – Upload from your PC**

1. Run in a cell: `from google.colab import files; files.upload()` and upload your project zip.
2. `!unzip -q your_project.zip && cd rehab_project` (or the folder name inside the zip).

### Step 3: Install and run

```python
!pip install -q fastdtw pyttsx3
!python run_on_kaggle.py
```

### Step 4: Download the model

```python
from google.colab import files
files.download("models/rehab_model.keras")
files.download("models/meta.json")
```

---

## Why it’s faster on Kaggle/Colab

| Environment | Typical device | Speed (approx) |
|-------------|----------------|-----------------|
| Your PC (no GPU) | CPU | ~170 ms/step → many hours for 50 epochs |
| Kaggle / Colab | GPU (P100, T4) | ~10–30 ms/step → often &lt; 1 hour |

`run_on_kaggle.py` enables GPU, uses mixed precision when possible, and increases batch size on GPU so the same 50-epoch run finishes much sooner.

---

## If you don’t use the runner script

- **Kaggle:** Set **Accelerator → GPU** and run `python train.py`. TensorFlow will use the GPU automatically. Data path: put your data in the dataset and in code set `DATA_DIR` to `/kaggle/input/<your-dataset-name>/` (and add `unified` or `custom` if needed).
- **Colab:** Set **Runtime → GPU**, then run `train.py`; paths can stay relative if you `%cd` into the project folder. For data on Drive, point `DATA_DIR` to the mounted path (e.g. `/content/drive/MyDrive/rehab_project_data`).

Using **Kaggle** or **Colab** with the steps above will move training off your PC and cut total time from hours to well under an hour in most cases.
