"""
Run training on Kaggle (or Colab) with GPU.
- Detects Kaggle/Colab environment and sets data paths.
- Uses GPU if available and increases batch size for speed.
- Run: python run_on_kaggle.py
"""
import os
import sys

# ── Environment detection ─────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/input")
IS_COLAB  = "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ

if IS_KAGGLE:
    # Kaggle: add your dataset in Notebook → Add Data → Your Datasets
    # Example: if dataset is "yourusername/rehab-data", data is at:
    #   /kaggle/input/rehab-data/
    _INPUT = "/kaggle/input"
    _WORK  = "/kaggle/working"
    # List dirs to find your dataset folder name
    _candidates = [d for d in os.listdir(_INPUT) if not d.startswith(".")]
    if _candidates:
        _DATA_ROOT = os.path.join(_INPUT, _candidates[0])
        # Support both: dataset contains "data/unified" or dataset IS "unified" or "custom"
        if os.path.isdir(os.path.join(_DATA_ROOT, "unified")):
            os.environ["REHAB_DATA_DIR"] = os.path.join(_DATA_ROOT, "unified")
        elif os.path.isdir(os.path.join(_DATA_ROOT, "custom")):
            os.environ["REHAB_DATA_DIR"] = os.path.join(_DATA_ROOT, "custom")
        else:
            os.environ["REHAB_DATA_DIR"] = _DATA_ROOT
    else:
        os.environ["REHAB_DATA_DIR"] = ""  # will use synthetic-only
    os.chdir(_WORK)
elif IS_COLAB:
    # Colab: mount Drive or upload data to /content/rehab_data (you create it)
    _WORK = "/content"
    if os.path.isdir("/content/drive/MyDrive"):
        _drive_data = "/content/drive/MyDrive/rehab_project_data"
        if os.path.isdir(_drive_data):
            os.environ["REHAB_DATA_DIR"] = _drive_data
        else:
            os.environ["REHAB_DATA_DIR"] = ""
    else:
        os.environ["REHAB_DATA_DIR"] = os.path.join(_WORK, "rehab_data") if os.path.isdir(os.path.join(_WORK, "rehab_data")) else ""
    os.chdir(_WORK)
else:
    os.environ["REHAB_DATA_DIR"] = ""

# ── Patch train config for GPU (before importing train) ──────────────────────
def _patch_train_for_gpu():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"  GPU detected: {[gpu.name for gpu in gpus]}")
        # Optional: use mixed precision for faster training
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("  Using mixed_float16 for faster training.")
        except Exception:
            pass
    else:
        print("  No GPU found — training will be slow.")

# ── Patch data dir in train.py ────────────────────────────────────────────────
_REHAB_DATA = os.environ.get("REHAB_DATA_DIR", "").strip()
import train as _train_mod
if _REHAB_DATA and os.path.isdir(_REHAB_DATA):
    _train_mod._UNIFIED = _REHAB_DATA
    _train_mod._CUSTOM  = _REHAB_DATA
    _train_mod.DATA_DIR = _REHAB_DATA
    print(f"  Using data directory: {_REHAB_DATA}")
else:
    print("  No external data dir set; will use data/unified, data/custom, or synthetic-only.")

# ── Optional: larger batch size on GPU for speed ──────────────────────────────
import tensorflow as tf
if tf.config.list_physical_devices("GPU"):
    if getattr(_train_mod, "BATCH_SIZE", None) is not None:
        _train_mod.BATCH_SIZE = 32  # or 64 if you have enough VRAM
        print(f"  Batch size set to {_train_mod.BATCH_SIZE} for GPU.")

# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Rehab model — Kaggle/Colab GPU training")
    print("=" * 60)
    _patch_train_for_gpu()
    print()
    _train_mod.train()
