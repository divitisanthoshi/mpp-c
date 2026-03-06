"""
First-Time Setup Script — Run ONCE before using the system.
Builds: 23-exercise dataset from 4 sources, reference seqs, trained model.
Usage:  python setup.py
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

ALL_EXERCISES = [
    "squat","deep_squat","hurdle_step","inline_lunge","side_lunge",
    "sit_to_stand","standing_leg_raise","shoulder_abduction",
    "shoulder_extension","shoulder_rotation","shoulder_scaption",
    "hip_abduction","trunk_rotation","leg_raise","reach_and_retrieve",
    "wall_pushup","heel_raise","bird_dog","glute_bridge","clamshell",
    "chin_tuck","marching_in_place","step_up",
]

def step(n, msg):
    print(f"\n{'='*65}\n  [{n}] {msg}\n{'='*65}")

def main():
    print("\n"+"="*65)
    print("  REHABILITATION MONITOR — FIRST-TIME SETUP")
    print("  23 Exercises | Custom + NTU + UI-PRMD + KIMORE")
    print("="*65)

    step(1,"Creating directories")
    for d in ([f"data/custom/{e}" for e in ALL_EXERCISES]+
              [f"data/unified/{e}" for e in ALL_EXERCISES]+
              ["data/ntu","data/uiprmd","data/kimore","data/reference","models","logs"]):
        os.makedirs(d, exist_ok=True)
    print(f"  done")

    step(2,"Generating custom synthetic data (23 exercises)")
    from scripts.generate_full_dataset import generate_exercise
    for ex in ALL_EXERCISES:
        n = generate_exercise(ex, f"data/custom/{ex}", n_per_quality=10)
        print(f"  {ex:<28s}  {n} seqs")

    step(3,"Building multi-source unified dataset")
    from scripts.build_dataset import build
    build(argparse.Namespace(synth_per_quality=15))

    step(4,"Generating reference sequences")
    from scripts.build_dataset import GENERATORS
    os.makedirs("data/reference", exist_ok=True)
    for ex in ALL_EXERCISES:
        gen = GENERATORS.get(ex)
        if gen:
            np.save(f"data/reference/{ex}_reference.npy", gen("good"))
            print(f"  {ex}")

    step(5,"Training model")
    from train import train
    train()

    step(6,"Sanity check")
    import tensorflow as tf
    model = tf.keras.models.load_model("models/rehab_model.keras", compile=False)
    preds = model.predict(np.zeros((1,60,99),"float32"), verbose=0)
    print(f"  output shapes: {[list(p.shape) for p in preds]}")
    print(f"  score sample:  {float(preds[0][0][0])*100:.1f}%")

    print("\n"+"="*65)
    print("  SETUP COMPLETE — run: python main.py")
    print("  Add real datasets to data/ntu|uiprmd|kimore then re-run")
    print("  scripts/build_dataset.py + train.py")
    print("="*65+"\n")

if __name__ == "__main__":
    main()
