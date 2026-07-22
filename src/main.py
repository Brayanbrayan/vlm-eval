"""
main.py — Entry point. Wires data loading, model, pipeline, and
metrics into a single end-to-end run.

Run with: uv run python src/main.py
"""

import time
from pathlib import Path

from data import load_subset
from model import Qwen3VL
from pipeline import run_pipeline
from metrics import compute_summary, save_summary

TRAJECTORIES_PATH = Path("results/trajectories.jsonl")
SUMMARY_PATH = Path("results/summary.json")



def main():
    print("Hello from vlm-eval!")
    run_start = time.time()

    print("Loading MMMU subset...", flush=True)
    samples = load_subset()
    total_expected = len(samples)  # reflects actual availability, not a hardcoded 100
    print(f"Loaded {total_expected} samples.", flush=True)

    print("Loading model...", flush=True)
    vlm = Qwen3VL()
    print(f"Model loaded on device: {vlm.device}", flush=True)

    run_pipeline(samples, vlm, TRAJECTORIES_PATH)

    total_runtime = time.time() - run_start
    summary = compute_summary(
        trajectories_path=TRAJECTORIES_PATH,
        model_id=vlm.model_id,
        total_expected=total_expected,
        total_runtime_seconds=total_runtime,
    )
    save_summary(summary, SUMMARY_PATH)



if __name__ == "__main__":
    main()

