"""
metrics.py — Aggregate trajectories.jsonl into summary.json.

Terminology note: "completed" here means a sample ran inference
without raising (error is None), matching the spec's own example
where completed + inference_errors == total_samples. Accuracy,
parse_failure_rate, and avg_inference_time are all computed over
this non-errored subset, since an errored sample never produced a
response to parse or a real inference time to average.
"""

import json
from pathlib import Path
from collections import defaultdict


def load_trajectories(trajectories_path: Path) -> list[dict]:
    records = []
    with open(trajectories_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def compute_summary(
    trajectories_path: Path,
    model_id: str,
    total_expected: int,
    total_runtime_seconds: float,
) -> dict:
    records = load_trajectories(trajectories_path)

    completed_records = [r for r in records if not r.get("error")]
    inference_errors = len(records) - len(completed_records)
    completed = len(completed_records)

    correct = sum(1 for r in completed_records if r.get("is_correct"))
    overall_accuracy = correct / completed if completed else 0.0

    per_subject_correct = defaultdict(int)
    per_subject_total = defaultdict(int)
    for r in completed_records:
        subject = r["subject"]
        per_subject_total[subject] += 1
        if r.get("is_correct"):
            per_subject_correct[subject] += 1

    per_subject_accuracy = {
        subject: per_subject_correct[subject] / per_subject_total[subject]
        for subject in per_subject_total
    }

    parse_failures = sum(1 for r in completed_records if not r.get("extraction_succeeded"))
    parse_failure_rate = parse_failures / completed if completed else 0.0

    inference_times = [
        r["inference_time_seconds"] for r in completed_records
        if r.get("inference_time_seconds") is not None
    ]
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0

    return {
        "model": model_id,
        "total_samples": total_expected,
        "completed": completed,
        "overall_accuracy": round(overall_accuracy, 4),
        "per_subject_accuracy": {k: round(v, 4) for k, v in per_subject_accuracy.items()},
        "parse_failure_rate": round(parse_failure_rate, 4),
        "inference_errors": inference_errors,
        "avg_inference_time_seconds": round(avg_inference_time, 3),
        "total_runtime_seconds": round(total_runtime_seconds, 1),
    }


def save_summary(summary: dict, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))