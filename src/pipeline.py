"""
pipeline.py — Evaluation loop with resumability.

Reads any existing trajectories.jsonl on startup to determine which
sample_ids already have a record. Every sample gets a record written
and flushed immediately after its attempt, success or failure, so
killing the process and restarting picks up exactly where it left
off with no lost state and no silent skips.
"""

import json
import time
from pathlib import Path

from parser import extract_answer


def load_completed_ids(trajectories_path: Path) -> set[str]:
    """A sample counts as 'complete' if it has any record, successful
    or errored. Retrying a sample that fails identically every time
    would stall a resume indefinitely, so a logged attempt counts as
    done regardless of outcome."""
    completed = set()
    if not trajectories_path.exists():
        return completed

    with open(trajectories_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                completed.add(record["sample_id"])
            except json.JSONDecodeError:
                continue  # partial line from a killed process, retry it

    return completed


def _build_prompt(question: str, choices: list[str]) -> str:
    return (
        f"{question}\n\n" + "\n".join(choices)
        + "\n\nAnswer with the letter of the correct option only."
    )


def run_pipeline(samples: list[dict], model, trajectories_path: Path) -> None:
    """samples: list of dicts with sample_id, subject, question, choices,
    correct_answer, images. model: an already-loaded Qwen3VL instance."""

    completed_ids = load_completed_ids(trajectories_path)
    print(f"Found {len(completed_ids)} completed samples, resuming.", flush=True)

    trajectories_path.parent.mkdir(parents=True, exist_ok=True)

    with open(trajectories_path, "a") as f:
        for sample in samples:
            sample_id = sample["sample_id"]
            if sample_id in completed_ids:
                continue

            prompt_sent = _build_prompt(sample["question"], sample["choices"])

            record = {
                "sample_id": sample_id,
                "subject": sample["subject"],
                "question": sample["question"],
                "choices": sample["choices"],
                "correct_answer": sample["correct_answer"],
                "prompt_sent": prompt_sent,
                "raw_model_response": None,
                "extracted_answer": None,
                "extraction_succeeded": False,
                "is_correct": False,
                "inference_time_seconds": None,
                "error": None,
            }

            start = time.time()
            try:
                raw_response, inference_time = model.generate(
                    image=sample["images"],
                    prompt=prompt_sent,
                )
                record["raw_model_response"] = raw_response
                record["inference_time_seconds"] = round(inference_time, 3)

                extracted, succeeded = extract_answer(raw_response)
                record["extracted_answer"] = extracted
                record["extraction_succeeded"] = succeeded
                record["is_correct"] = succeeded and extracted == sample["correct_answer"]

            except Exception as e:
                record["inference_time_seconds"] = round(time.time() - start, 3)
                record["error"] = str(e)

            f.write(json.dumps(record) + "\n")
            f.flush()

            status = "OK" if record["is_correct"] else (
                "ERROR" if record["error"] else "WRONG/UNPARSED"
            )
            print(f"[{sample_id}] {status} ({record['inference_time_seconds']}s)", flush=True)