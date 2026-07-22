"""
data.py — MMMU dataset loading and formatting.

Pulls the evaluation subset (first 25 samples from each of 4 subjects'
validation splits) and reshapes each sample into the flat dict shape
pipeline.py expects: sample_id, subject, question, choices,
correct_answer, images.
"""

import ast
import string
from datasets import load_dataset

SUBJECTS = ["Accounting", "Architecture_and_Engineering", "Art", "Biology"]
SAMPLES_PER_SUBJECT = 25
IMAGE_FIELDS = [f"image_{i}" for i in range(1, 8)]  # MMMU: image_1..image_7


def _parse_options(raw_options: str) -> list[str]:
    """MMMU stores options as a stringified Python list, e.g.
    "['3.5', '4.0', '4.5']". literal_eval is used instead of eval()
    since it only parses literals and can't execute arbitrary code."""
    try:
        parsed = ast.literal_eval(raw_options)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return []


def _extract_images(sample: dict) -> list:
    """Collect only the populated image_N fields, in order. The
    datasets library auto-decodes these to PIL.Image on access."""
    return [sample[field] for field in IMAGE_FIELDS if sample.get(field) is not None]


def load_subset(
    subjects: list[str] = SUBJECTS,
    samples_per_subject: int = SAMPLES_PER_SUBJECT,
) -> list[dict]:
    """Load and format the evaluation subset. Returns however many
    samples were actually available if a subject has fewer than
    requested, printing a warning to document that fact."""
    all_samples = []

    for subject in subjects:
        dataset = load_dataset("MMMU/MMMU", subject, split="validation")

        available = len(dataset)
        take = min(samples_per_subject, available)
        if available < samples_per_subject:
            print(
                f"WARNING: {subject} has only {available} validation samples, "
                f"requested {samples_per_subject}. Taking all {available}. "
                f"(document this in README)"
            )

        for i in range(take):
            raw = dataset[i]
            options = _parse_options(raw.get("options", "[]"))
            letters = string.ascii_uppercase[: len(options)]
            choices = [f"{letter}: {opt}" for letter, opt in zip(letters, options)]

            all_samples.append({
                "sample_id": raw["id"],
                "subject": subject,
                "question": raw["question"],
                "choices": choices,
                "correct_answer": raw["answer"],
                "images": _extract_images(raw),
            })

    return all_samples


if __name__ == "__main__":
    # Quick local check — no model needed, just confirms the dataset
    # actually parses into the expected shape before touching inference.
    samples = load_subset(subjects=["Accounting"], samples_per_subject=2)
    for s in samples:
        print(s["sample_id"], s["choices"], "->", s["correct_answer"], f"({len(s['images'])} images)")