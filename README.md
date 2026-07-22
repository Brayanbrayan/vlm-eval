
markdown
# VLM Evaluation Pipeline

## Overview

This pipeline evaluates the Qwen3-VL-2B-Instruct vision-language model on a 100-sample subset of the MMMU (Massive Multi-discipline Multimodal Understanding) benchmark. It demonstrates end-to-end evaluation capabilities including resumability, structured logging, and metrics aggregation.

## Setup

### Prerequisites

- Python 3.10 or higher
- 16GB+ RAM if running on CPU (GPU strongly recommended)
- CUDA GPU or Apple Silicon (MPS) recommended; CPU works but is slow and memory-constrained
- ~4.3GB disk space for model weights

### Installation

```bash
git clone 
cd vlm-eval

curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
```

### Running the Pipeline

```bash
# Full evaluation (100 samples)
uv run python src/main.py

```

## Project Structure

vlm-eval/
├── pyproject.toml
├── uv.lock
├── .python-version
├── src/
│ ├── main.py # entry point
│ ├── model.py # Qwen3-VL-2B-Instruct loading + inference
│ ├── data.py # MMMU subset loader (4 subjects, 25 samples each)
│ ├── parser.py # answer extraction from model responses
│ ├── pipeline.py # evaluation loop with resumability
│ └── metrics.py # metrics aggregation + summary generation
├── results/
│ ├── trajectories.jsonl
│ └── summary.json
└── README.md


## Design Decisions

### Resumability

`trajectories.jsonl` is written in append mode with an explicit flush after every single sample, not held in memory and written at the end. On startup, the pipeline reads any existing file, extracts the set of `sample_id`s already present, and skips them. A sample counts as "done" once it has any record at all, successful or errored, since retrying a sample that fails identically every time (an OOM on a specific image, for example) would stall a resume indefinitely rather than move forward.

Tested by killing the process mid-run and restarting; it resumes from the next unprocessed sample with no re-computation and no duplicate records.

### Model Loading and Device Placement

The model loads via `AutoModelForImageTextToText`, which dispatches to the correct Qwen3-VL architecture from the checkpoint's config rather than hardcoding a model class. Device selection falls back gracefully: CUDA if available, otherwise MPS (Apple Silicon), otherwise CPU. Precision is chosen per device: bfloat16 on CUDA, float16 on MPS, float32 on CPU, since defaulting to float32 everywhere would roughly double the ~4.3GB checkpoint's memory footprint.

`low_cpu_mem_usage=True` reduces peak RAM during weight deserialization by streaming weights in rather than allocating duplicate buffers. The processor is loaded with explicit `min_pixels`/`max_pixels` bounds (`256*28*28` to `1280*28*28`) to cap the number of visual patch tokens Qwen3-VL generates per image; without this bound, high-resolution MMMU images can drive the attention computation during vision prefill to allocate several GB on its own, independent of the model weights themselves.

### Answer Extraction

The parser tries several patterns in order of specificity before falling back to a bare letter search:

"The answer is B" -> B
"I think B is correct" -> B
"Option C" -> C
"B." -> B
"(A)" -> A


If nothing matches, it returns `(None, False)` rather than guessing, and the pipeline logs the failure and moves on. It never crashes on unparseable output.

### Error Handling

Every sample produces exactly one trajectory record, success or failure. Inference failures (OOM, malformed input, etc.) are caught, logged with the exception message in the `error` field, and the loop continues. No sample is ever silently skipped.

## Results

Model: Qwen/Qwen3-VL-2B-Instruct
Total Samples: 100
Completed: 100
Inference Errors: 0
Overall Accuracy: 27.00%
Parse Failure Rate: 41.00%
Avg Inference Time: 18.88s
Total Runtime: 1900.6s

Per-Subject Accuracy:
Art: 52.00%
Biology: 36.00%
Architecture & Engineering: 12.00%
Accounting: 8.00%


### Observations

Art scored highest by a wide margin (52%). Its questions are mostly direct visual recognition ("who painted this," "what is this called"), and the model typically answers with a single letter or a short phrase, leaving little room for parse failure.

Accounting and Architecture/Engineering scored lowest (8% and 12%). These require multi-step numeric reasoning, and the model's responses are long derivations rather than short answers.

### Known Issue: Response Truncation

A meaningful share of the 41% parse failure rate is not the model failing to reach an answer, it's the response being cut off by the `max_new_tokens` generation limit before it states a concluding letter. This shows up directly in the logged responses: several end mid-word (e.g. cut off at "...in the trans" where the model was clearly heading toward "transmembrane," or mid-calculation with no final statement). This pattern concentrates in the same subjects that score lowest, since Qwen answers quantitative, multi-step questions with long chain-of-thought before committing to a letter.


## Issues Encountered

**`torch_dtype` deprecation.** A recent `transformers` release renamed `torch_dtype` to `dtype` in `from_pretrained`. The old parameter name loaded silently with a warning, but caused inference to fail downstream. Fixed by renaming the parameter.

**`processor.apply_chat_template(..., return_dict=True, return_tensors="pt")` returning a dict without `.to()`.** Calling `.to(device)` directly on this return value crashed with `AttributeError: 'dict' object has no attribute 'to'`. Fixed by separating template rendering (`tokenize=False`, text only) from tensor construction (a separate `processor(text=..., images=...)` call), then moving only the tensor values in the resulting dict to the target device individually, leaving any non-tensor entries untouched.


**Cascade effect.** These three issues compounded: the dtype warning meant the model loaded in an unexpected precision, the `.to()` crash meant inputs never reached the correct device, and the resulting device mismatch caused every single `generate()` call to fail. The first full run showed `"inference_errors": 100, "completed": 0`, all 100 samples failing in the try/except boundary in `pipeline.py` rather than crashing the whole loop, which is exactly what that boundary is for. Isolating and fixing each bug in turn resolved the cascade.

**Local machine memory constraints.** The original development machine has 4GB total RAM; loading the ~4.3GB checkpoint in float32 on CPU (the default without an explicit dtype) requires roughly 8.5GB, which the OS killed before it could complete. Moved execution to Google Colab (T4 GPU, 15GB VRAM) while keeping the exact same codebase; nothing in `src/` differs between the local and cloud execution paths, only where it's run.

**Colab session persistence.** Colab's local disk is ephemeral across full session recycles, meaning both the downloaded model cache and `results/trajectories.jsonl` would be lost on a hard disconnect, not just a soft restart. Mounted Google Drive and set `HF_HOME` to a Drive path before loading the model, and cloned the repo onto Drive so `results/` also persists there. This means resumability holds not just within one live session but across a fully dead and restarted one.

## Dependencies

Core packages (pinned versions in `pyproject.toml`):

torch
transformers
accelerate
pillow
datasets
huggingface_hub


## License

This project is for evaluation purposes only. See individual library licenses for dependencies.
