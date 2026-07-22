# VLM Evaluation Pipeline

## Overview

This pipeline evaluates the Qwen3-VL-2B-Instruct vision-language model on a 100-sample subset of the MMMU (Massive Multi-discipline Multimodal Understanding) benchmark. It demonstrates end-to-end evaluation capabilities including resumability, structured logging, and metrics aggregation.

## Setup

### Prerequisites

- Python 3.10 or higher
- 16GB+ RAM (8GB minimum)
- GPU recommended (CUDA or MPS), but CPU works (slower)
- ~4GB disk space for model weights

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd vlm-eval

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Running the Pipeline

```bash
# Run full evaluation (100 samples)
uv run python src/main.py

# Run smoke test (single sample with demo image)
uv run python src/model.py

# Quick data load test
uv run python src/data.py
```

## Project Structure

```
vlm-eval/
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Locked dependencies for reproducibility
├── .python-version         # Python version (3.10)
├── src/
│   ├── main.py             # Entry point
│   ├── model.py            # Qwen3-VL-2B-Instruct loader + inference
│   ├── data.py             # MMMU subset loader (4 subjects, 25 samples each)
│   ├── parser.py           # Answer extraction from model responses
│   ├── pipeline.py         # Evaluation loop with resumability
│   └── metrics.py          # Metrics aggregation + summary generation
├── results/
│   ├── trajectories.jsonl  # Per-sample structured logs (append-only)
│   └── summary.json        # Aggregate metrics
└── README.md
```

## Design Decisions

### Resumability

The pipeline writes trajectories to `trajectories.jsonl` **immediately after each sample completes**. If interrupted:

1. On restart, it reads existing trajectories and extracts completed `sample_id`s
2. Skips any sample already logged (successful or errored)
3. Continues from the next unprocessed sample

To test: Run the pipeline, interrupt it (Ctrl+C), then run it again. It will resume from where it left off without re-running completed samples.

### Error Handling

- **Every sample produces a trajectory record**, even on error
- No silent skips — all 100 samples are accounted for
- Errors are logged with full traceback in the `error` field
- The pipeline continues processing after non-fatal errors

### Answer Extraction

The parser handles common Qwen response patterns:

```
"The answer is B" → "B"
"I think B is correct because..." → "B"
"Option C" → "C"
"B." → "B"
"B: The answer is..." → "B"
```

If extraction fails, the record includes `"extraction_succeeded": false` and continues.

### Memory Management

The model loads with:
- **4-bit quantization** (if `bitsandbytes` is available)
- **Device mapping**: CUDA → MPS → CPU fallback
- **`low_cpu_mem_usage=True`** to reduce peak memory during loading
- **`max_pixels=1280*28*28`** to prevent CUDA OOM on high-res images

## Results

### Summary Metrics

```
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
```

### Observations

**Best Performance: Art (52%)**
- Visual recognition tasks (identifying paintings, artists, styles)
- Questions are typically straightforward: "Who painted X?"
- Model outputs a single letter consistently

**Worst Performance: Accounting (8%)**
- Multi-step reasoning required
- Domain-specific knowledge (formulas, financial data)
- Questions are longer and more complex

**Parse Failures (41%)**
- Qwen often outputs full explanations instead of just the letter
- Common failure: "To determine the correct answer, we need to..." without including the letter
- This is expected behavior for instruction-tuned models

### Example Trajectory

```json
{
  "sample_id": "validation_Accounting_11",
  "subject": "Accounting",
  "question": "Donna Donie, CFA, has a client who believes...",
  "choices": ["A: $9.00", "B: $5.00", "C: the Maximum possible loss is unlimited"],
  "correct_answer": "A",
  "prompt_sent": "Donna Donie, CFA, has a client... Answer with the letter...",
  "raw_model_response": "To calculate the maximum possible loss...",
  "extracted_answer": "A",
  "extraction_succeeded": true,
  "is_correct": true,
  "inference_time_seconds": 14.178,
  "error": null
}
```

## Issues Encountered

### 1. Model Loading

**Issue**: First run downloads ~4GB of model weights.

**Resolution**: This is expected. The model is cached in `~/.cache/huggingface/` after first download.

### 2. CUDA Memory

**Issue**: Some high-resolution images (up to `max_pixels`) cause CUDA out of memory.

**Resolution**: Set `max_pixels=1280*28*28` in the processor. This balances image quality with memory constraints.

### 3. Long Inference Times

**Issue**: Some samples take 20-40 seconds on a 4GB GPU.

**Resolution**: This is expected for a 2B parameter model on limited hardware. The pipeline logs per-sample timing for analysis.

### 4. Parse Failures

**Issue**: 41% of responses couldn't be parsed.

**Resolution**: This is a model behavior, not a pipeline bug. The parser is permissive and logs failures. In production, you might use a more aggressive regex or a secondary model to extract answers.

## Dependencies

Key packages:

```
torch>=2.5.0
transformers>=4.46.0
qwen-vl-utils>=0.0.8
datasets>=3.0.0
pillow>=10.0.0
tqdm>=4.66.0
```

Full dependencies are pinned in `pyproject.toml`.

## Notes

- The pipeline is designed to be **swappable** — replace the model ID in `model.py` and the benchmark in `data.py`
- Results are saved in `results/` — trajectories are append-only, summary is overwritten
- The pipeline is **deterministic** with fixed random seeds

## Future Improvements

1. **Batch inference**: Process multiple samples per GPU pass for speed
2. **Quantization**: Add 4-bit or 8-bit quantization options
3. **Prompt engineering**: Experiment with chain-of-thought vs. direct answer prompts
4. **Model swapping**: Add support for other VLMs (Llama-3.2-Vision, Phi-3.5-Vision)
5. **Visualization**: Generate confusion matrices and error analysis plots

## License

This project is for evaluation purposes only. See individual library licenses for dependencies.
