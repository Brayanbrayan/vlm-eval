"""
model.py — Model loading and inference for Qwen3-VL-2B-Instruct.

Handles device placement (cuda -> mps -> cpu fallback) and exposes a
single run_inference() entry point that pipeline.py calls per sample.
"""

import sys
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def get_device() -> str:
    """Pick the best available device, falling back gracefully."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str) -> torch.dtype:
    """bf16 on CUDA, fp16 on MPS, fp32 on CPU."""
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_model(model_id: str = MODEL_ID):
    """Load model + processor onto the best available device."""
    device = get_device()
    dtype = get_dtype(device)

    print(f"--> Target device: [{device.upper()}] | Precision: [{dtype}]", flush=True)
    
    print("--> Loading processor...", flush=True)
    # 1. Cap max visual pixels to prevent memory spikes in attention layers
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,  # Prevents runaway image token sequence length
    )

    print("--> Loading model weights (using low_cpu_mem_usage)...", flush=True)
    
    # AutoModelForImageTextToText handles Qwen3-VL cleanly
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    
    if device in ["cpu", "mps"]:
        model.to(device)

    model.eval()
    print("--> Model successfully loaded into memory!", flush=True)

    return model, processor, device


def build_messages(images: list, question: str, choices: list[str]) -> list[dict]:
    """Construct the chat-format message list Qwen3-VL expects."""
    content = [{"type": "image", "image": img} for img in images]
    prompt_text = (
        f"{question}\n\n" + "\n".join(choices)
        + "\n\nAnswer with the letter of the correct option only."
    )
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def run_inference(model, processor, device: str, images: list, question: str,
                  choices: list[str], max_new_tokens: int = 128) -> tuple[str, str]:
    """Run one sample through the model. Returns (raw_response, prompt_sent)."""
    messages = build_messages(images, question, choices)

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    prompt_repr = (
        f"{question}\n\n" + "\n".join(choices)
        + "\n\nAnswer with the letter of the correct option only."
    )
    return output_text, prompt_repr


if __name__ == "__main__":
    from PIL import Image
    import requests
    from io import BytesIO

    print("--> Starting smoke test...", flush=True)
    url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    
    try:
        print("--> Fetching demo image...", flush=True)
        img = Image.open(BytesIO(requests.get(url, timeout=10).content))

        model, processor, device = load_model()

        print("--> Running model inference...", flush=True)
        response, prompt = run_inference(
            model, processor, device,
            images=[img],
            question="What animal is shown in this image?",
            choices=["A: cat", "B: dog", "C: rabbit", "D: bird"],
        )
        print("\n=== SMOKE TEST SUCCESS ===", flush=True)
        print("Raw response:", repr(response), flush=True)
        
    except Exception as e:
        print(f"\n[ERROR] Smoke test failed with exception: {e}", file=sys.stderr, flush=True)