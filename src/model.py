"""
model.py — Model loading and inference for Qwen3-VL-2B-Instruct.
"""

import time
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


class Qwen3VL:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-2B-Instruct"):
        self.model_id = model_id
        
        # Device detection
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        print(f"--> Target device: [{self.device.upper()}] | Precision: [{self.dtype}]", flush=True)

        print("--> Loading processor...", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )

        print("--> Loading model weights...", flush=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        
        #FIX 1: If not using device_map, move model to device explicitly
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("--> Model loaded successfully!", flush=True)

    def predict(self, image=None, prompt: str | None = None, max_new_tokens: int = 128):
        """Run inference and return (response_text, elapsed_seconds)."""
        # Build messages
        if isinstance(image, (list, tuple)):
            image_list = list(image)
        else:
            image_list = [image] if image is not None else []

        content = [{"type": "image", "image": img} for img in image_list]
        content.append({"type": "text", "text": prompt or ""})
        messages = [{"role": "user", "content": content}]

        #FIX 2: Apply template WITHOUT returning tensors first
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        #FIX 3: Process the text + images together, then move to device
        inputs = self.processor(
            text=text,
            images=image_list if image_list else None,
            return_tensors="pt",
        )

        #FIX 4: Move ALL tensors to the correct device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        start = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        #FIX 5: Decode only the generated portion
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response, round(time.time() - start, 3)

    # Alias for pipeline compatibility
    def generate(self, image=None, prompt: str | None = None, **kwargs):
        return self.predict(image=image, prompt=prompt, **kwargs)


if __name__ == "__main__":
    from PIL import Image
    import requests
    from io import BytesIO

    print("--> Smoke testing Qwen3VL class...", flush=True)
    vlm = Qwen3VL()
    
    url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    img = Image.open(BytesIO(requests.get(url, timeout=10).content))

    out, elapsed = vlm.predict(img, "What animal is shown in this image?")
    print("Response:", repr(out), flush=True)
    print("Elapsed seconds:", elapsed, flush=True)