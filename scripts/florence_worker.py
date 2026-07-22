"""Persistent Florence-2 inference worker with a line-delimited local protocol."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

PROTOCOL_PREFIX = "MOTO_LAPS_JSON:"


def respond(payload: dict[str, Any]) -> None:
    print(PROTOCOL_PREFIX + json.dumps(payload, ensure_ascii=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from transformers import (  # type: ignore[import-untyped]
            AutoModelForCausalLM,
            AutoProcessor,
        )

        if torch.cuda.is_available():
            device = "cuda:0"
            dtype = torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32
        processor = AutoProcessor.from_pretrained(
            args.model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        ).to(device)
        model.eval()
    except Exception as exc:
        respond({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 1
    respond({"status": "ready", "device": device})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                return 0
            encoded = base64.b64decode(request["image_jpeg"], validate=True)
            array = np.frombuffer(encoded, dtype=np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("JPEG payload could not be decoded")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = processor(
                text="<OCR>",
                images=Image.fromarray(rgb),
                return_tensors="pt",
            ).to(device, dtype)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=32,
                    do_sample=False,
                    num_beams=1,
                )
            raw = processor.batch_decode(generated, skip_special_tokens=False)[0]
            respond({"status": "result", "raw_text": raw})
        except Exception as exc:
            respond({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
