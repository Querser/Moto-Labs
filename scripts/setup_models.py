"""Download and verify the exact local OCR/VLM model snapshots."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

FLORENCE_REPOSITORY = "microsoft/Florence-2-base-ft"
FLORENCE_REVISION = "f6c1a25888ffc1d945ee8a1a77ac833c7303d46e"
FLORENCE_FILES = (
    "LICENSE",
    "README.md",
    "config.json",
    "configuration_florence2.py",
    "model.safetensors",
    "modeling_florence2.py",
    "preprocessor_config.json",
    "processing_florence2.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_ppocr(project_root: Path) -> Path:
    cache_dir = project_root / "models" / "paddlex"
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_dir)
    model_dir = cache_dir / "official_models" / "PP-OCRv6_medium_rec_onnx"
    detector_dir = cache_dir / "official_models" / "PP-OCRv5_mobile_det_onnx"
    if not (model_dir / "inference.onnx").is_file() or not (
        detector_dir / "inference.onnx"
    ).is_file():
        from app.vision.ocr import PaddleOcrV6DigitEngine

        engine = PaddleOcrV6DigitEngine(cache_dir=cache_dir)
        engine.close()
    model = model_dir / "inference.onnx"
    if not model.is_file():
        raise RuntimeError("PP-OCRv6 model download did not create inference.onnx")
    print(f"PP-OCRv6: {model} sha256={sha256(model)}")
    detector_model = detector_dir / "inference.onnx"
    if not detector_model.is_file():
        raise RuntimeError("PP-OCRv5 text detector download did not create inference.onnx")
    print(f"PP-OCRv5 detector: {detector_model} sha256={sha256(detector_model)}")
    return model


def prepare_florence(project_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    model_dir = project_root / "models" / "florence-2-base-ft"
    required = model_dir / "model.safetensors"
    if not required.is_file():
        snapshot_download(
            repo_id=FLORENCE_REPOSITORY,
            revision=FLORENCE_REVISION,
            local_dir=model_dir,
            allow_patterns=list(FLORENCE_FILES),
        )
    if not required.is_file():
        raise RuntimeError("Florence-2 download did not create model.safetensors")
    print(f"Florence-2: {required} sha256={sha256(required)}")
    return required


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-florence", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    prepare_ppocr(project_root)
    if not args.skip_florence:
        prepare_florence(project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
