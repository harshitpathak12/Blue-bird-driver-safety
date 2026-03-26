"""
Qwen-based OCR and validation for driving license crops.

Optimized for production (CPU): smaller image, fewer tokens, optional dynamic
quantization, and multi-threading. GPU: FP16/BF16, Flash Attention 2 when available.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from utils.logger import get_logger

_qlog = get_logger(__name__)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

_processor: AutoProcessor | None = None
_model: Qwen2_5_VLForConditionalGeneration | None = None
_device: str = "cpu"

# Production: smaller image and fewer tokens for lower latency
QWEN_MAX_SIZE = int(os.environ.get("QWEN_MAX_SIZE", "448"))
QWEN_MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "256"))
QWEN_QUANTIZE_CPU = os.environ.get("QWEN_QUANTIZE_CPU", "1").strip().lower() in ("1", "true", "yes")
# Dedicated thread count for Qwen inference (ignores OMP_NUM_THREADS which may be 1)
QWEN_CPU_THREADS = int(os.environ.get("QWEN_CPU_THREADS", "0"))

# Do NOT use angle-bracket placeholders — the model copies them literally.
_PROMPT = (
    "Read every line of text visible on this driving licence image.\n"
    "Output ONE JSON object only, no markdown, no explanation.\n"
    "Fields: text (full transcription as one string), is_valid (valid or invalid or unknown), "
    "confidence (0 to 1 number), reason (one short phrase).\n"
    "The text field must contain the real words from the image, not instructions or examples."
)


def is_placeholder_ocr_text(text: str) -> bool:
    """True if the model echoed a template instead of real OCR."""
    t = (text or "").strip()
    if not t or len(t) < 4:
        return True
    tl = t.lower()
    if "all visible text" in tl and len(t) < 80:
        return True
    if "brief reason" in tl and len(t) < 80:
        return True
    if tl in ("valid", "invalid", "unknown", "...", "n/a", "none"):
        return True
    if t.startswith("<") and ">" in t:
        return True
    if all(c in ".…_-– " for c in t):
        return True
    return False


@dataclass
class QwenOCRResult:
    text: str
    is_valid: str
    confidence: float
    reason: str


def _select_dtype() -> torch.dtype:
    """Pick the best dtype for the available GPU."""
    if not torch.cuda.is_available():
        return torch.float32
    cap = torch.cuda.get_device_capability()
    if cap >= (8, 0):
        return torch.bfloat16
    return torch.float16


def _load_qwen():
    """Lazy-load Qwen model and processor; CPU: optional quantization + threads."""
    global _processor, _model, _device
    if _processor is not None and _model is not None:
        return _processor, _model

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _select_dtype()

    # Use multiple threads for CPU inference. OMP_NUM_THREADS is often set to 1 by
    # runtimes (conda, systemd, etc.), so we use a dedicated QWEN_CPU_THREADS env var
    # and fall back to half of available cores for good throughput without starving YOLO.
    if _device == "cpu":
        n_threads = QWEN_CPU_THREADS or max(2, (os.cpu_count() or 4) // 2)
        torch.set_num_threads(n_threads)
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
        _qlog.info("Qwen CPU threads: %d (cores=%s)", n_threads, os.cpu_count())

    _qlog.info(
        "Loading %s on %s (dtype=%s, max_size=%s, max_tokens=%s)",
        MODEL_ID,
        _device,
        dtype,
        QWEN_MAX_SIZE,
        QWEN_MAX_NEW_TOKENS,
    )
    t0 = time.perf_counter()

    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
    }
    if _device == "cuda":
        model_kwargs["device_map"] = "auto"
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            _qlog.info("Using Flash Attention 2")
        except ImportError:
            _qlog.info("flash_attn not installed, using default attention")

    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, **model_kwargs
    )
    _model.eval()

    if _device == "cpu" and QWEN_QUANTIZE_CPU:
        try:
            _model = torch.ao.quantization.quantize_dynamic(
                _model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
            )
            _qlog.info("Applied dynamic quantization (int8) for CPU")
        except Exception as e:
            _qlog.warning("Quantization skipped: %s", e)

    if _device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    _qlog.info("Qwen model loaded in %.1fs", dt)

    return _processor, _model


def ensure_qwen_loaded() -> None:
    """Load Qwen processor + weights (for server startup preload)."""
    _load_qwen()


def qwen_ocr_and_validate(image_bgr: np.ndarray, max_size: int | None = None) -> QwenOCRResult:
    """Run Qwen OCR + validation on a license crop (BGR np.ndarray)."""
    if image_bgr is None or image_bgr.size == 0:
        return QwenOCRResult(text="", is_valid="unknown", confidence=0.0, reason="empty image")

    if max_size is None:
        max_size = QWEN_MAX_SIZE
    h, w = image_bgr.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)

    processor, model = _load_qwen()

    messages: list[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image", "image": pil_img},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = None, None
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
    except Exception:
        inputs = processor(
            images=[pil_img],
            text=[text_input],
            return_tensors="pt",
        )

    inputs = inputs.to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=QWEN_MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    json_str = generated_text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", json_str, flags=re.DOTALL)
    if fence:
        json_str = fence.group(1)
    else:
        bare = re.search(r"\{.*\}", json_str, flags=re.DOTALL)
        if bare:
            json_str = bare.group(0)

    data = None
    try:
        data = json.loads(json_str)
    except Exception:
        # JSON may be truncated (token limit). Try to extract the "text" field value
        # even from partial JSON like: {"text": "Indian Union Driving Licence...(cut off)
        text_match = re.search(
            r'"text"\s*:\s*"((?:[^"\\]|\\.)*)',
            json_str,
            flags=re.DOTALL,
        )
        if text_match:
            partial_text = text_match.group(1)
            partial_text = partial_text.replace("\\n", "\n").replace('\\"', '"')
            data = {"text": partial_text, "is_valid": "unknown", "confidence": 0.0, "reason": "partial OCR (truncated)"}
        else:
            raw = generated_text.strip()
            if len(raw) > 10 and not raw.startswith("{"):
                data = {"text": raw, "is_valid": "unknown", "confidence": 0.0, "reason": "raw OCR output"}
            else:
                return QwenOCRResult(
                    text=generated_text.strip(),
                    is_valid="unknown",
                    confidence=0.0,
                    reason="could not parse JSON from model output",
                )

    text = str(data.get("text", "")).strip()
    if is_placeholder_ocr_text(text):
        return QwenOCRResult(
            text="",
            is_valid="unknown",
            confidence=0.0,
            reason="model returned placeholder; retry or adjust image",
        )
    is_valid = str(data.get("is_valid", "unknown")).lower()
    if is_valid not in {"valid", "invalid", "unknown"}:
        is_valid = "unknown"
    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(data.get("reason", "")).strip()

    return QwenOCRResult(text=text, is_valid=is_valid, confidence=confidence, reason=reason)
