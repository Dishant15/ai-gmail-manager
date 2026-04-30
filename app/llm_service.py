"""
app/llm_service.py - Local Mistral LLM via HuggingFace Transformers + LangChain.

Device support:
  - "mps"  : Apple Silicon (M1/M2/M3/M4) via Metal Performance Shaders  ← recommended
  - "cuda" : NVIDIA GPU with 4-bit quantization via bitsandbytes
  - "cpu"  : Fallback, slow but always works
"""
from __future__ import annotations

from typing import Optional

import torch
from langchain_huggingface import HuggingFacePipeline
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from app.config import settings

_llm: Optional[HuggingFacePipeline] = None


def _detect_device() -> str:
    """
    Resolve the effective device to use for model loading.

    Priority:
      1. If HF_DEVICE is explicitly set to "mps", "cpu", or "cuda" — honour it.
      2. If HF_DEVICE="auto":
           - CUDA available  → "cuda"
           - MPS available   → "mps"   (Apple Silicon)
           - Otherwise       → "cpu"

    Note: "auto" in HuggingFace's device_map sense works well for CUDA but
    silently falls back to CPU on Apple Silicon, bypassing MPS entirely.
    This function fixes that by resolving "auto" ourselves.
    """
    requested = settings.hf_device.lower()

    if requested != "auto":
        return requested  # trust explicit config

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        logger.info("Apple Silicon detected — using MPS backend.")
        return "mps"
    return "cpu"


def _build_bnb_config() -> Optional[BitsAndBytesConfig]:
    """
    Return a 4-bit quantization config ONLY when:
      - HF_LOAD_IN_4BIT=true  AND
      - Running on CUDA (bitsandbytes does not support MPS or CPU)

    On Apple Silicon, quantization is unnecessary anyway — Mistral 7B in
    float16 fits comfortably within 24 GB of unified memory (~14 GB used).
    """
    if settings.hf_load_in_4bit and torch.cuda.is_available():
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return None


def _pick_dtype(device: str) -> torch.dtype:
    """
    Choose the right tensor dtype for each device:
      - MPS  → float16  (MPS supports float16; float32 works too but uses more memory)
      - CUDA → float16  (fast, half the memory of float32)
      - CPU  → float32  (MPS/CUDA ops not available; float16 on CPU is slow)
    """
    if device == "cpu":
        return torch.float32
    return torch.float16


def get_llm() -> HuggingFacePipeline:
    """
    Lazy-load the Mistral model. First call downloads weights from HuggingFace Hub
    (~14 GB on first run, then cached). Subsequent calls return instantly.
    """
    global _llm
    if _llm is not None:
        return _llm

    model_id = settings.hf_model_id
    device = _detect_device()
    bnb_config = _build_bnb_config()

    logger.info("Loading LLM: {}  |  device: {}  |  4-bit: {}", model_id, device, bnb_config is not None)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"trust_remote_code": True}

    if bnb_config:
        # CUDA + 4-bit: let accelerate distribute layers automatically
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        # MPS or CPU: load onto a single device, no accelerate distribution needed
        model_kwargs["torch_dtype"] = _pick_dtype(device)
        model_kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=settings.hf_max_new_tokens,
        do_sample=False,         # greedy decoding — deterministic, professional replies
        repetition_penalty=1.1,
        return_full_text=False,  # return only the generated part, not the full prompt
        pad_token_id=tokenizer.eos_token_id,
    )

    _llm = HuggingFacePipeline(pipeline=pipe)
    logger.info("LLM loaded successfully on {}.", device)
    return _llm


def generate_reply(
    email_subject: str,
    email_body: str,
    sender_name: Optional[str],
    context: str,
    system_prompt: str,
    reply_signature: str,
    max_tokens: int = 512,
) -> str:
    """
    Build a Mistral instruct prompt and generate an email reply.

    Prompt structure follows the Mistral instruct template:
    <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]
    """
    llm = get_llm()

    sender_label = sender_name or "the sender"
    context_block = (
        f"### Relevant Knowledge Base Context:\n{context}\n\n"
        if context
        else "### Knowledge Base Context:\n(No relevant documents found)\n\n"
    )

    user_message = (
        f"{context_block}"
        f"### Email to Reply To:\n"
        f"From: {sender_label}\n"
        f"Subject: {email_subject}\n\n"
        f"{email_body}\n\n"
        f"### Task:\n"
        f"Write a professional, helpful reply to the email above. "
        f"Use the knowledge base context if relevant. "
        f"Do NOT repeat the subject line or add 'Subject:' at the start. "
        f"Write only the reply body."
    )

    prompt = (
        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"
    )

    logger.debug("Generating reply for subject: '{}'", email_subject)

    raw_output: str = llm.invoke(prompt)

    # Strip any accidental prompt echo or special tokens
    for marker in ["[/INST]", "</s>", "<s>"]:
        raw_output = raw_output.split(marker)[-1]
    reply = raw_output.strip()

    return reply + reply_signature