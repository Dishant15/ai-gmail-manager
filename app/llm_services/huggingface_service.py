"""
app/llm_services/huggingface_service.py

LLM provider: HuggingFace Transformers (local model weights).

Loads the model configured in HF_MODEL_ID directly into Python process memory.
Supports CUDA (with 4-bit quantization), MPS (Apple Silicon, unstable for 7B+),
and CPU (recommended for Apple Silicon).

Apple Silicon warning:
  Mistral/Qwen 7B models loaded via HuggingFace frequently OOM on MPS and
  get stuck on CPU. Use the Ollama provider instead for Apple Silicon.

Env variables used:
  HF_MODEL_ID        — HuggingFace model repo (e.g. mistralai/Mistral-7B-Instruct-v0.3)
  HF_DEVICE          — auto | cpu | cuda | mps
  HF_LOAD_IN_4BIT    — true | false (CUDA only)
  HF_MAX_NEW_TOKENS  — max tokens to generate
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
from app.llm_services.base import build_user_message

_llm: Optional[HuggingFacePipeline] = None


def _detect_device() -> str:
    requested = settings.hf_device.lower()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_bnb_config() -> Optional[BitsAndBytesConfig]:
    """4-bit quantization — CUDA only."""
    if settings.hf_load_in_4bit and torch.cuda.is_available():
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return None


def _get_llm() -> HuggingFacePipeline:
    global _llm
    if _llm is not None:
        return _llm

    model_id = settings.hf_model_id
    device = _detect_device()
    bnb_config = _build_bnb_config()

    logger.info(
        "HuggingFace | Loading: {}  device: {}  4-bit: {}",
        model_id, device, bnb_config is not None,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if bnb_config:
        # CUDA + 4-bit
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
        )
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=settings.hf_max_new_tokens,
            do_sample=False, repetition_penalty=1.1,
            return_full_text=False, pad_token_id=tokenizer.eos_token_id,
            device_map="auto",
        )

    elif device == "mps":
        logger.warning(
            "HuggingFace | MPS selected — 7B models may OOM on Apple Silicon. "
            "Consider switching to LLM_PROVIDER=ollama."
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
        model = model.to("mps")
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=settings.hf_max_new_tokens,
            do_sample=False, repetition_penalty=1.1,
            return_full_text=False, pad_token_id=tokenizer.eos_token_id,
        )

    else:
        # CPU — recommended for Apple Silicon
        logger.info("HuggingFace | Loading on CPU (2–5 min on first load).")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32,
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=settings.hf_max_new_tokens,
            do_sample=False, repetition_penalty=1.1,
            return_full_text=False, pad_token_id=tokenizer.eos_token_id,
            device=-1,
        )

    _llm = HuggingFacePipeline(pipeline=pipe)
    logger.info("HuggingFace | LLM ready on {}.", device)
    return _llm


def generate_reply(
    email_subject: str,
    email_body: str,
    sender_name: str | None,
    context: str,
    system_prompt: str,
    reply_signature: str,
    max_tokens: int | None = 512,
) -> str:
    llm = _get_llm()
    user_message = build_user_message(email_subject, email_body, sender_name, context)

    # Mistral instruct prompt format
    prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"

    logger.debug("HuggingFace | Generating reply for: '{}'", email_subject)
    raw: str = llm.invoke(prompt)

    for marker in ["[/INST]", "</s>", "<s>"]:
        raw = raw.split(marker)[-1]

    return raw.strip() + reply_signature
