"""
app/llm_service.py - Local Mistral LLM via HuggingFace Transformers + LangChain.

Device support:
  - "cpu"  : Recommended for Apple Silicon M-series Macs (stable, fits in 24 GB RAM)
  - "cuda" : NVIDIA GPU with optional 4-bit quantization via bitsandbytes
  - "mps"  : Apple Silicon GPU — NOT recommended for 7B models due to MPS memory
             allocation limits causing OOM errors during inference

Apple Silicon note:
  Mistral 7B requires ~14 GB in float16 or ~28 GB in float32.
  MPS cannot reliably allocate this as inference buffers push total usage
  beyond macOS limits (model + KV cache + activations exceed ~30 GB cap).
  CPU inference in float32 is stable and fits within 24 GB unified RAM.
  Expect ~1-3 tokens/sec on CPU — adequate for email reply generation.
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
    Resolve the effective device.
    Priority: explicit config → CUDA → MPS → CPU.
    """
    requested = settings.hf_device.lower()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_bnb_config() -> Optional[BitsAndBytesConfig]:
    """4-bit quantization — CUDA only. Never used on MPS or CPU."""
    if settings.hf_load_in_4bit and torch.cuda.is_available():
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return None


def get_llm() -> HuggingFacePipeline:
    """
    Lazy-load the Mistral model. First call takes several minutes.
    Subsequent calls within the same process return instantly (singleton).
    """
    global _llm
    if _llm is not None:
        return _llm

    model_id = settings.hf_model_id
    device = _detect_device()
    bnb_config = _build_bnb_config()

    logger.info(
        "Loading LLM: {}  |  device: {}  |  4-bit: {}",
        model_id, device, bnb_config is not None,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if bnb_config:
        # ── CUDA + 4-bit quantization ──────────────────────────────────────
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=settings.hf_max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
            device_map="auto",
        )

    elif device == "mps":
        # ── Apple Silicon MPS ──────────────────────────────────────────────
        # Warn strongly — 7B models reliably OOM on MPS during inference.
        # User can override HF_DEVICE=mps in .env at their own risk.
        logger.warning(
            "MPS device selected. Mistral 7B may cause OOM errors during inference "
            "on Apple Silicon. Set HF_DEVICE=cpu in .env for stable operation."
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = model.to("mps")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=settings.hf_max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    else:
        # ── CPU (recommended for Apple Silicon) ────────────────────────────
        # float32 on CPU: ~28 GB peak during load, settles to ~14 GB at rest.
        # low_cpu_mem_usage=True streams weights in to avoid double-buffering.
        logger.info("Loading model on CPU — this takes 2-5 minutes on first load.")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=settings.hf_max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
            device=-1,   # -1 = CPU in HuggingFace pipeline API
        )

    _llm = HuggingFacePipeline(pipeline=pipe)
    logger.info("LLM loaded successfully. Inference device: {}", device)
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

    Prompt structure (Mistral instruct format):
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

    return raw_output.strip() + reply_signature