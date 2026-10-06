"""Phase 7: generate text with a small local Hugging Face model (no API, no Ollama)."""
from typing import Callable

from config import settings


class GenerationError(Exception):
    """Raised when the local model cannot be loaded or run."""


_model = None
_tokenizer = None
_backend_override: Callable[[list[dict]], str] | None = None


def set_generation_backend(function: Callable[[list[dict]], str] | None) -> None:
    """Replace the real model (used by unit tests so they don't need the LLM)."""
    global _backend_override
    _backend_override = function


def is_local_mode() -> bool:
    return settings.GENERATION_MODE.lower() == "local"


def _load_model():
    global _model, _tokenizer
    if _model is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL)
            _model = AutoModelForCausalLM.from_pretrained(settings.LLM_MODEL)
            _model.eval()
        except Exception as error:  # includes MemoryError-like and download problems
            _model, _tokenizer = None, None
            raise GenerationError(
                f"Could not load the local model '{settings.LLM_MODEL}'. The first run needs "
                "internet to download it once, and it needs free RAM. You can set "
                "GENERATION_MODE=retrieval_only to run without a model. "
                f"Details: {error}"
            ) from error
    return _model, _tokenizer


def generate(messages: list[dict], max_new_tokens: int | None = None) -> str:
    """Run the chat messages through the local model and return the reply text."""
    if _backend_override is not None:
        return _backend_override(messages)
    model, tokenizer = _load_model()
    try:
        import torch

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or settings.MAX_NEW_TOKENS,
                do_sample=False,  # deterministic answers are easier to debug
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception as error:
        raise GenerationError(f"The model failed while writing an answer: {error}") from error
