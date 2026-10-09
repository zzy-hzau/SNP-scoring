"LLM requests and JSON response parsing."
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from phenofuncrank_repro.interpretation.logging_utils import get_logger
from phenofuncrank_repro.interpretation.settings import settings

logger = get_logger(__name__)


def now_stamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def configured_model() -> str:
    "Return the bounded-agent model, for configured requests."
    return os.environ.get("RICE_PH_AGENT_LLM_MODEL", "deepseek-v4-flash")


def configured_thinking_mode() -> str:
    "Return the DeepSeek thinking mode for runtime calls."
    value = os.environ.get("RICE_PH_AGENT_LLM_THINKING", "disabled").strip().lower()
    return value if value in {"enabled", "disabled"} else "disabled"


def configured_llm_extra_body(
    extra_body: dict[str, Any] | None = None,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    "Return provider-specific controls for the configured OpenAI endpoint."
    endpoint = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).lower()
    model_name = (model or configured_model()).lower()
    is_qwen_vllm = (
        os.environ.get("RICE_PH_AGENT_LLM_BACKEND", "").lower() in {"qwen_vllm", "vllm"}
        or "qwen" in model_name
        or "127.0.0.1:8000" in endpoint
        or "localhost:8000" in endpoint
    )

    

    
    merged = dict(extra_body or {})
    if is_qwen_vllm:
        chat_kwargs = dict(merged.get("chat_template_kwargs") or {})
        chat_kwargs.setdefault(
            "enable_thinking", configured_thinking_mode() == "enabled",
        )
        merged["chat_template_kwargs"] = chat_kwargs
    else:
        merged.setdefault("thinking", {"type": configured_thinking_mode()})
    return merged


def _parse_json_object(text: str) -> dict[str, Any] | None:
    "Parse the one JSON object allowed from LLM calls."
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]

    
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]
    try:
        value = json.loads(cleaned)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def configured_api_key() -> str | None:
    """Load an LLM key from an environment variable or a protected file."""
    direct = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if direct:
        return direct.strip()
    locations = [
        os.environ.get("DEEPSEEK_API_KEY_FILE"),
        os.environ.get("OPENAI_API_KEY_FILE"),

        
        str(Path(os.environ.get("RICE_PH_AGENT_ROOT", Path(__file__).resolve().parents[1] / "runtime")) / "secrets" / "deepseek_api_key"),
    ]
    for location in locations:
        if not location:
            continue
        try:
            key = open(location, encoding="utf-8").read().strip()
            if key:
                return key
        except OSError:
            logger.warning("Configured LLM key file is not readable")
    return None


def complete_json(
    system: str,
    payload: dict[str, Any],
    *,
    max_tokens: int | None = None,
    extra_body: dict[str, Any] | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any] | None:
    "Return one validated JSON object, with a single repair retry."
    if max_tokens is None:
        max_tokens = settings.agent_max_output_tokens
    payload_text = f"{now_stamp()}\n" + json.dumps(payload, ensure_ascii=False, default=str)
    from phenofuncrank_repro.interpretation.logging_utils import increment as _increment
    _increment("llm_payload_chars", value=float(len(payload_text)))
    if len(payload_text) > settings.llm_max_prompt_chars:

        

        logger.warning(
            "LLM payload rejected: %d chars exceeds llm_max_prompt_chars=%d "
            "(system prefix %r); deterministic fallback will be used",
            len(payload_text), settings.llm_max_prompt_chars, system[:60],
        )
        _increment("llm_payload_overflow")
        return None

    

    
    endpoint = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    request_extra_body = configured_llm_extra_body(
        extra_body, model=model, base_url=endpoint,
    )
    try:
        from openai import OpenAI
        key = configured_api_key()

        

        if not key and ("127.0.0.1" in endpoint or "localhost" in endpoint):
            key = "EMPTY"
        if not key:
            return None
        client = OpenAI(
            api_key=key,
            base_url=endpoint,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )
        messages = [

            

            {"role": "system", "content": system},
            {"role": "user", "content": payload_text},
        ]
        response = client.chat.completions.create(
            model=model or configured_model(), temperature=0,
            max_tokens=max_tokens, messages=messages,
            extra_body=request_extra_body or {},
        )
        text = response.choices[0].message.content or ""
        if not text.strip():

            

            

            for suffix in (
                "\n\nRespond with only one valid JSON object.",
                "\n\nReturn the JSON now, no prose.",
                "\n\nJSON only.",
            ):
                retry_messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": payload_text + suffix},
                ]
                response = client.chat.completions.create(
                    model=model or configured_model(), temperature=0,
                    max_tokens=max_tokens, messages=retry_messages,
                    extra_body=request_extra_body or {},
                )
                text = response.choices[0].message.content or ""
                if text.strip():
                    break
        data = _parse_json_object(text)
        if data is not None:
            return data

        

        repair_messages = [
            {"role": "system", "content": "Return only one valid JSON object. Repair the supplied invalid JSON without adding facts."},
            {"role": "user", "content": text[:6000]},
        ]
        repaired = client.chat.completions.create(
            model=model or configured_model(), temperature=0, max_tokens=max_tokens,
            messages=repair_messages, extra_body=request_extra_body or {},
        )
        return _parse_json_object(repaired.choices[0].message.content or "")
    except Exception as exc:
        logger.info("Bounded LLM call unavailable: %s", exc)
        return None
