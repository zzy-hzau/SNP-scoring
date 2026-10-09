"""Source-extracted dependency used by the reported graph planning route."""
from __future__ import annotations
import json
import os
from string import Template
from typing import Any
from phenofuncrank_repro.interpretation.settings import settings

def registry() -> dict[str, Any]:
    if not settings.resource_file.is_file():
        raise RuntimeError(f"resource registry does not exist: {settings.resource_file}")
    raw = settings.resource_file.read_text(encoding="utf-8")
    variables = dict(os.environ)
    variables.setdefault("RICE_PH_AGENT_WORKSPACE", str(settings.workspace))
    variables.setdefault(
        "RICE_PH_AGENT_SCIENCE_ROOT",
        str(settings.workspace / "plant_caddlike_github_clean/trait_prediction_and_variant_prioritization"),
    )
    variables.setdefault("RICE_PH_AGENT_DATA_ROOT", str(settings.workspace / "data"))
    variables.setdefault("RICE_PH_AGENT_MODELS_ROOT", str(settings.workspace / "rice_ph_agent/rice_ph_agent"))
    expanded = Template(raw).safe_substitute(variables)
    return json.loads(expanded)


def reference_fasta_path(species: str) -> str | None:
    try:
        data = registry()
        config = data.get(species, {})
    except (OSError, RuntimeError, json.JSONDecodeError):
        return None
    path_text = config.get("reference_fasta")
    if not path_text:
        return None
    fasta = Path(path_text)
    if not fasta.is_file():
        return None

    if not fasta.with_suffix(fasta.suffix + ".fai").is_file():
        return None
    return str(fasta)

from pathlib import Path
