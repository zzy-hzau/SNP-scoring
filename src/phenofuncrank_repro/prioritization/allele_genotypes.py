"chromosome normalization for allele-aware downstream analysis."
from __future__ import annotations
import re

def norm_chrom(value: object, spec: dict) -> str:
    """Canonical chromosome label, preserving cotton A/D provenance."""
    text = str(value).strip().upper().replace("CHROMOSOME", "").replace("CHR", "")
    style = spec["chromosome_style"]
    if style == "cotton_ad":
        text = text.replace("_", "")
        if len(text) == 2 and text[0] in {"A", "D"} and text[1].isdigit():
            text = f"{text[0]}{int(text[1]):02d}"
        if len(text) != 3 or text[0] not in {"A", "D"} or not text[1:].isdigit() or not 1 <= int(text[1:]) <= 13:
            raise ValueError(f"unsupported cotton chromosome: {value!r}")
        return text
    if style == "chickpea_ca":
        text = text.replace("_", "")
        if text.startswith("CA") and text[2:].isdigit() and 1 <= int(text[2:]) <= 8:
            return f"Ca{int(text[2:])}"
        raise ValueError(f"unsupported chickpea chromosome: {value!r}")
    text = text.lstrip("0")
    if text.isdigit():
        num = int(text)
        if 1 <= num <= int(spec["chromosome_count"]):
            return str(num)
        raise ValueError(f"unsupported {spec['species']} chromosome: {value!r}")

    match = re.search(r"(?:^|\.)(\d*)(?:CH)?(\d+)$", text)
    if match:
        num = int(match.group(2))
        if 1 <= num <= int(spec["chromosome_count"]):
            return str(num)
    raise ValueError(f"unsupported {spec['species']} chromosome: {value!r}")
