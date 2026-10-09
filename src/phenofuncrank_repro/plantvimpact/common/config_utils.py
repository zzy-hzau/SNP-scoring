#!/usr/bin/env python3
"""Shared, dependency-light configuration helpers for pipeline wrappers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_.]*)\}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dotted_get(mapping: dict[str, Any], key: str, default: Any = None) -> Any:
    node: Any = mapping
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolve_template(value: Any, context: dict[str, Any]) -> Any:
    """Resolve ``{dotted.key}`` placeholders without evaluating code."""
    if not isinstance(value, str):
        return value

    result = value
    for _ in range(10):
        changed = False

        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            key = match.group(1)
            resolved = dotted_get(context, key, None)
            if resolved is None:
                return match.group(0)
            changed = True
            return str(resolved)

        next_result = _PLACEHOLDER.sub(replace, result)
        result = next_result
        if not changed:
            break
    unresolved = _PLACEHOLDER.findall(result)
    if unresolved:
        raise KeyError(
            f"unresolved configuration placeholders {unresolved} in {value!r}"
        )
    return result


def resolve_mapping(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: resolve_mapping(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_mapping(v, context) for v in value]
    return resolve_template(value, context)


def model_spec(registry: dict[str, Any], species: str, mode: str) -> dict[str, Any]:
    spec = dotted_get(registry, f"models.{species}.{mode}")
    if not isinstance(spec, dict):
        raise KeyError(f"model registry has no models.{species}.{mode} entry")
    return spec


def configured_models(registry: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for species, species_block in (registry.get("models") or {}).items():
        if not isinstance(species_block, dict):
            continue
        for mode in ("coding", "noncoding"):
            if isinstance(species_block.get(mode), dict):
                out.append((str(species), mode))
    return out


def parse_folds(value: str | Iterable[int]) -> list[int]:
    if not isinstance(value, str):
        return sorted({int(v) for v in value})
    folds: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            folds.extend(range(int(start), int(end) + 1))
        else:
            folds.append(int(item))
    return sorted(set(folds))


def folds_cli(value: str | Iterable[int]) -> str:
    return ",".join(str(v) for v in parse_folds(value))


def require_path(path: str | Path, *, kind: str = "any", label: str = "path") -> Path:
    result = Path(path)
    ok = result.exists()
    if kind == "file":
        ok = result.is_file()
    elif kind == "dir":
        ok = result.is_dir()
    if not ok:
        raise FileNotFoundError(f"{label} does not exist ({kind}): {result}")
    return result


def repo_root_from_script(script: str | Path) -> Path:
    from phenofuncrank_repro.common.paths import project_root
    return project_root(script)
