#!/usr/bin/env python3
"""Strict predictor policy shared by feature QA and model training."""
from __future__ import annotations

import re
from typing import Iterable


EXACT_FORBIDDEN = {
    "label",
    "new_label",
    "y_label",
    "p_mut",
    "p_base_eff",
    "callable_n",
    "an_obs",
    "callrate",
    "match_group",
    "opp_bin",
    "call_bin",
    "fold",
    "fold_id",
    "split",
    "chr",
    "chrom",
    "chromosome",
    "pos",
    "position",
    "ref",
    "alt",
    "snp_id",
    "variant_id",
}

TECHNICAL_PATTERNS = (
    re.compile(r"(^|_)(callable|callrate|callability|coverage)(_|$)", re.I),
    re.compile(r"(^|_)(opp|opportunity|call)_?bin(_|$)", re.I),
    re.compile(r"(^|_)(fold|split)(_|$)", re.I),
    re.compile(r"(^|_)(missing|status|reason|provenance)(_|$)", re.I),
    re.compile(r"(^|_)(p_mut|p_base_eff)(_|$)", re.I),
    re.compile(r"(^|_)(new_label|y_label|label)(_|$)", re.I),
)


def forbidden_reason(name: str) -> str | None:
    normalized = name.strip().lower()
    if normalized in EXACT_FORBIDDEN:
        return "exact forbidden label/opportunity/partition/coordinate identifier"
    for pattern in TECHNICAL_PATTERNS:
        if pattern.search(name):
            return f"technical/provenance pattern {pattern.pattern}"
    return None


def audit_predictors(
    features: Iterable[str],
    *,
    allowed_exceptions: Iterable[str] = (),
) -> list[dict[str, str]]:
    exceptions = {str(v) for v in allowed_exceptions}
    findings: list[dict[str, str]] = []
    for feature in features:
        name = str(feature)
        reason = forbidden_reason(name)
        if reason and name not in exceptions:
            findings.append({"feature": name, "reason": reason})
    return findings


def assert_predictors_allowed(
    features: Iterable[str],
    *,
    allowed_exceptions: Iterable[str] = (),
    context: str = "feature list",
) -> None:
    findings = audit_predictors(features, allowed_exceptions=allowed_exceptions)
    if findings:
        detail = "; ".join(f"{v['feature']}: {v['reason']}" for v in findings)
        raise ValueError(f"forbidden predictors in {context}: {detail}")
