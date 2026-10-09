"Four-layer variant interpretation benchmark metrics (SOP §7)."

from __future__ import annotations

import re
from typing import Any


REQUIRED_CASE_FIELDS = {
    "case_id", "species", "trait", "variant", "gold_papers", "gold_causal_answer",
}
REQUIRED_EVIDENCE_TYPES = {"variant_specific", "gene_trait", "mechanism", "background"}
EVIDENCE_TYPE_TO_LEVEL = {
    "variant_specific": 1,
    "gene_trait": 2,
    "mechanism": 3,
    "background": 4,
}
REQUIRED_GOLD_PAPER_KEYS = {"evidence_levels", "irrelevant_neighbors"}
LEGACY_GOLD_PAPER_KEYS = {"canonical", "supporting", "irrelevant_neighbors"}


def validate_case(case: dict[str, Any]) -> list[str]:
    """Return schema violations (empty list = valid)."""
    errors: list[str] = []
    missing = REQUIRED_CASE_FIELDS - set(case)
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
    required_evidence = case.get("required_evidence")
    if not isinstance(required_evidence, list) or not required_evidence:
        errors.append("required_evidence must be a non-empty list of {type, required}")
    else:
        for item in required_evidence:
            if not isinstance(item, dict):
                errors.append("required_evidence items must be dicts")
                continue
            etype = item.get("type")
            if etype not in REQUIRED_EVIDENCE_TYPES:
                errors.append(f"required_evidence.type must be one of {sorted(REQUIRED_EVIDENCE_TYPES)}")
            if "required" not in item or not isinstance(item.get("required"), bool):
                errors.append("required_evidence.required must be a bool")
    variant = case.get("variant") or {}
    for field in ("chromosome", "position", "gene"):
        if field not in variant:
            errors.append(f"variant missing {field}")
    gold_papers = case.get("gold_papers") or {}
    has_new = "evidence_levels" in gold_papers
    has_legacy = bool(LEGACY_GOLD_PAPER_KEYS & set(gold_papers))
    if not has_new and not has_legacy:
        errors.append("gold_papers must contain evidence_levels (or legacy canonical/supporting)")
    for key in REQUIRED_GOLD_PAPER_KEYS:
        if key not in gold_papers:
            errors.append(f"gold_papers missing {key}")
        elif key == "evidence_levels":
            if not isinstance(gold_papers[key], dict):
                errors.append("gold_papers.evidence_levels must be a dict")
        elif not isinstance(gold_papers[key], list):
            errors.append(f"gold_papers.{key} must be a list")
    if has_new:
        levels = gold_papers["evidence_levels"]
        if not isinstance(levels, dict) or not levels:
            errors.append("gold_papers.evidence_levels must be a non-empty dict")
        else:
            for level, pmids in levels.items():
                if not isinstance(pmids, list):
                    errors.append(f"gold_papers.evidence_levels.{level} must be a list")
    if has_legacy and "evidence_levels" not in gold_papers:
        for key in ("canonical", "supporting"):
            if key in gold_papers and not isinstance(gold_papers[key], list):
                errors.append(f"gold_papers.{key} must be a list")
    chain = case.get("gold_evidence_chain") or []
    if not isinstance(chain, list) or not chain:
        errors.append("gold_evidence_chain must be a non-empty list")
    for item in chain:
        for field in ("layer", "claim", "evidence_type", "source"):
            if field not in item:
                errors.append(f"gold_evidence_chain item missing {field}")
    return errors


def _match(record: dict[str, Any], paper_ids: list[str]) -> bool:
    """Match a returned record against paper IDs by PMID or DOI."""
    pmid = str(record.get("pmid") or "")
    doi = str(record.get("doi") or "").lower()
    for pid in paper_ids:
        pid = str(pid).strip()
        if pmid and pid == pmid:
            return True
        if doi and pid.lower() in doi:
            return True
        title = str(record.get("title") or "").lower()
        if title and pid.lower() in title:
            return True
    return False


def _top_k(records: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    return (records or [])[:k]


def _gold_levels(case: dict[str, Any]) -> dict[str, list[str]]:
    gold = case.get("gold_papers") or {}
    levels = gold.get("evidence_levels")
    if isinstance(levels, dict):
        return {str(k): [str(p) for p in (v or [])] for k, v in levels.items()}
    out: dict[str, list[str]] = {}
    canonical = gold.get("canonical") or []
    supporting = gold.get("supporting") or []
    if canonical:
        out["2"] = [str(p) for p in canonical]
    if supporting:
        out["3"] = [str(p) for p in supporting]
    return out


def classify_evidence_level(title: str, abstract: str = "", gene: str = "") -> int:
    text = f"{title} {abstract}".lower()
    variant_terms = (
        "allele-specific", "haplotype", "natural variant", "natural variation",
        "causal variant", "editing", "edit", "eqtl", "specific mutation",
        "genotype comparison",
    )
    effect_terms = (
        "height", "phenotype", "dwarf", "yield", "trait", "effect",
        "variation", "expression", "architecture", "grain", "seed",
        "tiller", "sterility", "flowering", "resistance",
    )
    gene_norm = re.sub(r"[^a-z0-9]", "", (gene or "").lower())
    text_norm = re.sub(r"[^a-z0-9]", "", text)
    if (
        any(k in text for k in variant_terms)
        and any(k in text for k in effect_terms)
        and (not gene_norm or gene_norm in text_norm)
    ):
        return 1
    if any(k in text for k in (
        "mutant", "knockout", "overexpression", "complementation",
        "rescue", "transgenic", "phenotype", "dwarf", "height",
    )):
        return 2
    if any(k in text for k in (
        "pathway", "signaling", "biosynthesis", "regulat", "expression",
        "hormone", "enzyme", "oxidase", "interaction",
    )):
        return 3
    return 4


def required_evidence_layers(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item for item in (case.get("required_evidence") or [])
        if item.get("type") in EVIDENCE_TYPE_TO_LEVEL
    ]


def evidence_completeness(
    records: list[dict[str, Any]], case: dict[str, Any],
) -> dict[str, Any]:
    layers = required_evidence_layers(case)
    required = [l for l in layers if l.get("required")]
    covered_levels: set[int] = set()
    for record in records or []:
        covered_levels.add(
            classify_evidence_level(
                str(record.get("title") or ""),
                str(record.get("abstract") or ""),
            )
        )
    covered: list[str] = []
    missing: list[str] = []
    for layer in required:
        level = EVIDENCE_TYPE_TO_LEVEL[layer["type"]]
        if level in covered_levels:
            covered.append(layer["type"])
        else:
            missing.append(layer["type"])
    optional = [l for l in layers if not l.get("required")]
    optional_covered = [
        l["type"] for l in optional
        if EVIDENCE_TYPE_TO_LEVEL[l["type"]] in covered_levels
    ]
    return {
        "required_layers": [l["type"] for l in required],
        "covered_layers": covered,
        "missing_layers": missing,
        "completeness": round(len(covered) / len(required), 4) if required else 0.0,
        "optional_layers": [l["type"] for l in optional],
        "optional_covered": optional_covered,
    }


_DOI_RE = re.compile(r"^10\.\d{4,9}/")


def _funnel_ident(rec: Any) -> dict[str, str]:
    if isinstance(rec, dict):
        return {
            "pmid": str(rec.get("pmid") or ""),
            "doi": str(rec.get("doi") or ""),
            "title": str(rec.get("title") or ""),
        }
    cand = getattr(rec, "candidate", None)
    if cand is not None:
        return {
            "pmid": str(getattr(cand, "pmid", None) or ""),
            "doi": str(getattr(cand, "doi", None) or ""),
            "title": str(getattr(cand, "title", None) or ""),
        }
    return {
        "pmid": str(getattr(rec, "pmid", None) or ""),
        "doi": str(getattr(rec, "doi", None) or ""),
        "title": str(getattr(rec, "title", None) or ""),
    }


def _funnel_norm_doi(doi: str) -> str:
    return (doi or "").lower().replace("https://doi.org/", "").replace("http://dx.doi.org/", "").strip()


def _gold_ids(case: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for pmlist in (case.get("gold_papers") or {}).get("evidence_levels", {}).values():
        out.extend(str(p) for p in pmlist)
    return out


def _rec_matches_gold(rec: Any, gold_id: str) -> bool:
    pm = _funnel_ident(rec)
    if _DOI_RE.match(gold_id):
        return bool(pm["doi"]) and _funnel_norm_doi(pm["doi"]) == _funnel_norm_doi(gold_id)
    return bool(pm["pmid"]) and pm["pmid"] == str(gold_id)


def funnel_metrics(
    case: dict[str, Any],
    *,
    metadata_candidates: list[Any] | None = None,
    retrieval_hits: list[Any] | None = None,
    final_records: list[dict[str, Any]] | None = None,
    rerank_k: int = 10,
) -> dict[str, Any]:
    """Funnel-stage metrics for one case (Candidate → Enrichment → Admission
    → Chunk → Rerank Top-K → Verification).

    ``metadata_candidates`` are RetrievalCandidate-like objects carrying
    ``metadata_rank`` / ``admission_rank`` / ``text_status`` / ``fulltext_status``.
    ``retrieval_hits`` are chunk-level hits (RetrievalHit-like).
    """
    golds = _gold_ids(case)
    rows: list[dict[str, Any]] = []
    for gid in golds:
        row = {
            "gold_id": gid,
            "candidate": False, "candidate_rank": None,
            "enriched": False, "text_status": "",
            "admitted": False, "ingested": False,
            "chunk": False, "rerank_top_k": False, "verified": False,
        }
        if metadata_candidates:
            for cand in metadata_candidates:
                if not _rec_matches_gold(cand, gid):
                    continue
                row["candidate"] = True
                row["candidate_rank"] = getattr(cand, "metadata_rank", None)
                status = str(getattr(cand, "text_status", "") or "")
                row["text_status"] = status
                row["enriched"] = status in {
                    "abstract_available", "abstract_enriched", "fulltext_available",
                }
                row["admitted"] = getattr(cand, "admission_rank", None) is not None
                row["ingested"] = getattr(cand, "admission_rank", None) is not None and (
                    getattr(cand, "fulltext_status", "") in {"ok", "abstract"}
                    or status == "fulltext_available"
                )
                break
        if retrieval_hits:
            row["chunk"] = any(_rec_matches_gold(h, gid) for h in retrieval_hits)
        if final_records is not None:
            row["rerank_top_k"] = any(
                _rec_matches_gold(r, gid) for r in final_records[:rerank_k]
            )
        rows.append(row)

    def _rate(key: str, base: str = "candidate") -> float:
        denom = sum(1 for r in rows if r[base])
        return round(sum(1 for r in rows if r[key]) / denom, 4) if denom else 0.0

    return {
        "gold_total": len(rows),
        "candidate_recall": round(sum(1 for r in rows if r["candidate"]) / len(rows), 4) if rows else 0.0,
        "enrichment_success": _rate("enriched"),
        "admission_survival": _rate("ingested"),
        "chunk_recall": _rate("chunk"),
        "rerank_recall_at_k": _rate("rerank_top_k"),
        "verified_evidence_recall": _rate("verified"),
        "per_gold": rows,
    }


def canonical_recall_at_k(records: list[dict[str, Any]], case: dict[str, Any], k: int) -> bool:
    levels = _gold_levels(case)
    gold = levels.get("1", []) + levels.get("2", [])
    return any(_match(r, gold) for r in _top_k(records, k))


canonical_hit_at_k = canonical_recall_at_k


def canonical_paper_recall_at_k(
    records: list[dict[str, Any]], case: dict[str, Any], k: int,
) -> float:
    """Paper-level L1/L2 recall: matched gold papers / all L1+L2 gold papers."""
    levels = _gold_levels(case)
    gold = levels.get("1", []) + levels.get("2", [])
    if not gold:
        return 1.0  
    matched = sum(
        1 for g in gold if any(_match(r, [g]) for r in _top_k(records, k))
    )
    return round(matched / len(gold), 4)


def evidence_recall_at_k(records: list[dict[str, Any]], case: dict[str, Any], k: int) -> bool:
    """L1: any gold paper (any evidence level) inside top-K."""
    gold = [p for pmids in _gold_levels(case).values() for p in pmids]
    return any(_match(r, gold) for r in _top_k(records, k))


def evidence_set_coverage(
    records: list[dict[str, Any]], case: dict[str, Any], k: int,
) -> dict[str, Any]:
    levels = _gold_levels(case)
    required = sorted(l for l, pmids in levels.items() if pmids)
    covered = [
        l for l in required
        if any(_match(r, levels[l]) for r in _top_k(records, k))
    ]
    return {
        "levels_required": required,
        "levels_covered": covered,
        "coverage": round(len(covered) / len(required), 4) if required else 0.0,
    }


def level3_penetration(records: list[dict[str, Any]], case: dict[str, Any], k: int) -> bool:
    """L1: any irrelevant neighbor inside top-K."""
    gold = (case.get("gold_papers") or {}).get("irrelevant_neighbors") or []
    return any(_match(r, gold) for r in _top_k(records, k))


def pmid_accuracy(records: list[dict[str, Any]], case: dict[str, Any]) -> dict[str, float | int]:
    """L1: correct citations vs total; title-correct-but-PMID-wrong count."""
    gold = set(p for pmids in _gold_levels(case).values() for p in pmids)
    gold_titles = {str(c.get("claim") or "").lower() for c in (case.get("gold_evidence_chain") or [])}
    correct = 0
    title_correct_pmid_wrong = 0
    total = len(records or [])
    for record in records or []:
        if _match(record, list(gold)):
            correct += 1
            continue
        title = str(record.get("title") or "").lower()
        if any(gt and (gt in title or title in gt) for gt in gold_titles):
            title_correct_pmid_wrong += 1
    return {
        "correct": correct,
        "total": total,
        "pmid_accuracy": round(correct / total, 4) if total else 0.0,
        "title_correct_pmid_wrong": title_correct_pmid_wrong,
    }


def citation_hallucination_rate(records: list[dict[str, Any]]) -> float:
    """L1: records failing existence/identity verification / total."""
    total = len(records or [])
    if not total:
        return 0.0
    bad = sum(
        1 for r in (records or [])
        if r.get("level1_exists") is False or r.get("level2_match") is False
    )
    return round(bad / total, 4)


def evidence_precision_and_completeness(
    proposals: list[dict[str, Any]], case: dict[str, Any],
) -> dict[str, float | int]:
    """L2: precision over extracted proposals; completeness over gold layers."""
    gold = (case.get("gold_evidence_chain") or [])
    gold_claims = {str(g.get("claim") or "").lower() for g in gold}
    correct = 0
    for p in proposals or []:
        claim = str(p.get("claim") or "").lower()
        if claim in gold_claims:
            correct += 1
    total = len(proposals or [])
    gold_layers = {str(g.get("layer") or "") for g in gold}
    covered_layers = {str(p.get("layer") or "") for p in proposals or []}
    return {
        "evidence_precision": round(correct / total, 4) if total else 0.0,
        "correct": correct,
        "total": total,
        "evidence_completeness": round(
            len(gold_layers & covered_layers) / len(gold_layers), 4,
        ) if gold_layers else 0.0,
        "gold_layers": len(gold_layers),
        "covered_layers": len(gold_layers & covered_layers),
    }


def evidence_leap_rate(claims: list[dict[str, Any]]) -> float:
    total_transitions = 0
    leaps = 0
    for claim in claims or []:
        steps = int(claim.get("steps") or 0)
        supported = int(claim.get("supported_steps") or 0)
        total_transitions += steps
        leaps += max(0, steps - supported)
    if not total_transitions:
        return 0.0
    return round(leaps / total_transitions, 4)


def chain_correctness(claim: dict[str, Any]) -> int:
    """L3: 0–3 for variant→gene, gene→mechanism, mechanism→trait."""
    steps = claim.get("steps") or {}
    return int(steps.get("variant_gene") or 0) + int(steps.get("gene_mechanism") or 0) + int(steps.get("mechanism_trait") or 0)


def aggregate_expert_scores(scores: list[dict[str, Any]]) -> dict[str, float]:
    """L3: mean overinterpretation / plausibility / completeness."""
    keys = ("overinterpretation", "biological_plausibility", "evidence_completeness")
    out: dict[str, float] = {}
    for key in keys:
        values = [float(s.get(key) or 0.0) for s in scores or []]
        out[f"mean_{key}"] = round(sum(values) / len(values), 4) if values else 0.0
    return out


def rejection_rate(decisions: list[dict[str, Any]]) -> float:
    """L4: error claims rejected (critic != ACCEPT) / all error claims."""
    total = len(decisions or [])
    if not total:
        return 0.0
    rejected = sum(
        1 for d in (decisions or [])
        if str(d.get("decision") or "").upper() != "ACCEPT"
    )
    return round(rejected / total, 4)


def critic_decision_accuracy(decisions: list[dict[str, Any]]) -> float:
    """L4: exact agreement with gold critic decisions."""
    total = len(decisions or [])
    if not total:
        return 0.0
    agree = sum(
        1 for d in (decisions or [])
        if str(d.get("decision") or "").upper() == str(d.get("gold") or "").upper()
    )
    return round(agree / total, 4)


def gap_routing_accuracy(identified: list[str], gold: list[str]) -> float:
    """A1: identified ∩ gold / gold (coverage) + efficiency note."""
    if not gold:
        return 0.0
    return round(len(set(identified) & set(gold)) / len(set(gold)), 4)


def causal_recall_at_20(ranked: list[str], gold: list[str]) -> float:
    if not gold:
        return 0.0
    top20 = set(ranked[:20])
    return round(len(top20 & set(gold)) / len(set(gold)), 4)
