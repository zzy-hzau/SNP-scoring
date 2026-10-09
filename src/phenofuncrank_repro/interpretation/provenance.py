"Claim-evidence provenance validation policy."

from __future__ import annotations

from typing import Any

from phenofuncrank_repro.interpretation.evidence_models import Claim


def validate_claim_support_rows(
    claim: Claim,
    supporting_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    claimed_ids = set(claim.supporting_evidence_ids)
    row_ids = {str(row["id"]) for row in supporting_rows}

    for evidence_id in sorted(claimed_ids - row_ids):
        issues.append({
            "severity": "error",
            "code": "missing_evidence_row",
            "message": f"claim cites evidence {evidence_id} that is not in the EvidenceStore",
        })
    if row_ids - claimed_ids:
        issues.append({
            "severity": "warning",
            "code": "unlisted_supporting_row",
            "message": "some fetched evidence rows were not explicitly cited by the claim",
        })

    for row in supporting_rows:
        if str(row.get("snp_key") or "") != claim.snp_key:
            issues.append({
                "severity": "error",
                "code": "snp_key_mismatch",
                "message": f"evidence {row.get('id')} belongs to {row.get('snp_key')} but claim is for {claim.snp_key}",
            })
        if str(row.get("predicate") or "") != "supports":
            issues.append({
                "severity": "warning",
                "code": "non_supporting_evidence",
                "message": f"evidence {row.get('id')} has predicate {row.get('predicate')}",
            })
        if str(row.get("evidence_level") or "E0") == "E0":
            issues.append({
                "severity": "warning",
                "code": "identity_only_evidence",
                "message": f"evidence {row.get('id')} is identity-level (E0) and adds no biological support",
            })

    causal = claim.claim_type.endswith("_causal") or str(
        (claim.payload or {}).get("causal_claim_allowed", False)
    ).lower() in {"1", "true", "yes"}
    if causal and not any(
        str(row.get("evidence_level")) in {"E5", "E6"} for row in supporting_rows
    ):
        issues.append({
            "severity": "error",
            "code": "causal_requires_experimental",
            "message": "causal claims require at least one E5/E6 (experimental/field) supporting evidence row",
        })
    return issues
