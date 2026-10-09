"""Convert existing DPC/CADD/LD tables into versioned EvidenceRecord objects."""
from __future__ import annotations

from typing import Any

import pandas as pd

from phenofuncrank_repro.interpretation.evidence_models import EvidenceLevel, EvidenceRecord, SNPIdentity
from phenofuncrank_repro.interpretation.evidence_store import EvidenceStore


ASSEMBLIES = {
    "rice": "IRGSP-1.0",
    "maize": "AGPv3.25",
    "cotton": "TM-1_CRI",
    "tomato": "SL4.0",
    "chickpea": "CDC_Frontier",
}


def assembly_for_species(species: str) -> str:
    """Use the deployment plant registry as the assembly source of truth."""
    from phenofuncrank_repro.interpretation.resources import registry
    return str(registry().get(species.lower(), {}).get("reference_assembly") or ASSEMBLIES.get(species.lower(), "unknown"))


def snp_identity_from_row(row: Any, species: str) -> SNPIdentity | None:
    values = row._asdict() if hasattr(row, "_asdict") else dict(row)
    required = ("chrom", "pos", "ref", "alt")
    if any(values.get(column) is None or pd.isna(values.get(column)) for column in required):
        return None
    try:
        return SNPIdentity(species, str(values.get("reference_build") or values.get("reference_assembly") or assembly_for_species(species)), str(values["chrom"]), int(values["pos"]), str(values["ref"]), str(values["alt"]))
    except (TypeError, ValueError):
        return None


def record_dpc_effects(effect: pd.DataFrame, store: EvidenceStore, *, species: str, trait: str, tool_run_id: str, artifact_id: str | None = None) -> list[str]:
    """Persist one E1 DPC perturbation observation per evaluable SNP."""
    snps: list[SNPIdentity] = []
    records: list[EvidenceRecord] = []
    for row in effect.itertuples(index=False):
        snp = snp_identity_from_row(row, species)
        if snp is None:
            continue
        snps.append(snp)
        value = getattr(row, "dpc_additive_effect_mean", None)
        direction = getattr(row, "dpc_direction_alt", None)
        try:
            value = float(value) if value is not None and not pd.isna(value) else None
        except (TypeError, ValueError):
            value = None
        records.append(EvidenceRecord(
            snp_key=snp.canonical_key,
            tool_run_id=tool_run_id,
            artifact_id=artifact_id,
            evidence_type="model_effect",
            subject="ALT_allele",
            predicate="predicted_to_change",
            object=trait,
            value=value,
            unit="trait_unit",
            direction=str(direction) if direction is not None and not pd.isna(direction) else None,
            confidence=None,
            scope={"trait": trait, "species": species, "background": "configured_dpcformer_background"},
            evidence_level=EvidenceLevel.E1,
            payload={"snp_id": getattr(row, "snp_id", None), "level1_core": getattr(row, "level1_core", None)},
        ))
    store.upsert_snps(snps)
    return store.add_evidence_batch(records)


def record_cadd_scores(scores: pd.DataFrame, store: EvidenceStore, *, species: str, tool_run_id: str, artifact_id: str | None = None) -> list[str]:

    

    for column in ("pos", "ref", "alt"):
        if column not in scores.columns and f"{column}_x" in scores.columns:
            scores = scores.rename(columns={f"{column}_x": column})
    snps: list[SNPIdentity] = []
    records: list[EvidenceRecord] = []
    for row in scores.itertuples(index=False):
        snp = snp_identity_from_row(row, species)
        if snp is None:
            continue
        value = getattr(row, "pvi_score", getattr(row, "cadd_prob", None))
        try:
            value = float(value) if value is not None and not pd.isna(value) else None
        except (TypeError, ValueError):
            value = None
        if value is None:
            continue
        snps.append(snp)
        records.append(EvidenceRecord(
            snp_key=snp.canonical_key,
            tool_run_id=tool_run_id,
            artifact_id=artifact_id,
            evidence_type="functional_score",
            subject="SNP",
            predicate="has_functional_plausibility_score",
            object="PlantVImpact_probability",
            value=value,
            unit="probability",
            scope={"species": species},
            evidence_level=EvidenceLevel.E1,
            payload={"priority_rank": getattr(row, "priority_rank", None), "variant_key": getattr(row, "variant_key", None)},
        ))
    store.upsert_snps(snps)
    return store.add_evidence_batch(records)


def record_ld_proxies(proxies: pd.DataFrame, store: EvidenceStore, *, species: str, trait: str, tool_run_id: str, identities: dict[str, SNPIdentity], artifact_id: str | None = None) -> list[str]:
    """Persist LD as a relationship observation without upgrading causal evidence."""
    evidence_ids: list[str] = []
    for row in proxies.itertuples(index=False):
        proxy = identities.get(str(getattr(row, "proxy_snp_id")))
        core = identities.get(str(getattr(row, "core_snp_id")))
        if proxy is None or core is None:
            continue
        store.upsert_snp(proxy)
        evidence_ids.append(store.add_evidence(EvidenceRecord(
            snp_key=proxy.canonical_key,
            tool_run_id=tool_run_id,
            artifact_id=artifact_id,
            evidence_type="ld_proxy",
            subject=proxy.canonical_key,
            predicate="in_ld_with",
            object=core.canonical_key,
            value=float(getattr(row, "r2_to_core")),
            unit="r2",
            scope={"trait": trait, "species": species, "within_submitted_panel": True},
            evidence_level=EvidenceLevel.E1,
        )))
    return evidence_ids


def record_functional_annotation(annotation: dict[str, Any], snp: SNPIdentity, store: EvidenceStore, *, tool_run_id: str, artifact_id: str | None = None) -> list[str]:
    """Persist machine-readable SNPFunctionExplorer output without inferring causality."""
    store.upsert_snp(snp)
    evidence_ids: list[str] = []
    route = annotation.get("route") or annotation.get("region_type")
    if route:
        evidence_ids.append(store.add_evidence(EvidenceRecord(
            snp_key=snp.canonical_key, tool_run_id=tool_run_id, artifact_id=artifact_id,
            evidence_type="functional_annotation", subject="SNP", predicate="annotated_as", object=str(route),
            scope={"species": snp.species, "assembly": snp.assembly}, evidence_level=EvidenceLevel.E1,
            payload={
                "region_type": annotation.get("region_type"),
                "functional_hypothesis": annotation.get("functional_hypothesis"),
                "signal": annotation.get("signal"),
                "consequence": annotation.get("consequence") or annotation.get("variant_consequence"),
                "molecular_effect": annotation.get("molecular_effect"),
                "expression_effect": annotation.get("expression_effect"),
                "protein_effect": annotation.get("protein_effect"),
                "subgenome": annotation.get("subgenome"),
                "homeolog": annotation.get("homeolog"),
            },
        )))
    gene_id = annotation.get("gene_id")
    if gene_id:
        evidence_ids.append(store.add_evidence(EvidenceRecord(
            snp_key=snp.canonical_key, tool_run_id=tool_run_id, artifact_id=artifact_id,
            evidence_type="gene_link", subject="SNP", predicate="annotated_to_gene", object=str(gene_id),
            scope={"species": snp.species, "assembly": snp.assembly}, evidence_level=EvidenceLevel.E1,
            payload={"route": route, "subgenome": annotation.get("subgenome"), "homeolog": annotation.get("homeolog")},
        )))
    return evidence_ids


def record_literature_hits(articles: list[dict[str, Any]], snp: SNPIdentity, store: EvidenceStore, *, tool_run_id: str, query: str, artifact_id: str | None = None) -> list[str]:
    "Link retrieved literature to a SNP as E4 retrieval evidence."
    store.upsert_snp(snp)
    evidence_ids: list[str] = []
    for article in articles:
        identifier = article.get("pmid") or article.get("doi") or article.get("title")
        if not identifier:
            continue
        evidence_ids.append(store.add_evidence(EvidenceRecord(
            snp_key=snp.canonical_key, tool_run_id=tool_run_id, artifact_id=artifact_id,
            evidence_type="literature", subject="SNP", predicate="has_retrieved_literature", object=str(identifier),
            scope={"query": query, "provider": article.get("source") or "multi_source"}, evidence_level=EvidenceLevel.E4,
            direction=article.get("trait_direction") or article.get("direction"),
            confidence=(article.get("relevance_assessment") or {}).get("score"),
            payload=article,
        )))
    return evidence_ids


def record_population_metrics(
    metrics: pd.DataFrame,
    store: EvidenceStore,
    *,
    species: str,
    trait: str,
    tool_run_id: str,
    artifact_id: str | None = None,
) -> list[str]:
    """Persist per-cohort allele frequencies as SNP-population evidence.

    One ``population_metric`` record per SNP per cohort (cultivar / wild).
    AF is the value; AC/AN/callability/He travel in the payload.  The variant
    graph consumes these as ``OBSERVED_IN`` edges and derives
    ``DIFFERENT_FROM`` (domestication shift) from wild-vs-cultivar pairs.
    """
    snps: list[SNPIdentity] = []
    records: list[EvidenceRecord] = []
    for row in metrics.itertuples(index=False):
        snp = snp_identity_from_row(row, species)
        if snp is None:
            continue
        snps.append(snp)

        af_cultivar = _row_float(row, "AF_cultivar")
        if af_cultivar is not None:
            records.append(EvidenceRecord(
                snp_key=snp.canonical_key,
                tool_run_id=tool_run_id,
                artifact_id=artifact_id,
                evidence_type="population_metric",
                subject="SNP",
                predicate="observed_in",
                object="cultivar",
                value=af_cultivar,
                unit="AF",
                scope={
                    "species": species,
                    "trait": trait,
                    "population": "cultivar",
                    "source": "cultivar_genotype_memmap",
                },
                evidence_level=EvidenceLevel.E3,
                payload={
                    "he": _row_float(row, "He"),
                    "sample_size": _row_float(row, "n_samples"),
                    "ac": _row_float(row, "AC"),
                    "an": _row_float(row, "AN"),
                    "callable_rate": _row_float(row, "callable_rate"),
                },
            ))

        af_wild = _row_float(row, "AF_wild")
        if af_wild is not None:
            records.append(EvidenceRecord(
                snp_key=snp.canonical_key,
                tool_run_id=tool_run_id,
                artifact_id=artifact_id,
                evidence_type="population_metric",
                subject="SNP",
                predicate="observed_in",
                object="wild",
                value=af_wild,
                unit="AF",
                scope={
                    "species": species,
                    "trait": trait,
                    "population": "wild",
                    "source": "wild_vcf",
                },
                evidence_level=EvidenceLevel.E3,
                payload={
                    "delta_af": _row_float(row, "delta_AF"),
                    "sample_size": _row_float(row, "n_wild"),
                    "ac": _row_float(row, "AC_wild"),
                    "an": _row_float(row, "AN_wild"),
                    "callable_rate": _row_float(row, "callable_rate_wild"),
                },
            ))

    store.upsert_snps(snps)
    return store.add_evidence_batch(records)


def _row_float(row: Any, column: str) -> float | None:
    try:
        value = getattr(row, column, None)
    except AttributeError:
        return None
    try:
        if value is None or value != value:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
