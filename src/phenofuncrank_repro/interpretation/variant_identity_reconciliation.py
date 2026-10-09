"Deterministic Variant Identity Reconciliation."

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


IdentityStatus = Literal["EXACT", "RECONCILED", "AMBIGUOUS", "MISMATCH"]
MappingBasis = Literal[
    "coordinate", "ref_alt", "named_allele", "amino_acid_change",
    "paper_explicit_mapping", "curated_alias", "assembly_liftover",
    "transcript_codon", "protein_consequence",
]


class CanonicalVariant(BaseModel):
    species: str
    assembly: str
    chrom: str
    pos: int
    ref: str
    alt: str
    gene: str = ""
    transcript: str = ""
    protein_change: str = ""

    @model_validator(mode="after")
    def _validate_variant(self) -> "CanonicalVariant":
        if self.pos <= 0:
            raise ValueError("canonical variant position must be positive")
        if not self.ref or not self.alt:
            raise ValueError("canonical variant requires ref and alt")
        return self

    @property
    def key(self) -> str:
        return f"{self.species}|{self.assembly}|{self.chrom}|{self.pos}|{self.ref}>{self.alt}"


class VariantMention(BaseModel):

    exact_coordinate: str = ""
    ref_alt: str = ""
    rs_id: str = ""
    vg_id: str = ""
    snp_id: str = ""
    amino_acid_change: str = ""
    named_allele: str = ""
    haplotype: str = ""
    mutant_line: str = ""
    structural_variant: str = ""
    identity_basis: list[str] = Field(default_factory=list)

    def informative_values(self) -> list[str]:
        values = [
            self.exact_coordinate, self.ref_alt, self.rs_id, self.vg_id,
            self.snp_id, self.amino_acid_change, self.named_allele,
            self.haplotype, self.mutant_line, self.structural_variant,
        ]
        return [value.strip() for value in values if str(value or "").strip()]


class MappingEvidence(BaseModel):
    "An externally mapping claim supplied to the resolver."

    basis: MappingBasis
    source_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    target_variant_key: str = Field(min_length=1)
    assembly_from: str = ""
    assembly_to: str = ""
    confidence: float = Field(ge=0, le=1)


class ReconciliationResult(BaseModel):
    status: IdentityStatus
    canonical_variant: CanonicalVariant
    paper_mention: VariantMention
    basis: list[str] = Field(default_factory=list)
    supporting_source: str = ""
    supporting_quote: str = ""
    assembly_from: str = ""
    assembly_to: str = ""
    confidence: float = Field(ge=0, le=1)
    reason: str
    provenance_complete: bool = False


_COORD_RE = re.compile(
    r"(?P<chrom>(?:chr(?:omosome)?\s*)?[A-Za-z0-9]+)\s*[:\-]\s*"
    r"(?P<pos>\d[\d,]*)\s*(?P<alleles>[ACGTN]+\s*(?:>|/|→|to)\s*[ACGTN]+)?",
    re.I,
)
_ALLELE_RE = re.compile(r"\b([ACGTN]+)\s*(?:>|/|→|to)\s*([ACGTN]+)\b", re.I)
_AA_RE = re.compile(r"\b(?:p\.)?([A-Z][a-z]{2}|[A-Z])?(\d+)([A-Z][a-z]{2}|[A-Z*])\b")
_ID_RE = re.compile(r"\b(?:rs\d+|VG[A-Za-z0-9_.-]+|SNP[_-]?[A-Za-z0-9_.-]+)\b", re.I)
_STRUCTURAL_RE = re.compile(
    r"\b(?:\d+[- ]?bp\s+)?(?:insertion|deletion|indel|transposon|retrotransposon|"
    r"tandem repeat|haplotype|promoter variant|edit(?:ed|ing)?)\b",
    re.I,
)
_GENERIC = {
    "allele", "variant", "mutation", "snp", "polymorphism", "mutant",
}
_ALLOWED_RECONCILIATION_BASIS = {
    "coordinate", "ref_alt", "paper_explicit_mapping", "curated_alias",
    "assembly_liftover", "transcript_codon", "protein_consequence",
}


def _norm(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text)


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", _norm(value))


def _chrom(value: str) -> str:
    return re.sub(r"^(?:chr(?:omosome)?)[ ]*", "", _norm(value))


def _quote_contains(value: str, quote: str) -> bool:
    value_compact = _compact(value)
    quote_compact = _compact(quote)
    if len(value_compact) >= 3 and value_compact in quote_compact:
        return True
    tokens = [token for token in re.findall(r"[a-z0-9]+", _norm(value)) if len(token) >= 3]
    return bool(tokens) and sum(token in _norm(quote) for token in tokens) / len(tokens) >= 0.8


def parse_variant_mention(text: str, *, gene: str = "") -> VariantMention:
    """Parse explicit identity forms without deciding equivalence."""
    text = str(text or "")
    coord = _COORD_RE.search(text)
    bare_coordinate = None
    if not coord:

        

        
        bare_coordinate = re.search(r"(?<![A-Za-z0-9])\d[\d,]{4,}(?![A-Za-z0-9])", text)
    exact_coordinate = coord.group(0) if coord else (bare_coordinate.group(0) if bare_coordinate else "")
    ref_alt = ""
    if coord and coord.group("alleles"):
        ref_alt = coord.group("alleles")
    else:
        pair = _ALLELE_RE.search(text)
        if pair:
            ref_alt = f"{pair.group(1)}>{pair.group(2)}"
    ids = [match.group(0) for match in _ID_RE.finditer(text)]
    aa = _AA_RE.search(text)
    structural = _STRUCTURAL_RE.search(text)

    named = ""
    coordinate_chrom = _compact(coord.group("chrom")) if coord else ""
    identifier_values = {_compact(value) for value in ids}
    for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9_.-]{1,24}\b", text):
        value = match.group(0)
        compact_value = _compact(value)
        if (
            compact_value == _compact(gene)
            or _norm(value) in _GENERIC
            or compact_value == coordinate_chrom
            or compact_value in identifier_values
            or ":" in value
        ):
            continue
        if re.search(r"[-_]\d|\d[-_]|[A-Z]{2,}\d|\b(?:d|q|sd|slr)\d", value, re.I):
            named = value
            break
    basis: list[str] = []
    if exact_coordinate:
        basis.append("coordinate")
    if ref_alt:
        basis.append("ref_alt")
    if ids:
        basis.append("named_allele")
    if aa:
        basis.append("amino_acid_change")
    if named:
        basis.append("named_allele")
    if structural:
        basis.append("named_allele")
    return VariantMention(
        exact_coordinate=exact_coordinate,
        ref_alt=ref_alt,
        rs_id=next((x for x in ids if x.lower().startswith("rs")), ""),
        vg_id=next((x for x in ids if x.lower().startswith("vg")), ""),
        snp_id=next((x for x in ids if x.lower().startswith("snp")), ""),
        amino_acid_change=aa.group(0) if aa else "",
        named_allele=named,
        structural_variant=structural.group(0) if structural else "",
        identity_basis=sorted(set(basis)),
    )


def _mention_variant_class(mention: VariantMention) -> str:
    text = _norm(" ".join(mention.informative_values()))
    if any(term in text for term in ("insertion", "deletion", "indel", "transposon", "repeat")):
        return "structural"
    if mention.ref_alt or mention.exact_coordinate or mention.rs_id or mention.vg_id:
        return "snv_or_small_variant"
    if mention.amino_acid_change:
        return "protein_change"
    if mention.named_allele or mention.haplotype or mention.mutant_line:
        return "named_variant"
    return "generic"


def _explicit_coordinate(mention: VariantMention) -> tuple[str, int, str, str] | None:
    raw = mention.exact_coordinate
    match = _COORD_RE.search(raw)
    if not match:
        bare = re.fullmatch(r"\d[\d,]{4,}", raw.strip())
        if not bare:
            return None
        ref, alt = "", ""
        if mention.ref_alt:
            pair = _ALLELE_RE.search(mention.ref_alt)
            if pair:
                ref, alt = pair.group(1), pair.group(2)
        return "", int(bare.group(0).replace(",", "")), ref.upper(), alt.upper()
    ref, alt = "", ""
    if match.group("alleles"):
        pair = _ALLELE_RE.search(match.group("alleles"))
        if pair:
            ref, alt = pair.group(1), pair.group(2)
    if not ref and mention.ref_alt:
        pair = _ALLELE_RE.search(mention.ref_alt)
        if pair:
            ref, alt = pair.group(1), pair.group(2)
    return (
        match.group("chrom"),
        int(match.group("pos").replace(",", "")),
        ref.upper(),
        alt.upper(),
    )


def _result(
    status: IdentityStatus,
    canonical: CanonicalVariant,
    mention: VariantMention,
    *,
    basis: list[str],
    reason: str,
    confidence: float,
    mapping: MappingEvidence | None = None,
) -> ReconciliationResult:
    return ReconciliationResult(
        status=status,
        canonical_variant=canonical,
        paper_mention=mention,
        basis=sorted(set(basis)),
        supporting_source=mapping.source_id if mapping else "",
        supporting_quote=mapping.quote if mapping else "",
        assembly_from=mapping.assembly_from if mapping else "",
        assembly_to=mapping.assembly_to if mapping else "",
        confidence=confidence,
        reason=reason,
        provenance_complete=bool(
            status in {"EXACT", "RECONCILED"}
            and mapping is not None
            and mapping.source_id
            and mapping.source_type
            and mapping.quote
            and mapping.target_variant_key == canonical.key
        ),
    )


def reconcile_variant_identity(
    canonical: CanonicalVariant,
    mention: VariantMention,
    *,
    paper_id: str,
    paper_quote: str,
    paper_assembly: str = "",
    paper_gene: str = "",
    mapping_evidence: list[MappingEvidence] | None = None,
) -> ReconciliationResult:
    """Reconcile one paper mention against one canonical variant.

    Promotion rules are deliberately asymmetric: explicit contradiction can
    produce MISMATCH; absence of a mapping produces AMBIGUOUS; only an exact
    identity or a source-backed mapping can produce EXACT/RECONCILED.
    """
    if not paper_id or not paper_quote.strip():
        return _result(
            "AMBIGUOUS", canonical, mention, basis=[],
            reason="paper_id and exact supporting quote are required for identity audit",
            confidence=0.0,
        )
    if paper_gene and canonical.gene and _norm(paper_gene) != _norm(canonical.gene):
        return _result(
            "MISMATCH", canonical, mention, basis=["gene_identity"],
            reason="paper source is annotated to a different gene than the requested variant",
            confidence=0.99,
        )
    values = mention.informative_values()
    if not values:
        return _result(
            "AMBIGUOUS", canonical, mention, basis=[],
            reason="paper quote contains no specific variant mention",
            confidence=0.1,
        )
    if not any(_quote_contains(value, paper_quote) for value in values):
        return _result(
            "MISMATCH", canonical, mention, basis=[],
            reason="structured paper mention is not present in the supporting quote",
            confidence=0.98,
        )

    explicit = _explicit_coordinate(mention)
    if explicit:
        observed_chrom, observed_pos, observed_ref, observed_alt = explicit
        chrom_same = _chrom(observed_chrom) == _chrom(canonical.chrom)
        pos_same = observed_pos == canonical.pos
        allele_present = bool(observed_ref or observed_alt)
        alleles_same = (
            observed_ref == canonical.ref.upper() and observed_alt == canonical.alt.upper()
            if allele_present else True
        )
        if chrom_same and pos_same and alleles_same and (not paper_assembly or paper_assembly == canonical.assembly):
            mapping = MappingEvidence(
                basis="coordinate", source_id=paper_id, source_type="paper",
                quote=paper_quote, target_variant_key=canonical.key,
                assembly_from=paper_assembly or canonical.assembly,
                assembly_to=canonical.assembly, confidence=1.0,
            )
            return _result(
                "EXACT", canonical, mention,
                basis=["coordinate", "ref_alt"] if allele_present else ["coordinate"],
                reason="paper coordinate and allele match the canonical variant",
                confidence=1.0, mapping=mapping,
            )
        mapping_can_explain_coordinate = any(
            mapping.basis in {"assembly_liftover", "transcript_codon"}
            and mapping.target_variant_key == canonical.key
            and mapping.source_id
            and mapping.source_type
            and mapping.quote
            and any(_quote_contains(value, paper_quote) for value in values)
            for mapping in (mapping_evidence or [])
        )
        if chrom_same and pos_same and allele_present and not alleles_same:
            if not mapping_can_explain_coordinate:
                return _result(
                    "MISMATCH", canonical, mention, basis=["coordinate", "ref_alt"],
                    reason="paper coordinate matches but REF/ALT conflicts with canonical variant",
                    confidence=0.99,
                )
        if chrom_same and not pos_same:
            if not mapping_can_explain_coordinate:
                return _result(
                    "MISMATCH", canonical, mention, basis=["coordinate"],
                    reason="paper coordinate explicitly differs from canonical position",
                    confidence=0.99,
                )

    if _mention_variant_class(mention) == "structural" and len(canonical.ref) == len(canonical.alt) == 1:
        return _result(
            "MISMATCH", canonical, mention, basis=["named_allele"],
            reason="paper explicitly describes a structural variant while canonical target is an SNV",
            confidence=0.97,
        )

    for mapping in mapping_evidence or []:
        if (
            mapping.basis in _ALLOWED_RECONCILIATION_BASIS
            and mapping.target_variant_key == canonical.key
            and mapping.source_id
            and mapping.source_type
            and mapping.quote
            and any(_quote_contains(value, paper_quote) for value in values)
        ):
            return _result(
                "RECONCILED", canonical, mention,
                basis=[mapping.basis],
                reason="paper-defined mention is linked to canonical variant by explicit mapping evidence",
                confidence=mapping.confidence, mapping=mapping,
            )

    return _result(
        "AMBIGUOUS", canonical, mention,
        basis=["named_allele" if len(values) else "gene_context_only"],
        reason="specific paper mention is present, but no auditable mapping to the canonical variant was supplied",
        confidence=0.35,
    )


__all__ = [
    "CanonicalVariant", "VariantMention", "MappingEvidence",
    "ReconciliationResult", "parse_variant_mention", "reconcile_variant_identity",
]
