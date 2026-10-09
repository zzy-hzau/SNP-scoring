"Verify variant identity and source evidence."

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from phenofuncrank_repro.interpretation.evidence_contract import _L1_TYPES, _L2_TYPES, _L3_TYPES, _proximity_level
from phenofuncrank_repro.interpretation.schemas.literature import CandidateContext, EvidenceGapIntent, EvidenceProposal


IdentityStatus = Literal["EXACT", "RECONCILED", "AMBIGUOUS", "MISMATCH"]
VerificationDecision = Literal["PASS", "PARTIAL", "BLOCK"]


@dataclass(frozen=True)
class IdentityCheck:
    status: IdentityStatus
    requested: str
    observed: str
    matched_terms: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class ProposalVerification:
    decision: VerificationDecision
    relation: str
    species_identity: IdentityCheck
    evidence_type_valid: bool
    variant_identity: IdentityCheck
    gene_identity: IdentityCheck
    trait_identity: IdentityCheck
    required_level: str
    observed_level: str
    effective_level: str
    reasons: tuple[str, ...] = ()

    @property
    def trusted(self) -> bool:
        return self.decision == "PASS"

    @property
    def level3_support(self) -> str:
        if self.decision == "PASS":
            return "yes"
        if self.decision == "PARTIAL":
            return "uncertain"
        return "no"


_VARIANT_RELATIONS = {
    "variant_functional_effect",
    "variant_trait",
    "variant_expression",
    "variant_mechanism",
    "variant_protein_function",
}
_RELATION_LEVELS = {
    "variant_functional_effect": "L1",
    "variant_trait": "L1",
    "variant_expression": "L1",
    "variant_mechanism": "L1",
    "variant_protein_function": "L1",
    "gene_trait": "L2",
    "gene_mechanism": "L3",
}

_EXPRESSION_TERMS = (
    "expression", "transcript", "mrna", "mRNA", "eqtl", "e-qtl", "ase",
    "allele-specific", "reporter", "promoter activity", "transcript abundance",
)
_MECHANISM_TERMS = (
    "pathway", "signaling", "signalling", "biochemical", "enzyme", "protein",
    "interaction", "mechanism", "regulat", "biosynthesis", "activity",
)
_GENE_TRAIT_EXPERIMENT_TERMS = (
    "mutant", "knockout", "loss-of-function", "overexpression", "over-expression",
    "complement", "rescue", "transgenic", "phenotype", "caused",
)
_TRAIT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "plant_height": ("plant height", "height", "stature"),
    "seed_shattering": ("seed shattering", "shattering", "shatter"),
    "apical_dominance": ("apical dominance", "branching dominance"),
    "plant_width": ("plant width", "canopy width", "leaf width"),
    "fruit_locule_number": ("fruit locule number", "locule number", "locules"),
    "grain_size": ("grain size", "grain length", "grain width", "grain weight"),
    "flowering_time": ("flowering time", "flowering", "heading date"),
    "yield_per_plant": ("yield per plant", "yield"),
    "grain_width_weight": ("grain width", "grain weight"),
    "salt_tolerance": ("salt tolerance", "salinity tolerance", "salt sensitivity"),
}
_SPECIES_ALIASES: dict[str, tuple[str, ...]] = {
    "rice": ("rice", "oryza sativa", "oryza"),
    "maize": ("maize", "corn", "zea mays", "zea"),
    "cotton": ("cotton", "gossypium"),
    "chickpea": ("chickpea", "chick pea", "cicer arietinum", "cicer"),
    "tomato": ("tomato", "solanum lycopersicum", "solanum"),
}


_SPECIES_CONFLICT_ALIASES: dict[str, tuple[str, ...]] = {
    "arabidopsis": ("arabidopsis", "arabidopsis thaliana", "thaliana"),
    "barley": ("barley", "hordeum vulgare", "hordeum"),
    "wheat": ("wheat", "triticum aestivum", "triticum"),
    "soybean": ("soybean", "glycine max", "glycine"),
    "poplar": ("poplar", "populus", "populus trichocarpa"),
    "psathyrostachys": ("psathyrostachys", "psathyrostachys juncea"),
}
_DIRECTIONAL_RELATED = {
    "dwarf", "dwarfism", "tillering", "biomass", "vigor", "growth", "architecture",
}
_VARIANT_TERM_RE = re.compile(
    r"(?:\b(?:snp|variant|mutation|allele|haplotype|insertion|deletion|edit|edited|"
    r"substitution|polymorphism|e?qtl)\b|[ACGTN]+\s*[>/→-]\s*[ACGTN]+)",
    re.I,
)
_COORD_RE = re.compile(
    r"(?P<chrom>(?:chr(?:omosome)?\s*)?[0-9]+[A-Za-z]?)\s*[:\-]\s*"
    r"(?P<pos>[0-9]{3,})\s*(?P<alleles>[ACGTN]+\s*(?:>|/|→|to)\s*[ACGTN]+)?",
    re.I,
)
_ALLELE_RE = re.compile(r"\b([ACGTN]+)\s*(?:>|/|→|to)\s*([ACGTN]+)\b", re.I)
_GENE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{1,}")


def _norm(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("–", "-").replace("—", "-").replace("′", "'")
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _text(proposal: EvidenceProposal) -> str:
    return " ".join(
        str(value or "")
        for value in (
            proposal.claim,
            proposal.chunk.quote,
            proposal.variant_or_allele,
            proposal.gene,
            proposal.trait,
        )
    )


def _source_grounded_text(proposal: EvidenceProposal) -> str:
    "Return text that can establish an identity endpoint."
    source = proposal.source if isinstance(proposal.source, dict) else {}
    return " ".join(
        str(value or "")
        for value in (
            proposal.chunk.quote,
            source.get("title", ""),
            source.get("abstract", ""),
            source.get("fulltext", ""),
        )
    )


def _canonical_trait(value: str) -> str:
    text = _norm(value)
    if not text:
        return ""
    if text in _TRAIT_SYNONYMS:
        return text
    for canonical, terms in _TRAIT_SYNONYMS.items():
        if text in {_norm(term) for term in terms}:
            return canonical
    return text.replace(" ", "_")


def _canonical_species(value: str) -> str:
    text = _norm(value)
    if not text:
        return ""
    for canonical, terms in _SPECIES_ALIASES.items():
        if text == canonical or text in {_norm(term) for term in terms}:
            return canonical
    return text


def _species_identity(proposal: EvidenceProposal, candidate: CandidateContext) -> IdentityCheck:
    requested = _canonical_species(candidate.species)
    returned = _canonical_species(proposal.species)
    grounded = _norm(_source_grounded_text(proposal))
    if str(proposal.species_match or "unknown") == "other":
        return IdentityCheck("MISMATCH", candidate.species, proposal.species, (), "extractor marked a different species")
    if returned and requested and returned != requested:
        return IdentityCheck("MISMATCH", candidate.species, proposal.species, (), "species identity mismatch")


    

    
    requested_terms = _SPECIES_ALIASES.get(requested, ())
    requested_in_source = any(_norm(term) in grounded for term in requested_terms)
    other_terms = tuple(
        term
        for species, aliases in _SPECIES_CONFLICT_ALIASES.items()
        if species != requested
        for term in aliases
        if _norm(term) in grounded
    )
    if other_terms and not requested_in_source:
        return IdentityCheck(
            "MISMATCH",
            candidate.species,
            proposal.species or "source text",
            other_terms,
            "source explicitly names a non-target species",
        )
    if re.search(r"\bortholog(?:ue|ous)?\b", grounded) and not requested_in_source:
        return IdentityCheck(
            "AMBIGUOUS",
            candidate.species,
            proposal.species or "ortholog source",
            ("ortholog",),
            "ortholog evidence does not establish same-species identity",
        )
    if str(proposal.species_match or "unknown") == "ortholog":
        return IdentityCheck("AMBIGUOUS", candidate.species, proposal.species, (), "ortholog evidence is not same-species evidence")
    if returned == requested and requested:
        return IdentityCheck("EXACT", candidate.species, proposal.species, (proposal.species,), "same species")
    return IdentityCheck("AMBIGUOUS", candidate.species, proposal.species, (), "species not explicitly resolved")


def _trait_identity(proposal: EvidenceProposal, candidate: CandidateContext) -> IdentityCheck:
    requested = _canonical_trait(candidate.trait)
    quote = _norm(_source_grounded_text(proposal))
    terms = _TRAIT_SYNONYMS.get(requested, (requested.replace("_", " "),))
    exact_terms = tuple(term for term in terms if _norm(term) in quote)
    if exact_terms:
        return IdentityCheck("EXACT", candidate.trait, proposal.chunk.quote, exact_terms, "canonical trait term present in source")

    related = tuple(term for term in _DIRECTIONAL_RELATED if term in quote)
    if related:
        return IdentityCheck("MISMATCH", candidate.trait, proposal.chunk.quote, related, "related but non-equivalent trait term")
    if any(term in quote for term in ("trait", "phenotype", "associated")):
        return IdentityCheck("MISMATCH", candidate.trait, proposal.chunk.quote, (), "source trait is not the requested canonical trait")
    return IdentityCheck("AMBIGUOUS", candidate.trait, proposal.chunk.quote, (), "no source-grounded trait endpoint")


def _gene_identity(proposal: EvidenceProposal, candidate: CandidateContext) -> IdentityCheck:
    requested = str(candidate.gene or "").strip()
    target = _compact(requested)
    raw_quote = str(proposal.chunk.quote or "")
    quote = _compact(raw_quote)
    returned = str(proposal.gene or "").strip()
    observed = returned or proposal.chunk.quote
    if returned and _compact(returned) != target:
        return IdentityCheck("MISMATCH", requested, returned, (), "extractor named a different gene")
    if target and re.search(rf"(?<![A-Za-z0-9]){re.escape(requested)}(?![A-Za-z0-9])", raw_quote, re.I):
        return IdentityCheck("EXACT", requested, observed, (requested,), "target gene explicitly occurs in quote")
    if returned and _compact(returned) == target:
        return IdentityCheck("AMBIGUOUS", requested, returned, (returned,), "gene returned by extractor but absent from quote")
    if any(token in _norm(proposal.chunk.quote) for token in ("near ", "nearby", "adjacent", "neighbor", "neighbour", "locus")):
        return IdentityCheck("AMBIGUOUS", requested, proposal.chunk.quote, (), "proximity/locus context is not gene identity")
    return IdentityCheck("AMBIGUOUS", requested, "", (), "target gene not explicitly identified")


def _coordinate_identity(proposal: EvidenceProposal, candidate: CandidateContext) -> IdentityCheck:
    requested = f"{candidate.chrom}:{candidate.pos} {candidate.ref}>{candidate.alt}".strip()
    quote = proposal.chunk.quote
    normalized_quote = quote.lower().replace("chromosome", "chr")
    if not candidate.pos or not candidate.chrom:
        return IdentityCheck("AMBIGUOUS", requested, proposal.variant_or_allele or quote, (), "paper-defined variant requires alias reconciliation")
    chrom = _norm(candidate.chrom).replace("chromosome ", "").replace(" ", "")
    pos = str(candidate.pos)
    ref, alt = _norm(candidate.ref), _norm(candidate.alt)
    coords = list(_COORD_RE.finditer(normalized_quote))
    exact_coord = False
    explicit_conflict = False
    matched: list[str] = []
    for match in coords:
        observed_chrom = match.group("chrom").lower().replace("chromosome", "").replace("chr", "").replace(" ", "")
        observed_pos = match.group("pos")
        if observed_pos != pos:
            continue
        if observed_chrom != chrom.replace("chr", ""):
            explicit_conflict = True
            continue
        exact_coord = True
        matched.append(match.group(0))
        pair = match.group("alleles") or ""
        if pair:
            alleles = _ALLELE_RE.search(pair)
            if alleles and (_norm(alleles.group(1)) != ref or _norm(alleles.group(2)) != alt):
                explicit_conflict = True
    pairs = [(_norm(a), _norm(b)) for a, b in _ALLELE_RE.findall(normalized_quote)]
    if exact_coord and (not pairs or (ref, alt) in pairs):
        return IdentityCheck("EXACT", requested, proposal.variant_or_allele or quote, tuple(matched), "coordinate and allele match")
    if explicit_conflict or any(a == ref and b != alt for a, b in pairs) or any(a != ref and b == alt for a, b in pairs):
        return IdentityCheck("MISMATCH", requested, proposal.variant_or_allele or quote, tuple(matched), "coordinate or allele conflicts with requested variant")

    

    
    if any(term in normalized_quote for term in ("insertion", "deletion", "transposon", "indel")):
        return IdentityCheck("MISMATCH", requested, proposal.variant_or_allele or quote, (), "different variant class from requested SNV")

    
    snp_id = str(candidate.snp_id or "").strip()
    if snp_id and _compact(snp_id) in _compact(quote):
        return IdentityCheck("EXACT", requested, snp_id, (snp_id,), "requested variant identifier present")
    if _VARIANT_TERM_RE.search(quote):
        return IdentityCheck("AMBIGUOUS", requested, proposal.variant_or_allele or quote, (), "variant language present without exact coordinate/allele")
    return IdentityCheck("AMBIGUOUS", requested, "", (), "no exact variant identity")


def _paper_variant_identity(proposal: EvidenceProposal, candidate: CandidateContext) -> IdentityCheck:
    requested = candidate.variant_description or candidate.snp_id
    quote = _norm(proposal.chunk.quote)
    description = _norm(candidate.variant_description or "")
    snp_id = _norm(candidate.snp_id)
    mention_values = proposal.variant_mention.informative_values()
    observed = proposal.variant_or_allele or (
        " ".join(mention_values) if mention_values else proposal.chunk.quote
    )

    

    
    if snp_id and _compact(snp_id) in _compact(quote):
        return IdentityCheck("EXACT", requested, observed, (candidate.snp_id,), "canonical variant identifier present")
    distinctive = [term for term in re.findall(r"[a-z0-9]{4,}", description) if term not in {"variant", "allele", "versus", "expression", "associated"}]
    matched = tuple(term for term in distinctive if term in quote)
    if matched:
        return IdentityCheck("AMBIGUOUS", requested, observed, matched, "paper alias is present but explicit mapping provenance is absent")
    if any(term in quote for term in ("allele", "haplotype", "insertion", "mutation", "snp", "edit")):
        return IdentityCheck("AMBIGUOUS", requested, observed, (), "variant class present but alias is not uniquely mapped")
    return IdentityCheck("AMBIGUOUS", requested, "", (), "paper-defined variant is not explicitly identified")


def variant_identity(proposal: EvidenceProposal, candidate: CandidateContext) -> IdentityCheck:
    if candidate.pos and candidate.chrom:
        return _coordinate_identity(proposal, candidate)
    return _paper_variant_identity(proposal, candidate)


def _has_expression_endpoint(text: str) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in _EXPRESSION_TERMS)


def _has_mechanism_endpoint(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _MECHANISM_TERMS)


def _explicit_variant_link(text: str) -> bool:
    lowered = text.lower()
    if "e-qtl" in lowered or "eqtl" in lowered or "allele-specific" in lowered:
        return True
    if not _VARIANT_TERM_RE.search(text):
        return False
    return bool(re.search(
        r"(?:allele|variant|mutation|snp|haplotype|insertion|edit).{0,100}"
        r"(?:expression|transcript|phenotype|height|stature|trait|activity|function|effect|reporter|promoter)",
        lowered,
    ) or re.search(
        r"(?:expression|transcript|phenotype|height|stature|trait|activity|function|effect|reporter|promoter).{0,100}"
        r"(?:allele|variant|mutation|snp|haplotype|insertion|edit)",
        lowered,
    ))


_APPROVED_MENTION_TYPES = {
    "MUTANT", "EDITED_ALLELE", "INDEL", "TE_INSERTION",
    "NAMED_ALLELE", "HAPLOTYPE",
}


def _has_approved_variant_mention(proposal: EvidenceProposal) -> bool:
    mention = proposal.variant_mention
    mention_type = str(mention.mention_type or "NONE").upper()
    raw_parts = [
        str(mention.raw_mention or ""),
        str(mention.structural_variant or ""),
        str(mention.structural_event or ""),
        str(mention.named_allele or ""),
        str(mention.haplotype or ""),
        str(mention.mutant_line or ""),
    ]
    raw = " ".join(raw_parts).lower()
    if mention_type == "STRUCTURAL_VARIANT":
        return "tandem repeat" in raw or "copy-number" in raw or "copy number" in raw
    if mention_type == "SNV":
        return bool(
            mention.ref_alt or mention.exact_coordinate
            or mention.snp_id or mention.rs_id or mention.vg_id
        )
    if mention_type in _APPROVED_MENTION_TYPES:
        return True

    
    return any(
        term in raw
        for term in (
            "crispr", "edit", "editing", "mutant", "mutagen",
            "insertion", "deletion", "transposon", "retrotransposon",
            "tandem repeat",
        )
    )


def _evidence_type_contract(proposal: EvidenceProposal, target: str, text: str) -> tuple[bool, str]:
    """Check whether the declared type matches the requested relation.

    This is intentionally narrower than general semantic entailment.  It
    catches type/endpoint contradictions such as a mechanism sentence being
    declared ``mutant_phenotype`` while leaving nuanced interpretation to the
    independent Judge.
    """
    et = str(proposal.evidence_type or "").strip().lower()
    lowered = text.lower()
    if target in _VARIANT_RELATIONS:
        if et not in (_L1_TYPES | _L2_TYPES | _L3_TYPES):
            return False, "unregistered_evidence_type"
        if target == "variant_expression" and et not in {"eqtl", "allele_specific", "regulatory", "expression"}:
            return False, "variant_expression_type_mismatch"
        if target == "variant_trait" and et not in {
            "variant_specific", "allele_specific", "trait_association", "mutant_phenotype",
            "functional_validation", "complementation", "rescue", "transgenic",
        }:
            return False, "variant_trait_type_mismatch"
        return True, "variant_relation_type_allowed"
    if target == "gene_trait":
        if et not in _L2_TYPES:
            return False, "gene_trait_requires_L2_type"
        if et in {"mutant_phenotype", "functional_validation", "complementation", "rescue", "transgenic"}:
            if not any(term in lowered for term in _GENE_TRAIT_EXPERIMENT_TERMS):
                return False, "gene_trait_experiment_endpoint_missing"
        return True, "gene_trait_type_allowed"
    if target == "gene_mechanism":
        if et not in _L3_TYPES and et not in _L2_TYPES:
            return False, "gene_mechanism_requires_mechanism_type"
        if not _has_mechanism_endpoint(text):
            return False, "gene_mechanism_endpoint_missing"
        return True, "gene_mechanism_type_allowed"
    return False, "unknown_target_relation"


def verify_proposal(
    proposal: EvidenceProposal,
    gap: EvidenceGapIntent,
    candidate: CandidateContext,
    *,
    resolved_identity: Any | None = None,
) -> ProposalVerification:
    relation = str(proposal.supported_relation or gap.missing_link or "").strip().lower()
    target = str(gap.missing_link or "").strip().lower()
    if relation != target:
        variant = IdentityCheck("AMBIGUOUS", candidate.variant_description or candidate.snp_id, proposal.variant_or_allele, (), "proposal relation does not match requested gap")
        gene = _gene_identity(proposal, candidate)
        trait = _trait_identity(proposal, candidate)
        species = _species_identity(proposal, candidate)
        level = _proximity_level(proposal.model_dump(mode="json"))
        return ProposalVerification("BLOCK", relation, species, False, variant, gene, trait, _RELATION_LEVELS.get(target, "L4"), level, level, ("supported_relation_mismatch",))

    text = _text(proposal)
    species = _species_identity(proposal, candidate)
    variant = variant_identity(proposal, candidate)
    if resolved_identity is not None:
        resolved_status = str(getattr(resolved_identity, "status", "AMBIGUOUS"))
        if resolved_status not in {"EXACT", "RECONCILED", "AMBIGUOUS", "MISMATCH"}:
            resolved_status = "AMBIGUOUS"
        variant = IdentityCheck(
            resolved_status,
            candidate.variant_description or candidate.snp_id,
            proposal.variant_or_allele or proposal.chunk.quote,
            tuple(getattr(resolved_identity, "basis", ()) or ()),
            str(getattr(resolved_identity, "reason", "resolver result supplied by A1")),
        )
    gene = _gene_identity(proposal, candidate)
    trait = _trait_identity(proposal, candidate)
    evidence_type_valid, evidence_type_reason = _evidence_type_contract(proposal, target, text)
    observed_level = _proximity_level(proposal.model_dump(mode="json"))
    required_level = _RELATION_LEVELS.get(target, "L4")
    reasons: list[str] = []
    hard_block = False
    partial = False

    if not evidence_type_valid:
        hard_block = True
        reasons.append(evidence_type_reason)

    if species.status == "MISMATCH":
        hard_block = True
        reasons.append("species_identity_mismatch")
    elif species.status == "AMBIGUOUS":
        partial = True
        reasons.append("species_identity_ambiguous")

    if gene.status == "MISMATCH":
        hard_block = True
        reasons.append("gene_identity_mismatch")
    elif gene.status == "AMBIGUOUS":
        partial = True
        reasons.append("gene_identity_ambiguous")

    if target in _VARIANT_RELATIONS:
        if variant.status == "MISMATCH":
            hard_block = True
            reasons.append("variant_identity_mismatch")
        elif variant.status == "AMBIGUOUS":
            partial = True
            reasons.append("variant_identity_ambiguous")
        explicit_link = _explicit_variant_link(text) or _has_approved_variant_mention(proposal)
        if not explicit_link:
            partial = True
            reasons.append("variant_relation_not_explicit")

    if target in {"variant_expression"} and not _has_expression_endpoint(text):
        hard_block = True
        reasons.append("expression_endpoint_missing")
    if target == "variant_trait":
        if trait.status == "MISMATCH":
            hard_block = True
            reasons.append("trait_identity_mismatch")
        elif trait.status == "AMBIGUOUS":
            partial = True
            reasons.append("trait_identity_ambiguous")
        if not _has_expression_endpoint(text) and not any(term in text.lower() for term in ("phenotype", "trait", "height", "stature", "dwarf", "shatter", "yield", "grain")):
            hard_block = True
            reasons.append("trait_endpoint_missing")
    if target == "gene_trait":
        if trait.status == "MISMATCH":
            hard_block = True
            reasons.append("trait_identity_mismatch")
        elif trait.status == "AMBIGUOUS":
            partial = True
            reasons.append("trait_identity_ambiguous")
        if observed_level != "L2":
            hard_block = True
            reasons.append("gene_trait_requires_L2")
    if target == "gene_mechanism":
        if not _has_mechanism_endpoint(text):
            hard_block = True
            reasons.append("mechanism_endpoint_missing")
        if observed_level not in {"L2", "L3"}:
            hard_block = True
            reasons.append("mechanism_requires_L3_or_experimental")

    if hard_block:
        decision: VerificationDecision = "BLOCK"
    elif partial:
        decision = "PARTIAL"
    else:
        decision = "PASS"
        reasons.append("identity_and_relation_contract_pass")


    

    
    effective_level = observed_level
    if (
        decision == "PASS"
        and target in _VARIANT_RELATIONS
        and variant.status in {"EXACT", "RECONCILED"}
        and gene.status == "EXACT"
        and explicit_link
    ):
        effective_level = "L1"
        reasons.append("verified_variant_level_override")

    return ProposalVerification(
        decision, relation, species, evidence_type_valid, variant, gene, trait, required_level, observed_level,
        effective_level, tuple(dict.fromkeys(reasons)),
    )


def apply_verification(proposal: EvidenceProposal, result: ProposalVerification) -> EvidenceProposal:
    if result.decision == "PASS":
        return proposal.model_copy(update={
            "relation_match": "exact",
            "support": "direct",
            "directness": "direct",
            "verification_status": "reconciled" if result.variant_identity.status == "RECONCILED" else "passed",
        })
    if result.decision == "PARTIAL":
        return proposal.model_copy(update={
            "relation_match": "partial",
            "support": "partial",
            "directness": "partial",
            "verification_status": "pending",
        })
    return proposal.model_copy(update={
        "relation_match": "none",
        "support": "indirect",
        "directness": "indirect",
        "verification_status": "refuted",
    })


def verification_payload(result: ProposalVerification) -> dict[str, Any]:
    payload = asdict(result)
    for key in ("species_identity", "variant_identity", "gene_identity", "trait_identity"):
        payload[key] = asdict(result.__getattribute__(key))
    payload["level3_support"] = result.level3_support
    payload["trusted"] = result.trusted
    return payload
