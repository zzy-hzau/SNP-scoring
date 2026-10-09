"Normalize variant mentions in source passages."

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

from pydantic import BaseModel, Field

from phenofuncrank_repro.interpretation.schemas.literature import PaperVariantMention


MENTION_TYPES = {
    "NONE",
    "SNV",
    "INDEL",
    "TE_INSERTION",
    "NAMED_ALLELE",
    "HAPLOTYPE",
    "MUTANT",
    "EDITED_ALLELE",
    "PROTEIN_CHANGE",
    "PROMOTER_VARIANT",
    "REGULATORY_ALLELE",
    "STRUCTURAL_VARIANT",
    "GENERIC_VARIANT",
}


class UnifiedVariantMention(BaseModel):
    "A normalized observation of one paper-side variant mention."

    mention_type: str = "NONE"
    raw_mention: str = ""
    coordinate_text: str = ""
    chromosome: str = ""
    position: int | None = None
    ref: str = ""
    alt: str = ""
    protein_change: str = ""
    named_allele: str = ""
    structural_event: str = ""
    gene: str = ""
    relation: str = ""
    source_quote: str = ""
    identity_basis: list[str] = Field(default_factory=list)
    rs_id: str = ""
    vg_id: str = ""
    snp_id: str = ""
    haplotype: str = ""
    mutant_line: str = ""

    def informative_values(self) -> list[str]:
        values = (
            self.raw_mention,
            self.exact_coordinate,
            self.ref_alt,
            self.protein_change,
            self.named_allele,
            self.haplotype,
            self.mutant_line,
            self.structural_event,
            self.rs_id,
            self.vg_id,
            self.snp_id,
        )
        return [str(value).strip() for value in values if str(value or "").strip()]

    @property
    def exact_coordinate(self) -> str:
        if not self.chromosome or self.position is None:
            return ""
        return f"{self.chromosome}:{self.position}"

    @property
    def ref_alt(self) -> str:
        if not self.ref or not self.alt:
            return ""
        return f"{self.ref}>{self.alt}"

    def to_paper_variant_mention(self) -> PaperVariantMention:
        return PaperVariantMention(
            exact_coordinate=self.coordinate_text or self.exact_coordinate,
            ref_alt=self.ref_alt,
            rs_id=self.rs_id,
            vg_id=self.vg_id,
            snp_id=self.snp_id,
            amino_acid_change=self.protein_change,
            named_allele=self.named_allele,
            haplotype=self.haplotype,
            mutant_line=self.mutant_line,
            structural_variant=self.raw_mention if self.structural_event else "",
            identity_basis=list(self.identity_basis),
            mention_type=self.mention_type,
            raw_mention=self.raw_mention,
            chromosome=self.chromosome,
            position=self.position,
            protein_change=self.protein_change,
            structural_event=self.structural_event,
            gene=self.gene,
            relation=self.relation,
            source_quote=self.source_quote,
            coordinate_text=self.coordinate_text,
        )


def normalize_variant_quote(text: str) -> str:
    value = html.unescape(str(text or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\u00a0", " ")
    value = value.replace("\u2013", "-").replace("\u2014", "-")
    value = value.replace("\u2212", "-")
    value = value.replace("\u2192", ">")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


_COORD_RE = re.compile(
    r"(?P<chrom>(?:chr(?:omosome)?\s*)?[A-Za-z][A-Za-z0-9_.-]*|"
    r"(?:chr(?:omosome)?\s*)?[0-9]+)\s*:\s*"
    r"(?P<pos>\d[\d,]*)"
    r"|(?<![A-Za-z0-9])(?P<bare_pos>\d[\d,]{4,})(?![A-Za-z0-9])",
    re.I,
)
_REF_ALT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<ref>[ACGTN]{1,2})\s*"
    r"(?:>|/|->|to)\s*(?P<alt>[ACGTN]{1,2})(?![A-Za-z0-9])",
    re.I,
)
_C_CHANGE_RE = re.compile(
    r"(?<![A-Za-z0-9])c\.\s*(?:\d+|[-+]?\d+)(?:[ACGTN]+)?"
    r"(?:>|/|->|to|del|ins)[ACGTN-]+(?![A-Za-z0-9])",
    re.I,
)
_P_CHANGE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:p\.)?(?:[A-Z][a-z]{2}|[A-Z])\d+"
    r"(?:[A-Z][a-z]{2}|[A-Z*]|del|ins)(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])(?:[A-Z][a-z]{2}|[A-Z])\d+"
    r"(?:[A-Z][a-z]{2}|[A-Z*])(?![A-Za-z0-9])",
)
_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:rs\d+|VG[A-Za-z0-9_.-]+|SNP[_-]?[A-Za-z0-9_.-]+)"
    r"(?![A-Za-z0-9])",
    re.I,
)
_NAMED_RE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9_.\-/]{1,40})\s+"
    r"(?P<kind>alleles?|haplotypes?|mutants?(?:\s+lines?)?|mutations?|locus|lines?)\b",
    re.I,
)
_ALLELE_LABEL_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<phrase>(?i:allele)\s+"
    r"(?:[A-Z]|\d+|[A-Za-z][A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*))"
    r"(?![A-Za-z0-9])",
)
_QTL_HAPLOTYPE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:q[A-Za-z0-9]+/[A-Za-z0-9][A-Za-z0-9_-]+|"
    r"q[A-Za-z]{1,8}\d+(?:[-_][A-Za-z0-9]+)?)(?![A-Za-z0-9])",
    re.I,
)
_STRUCTURAL_RE = re.compile(
    r"(?P<phrase>(?:\d+\s*[- ]?bp\s+)?(?:insertion|deletion|indel|"
    r"transposable element|retrotransposon|transposon|tandem repeat|"
    r"structural variant|copy-number variant))",
    re.I,
)
_TE_NAME_RE = re.compile(
    r"(?:transposable element|retrotransposon|transposon)\s*"
    r"(?:\(|named\s+)?(?P<name>(?!insertion\b|element\b)[A-Za-z][A-Za-z0-9_.-]{2,40})\)?",
    re.I,
)
_PROMOTER_RE = re.compile(
    r"(?P<phrase>(?:promoter|regulatory|upstream|5['’]?\s*regulatory)"
    r"[^.;,]{0,70}(?:snp|variant|allele|mutation|insertion|deletion|polymorphism))",
    re.I,
)
_EDITED_RE = re.compile(
    r"(?P<phrase>(?:CRISPR(?:[/ -]?Cas9)?|gene[- ]edit(?:ed|ing)|edit(?:ed|ing))"
    r"[^.;,]{0,120}(?:allele|mutant|mutation|editing|genes?|system|line))",
    re.I,
)
_GENERIC_RE = re.compile(
    r"(?P<phrase>(?:single[- ]?nucleotide\s+(?:polymorphism|mutation|variant)|"
    r"single[- ]?base(?:pair)?\s+(?:mutation|deletion|insertion)|"
    r"(?:(?:derived\s+)?natural)\s+(?:variation|alleles?|haplotypes?)|"
    r"(?:natural|genetic)\s+variation|"
    r"\b(?:SNP|SNV|variant|polymorphism|haplotype|allele|mutation|"
    r"mutants?|insertion|deletion)\b))",
    re.I,
)
_GENERIC_NAMED_PREFIXES = {
    "base", "single", "nucleotide", "single base", "single-base",
    "gene", "natural", "genetic", "rice", "weedy", "of",
    "edited", "targeted", "generated", "new", "that",
}


def _clean_span(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" \t,;:()[]")


def _coordinate_fields(match: re.Match[str]) -> tuple[str, int | None, str]:
    chrom = (match.group("chrom") or "").strip()
    raw_pos = (match.group("pos") or match.group("bare_pos") or "").replace(",", "")
    if not raw_pos:
        return "", None, ""
    try:
        position = int(raw_pos)
    except ValueError:
        return "", None, ""
    chrom = re.sub(r"^chromosome\s*", "chr", chrom, flags=re.I)
    if chrom and not chrom.lower().startswith("chr"):
        chrom = chrom
    return chrom, position, _clean_span(match.group(0))


def _basis(mention: UnifiedVariantMention) -> list[str]:
    values: list[str] = []
    if mention.chromosome and mention.position is not None:
        values.append("coordinate")
    if mention.ref and mention.alt:
        values.append("ref_alt")
    if mention.protein_change:
        values.append("protein_change")
    if mention.named_allele or mention.haplotype or mention.mutant_line:
        values.append("named_allele")
    if mention.structural_event:
        values.append("structural_event")
    if mention.mention_type in {"SNV", "GENERIC_VARIANT"} and not values:
        values.append("paper_variant_term")
    return sorted(set(values))


def _new(
    *,
    quote: str,
    raw: str,
    mention_type: str = "GENERIC_VARIANT",
    chromosome: str = "",
    position: int | None = None,
    ref: str = "",
    alt: str = "",
    protein_change: str = "",
    named_allele: str = "",
    structural_event: str = "",
    gene: str = "",
    relation: str = "",
    haplotype: str = "",
    mutant_line: str = "",
    rs_id: str = "",
    vg_id: str = "",
    snp_id: str = "",
    coordinate_text: str = "",
) -> UnifiedVariantMention:
    mention = UnifiedVariantMention(
        mention_type=mention_type if mention_type in MENTION_TYPES else "GENERIC_VARIANT",
        raw_mention=_clean_span(raw),
        coordinate_text=_clean_span(coordinate_text),
        chromosome=chromosome,
        position=position,
        ref=ref.upper(),
        alt=alt.upper(),
        protein_change=_clean_span(protein_change),
        named_allele=_clean_span(named_allele),
        structural_event=_clean_span(structural_event),
        gene=gene,
        relation=relation,
        source_quote=quote,
        haplotype=_clean_span(haplotype),
        mutant_line=_clean_span(mutant_line),
        rs_id=_clean_span(rs_id),
        vg_id=_clean_span(vg_id),
        snp_id=_clean_span(snp_id),
    )
    return mention.model_copy(update={"identity_basis": _basis(mention)})


def _deduplicate(mentions: list[UnifiedVariantMention]) -> list[UnifiedVariantMention]:
    result: list[UnifiedVariantMention] = []
    keys: set[tuple[Any, ...]] = set()
    for mention in mentions:
        key = (
            mention.mention_type,
            mention.raw_mention.lower(),
            mention.exact_coordinate.lower(),
            mention.ref_alt.lower(),
            mention.protein_change.lower(),
            mention.named_allele.lower(),
            mention.haplotype.lower(),
            mention.mutant_line.lower(),
            mention.structural_event.lower(),
        )
        if not any(key) or key in keys:
            continue
        keys.add(key)
        result.append(mention)
    return result


def normalize_variant_mentions(
    text: str,
    *,
    gene: str = "",
    relation: str = "",
) -> list[UnifiedVariantMention]:
    quote = normalize_variant_quote(text)
    if not quote:
        return []
    observed_gene = gene if gene and re.search(re.escape(gene), quote, re.I) else ""
    mentions: list[UnifiedVariantMention] = []

    coordinates = list(_COORD_RE.finditer(quote))
    ref_alt_matches = list(_REF_ALT_RE.finditer(quote))
    protein_matches = list(_P_CHANGE_RE.finditer(quote))
    c_matches = list(_C_CHANGE_RE.finditer(quote))


    

    for coord in coordinates:
        chrom, position, raw_coord = _coordinate_fields(coord)
        if position is None:
            continue
        nearest = min(ref_alt_matches, key=lambda m: abs(m.start() - coord.end()), default=None)
        ref = alt = ""
        if nearest is not None and abs(nearest.start() - coord.end()) <= 180:
            ref, alt = nearest.group("ref"), nearest.group("alt")
        nearest_protein = min(protein_matches, key=lambda m: abs(m.start() - coord.end()), default=None)
        protein = (
            nearest_protein.group(0)
            if nearest_protein is not None and abs(nearest_protein.start() - coord.end()) <= 260
            else ""
        )
        nearby_named_candidates = [
            item for item in _NAMED_RE.finditer(quote)
            if item.group("name").strip().lower() not in _GENERIC_NAMED_PREFIXES
        ]
        nearby_named = min(
            nearby_named_candidates,
            key=lambda m: (
                0 if m.group("kind").lower().startswith(("allele", "haplotype")) else
                1 if m.group("kind").lower().startswith(("mutant", "mutation")) else 2,
                abs(m.start() - coord.end()),
            ),
            default=None,
        )
        named_allele = ""
        haplotype = ""
        mutant_line = ""
        if nearby_named is not None and abs(nearby_named.start() - coord.end()) <= 300:
            named_phrase = _clean_span(nearby_named.group(0))
            named_kind = nearby_named.group("kind").lower()
            if named_kind.startswith("haplotype"):
                haplotype = named_phrase
            elif named_kind.startswith("mutant") or named_kind.startswith("mutation") or named_kind.startswith("line"):
                mutant_line = named_phrase
            else:
                named_allele = named_phrase
        raw = raw_coord
        if ref and alt:
            raw = f"{raw_coord} {ref.upper()}>{alt.upper()}"
        kind = "PROTEIN_CHANGE" if protein and not (ref and alt) else "SNV"
        mentions.append(_new(
            quote=quote, raw=raw, mention_type=kind,
            chromosome=chrom, position=position, ref=ref, alt=alt,
            protein_change=protein, named_allele=named_allele,
            haplotype=haplotype, mutant_line=mutant_line,
            gene=observed_gene, relation=relation,
            coordinate_text=raw_coord,
        ))

    for match in c_matches:
        mentions.append(_new(
            quote=quote, raw=match.group(0), mention_type="INDEL" if any(
                word in match.group(0).lower() for word in ("del", "ins")
            ) else "SNV", protein_change="", gene=observed_gene, relation=relation,
        ))

    for match in protein_matches:
        mentions.append(_new(
            quote=quote, raw=match.group(0), mention_type="PROTEIN_CHANGE",
            protein_change=match.group(0), gene=observed_gene, relation=relation,
        ))

    for match in _ID_RE.finditer(quote):
        value = match.group(0)
        lower = value.lower()
        mentions.append(_new(
            quote=quote, raw=value, mention_type="SNV",
            rs_id=value if lower.startswith("rs") else "",
            vg_id=value if lower.startswith("vg") else "",
            snp_id=value if lower.startswith("snp") else "",
            gene=observed_gene, relation=relation,
        ))


    
    for match in _NAMED_RE.finditer(quote):
        name = _clean_span(match.group("name"))
        if name.lower() in _GENERIC_NAMED_PREFIXES:
            continue
        kind = match.group("kind").lower()
        phrase = _clean_span(match.group(0))
        if kind.startswith("haplotype"):
            mention_type = "HAPLOTYPE"
            haplotype = phrase
            named = ""
            mutant = ""
        elif kind.startswith("mutant") or kind.startswith("mutation") or kind.startswith("line"):
            mention_type = "MUTANT"
            haplotype = ""
            named = ""
            mutant = phrase
        else:
            mention_type = "NAMED_ALLELE"
            haplotype = ""
            named = phrase
            mutant = ""
        mentions.append(_new(
            quote=quote, raw=phrase, mention_type=mention_type,
            named_allele=named, haplotype=haplotype, mutant_line=mutant,
            gene=observed_gene, relation=relation,
        ))

    for match in _ALLELE_LABEL_RE.finditer(quote):
        phrase = _clean_span(match.group("phrase"))
        if phrase.lower() in {m.raw_mention.lower() for m in mentions}:
            continue
        mentions.append(_new(
            quote=quote, raw=phrase, mention_type="NAMED_ALLELE",
            named_allele=phrase, gene=observed_gene, relation=relation,
        ))

    for match in _QTL_HAPLOTYPE_RE.finditer(quote):
        phrase = _clean_span(match.group(0))
        context = quote[max(0, match.start() - 45):min(len(quote), match.end() + 45)].lower()

        

        if "/" not in phrase and not any(
            term in context for term in ("qtl", "haplotype", "allele", "locus")
        ):
            continue
        if phrase.lower() in {m.raw_mention.lower() for m in mentions}:
            continue
        mentions.append(_new(
            quote=quote, raw=phrase, mention_type="HAPLOTYPE",
            haplotype=phrase, gene=observed_gene, relation=relation,
        ))


    for match in _STRUCTURAL_RE.finditer(quote):
        phrase = _clean_span(match.group("phrase"))
        lower = phrase.lower()
        event = "insertion" if "insertion" in lower else "deletion" if "deletion" in lower else "structural_variant"
        if "transposable" in lower or "retrotransposon" in lower or "transposon" in lower:
            kind = "TE_INSERTION"
            if event == "structural_variant" and (
                "within" in quote.lower() or "promoter" in quote.lower() or "insertion" in quote.lower()
            ):
                event = "insertion"
        elif "tandem repeat" in lower or "copy-number" in lower:
            kind = "STRUCTURAL_VARIANT"
        else:
            kind = "INDEL" if event in {"insertion", "deletion"} else "STRUCTURAL_VARIANT"
        mentions.append(_new(
            quote=quote, raw=phrase, mention_type=kind,
            structural_event=event, gene=observed_gene, relation=relation,
        ))

    for match in _TE_NAME_RE.finditer(quote):
        name = _clean_span(match.group("name"))
        context_start = max(0, match.start() - 80)
        context_end = min(len(quote), match.end() + 100)
        context = quote[context_start:context_end]
        raw = _clean_span(f"transposable element {name}")
        mentions.append(_new(
            quote=quote, raw=raw, mention_type="TE_INSERTION",
            named_allele=name, structural_event="insertion",
            gene=observed_gene, relation=relation,
        ))

    for match in _PROMOTER_RE.finditer(quote):
        phrase = _clean_span(match.group("phrase"))
        lower = phrase.lower()
        kind = "PROMOTER_VARIANT" if "promoter" in lower else "REGULATORY_ALLELE"
        mentions.append(_new(
            quote=quote, raw=phrase, mention_type=kind,
            structural_event="insertion" if "insertion" in lower else "",
            gene=observed_gene, relation=relation,
        ))

    for match in _EDITED_RE.finditer(quote):
        phrase = _clean_span(match.group("phrase"))
        mentions.append(_new(
            quote=quote, raw=phrase, mention_type="EDITED_ALLELE",
            gene=observed_gene, relation=relation,
        ))


    

    for match in _GENERIC_RE.finditer(quote):
        phrase = _clean_span(match.group("phrase"))
        lower = phrase.lower()
        if any(m.raw_mention.lower() == phrase.lower() for m in mentions):
            continue
        if "haplotype" in lower:
            kind = "HAPLOTYPE"
        elif "insertion" in lower or "deletion" in lower:
            kind = "INDEL"
        elif "snp" in lower or "single-nucleotide" in lower or "single nucleotide" in lower:
            kind = "SNV"
        elif "natural" in lower and "allele" in lower:
            kind = "NAMED_ALLELE"
        else:
            kind = "GENERIC_VARIANT"
        mentions.append(_new(
            quote=quote, raw=phrase, mention_type=kind,
            structural_event=("insertion" if "insertion" in lower else "deletion" if "deletion" in lower else ""),
            gene=observed_gene, relation=relation,
        ))

    return _deduplicate(mentions)


def select_normalized_variant_mention(
    text: str,
    *,
    candidate_pos: int | None = None,
    candidate_chrom: str = "",
    candidate_ref: str = "",
    candidate_alt: str = "",
    variant_description: str = "",
    gene: str = "",
    relation: str = "",
) -> UnifiedVariantMention:
    """Select the most informative observation without resolving identity."""
    mentions = normalize_variant_mentions(text, gene=gene, relation=relation)
    if not mentions:
        return UnifiedVariantMention(source_quote=normalize_variant_quote(text), gene="" if not gene else gene)
    description = normalize_variant_quote(variant_description).lower()
    wanted_ref_alt = f"{candidate_ref}>{candidate_alt}".lower() if candidate_ref and candidate_alt else ""
    wanted_chrom = re.sub(r"^chr", "", str(candidate_chrom).lower())

    def score(mention: UnifiedVariantMention) -> tuple[int, int, int]:
        value = 0
        if candidate_pos is not None and mention.position == candidate_pos:
            value += 120
        if wanted_chrom and mention.chromosome.lower().replace("chr", "") == wanted_chrom:
            value += 20
        if wanted_ref_alt and mention.ref_alt.lower() == wanted_ref_alt:
            value += 60
        if mention.named_allele and mention.named_allele.lower() in description:
            value += 30
        if mention.protein_change and mention.protein_change.lower() in description:
            value += 30
        type_value = {
            "SNV": 10, "INDEL": 10, "TE_INSERTION": 10,
            "NAMED_ALLELE": 8, "HAPLOTYPE": 8, "MUTANT": 8,
            "EDITED_ALLELE": 8,
            "PROTEIN_CHANGE": 7, "GENERIC_VARIANT": 1,
        }.get(mention.mention_type, 0)
        return value + type_value, len(mention.informative_values()), len(mention.raw_mention)

    return max(mentions, key=score)


__all__ = [
    "MENTION_TYPES",
    "UnifiedVariantMention",
    "normalize_variant_quote",
    "normalize_variant_mentions",
    "select_normalized_variant_mention",
]
