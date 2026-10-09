"source-level query expansion for plant literature discovery."

from __future__ import annotations

import re


HISTORICAL_SOURCE_TERMS: dict[str, tuple[str, ...]] = {
    "SD1": ("sd-1 green revolution rice", "sd-1 gibberellin 20-oxidase", "rice semidwarf sd-1"),
    "D2": ("d2 ebisu dwarf rice", "D2 CYP90D2 rice", "rice brassinosteroid d2"),
    "D61": ("d61 OsBRI1 rice", "D61 brassinosteroid rice", "rice dwarf d61"),
    "D18": ("d18-AD OsGA3ox2 rice", "D18 gibberellin rice", "rice d18 mutant"),
    "D14": ("d14 strigolactone rice", "D14 mutant tillering rice", "rice d14"),
    "D27": ("d27 strigolactone rice", "D27 tillering rice", "rice d27 mutant"),
    "D53": ("d53 strigolactone rice", "D53 repressor rice", "rice d53 mutant"),
    "D10": ("d10 CCD8 rice", "D10 strigolactone rice", "rice d10 tillering"),
    "D3": ("D3 strigolactone rice", "D3 D14 rice", "rice D3 mutant tillering"),
    "MAPK6": ("dsg1 MAPK6 rice", "OsMAPK6 dwarf rice", "rice MAPK6 mutant"),
    "THIS1": ("THIS1 rice tillering", "this1 rice plant height", "rice THIS1 mutant"),
    "GID1": ("gid1 gibberellin rice", "GID1 GA receptor rice", "rice gid1 dwarf"),
    "EUI1": ("eui1 rice internode", "EUI1 gibberellin rice", "rice eui mutant"),
    "WAK10": ("WAK10 stem height rice", "WAK10 ectodomain rice", "rice WAK10 cell wall"),
    "qSH1": ("qSH1 seed shattering", "qsh1 edited rice", "seed shattering rice"),
    "GL4": ("GL4 African rice", "GL4 SH4 orthologue", "GL4 seed shattering"),
    "TGW2": ("qTGW2", "TGW2 OsCNR1 promoter", "TGW2 grain width"),
    "WUSCHEL": ("lc locule number tomato", "WUSCHEL lc QTL", "tomato locule number"),
    "SH4": ("SH4 C9 weedy rice", "SH4 qSH1 seed shattering", "weedy rice seed shattering"),
    "GS3": ("GS3 OsPIN5b OsMYB30 CRISPR", "GS3 edited rice grain size", "GS3 target site rice"),
    "tb1": ("tb1 Hopscotch transposon", "teosinte branched1 maize", "tb1 enhancer maize"),
    "ZmCCT": ("ZmCCT CACTA insertion", "ZmCCT photoperiod maize", "maize flowering CACTA"),
    "ZmHKT1": ("ZmHKT1 retrotransposon", "ZmHKT1 salt tolerance maize", "maize HKT1 insertion"),
    "IPA1": ("IPA1 qWS8 ipa1-2D", "OsSPL14 IPA1 haplotype", "IPA1 tandem repeat rice"),
    "LRR_receptor_kinase_cluster": ("qGY2-1 Dongxiang wild rice", "rice yield QTL qGY2-1", "LRR receptor kinase rice QTL"),
    "OsBRI1": ("D61 OsBRI1 rice", "d61 brassinosteroid rice", "OsBRI1 mutant rice"),
    "OsGA3ox2": ("D18 d18-AD rice", "OsGA3ox2 d18 mutant", "rice gibberellin 3 beta hydroxylase"),
    "SLR1": ("slr1-1 rice", "SLR1 DELLA rice", "slender rice GA response"),
    "OsPYL9": ("OsPYL9 CRISPR rice", "PYL9 ABA receptor rice", "OsPYL9 drought rice"),
    "CabHLH10": ("CabHLH10 favorable allele chickpea", "CabHLH10 NIL chickpea", "chickpea yield CabHLH10"),
    "ZmDREB2.7": ("ZmDREB2.7 favorable allele", "ZmDREB2.7 promoter maize", "maize drought ZmDREB2.7"),
}

_GENERIC = {
    "allele", "alleles", "variant", "variants", "mutation", "mutations",
    "snp", "natural", "versus", "rice", "maize", "tomato", "chickpea",
    "cultivated", "progenitor", "reported", "target", "sites", "site",
    "high", "low", "width", "weight", "trait", "phenotype", "functional",
    "validation", "genome", "editing", "crispr", "homozygous", "heterozygous",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _gene_terms(gene: str) -> tuple[str, ...]:
    target = _key(gene)
    for name, terms in HISTORICAL_SOURCE_TERMS.items():
        if _key(name) == target:
            return terms
    return ()


def _with_species(phrase: str, species: str) -> str:
    """Append species once without changing the source term itself."""
    phrase = _norm(phrase)
    species = _norm(species)
    if not species:
        return phrase
    phrase_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9_.-]+", phrase)}
    species_tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9_.-]+", species)]
    if species_tokens and all(token in phrase_tokens for token in species_tokens):
        return phrase
    return " ".join(part for part in (phrase, species) if part)


def build_source_discovery_terms(
    *,
    gene: str,
    species: str,
    trait: str,
    variant_description: str = "",
    max_terms: int = 4,
) -> list[str]:
    "Return paper-discovery queries."
    gene = _norm(gene)
    species = _norm(species)
    trait = _norm(trait).replace("_", " ")
    terms: list[str] = []

    for phrase in _gene_terms(gene):
        terms.append(_with_species(phrase, species))


    

    description = _norm(variant_description)
    tokens = [
        token for token in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{1,30}", description)
        if token.lower() not in _GENERIC
    ]
    for token in tokens[:3]:
        terms.append(_with_species(" ".join(part for part in (gene, token) if part), species))


    

    if not terms and gene:
        terms.append(" ".join(part for part in (gene, trait, species) if part))

    output: list[str] = []
    seen: set[str] = set()
    for term in terms:
        cleaned = _norm(term)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
        if len(output) >= max(1, max_terms):
            break
    return output


__all__ = ["HISTORICAL_SOURCE_TERMS", "build_source_discovery_terms"]
