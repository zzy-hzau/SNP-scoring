"facade for the unified V17 variant normalizer."

from __future__ import annotations

from phenofuncrank_repro.interpretation.schemas.literature import PaperVariantMention
from phenofuncrank_repro.interpretation.variant_mention_normalizer import normalize_variant_mentions, select_normalized_variant_mention


def parse_variant_mentions(text: str, *, gene: str = "") -> list[PaperVariantMention]:
    return [mention.to_paper_variant_mention() for mention in normalize_variant_mentions(text, gene=gene)]


def select_variant_mention(
    text: str,
    *,
    candidate_pos: int | None = None,
    candidate_chrom: str = "",
    candidate_ref: str = "",
    candidate_alt: str = "",
    variant_description: str = "",
    gene: str = "",
) -> PaperVariantMention:
    return select_normalized_variant_mention(
        text,
        candidate_pos=candidate_pos,
        candidate_chrom=candidate_chrom,
        candidate_ref=candidate_ref,
        candidate_alt=candidate_alt,
        variant_description=variant_description,
        gene=gene,
    ).to_paper_variant_mention()


__all__ = ["parse_variant_mentions", "select_variant_mention"]
