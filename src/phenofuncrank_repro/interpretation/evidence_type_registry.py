"Evidence type ontology registry (SOP §4.7)."

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FROZEN_VERSION = "v1"


EVIDENCE_TYPE_ALIASES: dict[str, str] = {
    "variant_linked_eqtl": "eqtl",
    "allele_specific_expression": "allele_specific",
    "reporter_assay": "regulatory",
    "regulatory_assay": "regulatory",
    "natural_allele": "variant_specific",
    "precise_editing": "variant_specific",
    "variant_functional_assay": "variant_specific",
    "variant_specific_phenotype": "variant_specific",
    "protein_activity_assay": "molecular_mechanism",
}


CurationKind = Literal["standard", "extension", "mixed"]
GoldCategory = Literal["variant_specific", "gene_function", "phenotype_association", "mechanism", "population_evidence"]


@dataclass(frozen=True)
class EvidenceTypeDef:
    """One registered evidence type with ontology provenance."""

    name: str
    label: str
    ontology_source: str
    term_id: str
    definition: str
    curation: CurationKind
    gold_category: GoldCategory
    frozen_version: str = FROZEN_VERSION


EVIDENCE_TYPES: dict[str, EvidenceTypeDef] = {

    "functional_validation": EvidenceTypeDef(
        "functional_validation", "Functional validation",
        "Gene Ontology evidence codes", "IMP/IGI/IDA/IPI/IEP",
        "Experimental evidence for gene function (mutant, knockout, overexpression, complementation).",
        "standard", "gene_function",
    ),
    "gene_function": EvidenceTypeDef(
        "gene_function", "Gene function",
        "Gene Ontology (function) + curated gene databases", "GO molecular function; gene database annotations",
        "Gene-level functional evidence (annotation, domain, or experimental function).",
        "mixed", "gene_function",
    ),
    "mutant_phenotype": EvidenceTypeDef(
        "mutant_phenotype", "Mutant phenotype",
        "Plant Ontology / Trait Ontology", "PO phenotype structure; TO trait terms",
        "Observed phenotype of a mutant or transgenic line (height, sterility, architecture...).",
        "standard", "phenotype_association",
    ),
    "expression": EvidenceTypeDef(
        "expression", "Expression evidence",
        "Experimental Factor Ontology", "EFO expression terms",
        "Gene/transcript expression measurements (qPCR, microarray, RNA-seq, in situ).",
        "standard", "mechanism",
    ),
    "rna_seq": EvidenceTypeDef(
        "rna_seq", "RNA-seq evidence",
        "Experimental Factor Ontology", "EFO RNA-seq terms",
        "Expression or differential-expression evidence from RNA-seq.",
        "standard", "mechanism",
    ),
    "eqtl": EvidenceTypeDef(
        "eqtl", "Expression QTL",
        "Experimental Factor Ontology / Sequence Ontology", "EFO QTL terms; SO:0001626",
        "Variant-to-expression association (expression quantitative trait locus).",
        "standard", "mechanism",
    ),
    "regulatory": EvidenceTypeDef(
        "regulatory", "Regulatory evidence",
        "Project extension (PlantRegMap/JASPAR semantics)", "RPA:REG-001",
        "Regulatory interaction or cis-regulatory evidence for a gene/locus.",
        "extension", "mechanism",
    ),
    "promoter_motif": EvidenceTypeDef(
        "promoter_motif", "Promoter motif",
        "Project extension (JASPAR/PlantRegMap)", "RPA:MOTIF-001",
        "Transcription-factor binding motif found in a promoter window.",
        "extension", "mechanism",
    ),
    "tfbs_overlap": EvidenceTypeDef(
        "tfbs_overlap", "TFBS overlap",
        "Project extension (PlantRegMap genome tracks)", "RPA:TFBS-001",
        "Variant overlaps an annotated transcription-factor binding site.",
        "extension", "mechanism",
    ),
    "trait_association": EvidenceTypeDef(
        "trait_association", "Trait association",
        "Trait Ontology + Experimental Factor Ontology", "TO trait terms; EFO association terms",
        "Statistical association between locus and a trait (population or mapping study).",
        "standard", "phenotype_association",
    ),
    "gwas": EvidenceTypeDef(
        "gwas", "GWAS evidence",
        "Experimental Factor Ontology", "EFO GWAS terms",
        "Genome-wide association study hit for a trait.",
        "standard", "phenotype_association",
    ),
    "qtl": EvidenceTypeDef(
        "qtl", "QTL evidence",
        "Trait Ontology + Experimental Factor Ontology", "TO/EFO QTL terms",
        "Quantitative trait locus mapping hit for a trait.",
        "standard", "phenotype_association",
    ),
    "molecular_mechanism": EvidenceTypeDef(
        "molecular_mechanism", "Molecular mechanism",
        "Gene Ontology (BP/MF) + documented extension", "GO biological process/molecular function + RPA:MECH-001",
        "Molecular mechanism statement (pathway, enzymatic activity, signaling).",
        "mixed", "mechanism",
    ),
    "population_evidence": EvidenceTypeDef(
        "population_evidence", "Population evidence",
        "Experimental Factor Ontology + project extension", "EFO population terms + RPA:POP-001",
        "Allele frequency, selection signal or population-level evidence.",
        "mixed", "population_evidence",
    ),
    "pathway": EvidenceTypeDef(
        "pathway", "Pathway evidence",
        "Gene Ontology / Reactome", "GO pathways; Reactome pathway terms",
        "Pathway membership or pathway-level evidence.",
        "standard", "mechanism",
    ),
    "go": EvidenceTypeDef(
        "go", "GO annotation",
        "Gene Ontology", "GO terms",
        "Gene Ontology functional annotation.",
        "standard", "gene_function",
    ),
    "complementation": EvidenceTypeDef(
        "complementation", "Complementation",
        "Gene Ontology evidence codes", "IGI/IMP",
        "Mutant phenotype rescued by wild-type transgene (functional complementation).",
        "standard", "gene_function",
    ),
    "rescue": EvidenceTypeDef(
        "rescue", "Genetic rescue",
        "Gene Ontology evidence codes", "IGI/IMP",
        "Phenotype rescue establishing gene-function causality.",
        "standard", "gene_function",
    ),
    "transgenic": EvidenceTypeDef(
        "transgenic", "Transgenic evidence",
        "Gene Ontology evidence codes", "IGI/IDA",
        "Transgenic overexpression/knockdown phenotype evidence.",
        "standard", "gene_function",
    ),
    "variant_specific": EvidenceTypeDef(
        "variant_specific", "Variant-specific evidence",
        "Project extension", "RPA:VAR-001",
        "Evidence specific to a named variant/allele (not the gene in general).",
        "extension", "variant_specific",
    ),
    "allele_specific": EvidenceTypeDef(
        "allele_specific", "Allele-specific evidence",
        "Project extension", "RPA:VAR-002",
        "Allele-specific expression, effect or genotype-phenotype evidence.",
        "extension", "variant_specific",
    ),
    "ld_proxy": EvidenceTypeDef(
        "ld_proxy", "LD proxy",
        "Population genetics (standard)", "RPA:LD-001",
        "Linkage-disequilibrium proxy relationship between variants.",
        "standard", "variant_specific",
    ),

    "model_effect": EvidenceTypeDef(
        "model_effect", "DPC model effect",
        "Project extension (DPCformer)", "RPA:MODEL-001",
        "Deterministic DPCformer predicted additive effect.",
        "extension", "variant_specific",
    ),
    "variant_context": EvidenceTypeDef(
        "variant_context", "Variant context",
        "Project extension (VCF/GFF audit)", "RPA:VAR-003",
        "Variant position, alleles and genomic context after normalization.",
        "extension", "variant_specific",
    ),
    "functional_score": EvidenceTypeDef(
        "functional_score", "Functional score",
        "Project extension (PlantVImpact)", "RPA:SCORE-001",
        "Computational functional impact score (CADD-like).",
        "extension", "variant_specific",
    ),
    "motif_scan": EvidenceTypeDef(
        "motif_scan", "Motif scan",
        "Project extension (JASPAR/FIMO)", "RPA:MOTIF-002",
        "Transcription-factor motif scan result on sequence context.",
        "extension", "mechanism",
    ),
    "tfbs_density": EvidenceTypeDef(
        "tfbs_density", "TFBS density",
        "Project extension (PlantRegMap)", "RPA:TFBS-002",
        "Density of transcription-factor binding sites in a window.",
        "extension", "mechanism",
    ),
    "regulatory_annotation": EvidenceTypeDef(
        "regulatory_annotation", "Regulatory annotation",
        "Project extension (PlantRegMap)", "RPA:REG-002",
        "Annotated regulatory element overlap.",
        "extension", "mechanism",
    ),
    "rna_structure": EvidenceTypeDef(
        "rna_structure", "RNA structure",
        "Project extension (RNAfold)", "RPA:RNA-001",
        "Predicted RNA secondary structure change caused by the variant.",
        "extension", "mechanism",
    ),
    "plantregmap_search": EvidenceTypeDef(
        "plantregmap_search", "PlantRegMap search",
        "PlantRegMap", "RPA:REG-003",
        "PlantRegMap regulatory interaction search result.",
        "standard", "mechanism",
    ),
    "protein_domain": EvidenceTypeDef(
        "protein_domain", "Protein domain",
        "InterPro / Pfam", "InterPro entries; PFAM domains",
        "Protein domain annotation from InterPro/Pfam.",
        "standard", "gene_function",
    ),
    "protein_structure": EvidenceTypeDef(
        "protein_structure", "Protein structure",
        "Project extension (AlphaFold)", "RPA:STRUCT-001",
        "Predicted protein structure evidence.",
        "extension", "gene_function",
    ),
    "protein_structure_quality": EvidenceTypeDef(
        "protein_structure_quality", "Protein structure quality",
        "Project extension (AlphaFold pLDDT)", "RPA:STRUCT-002",
        "Model confidence for predicted protein structure.",
        "extension", "gene_function",
    ),
    "cds_translation": EvidenceTypeDef(
        "cds_translation", "CDS translation",
        "Project extension (GFF/CDS)", "RPA:PROT-001",
        "Translation consequence of a coding variant.",
        "extension", "variant_specific",
    ),
    "sequence_homology": EvidenceTypeDef(
        "sequence_homology", "Sequence homology",
        "NCBI BLAST (standard)", "RPA:HOM-001",
        "Homology evidence from sequence search.",
        "standard", "gene_function",
    ),
    "functional_annotation": EvidenceTypeDef(
        "functional_annotation", "Functional annotation",
        "Mixed (GFF + curated sources)", "RPA:ANN-001",
        "Combined functional annotation for a gene/protein.",
        "mixed", "gene_function",
    ),
    "gene_context": EvidenceTypeDef(
        "gene_context", "Gene context",
        "Genome annotation (GFF)", "RPA:GENE-001",
        "Variant-to-gene genomic context from reference annotation.",
        "standard", "gene_function",
    ),
    "gene_link": EvidenceTypeDef(
        "gene_link", "Gene link",
        "Project extension (knowledge graph)", "RPA:GENE-002",
        "Curated or inferred gene linkage evidence.",
        "extension", "gene_function",
    ),
    "known_gene_proximity": EvidenceTypeDef(
        "known_gene_proximity", "Known-gene proximity",
        "Genome annotation", "RPA:GENE-003",
        "Proximity of the variant to a known gene.",
        "standard", "gene_function",
    ),
    "uniprot_mapping": EvidenceTypeDef(
        "uniprot_mapping", "UniProt mapping",
        "UniProtKB", "RPA:UNIPROT-001",
        "Gene-to-UniProt accession mapping.",
        "standard", "gene_function",
    ),
    "gene_alias": EvidenceTypeDef(
        "gene_alias", "Gene alias",
        "Gene symbol registries", "RPA:GENE-004",
        "Alternative gene symbol/identifier mapping.",
        "standard", "gene_function",
    ),
    "gene_database": EvidenceTypeDef(
        "gene_database", "Gene database entry",
        "Curated gene databases", "RPA:GENE-005",
        "Gene entry from a curated database.",
        "standard", "gene_function",
    ),
    "pathway_annotation": EvidenceTypeDef(
        "pathway_annotation", "Pathway annotation",
        "Reactome / KEGG", "RPA:PATH-001",
        "Pathway annotation from curated pathway databases.",
        "standard", "mechanism",
    ),
    "pathway_membership": EvidenceTypeDef(
        "pathway_membership", "Pathway membership",
        "Reactome / KEGG", "RPA:PATH-002",
        "Gene membership in a curated pathway.",
        "standard", "mechanism",
    ),
    "pathway_projection": EvidenceTypeDef(
        "pathway_projection", "Pathway projection",
        "Project extension (orthology projection)", "RPA:PATH-003",
        "Pathway evidence projected from an orthologous gene.",
        "extension", "mechanism",
    ),
    "go_annotation": EvidenceTypeDef(
        "go_annotation", "GO annotation",
        "Gene Ontology", "RPA:GO-001",
        "Gene Ontology annotation for a gene.",
        "standard", "gene_function",
    ),
    "kegg_pathway": EvidenceTypeDef(
        "kegg_pathway", "KEGG pathway",
        "KEGG", "RPA:KEGG-001",
        "KEGG pathway annotation.",
        "standard", "mechanism",
    ),
    "kegg_pathway_annotation": EvidenceTypeDef(
        "kegg_pathway_annotation", "KEGG pathway annotation",
        "KEGG", "RPA:KEGG-002",
        "KEGG pathway annotation detail.",
        "standard", "mechanism",
    ),
    "kegg_pathway_membership": EvidenceTypeDef(
        "kegg_pathway_membership", "KEGG pathway membership",
        "KEGG", "RPA:KEGG-003",
        "Gene membership in a KEGG pathway.",
        "standard", "mechanism",
    ),
    "literature": EvidenceTypeDef(
        "literature", "Literature evidence",
        "PubMed / EuropePMC / OpenAlex / Semantic Scholar", "RPA:LIT-001",
        "Article-level literature evidence (title/metadata/abstract).",
        "standard", "gene_function",
    ),
    "population_metric": EvidenceTypeDef(
        "population_metric", "Population metric",
        "Population genetics (standard)", "RPA:POP-002",
        "Allele frequency, heterozygosity or selection metric.",
        "standard", "population_evidence",
    ),
    "user_hint": EvidenceTypeDef(
        "user_hint", "User annotation hint",
        "Project extension (user-supplied)", "RPA:USER-001",
        "Unverified user-supplied annotation; never treated as independent evidence.",
        "extension", "gene_function",
    ),
    "annotation_conflict": EvidenceTypeDef(
        "annotation_conflict", "Annotation conflict",
        "Project extension (GFF audit)", "RPA:AUDIT-001",
        "Audit record of conflicting candidate vs reference annotation.",
        "extension", "variant_specific",
    ),
    "sequence_context_alignment": EvidenceTypeDef(
        "sequence_context_alignment", "Sequence context alignment",
        "Project extension (local alignment)", "RPA:HOM-002",
        "Alignment of sequence context against reference/panels.",
        "extension", "variant_specific",
    ),
    "tf_family": EvidenceTypeDef(
        "tf_family", "TF family",
        "PlantTFDB (standard)", "RPA:TF-001",
        "Transcription-factor family classification.",
        "standard", "gene_function",
    ),
    "tf_family_inference": EvidenceTypeDef(
        "tf_family_inference", "TF family inference",
        "Project extension (classifier)", "RPA:TF-002",
        "Predicted transcription-factor family from sequence features.",
        "extension", "gene_function",
    ),
}


CLAIM_TYPE_REQUIREMENTS: dict[str, list[str]] = {
    "expression": ["eqtl", "rna_seq", "expression", "promoter_motif", "allele_specific"],
    "function": ["functional_validation", "mutant_phenotype", "complementation", "rescue", "transgenic"],
    "trait": [
        "variant_specific", "allele_specific", "trait_association",
        "gwas", "qtl", "mutant_phenotype",
    ],
    "regulatory": ["regulatory", "eqtl", "promoter_motif", "tfbs_overlap"],
}


EVIDENCE_PROXIMITY: dict[str, int] = {

    "variant_specific": 1,
    "allele_specific": 1,
    "eqtl": 1,
    "ld_proxy": 1,

    "mutant_phenotype": 2,
    "functional_validation": 2,
    "trait_association": 2,
    "gwas": 2,
    "qtl": 2,
    "complementation": 2,
    "rescue": 2,
    "transgenic": 2,
    "population_evidence": 2,

    "molecular_mechanism": 3,
    "expression": 3,
    "rna_seq": 3,
    "regulatory": 3,
    "promoter_motif": 3,
    "tfbs_overlap": 3,
    "pathway": 3,
    "pathway_annotation": 3,
    "pathway_membership": 3,
    "pathway_projection": 3,
    "kegg_pathway": 3,
    "kegg_pathway_annotation": 3,
    "kegg_pathway_membership": 3,
    "regulatory_annotation": 3,
    "motif_scan": 3,
    "tfbs_density": 3,
    "plantregmap_search": 3,
    "rna_structure": 3,
    "population_metric": 3,

    "gene_function": 4,
    "gene_context": 4,
    "gene_link": 4,
    "known_gene_proximity": 4,
    "uniprot_mapping": 4,
    "gene_alias": 4,
    "gene_database": 4,
    "protein_domain": 4,
    "protein_structure": 4,
    "protein_structure_quality": 4,
    "cds_translation": 4,
    "sequence_homology": 4,
    "functional_annotation": 4,
    "go": 4,
    "go_annotation": 4,
    "literature": 4,
    "user_hint": 4,
    "annotation_conflict": 4,
    "sequence_context_alignment": 4,
    "tf_family": 4,
    "tf_family_inference": 4,
}


GAP_LAYER_TYPES: dict[str, list[str]] = {
    "gene_function": ["gene_function", "functional_validation", "mutant_phenotype"],
    "trait_evidence": [
        "variant_specific", "allele_specific", "trait_association",
        "mutant_phenotype", "gwas", "qtl",
    ],
    "phenotype": [
        "variant_specific", "allele_specific", "trait_association",
        "mutant_phenotype", "gwas", "qtl",
    ],
    "variant_specific": ["variant_specific", "allele_specific"],
    "regulatory": ["expression", "eqtl", "promoter_motif", "regulatory"],
    "variant_ld": ["ld_proxy"],
    "pathway": ["pathway", "go"],
}

GAP_LAYER_TEMPLATES: dict[str, list[str]] = {
    "gene_function": [
        "{gene} function {species}",
        "{gene} protein characterization {species}",
        "{gene} mutant {species}",
        "{gene} knockout {species}",
        "{gene} cloning {species}",
        "{gene} gene isolation {species}",
    ],
    "trait_evidence": [
        "{gene} {trait} {species}",
        "{gene} mutant phenotype {species}",
        "{gene} {trait_term} {species}",
        "{gene} cloning {species}",
        "{gene} mutant characterization {species}",
    ],
    "phenotype": [
        "{gene} {trait} {species}",
        "{gene} mutant phenotype {species}",
        "{gene} {trait_term} {species}",
        "{gene} cloning {species}",
    ],
    "variant_specific": [
        "{snp_id} {gene} {species}",
        "{gene} {pos} {ref}>{alt}",
        "{gene} allele {species}",
    ],
    "regulatory": [
        "{gene} expression regulation {species}",
        "{gene} promoter {species}",
        "{gene} eQTL {species}",
        "{gene} transcription factor binding",
    ],
    "variant_ld": [
        "{gene} LD {species}",
        "{snp_id} linkage disequilibrium",
    ],
    "pathway": [
        "{gene} pathway {species}",
        "{gene} GO {species}",
    ],
}


def is_registered(evidence_type: str) -> bool:
    return evidence_type in EVIDENCE_TYPES


def canonical_evidence_type(evidence_type: str) -> str:
    """Return the frozen registry identifier for a public policy alias."""
    value = str(evidence_type or "").strip().lower()
    return EVIDENCE_TYPE_ALIASES.get(value, value)


def registered_types() -> list[str]:
    return sorted(EVIDENCE_TYPES)


def registry_stats() -> dict[str, int]:
    counts: dict[str, int] = {"standard": 0, "extension": 0, "mixed": 0}
    for entry in EVIDENCE_TYPES.values():
        counts[entry.curation] = counts.get(entry.curation, 0) + 1
    return {
        "total": len(EVIDENCE_TYPES),
        "frozen_version": FROZEN_VERSION,
        **counts,
    }


def unregistered_types(candidate: list[str]) -> list[str]:
    return sorted({t for t in candidate if t and not is_registered(t)})
