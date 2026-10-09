"""Join a position table to the original allele/consequence universe and GFF transcript context."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from phenofuncrank_repro.plantvimpact.native_upstream.candidate_universe import open_text, parse_attrs, clean_gene_id


def transcript_context(gff):
    by_gene = {}
    with open_text(str(gff)) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            attrs = parse_attrs(fields[8])
            if fields[2].lower() in {"mrna", "transcript", "rna"}:
                transcript = next((attrs[k] for k in ["ID", "transcript_id", "Name"] if k in attrs), None)
                gene = next((attrs[k] for k in ["Parent", "Locus_id", "locus_id", "gene_id", "gene"] if k in attrs), None)
                if transcript and gene:
                    by_gene.setdefault(clean_gene_id(gene), set()).add(clean_gene_id(transcript))
            elif fields[2].lower() == "gene":
                gene = next((attrs[k] for k in ["gene_id", "gene", "locus_tag", "ID", "Name"] if k in attrs), None)
                for key in ["Transcript variants", "transcript_variants", "transcript_id"]:
                    if gene and key in attrs:
                        by_gene.setdefault(clean_gene_id(gene), set()).update(clean_gene_id(x) for x in attrs[key].split(","))
    return {gene: ";".join(sorted(values)) for gene, values in by_gene.items()}


def select(universe, positions, gff, kind, output):
    positions = pd.read_csv(positions, sep="\t", dtype={"Chr": str})[["Chr", "Pos"]].drop_duplicates()
    positions["Pos"] = positions.Pos.astype("int64")
    context = transcript_context(gff)
    dataset = ds.dataset(universe, format="parquet")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    rows, seen = 0, set()
    try:
        for chrom, group in positions.groupby("Chr", sort=False):
            mask = (ds.field("Chr") == chrom) & ds.field("Pos").isin(group.Pos.to_numpy())
            for batch in dataset.scanner(filter=mask, batch_size=100000).to_batches():
                frame = batch.to_pandas()
                if frame.empty:
                    continue
                frame["transcript_ids"] = frame.GeneID.map(context).fillna("")

                frame["consequence"] = "missense_variant" if kind == "coding" else frame.region_type.astype(str)
                frame["annotation_scope"] = "native_gene_level_CDS" if kind == "coding" else "native_regulatory_interval"
                seen.update(zip(frame.Chr.astype(str), frame.Pos.astype(int)))
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output, table.schema)
                writer.write_table(table)
                rows += len(frame)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("No eligible alleles at the requested positions in the native universe")
    manifest = dict(requested_positions=len(positions), retained_positions=len(seen), allele_annotation_rows=rows, kind=kind, coding_scope="source missense-only; synonymous and stop changes excluded", transcript_scope="GFF gene-to-transcript context; native consequences preserved")
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", required=True)
    p.add_argument("--positions", required=True, help="TSV with Chr and Pos; exact source chromosome names")
    p.add_argument("--gff", required=True)
    p.add_argument("--kind", choices=["coding", "noncoding"], required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    select(a.universe, a.positions, a.gff, a.kind, a.output)


if __name__ == "__main__":
    main()
