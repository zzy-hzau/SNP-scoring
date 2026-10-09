"""Convert the supplied DPCformer CSV/MAP format through the native PED input builder."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
from . import inputs_diploid_ped, inputs_cotton_vcf


def prepare_csv(csv_path: Path, map_path: Path, outdir: Path, phenotype_column: str,
                sample_column: str = 'individual_id', genotype_column: str = 'genotype',
                species: str = 'diploid') -> dict:
    data = pd.read_csv(csv_path, sep=None, engine='python', dtype={sample_column: str})
    required = {sample_column, genotype_column, phenotype_column}
    if not required.issubset(data.columns):
        raise ValueError(f'Missing CSV columns: {sorted(required-set(data.columns))}')
    if data[sample_column].isna().any() or not data[sample_column].is_unique:
        raise ValueError('Sample identifiers must be present and unique')
    if data[sample_column].astype(str).str.contains(r'\s').any():
        raise ValueError('PED sample identifiers cannot contain whitespace')
    locus_map = pd.read_csv(map_path, sep=r'\s+', header=None)
    if locus_map.shape[1] != 4:
        raise ValueError('Provide a four-column MAP: chromosome, SNP ID, cM, position')
    sequences = []
    for sample, raw in zip(data[sample_column], data[genotype_column]):
        if pd.isna(raw):
            raise ValueError(f'Missing genotype string for {sample}')
        sequence = ''.join(str(raw).upper().split())
        if len(sequence) != 2*len(locus_map):
            raise ValueError(f'{sample}: expected exactly {2*len(locus_map)} alleles, got {len(sequence)}')
        if set(sequence)-set('ACGT0N'):
            raise ValueError(f'{sample}: genotype contains unsupported allele characters')
        sequences.append(sequence)
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ped = outdir/'csv_source.ped'
    phenotype = outdir/'csv_source_phenotypes.tsv'
    data.drop(columns=[genotype_column]).to_csv(phenotype, sep='\t', index=False)
    with ped.open('w', encoding='utf-8') as output:
        for sample, sequence in zip(data[sample_column], sequences):
            output.write(f'{sample} {sample} 0 0 0 -9 '+ ' '.join(sequence)+'\n')
    previous_arguments = sys.argv
    try:
        sys.argv = ['inputs_diploid_ped', '--ped', str(ped), '--map', str(map_path.resolve()),
                    '--pheno', str(phenotype), '--sample-id-col', sample_column,
                    '--outdir', str(outdir), '--aggregate-duplicates', 'none', '--species', species]
        inputs_diploid_ped.main()
    finally:
        sys.argv = previous_arguments
    if species == 'cotton':
        snps = pd.read_csv(outdir/'cultivar.snp_map.tsv.gz', sep='\t')
        numbers = snps.Chr.map(inputs_cotton_vcf.chrom_to_numeric)
        if not numbers.between(1,26).all():
            raise ValueError('Cotton chromosome labels must identify A1..A13/D1..D13 or 1..26')
        snps['Chr'] = numbers.map(lambda n: f'A{int(n):02d}' if n <= 13 else f'D{int(n)-13:02d}')
        snps['chromosome'] = numbers
        snps['group_id'] = snps.Chr.map(inputs_cotton_vcf.chrom_to_group)
        snps.to_csv(outdir/'cultivar.snp_map.tsv.gz', sep='\t', index=False, compression='gzip')
    metadata_path = outdir/'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    if species == 'cotton':
        metadata['n_groups'] = int(snps.group_id.max())
        metadata['chromosomes'] = snps.Chr.drop_duplicates().tolist()
    metadata['csv_input_contract'] = {
        'alleles_per_locus': 2,
        'biallelic_reference_orientation': 'Inferred allele labels, not reference-genome REF/ALT',
        'non_biallelic_or_monomorphic_policy': 'Native PED builder records these loci as unusable code 0',
        'chromosome_grouping': 'cotton homologous A/D pairs' if species == 'cotton' else 'separate MAP chromosomes',
        'minimum_valid_phenotype_n_for_native_training': 100,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--pheno-col', required=True)
    parser.add_argument('--sample-id-col', default='individual_id')
    parser.add_argument('--genotype-col', default='genotype')
    parser.add_argument('--species', default='diploid')
    args = parser.parse_args()
    prepare_csv(args.csv, args.map, args.outdir, args.pheno_col,
                args.sample_id_col, args.genotype_col, args.species)


if __name__ == '__main__':
    main()
