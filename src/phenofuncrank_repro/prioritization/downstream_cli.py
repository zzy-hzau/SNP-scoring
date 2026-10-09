"""Core-reference and allele-aware LD entry points; consumes upstream effect/genotype data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from phenofuncrank_repro.prioritization.core_reference_current import canonical_core_reference
from phenofuncrank_repro.prioritization.genotype_ld_current import candidate_ld, annotate_proxy_direction, expand_uploaded_ld


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    core = sub.add_parser('core', help='Select the real-allele upper fraction from a complete supplied trait-locus universe')
    core.add_argument('--effects', type=Path, required=True)
    core.add_argument('--fraction', type=float, default=0.05)
    core.add_argument('--output', type=Path, required=True)
    ld = sub.add_parser('ld', help='Compute genotype QC, signed support edges and candidate novelty LD')
    ld.add_argument('--candidates', type=Path, required=True)
    ld.add_argument('--spec', type=Path, required=True)
    ld.add_argument('--vcf', type=Path)
    ld.add_argument('--output-dir', type=Path, required=True)
    ld.add_argument('--novelty-seed-index', type=int, action='append', default=[], help='Optional per-seed M novelty vectors; no dense all-pairs matrix is created')
    expand = sub.add_parser('expand-vcf', help='Expand a supplied Core set against submitted genotypes')
    expand.add_argument('--core', type=Path, required=True)
    expand.add_argument('--panel', type=Path, required=True)
    expand.add_argument('--spec', type=Path, required=True)
    expand.add_argument('--vcf', type=Path, required=True)
    expand.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'core':
        table = pd.read_csv(args.effects, sep='\t')
        result = canonical_core_reference(table, args.fraction)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, sep='\t', index=False)
        metadata = {'source_sha256': hashlib.sha256(args.effects.read_bytes()).hexdigest(), 'supplied_rows': len(table), 'selected_rows': len(result), 'fraction': args.fraction, 'calibration_requirement': 'The supplied input must be the complete matching trait locus universe; a sampled cohort is not a global reference.'}
        args.output.with_suffix(args.output.suffix + '.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        return
    spec = json.loads(args.spec.read_text(encoding='utf-8'))
    if spec.get('inputs_dir') and not Path(spec['inputs_dir']).is_absolute():
        spec['inputs_dir'] = str(args.spec.resolve().parent / spec['inputs_dir'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == 'expand-vcf':
        core_frame = pd.read_csv(args.core, sep='\t')
        panel_frame = pd.read_csv(args.panel, sep='\t')
        expanded, pairs, audit = expand_uploaded_ld(core_frame, panel_frame, spec, args.vcf, return_audit=True)
        expanded.to_csv(args.output_dir / 'expanded_candidates.tsv', sep='\t', index=False)
        pairs.to_csv(args.output_dir / 'signed_support_pairs.tsv', sep='\t', index=False)
        audit.to_csv(args.output_dir / 'pair_audit.tsv', sep='\t', index=False)
        return
    frame = pd.read_csv(args.candidates, sep='\t')
    panel = candidate_ld(frame, spec, args.vcf)
    annotated = panel.annotate(frame)
    pairs = panel.support_pairs(annotated, spec)
    annotated = annotate_proxy_direction(annotated, pairs)
    annotated.to_csv(args.output_dir / 'candidate_QC_and_direction.tsv', sep='\t', index=False)
    pairs.to_csv(args.output_dir / 'signed_support_pairs.tsv', sep='\t', index=False)
    for index in args.novelty_seed_index:
        np.save(args.output_dir / f'candidate_novelty_r2_seed{index}.npy', panel(index))


if __name__ == '__main__':
    main()
