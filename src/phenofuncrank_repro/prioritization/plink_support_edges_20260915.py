"Explicit PLINK2 unphased-r2 producer for the paper profile."
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess


def command(plink, panel_prefix, panel_format, core_ids, chromosome, output_prefix, threads=4, memory_mb=1024):
    return [str(plink), '--'+panel_format, str(panel_prefix), '--chr', str(chromosome),
            '--ld-snp-list', str(core_ids), '--ld-window', '999999',
            '--ld-window-kb', '999999', '--ld-window-r2', '0.6',
            '--r2-unphased', '--allow-no-sex', '--allow-extra-chr',
            '--threads', str(threads), '--memory', str(memory_mb), '--out', str(output_prefix)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plink',type=Path,required=True)
    p.add_argument('--panel-prefix',type=Path,required=True)
    p.add_argument('--panel-format',choices=['bfile','pfile'],required=True)
    p.add_argument('--core-ids',type=Path,required=True)
    p.add_argument('--chromosome',required=True)
    p.add_argument('--output-prefix',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--memory-mb',type=int,default=1024)
    args=p.parse_args()
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(command(args.plink,args.panel_prefix,args.panel_format,args.core_ids,args.chromosome,args.output_prefix,args.threads,args.memory_mb),check=True)


if __name__=='__main__':
    main()
