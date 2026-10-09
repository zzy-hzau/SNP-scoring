"""Author's actual chickpea PLINK/PC1-PC5/GEMMA linear-model workflow."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd


def export_original_codes(codes, variants, phenotypes, output_prefix):
    """Source canonical0=missing,1=homREF,2=hetero,3=homALT;BIM allele1=REF."""
    codes=np.asarray(codes,dtype=np.uint8)
    if codes.shape!=(len(variants),len(phenotypes)) or codes.max()>3:raise ValueError('Require source variant-by-sample codes0..3 in supplied row order')
    prefix=Path(output_prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    states=np.array([1,0,2,3],dtype=np.uint8)[codes]
    if states.shape[1]%4:states=np.pad(states,((0,0),(0,4-states.shape[1]%4)),constant_values=1)
    grouped=states.reshape(len(variants),-1,4)
    packed=grouped[:,:,0]|(grouped[:,:,1]<<2)|(grouped[:,:,2]<<4)|(grouped[:,:,3]<<6)
    prefix.with_suffix('.bed').write_bytes(bytes.fromhex('6c1b01')+packed.tobytes())
    bim=pd.DataFrame({0:variants['chrom'].astype(str).str.replace('Ca','',regex=False),1:variants['snp_id'],2:0,3:variants['pos'],4:variants['ref'],5:variants['alt']})
    bim.to_csv(prefix.with_suffix('.bim'),sep='\t',index=False,header=False)
    aliases=[f'TS{i:04d}' for i in range(len(phenotypes))]
    values=pd.to_numeric(phenotypes['Hundred_seed_weight'],errors='coerce').to_numpy(dtype=float,copy=True);values[~np.isfinite(values)]=-9.
    fam=pd.DataFrame({0:aliases,1:aliases,2:0,3:0,4:0,5:values});fam.to_csv(prefix.with_suffix('.fam'),sep='\t',index=False,header=False)
    return aliases


def command(args, directory, name, blas_threads=1):
    env=dict(os.environ,OPENBLAS_NUM_THREADS=str(blas_threads),OMP_NUM_THREADS=str(blas_threads))
    with (directory/(name+'.stdout.log')).open('w') as log:subprocess.run([str(a) for a in args],cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def run_pipeline(input_bfile, phenotypes, plink, gemma, output_directory, memory_mb=8000, threads=4):
    out=Path(output_directory).resolve();out.mkdir(parents=True,exist_ok=True)
    bfile=Path(input_bfile).resolve();qc=out/'chickpea_qc'

    command([plink,'--bfile',bfile,'--maf','0.05','--geno','0.05','--allow-extra-chr','--make-bed','--out',qc,'--memory',memory_mb,'--threads',threads],out,'QC')
    fam=pd.read_csv(qc.with_suffix('.fam'),sep=r'\s+',header=None,dtype={0:str,1:str})
    bim=pd.read_csv(qc.with_suffix('.bim'),sep=r'\s+',header=None,dtype={0:str,1:str})
    bim[0]=bim[0].str.replace('Ca','',regex=False).astype(int);bim.to_csv(qc.with_suffix('.bim'),sep='\t',index=False,header=False)
    lookup=phenotypes.set_index('alias')
    if not lookup.index.is_unique:raise ValueError('Source sample aliases must be unique')
    pca=out/'chickpea_qc_pca'
    command([plink,'--bfile',qc,'--pca','5','--out',pca,'--memory',memory_mb,'--threads',threads],out,'PCA',threads)
    eig=pd.read_csv(pca.with_suffix('.eigenvec'),sep=r'\s+',header=None,dtype={0:str,1:str})
    if eig[1].tolist()!=fam[1].tolist() or eig.shape[1]!=7:raise ValueError('PCA rows must match FAM order and contain exactly fivePCs')
    cov=out/'PC1_5.txt';eig.iloc[:,2:].to_csv(cov,sep='\t',index=False,header=False)
    traits={'HSW':'Hundred_seed_weight','PW':'Plant_width'};summaries=[]
    for trait,column in traits.items():
        records=[];values=pd.to_numeric(lookup.reindex(fam[1])[column],errors='coerce').to_numpy(dtype=float,copy=True);values[~np.isfinite(values)]=-9.
        for chrom in sorted(bim[0].unique()):
            prefix=out/f'{trait}_chr{chrom}'
            command([plink,'--bfile',qc,'--chr',str(chrom),'--make-bed','--out',prefix,'--memory',memory_mb,'--threads',threads],out,f'split_{trait}_{chrom}')
            ff=pd.read_csv(prefix.with_suffix('.fam'),sep=r'\s+',header=None,dtype={0:str,1:str})
            if ff[1].tolist()!=fam[1].tolist():raise ValueError('Chromosome split changed sample order')
            ff[5]=values;ff.to_csv(prefix.with_suffix('.fam'),sep='\t',index=False,header=False)
            name=f'{trait}_chr{chrom}_PC_only'
            command([gemma,'-bfile',prefix,'-c',cov,'-lm','-o',name],out,name)
            records.append(pd.read_csv(out/'output'/(name+'.assoc.txt'),sep=r'\s+'))
        merged=pd.concat(records,ignore_index=True);merged.to_csv(out/f'{trait}_PC_only_merged.tsv',sep='\t',index=False)
        summaries.append({'trait':trait,'complete_phenotype_n':int((values!=-9).sum()),'retained_QC_markers':len(bim),'native_tested_variants':len(merged)})
    receipt={'primary_model':'GEMMA additive linear model:intercept andPC1-PC5,no kinship','MAF_min':.05,'missing_genotype_fraction_max':.05,'PCA_all_QC_markers':True,'additional_PCA_pruning':False,'GEMMA_scientific_flags':['-bfile','-c','-lm','-o'],'kinship_or_random_effect_added':False,'supplied_PC_columns':5,'source_GEMMA0985_effective_covariates':6,'resource_memory_mb':memory_mb,'resource_threads':threads,'traits':summaries}
    (out/'primary_workflow_scope.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--bfile',type=Path)
    inputs.add_argument('--codes',type=Path,help='Source canonical0..3 NPY in supplied phenotype row order')
    inputs.add_argument('--merge-prefix-list',type=Path,help='Source chromosome bfile prefix list;native --merge-list route')
    parser.add_argument('--variants',type=Path,help='Required with --codes:chrom,pos,snp_id,ref,alt')
    parser.add_argument('--phenotypes',type=Path,required=True,help='TSV:alias,Hundred_seed_weight,Plant_width')
    parser.add_argument('--plink',required=True);parser.add_argument('--gemma',required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    parser.add_argument('--memory-mb',type=int,default=8000);parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    phenotypes=pd.read_csv(args.phenotypes,sep='\t',dtype={'alias':str})
    bfile=args.bfile
    if args.codes:
        if not args.variants:parser.error('--variants is required with --codes')
        bfile=args.output_directory.resolve()/'prepared_input'
        phenotypes['alias']=export_original_codes(np.load(args.codes),pd.read_csv(args.variants,sep='\t'),phenotypes,bfile)
    elif args.merge_prefix_list:
        args.output_directory.mkdir(parents=True,exist_ok=True);bfile=args.output_directory.resolve()/'merged_input'
        command([args.plink,'--merge-list',args.merge_prefix_list.resolve(),'--allow-extra-chr','--make-bed','--out',bfile,'--memory',args.memory_mb,'--threads',args.threads],args.output_directory.resolve(),'merge_corrected')
    run_pipeline(bfile,phenotypes,args.plink,args.gemma,args.output_directory,args.memory_mb,args.threads)


if __name__=='__main__':main()
