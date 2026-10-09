#!/usr/bin/env python3
"PlantDeepSEA/RiceVarMap/PICNC comparator recipes with IO."
from phenofuncrank_repro.common.paths import project_root, package_dir
from pathlib import Path
import sys,json,hashlib,concurrent.futures
import numpy as np,pandas as pd,requests
ROOT = project_root(__file__)
O=RES=GENOME=CHECKPOINT=None
DEVICE="cpu"
POOL_TAGS=("maize_coding","rice_coding","rice_noncoding")
CHECKPOINT_SHA256=None

def get_pool(species):
    paths=[O/(tag+"_pool.parquet") for tag in sorted(POOL_TAGS)
           if tag.startswith(species+"_")]
    return pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)


def ricevarmap():
    rice=get_pool('rice');rice['var']='vg'+rice.chr_num.astype(str).str.zfill(2)+rice.Pos.astype(int).astype(str).str.zfill(8)
    wanted=set(rice['var']);rows=[];status=[]
    def one(chrom):
        filename=f'chr{chrom:02d}_snpeff_coovar_polyphen_sift_merge_anno.csv.gz';p=RES/filename;url='https://ricevarmap.ncpgr.cn/media/'+filename
        if not p.exists():
            r=requests.get(url,timeout=90,stream=True);r.raise_for_status()
            tmp=p.with_suffix('.partial')
            with tmp.open('wb') as f:
                for block in r.iter_content(1024*1024):f.write(block)
            tmp.replace(p)
        use=['var','var_ref','alt','gene','transcript','snpeff_anno','sift_score','polyphen_score'];hit=[]
        for chunk in pd.read_csv(p,usecols=use,dtype=str,chunksize=200000):
            q=chunk[chunk['var'].isin(wanted)]
            if len(q):hit.append(q)
        d=pd.concat(hit,ignore_index=True) if hit else pd.DataFrame(columns=use)
        print('RICEVARMAP_CHR',chrom,'annotation_rows',len(d),flush=True)
        return d,dict(chromosome=chrom,url=url,size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        for d,s in ex.map(one,range(1,13)):rows.append(d);status.append(s)
    anno=pd.concat(rows,ignore_index=True);anno['SIFT']=1-pd.to_numeric(anno.sift_score,errors='coerce');anno['PolyPhen2']=pd.to_numeric(anno.polyphen_score,errors='coerce')
    exact=rice[['snp_id','var','Ref','Alt']].merge(anno,left_on=['var','Ref','Alt'],right_on=['var','var_ref','alt'],how='left')
    exact.to_csv(O/'RiceVarMap_allele_annotation_rows.tsv',sep='\t',index=False)
    exact.groupby('snp_id')[['SIFT','PolyPhen2']].max().reset_index().to_csv(O/'RiceVarMap_scores.tsv',sep='\t',index=False)
    (O/'RiceVarMap_sources.json').write_text(json.dumps(status,indent=2))


def picnc():
    d=pd.read_csv(RES/'SNP_PNC.csv.gz');d['snp_id']=d.CHROM.astype(str)+'_'+d.POS.astype(str)+'_'+d.REF+'_'+d.ALT
    assert not d.snp_id.duplicated().any()
    p=get_pool('maize');j=p[['snp_id','role']].merge(d[['snp_id','predicted_PNC']],on='snp_id',how='left',validate='one_to_one');j.rename(columns={'predicted_PNC':'PICNC'}).to_csv(O/'PICNC_scores.tsv',sep='\t',index=False)
    print('PICNC_COVERAGE',j.groupby('role').predicted_PNC.count().to_dict(),flush=True)


def plantdeepsea():
    import torch,pysam
    sys.path.insert(0,str(ROOT/"third_party/selene"))
    from phenofuncrank_repro.vendor.deeperdeepsea import DeeperDeepSEA
    torch.set_num_threads(4);device=DEVICE
    checkpoint=CHECKPOINT
    if CHECKPOINT_SHA256 and hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA256:
        raise ValueError('Published checkpoint hash differs from frozen profile')

    
    raw=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state={k.replace('module.model.',''):v for k,v in raw['state_dict'].items()}
    model=DeeperDeepSEA(1000,15);model.load_state_dict(state,strict=True);model.to(device).eval()
    genome=GENOME;fa=pysam.FastaFile(str(genome))
    def chrom_name(n):
        for name in [f'chr{n:02d}',f'chr{n}',str(n)]:
            if name in fa.references:return name
        raise ValueError(n)
    p=get_pool('rice');data=[];ident=[];audit=[]
    lut=np.full((256,4),.25,dtype=np.float32)
    for i,b in enumerate('ACGT'):lut[ord(b)]=np.eye(4,dtype=np.float32)[i]
    for row in p.itertuples(index=False):
        chrom=chrom_name(int(row.chr_num));pos0=int(row.Pos)-1;start=pos0-500;end=start+1000
        if start<0 or end>fa.get_reference_length(chrom):audit.append(dict(snp_id=row.snp_id,status='out_of_bounds'));continue
        seq=fa.fetch(chrom,start,end).upper()
        if seq[500]!=row.Ref:audit.append(dict(snp_id=row.snp_id,status='reference_mismatch',expected=row.Ref,genome=seq[500]));continue
        ref=lut[np.frombuffer(seq.encode(),dtype=np.uint8)].T.copy();alt=ref.copy();alt[:,500]=lut[ord(row.Alt)]
        data.extend([ref,alt]);ident.append(row.snp_id);audit.append(dict(snp_id=row.snp_id,status='PASS'))
    predictions=[]
    with torch.no_grad():
        for i in range(0,len(data),32):
            x=torch.from_numpy(np.stack(data[i:i+32])).to(device)
            pred=(model(x)+model(torch.flip(x,dims=[1,2])))/2
            predictions.append(pred.cpu().numpy())
            if i%1024==0:print('PDS_BATCH',i,len(data),flush=True)
    a=np.concatenate(predictions);ref,alt=a[0::2],a[1::2];diff=np.abs(alt-ref)

    result=pd.DataFrame({'snp_id':ident,'PlantDeepSEA':diff.max(axis=1),'PlantDeepSEA_mean_abs':diff.mean(axis=1)})
    for i in range(15):result[f'abs_delta_target_{i+1}']=diff[:,i]
    result.to_csv(O/'PlantDeepSEA_scores.tsv',sep='\t',index=False);pd.DataFrame(audit).to_csv(O/'PlantDeepSEA_sequence_audit.tsv',sep='\t',index=False)
    (O/'PlantDeepSEA_protocol.json').write_text(json.dumps(dict(model='published rice Zhenshan97 15-target PlantDeepSEA checkpoint',model_source='http://plantdeepsea.ncpgr.cn/model_and_data/models/zs_model.zip',checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),sequence_reference='supplied_genome_fasta',reference_background='Nipponbare IRGSP1.0 sequences applied to the frozen ZS97 checkpoint',sequence_length=1000,variant_zero_based_index=500,base_order='ACGT',reverse_complement='mean of forward and reverse-complement probabilities before ref-alt difference',tissue_aggregation='max absolute ref-alt probability difference over 15 targets',tuned_on_benchmark=False,scored=len(result),input=len(p)),indent=2))
    print('PDS_FINISHED',len(result),flush=True)


def main():
    global O,RES,GENOME,CHECKPOINT,DEVICE,POOL_TAGS,CHECKPOINT_SHA256
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method",choices=["picnc","ricevarmap","plantdeepsea"])
    parser.add_argument("--pool-dir",required=True,type=Path)
    parser.add_argument("--resources-dir",required=True,type=Path)
    parser.add_argument("--out-dir",required=True,type=Path)
    parser.add_argument("--genome",type=Path)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--pool-tags",nargs="+",
                        default=["maize_coding","rice_coding","rice_noncoding"])
    args=parser.parse_args()
    O=args.out_dir
    O.mkdir(parents=True,exist_ok=True)
    RES=args.resources_dir
    GENOME=args.genome;CHECKPOINT=args.checkpoint;DEVICE=args.device
    POOL_TAGS=args.pool_tags;CHECKPOINT_SHA256=args.checkpoint_sha256

    input_dir=args.pool_dir
    def supplied_pool(species):
        paths=[input_dir/(tag+"_pool.parquet") for tag in sorted(POOL_TAGS)
               if tag.startswith(species+"_")]
        return pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    globals()["get_pool"]=supplied_pool
    if args.method=="plantdeepsea" and (not GENOME or not CHECKPOINT):
        parser.error("PlantDeepSEA requires --genome and --checkpoint")
    globals()[args.method]()


if __name__=="__main__":
    main()
