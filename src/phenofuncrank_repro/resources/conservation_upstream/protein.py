"""Original proteome preparation -> OrthoFinder -> untrimmed orthogroup MSA."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
from phenofuncrank_repro.resources.conservation_upstream.common import configuration, input_path, acquire, command, plain_file

STEPS=['prepare','targets','panels','orthogroups','msa']
NATIVE=Path(__file__).with_name('native')


def records(path):
    header=None;sequence=[]
    with open(path) as handle:
        for line in handle:
            if line.startswith('>'):
                if header is not None:yield header,''.join(sequence)
                header=line[1:].strip().split()[0];sequence=[]
            else:sequence.append(line.strip())
    if header is not None:yield header,''.join(sequence)


def sequence_ids(result):
    mapping={}
    with (result/'WorkingDirectory/SequenceIDs.txt').open() as handle:
        for line in handle:
            if ': ' not in line:continue
            key,value=line.rstrip('\n').split(': ',1);value=value.split()[0]
            mapping[key]=value;mapping['Species'+key]=value
            if '|' in value:
                prefix,rest=value.split('|',1);mapping[prefix+'_'+prefix+'|'+rest]=value;mapping[prefix+'|'+rest]=value
    return mapping


def restore_orthogroup(source, output, mapping):
    with open(source) as fi, open(output,'w') as fo:
        for line in fi:
            if line.startswith('>'):
                old=line[1:].strip().split()[0]
                new=mapping.get(old,mapping.get(old.removeprefix('Species'),old))
                fo.write('>'+new+'\n')
            else:fo.write(line)


def run(config,step,*,result=None,dry_run=False):
    root=config['_output'];prepared=root/'prepared';prepared.mkdir(exist_ok=True)
    if step=='prepare':
        for item in config['proteomes']:
            source=root/'raw'/item.get('filename',item['name']+'.fa.gz')
            if not dry_run:source=acquire(config,item,source)
            if config['preparation']=='cotton_longest_gene':
                continue

            command(config,[sys.executable,'-B',NATIVE/'standardize_protein.py','--input',source,'--prefix',item['name'],'--out',prepared/(item['name']+'.fa')],dry_run=dry_run)
        return
    if step=='targets':
        for item in config['targets']:
            folder=root/'target_proteins'/item['name'];folder.mkdir(parents=True,exist_ok=True)
            genome=folder/'reference.fa';gff=folder/'annotation.gff';raw=folder/'protein.raw.fa'
            if item.get('protein_fasta'):
                if not dry_run:
                    plain_file(input_path(config,item['protein_fasta']),raw)
            elif not dry_run:
                plain_file(input_path(config,item['reference']),genome)
                plain_file(input_path(config,item['gff']),gff)
            if not item.get('protein_fasta'):
                command(config,['gffread',gff,'-g',genome,'-y',raw],dry_run=dry_run)
            if config['preparation']=='cotton_longest_gene':
                continue
            if not dry_run:
                with raw.open() as fi, (prepared/(item['name']+'.fa')).open('w') as fo:
                    for line in fi:
                        if line.startswith('>'):
                            token=line[1:].split()[0].removeprefix('transcript:').removeprefix('mRNA:')
                            if item.get('protein_fasta') and '|' in token:token=token.split('|')[-1]
                            fo.write('>'+item['name']+'|'+token+'\n')
                        else:fo.write(line.upper())
        return
    if step=='panels':
        folder=root/'input';folder.mkdir(exist_ok=True)
        if config['preparation']=='cotton_longest_gene':
            from phenofuncrank_repro.resources.conservation_upstream.native.cotton_proteins import prepare as native_prepare
            target=config['targets'][0];sources=[(target['name'],root/'target_proteins'/target['name']/'protein.raw.fa')]
            sources += [(item['name'],acquire(config,item,root/'raw'/item.get('filename',item['name']+'.fa.gz'))) for item in config['proteomes']] if not dry_run else []
            if not dry_run:
                native_prepare(sources,root/'native_raw',folder,root/'qc')
        else:
            for name in [item['name'] for item in config['proteomes']]+[item['name'] for item in config['targets']]:
                if not dry_run:shutil.copyfile(prepared/(name+'.fa'),folder/(name+'.fa'))
        return
    if step=='orthogroups':

        argv=['orthofinder','-f',root/'input','-o',root/'orthofinder','-S','diamond','-M','msa','-T','fasttree','-t',config.get('threads',32),'-a',config.get('analysis_threads',32)]
        if config.get('orthofinder_aligner'):argv += ['-A',config['orthofinder_aligner']]
        command(config,argv,dry_run=dry_run);return
    if step=='msa':
        result=Path(result).resolve() if result else sorted((root/'orthofinder').glob('Results_*'))[-1]
        mapping=sequence_ids(result);folder=root/'msa_untrimmed';folder.mkdir(exist_ok=True);restored=root/'orthogroup_sequences';restored.mkdir(exist_ok=True)
        ids=set()
        for target in config['targets']:
            ids.update(header for header,_ in records(root/'input'/(target['name']+'.fa')))
        aligner=config.get('msa_aligner','source_preferred')
        if aligner=='source_preferred':aligner='famsa' if shutil.which(config.get('tools',{}).get('famsa','famsa')) else 'mafft'
        included=[]
        for source in sorted((result/'Orthogroup_Sequences').glob('OG*.fa')):
            target=restored/source.name
            restore_orthogroup(source,target,mapping)
            members=list(records(target))
            if config.get('target_groups_only',True) and not any(header in ids for header,_ in members):continue
            out=folder/source.name
            if len(members)<=1:
                if not dry_run:shutil.copyfile(target,out)
            elif aligner=='famsa':
                argv=['famsa']
                if config.get('famsa_threads') is not None:argv += ['-t',config['famsa_threads']]
                command(config,argv+[target,out],dry_run=dry_run)
            else:
                argv=['mafft','--auto']
                if config.get('mafft_anysymbol',False):argv += ['--anysymbol']
                argv += ['--thread',config.get('msa_threads',1),target]
                command(config,argv,stdout=out,dry_run=dry_run)
            included.append({'orthogroup':source.stem,'sequences':len(members),'alignment':str(out.relative_to(root)),'aligner':'singleton_copy' if len(members)<=1 else aligner})
        if not dry_run:
            (root/'msa_manifest.json').write_text(json.dumps({'result':str(result),'trimmed':False,'orthogroups':included,'target_groups_only':config.get('target_groups_only',True)},indent=2)+'\n')
            (root/'profile_used.json').write_text(json.dumps({k:v for k,v in config.items() if not k.startswith('_')},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--step',choices=STEPS+['all'],default='all');p.add_argument('--result',type=Path);p.add_argument('--threads',type=int);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();config=configuration(a.config,a.output_dir)
    if a.threads:config['threads']=a.threads;config['analysis_threads']=a.threads;config['famsa_threads']=a.threads;config['msa_threads']=a.threads
    for step in STEPS if a.step=='all' else [a.step]:run(config,step,result=a.result,dry_run=a.dry_run)


if __name__=='__main__':main()
