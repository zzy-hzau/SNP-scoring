"""Multi-species genomes -> Cactus HAL -> target MAF -> PHAST WIG tracks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
from phenofuncrank_repro.resources.conservation_upstream.common import configuration, input_path, open_text, acquire, command

STEPS = ['prepare', 'seqfile', 'align', 'maf', 'model', 'scores']


def prepare(config):
    root = config['_output']
    if config.get('fasta_policy') == 'cotton_primary':
        from phenofuncrank_repro.resources.conservation_upstream.native.cotton_genomes import prepare as native_prepare
        sources = {item['name']: acquire(config, item, root/'raw'/item.get('filename', item['name']+'.fa.gz')) for item in config['genomes']}
        native_prepare(sources['tm1'], [(name,sources[name],config['primary_counts'][name]) for name in ['garb','grai','gbar','gtom']], root/'genomes', root/'qc')
        for path in (root/'genomes').glob('*.primary.fa'):
            command(config, ['samtools','faidx',path])
        return
    for item in config['genomes']:
        source = acquire(config, item, root/'raw'/item.get('filename', item['name']+'.fa.gz'))
        folder = root/'genomes'/item['name'];folder.mkdir(parents=True, exist_ok=True)
        target = folder/'genome.fa'

        with open_text(source) as fi, target.open('w') as fo:
            for line in fi:
                fo.write('>'+line[1:].split()[0]+'\n' if line.startswith('>') else line.upper())
        command(config, ['samtools','faidx',target])


def genome_file(config, item):
    root=config['_output']
    if config.get('fasta_policy')=='cotton_primary':
        return root/'genomes'/(item['name']+'.primary.fa')
    return root/'genomes'/item['name']/'genome.fa'


def seqfile(config):
    path=config['_output']/'panel.seqfile'
    with path.open('w') as handle:
        handle.write(config['tree'].strip()+'\n')
        for item in config['genomes']:
            handle.write(item['name']+'\t'+str(genome_file(config,item))+'\n')
    return path


def sequence_names(config, selected=None):
    if selected:
        return selected
    if config.get('sequences'):
        return config['sequences']
    target=next(item for item in config['genomes'] if item['name']==config['target'])
    return [line.split('\t')[0] for line in Path(str(genome_file(config,target))+'.fai').read_text().splitlines()]


def normalize_wig(source, output, sequence):
    with open(source) as fi, open(output,'w') as fo:
        for line in fi:
            if line.startswith(('fixedStep','variableStep')):
                line=re.sub(r'chrom=\S+', 'chrom='+sequence, line)
            fo.write(line)


def run(config, step, *, selected=None, dry_run=False):
    root=config['_output'];cores=config.get('threads',32)
    if step=='prepare':
        if dry_run:
            print(json.dumps({'prepare':config['genomes'],'fasta_policy':config.get('fasta_policy','native_first_token_uppercase')},indent=2));return
        prepare(config);return
    if step=='seqfile':
        if not dry_run:seqfile(config)
        return
    if step=='align':
        if config.get('align_by_sequence'):
            import pysam
            folder=root/'hal_by_seq';folder.mkdir(exist_ok=True)
            for sequence in sequence_names(config,selected):
                target_item=next(item for item in config['genomes'] if item['name']==config['target'])
                split=root/('target_'+sequence+'.fa');panel=root/('panel_'+sequence+'.seqfile')
                if not dry_run:
                    with pysam.FastaFile(str(genome_file(config,target_item))) as fa:
                        text=fa.fetch(sequence)
                    split.write_text('>'+sequence+'\n'+'\n'.join(text[i:i+80] for i in range(0,len(text),80))+'\n')
                    with panel.open('w') as handle:
                        handle.write(config['tree'].strip()+'\n')
                        for item in config['genomes']:
                            handle.write(item['name']+'\t'+str(split if item['name']==config['target'] else genome_file(config,item))+'\n')
                argv=['cactus',root/('jobstore_align_'+sequence),panel,folder/(sequence+'.hal'),'--batchSystem','single_machine','--maxCores',cores,'--workDir',root/('work_align_'+sequence)]
                if config.get('default_memory'):argv += ['--defaultMemory',config['default_memory']]
                if config.get('max_memory'):argv += ['--maxMemory',config['max_memory']]
                command(config,argv,dry_run=dry_run)
            return
        argv=['cactus',root/'jobstore_align',root/'panel.seqfile',root/'panel.hal','--batchSystem','single_machine','--maxCores',cores,'--workDir',root/'work_align']
        if config.get('max_memory'):argv += ['--maxMemory',config['max_memory']]
        if config.get('default_memory'):argv += ['--defaultMemory',config['default_memory']]
        if config.get('cactus_config'):argv += ['--configFile',input_path(config,config['cactus_config'])]
        if config.get('restart'):argv += ['--restart']
        command(config,argv,dry_run=dry_run);return
    if step=='maf':
        folder=root/'maf';folder.mkdir(exist_ok=True)
        for sequence in sequence_names(config,selected):

            hal=root/'hal_by_seq'/(sequence+'.hal') if config.get('align_by_sequence') else root/'panel.hal'
            command(config,['cactus-hal2maf',root/('jobstore_maf_'+sequence),hal,folder/(sequence+'.maf'),'--refGenome',config['target'],'--refSequence',sequence,'--chunkSize',1000000,'--batchSystem','single_machine','--maxCores',config.get('maf_threads',4),'--workDir',root/('work_maf_'+sequence)],dry_run=dry_run)
        return
    if step=='model':
        folder=root/'model';folder.mkdir(exist_ok=True);tree=folder/'tree.nh';merged=folder/'all.maf'
        mafs=[input_path(config,p) for p in config.get('model_mafs',[])] or sorted((root/'maf').glob('*.maf'))
        if not dry_run:
            if not mafs:raise FileNotFoundError('No target-coordinate MAFs for phyloFit')
            tree.write_text(config['tree'].strip()+'\n')
            with merged.open('w') as fo:
                for index,path in enumerate(mafs):
                    with open_text(path) as fi:
                        for line in fi:
                            if index and line.startswith('#'):continue
                            fo.write(line)

        argv=['phyloFit','--tree',tree,'--subst-mod',config['phylofit_model'],'--msa-format','MAF','--out-root',folder/'neutral']
        if config.get('phylofit_seed') is not None:argv += ['--seed',config['phylofit_seed']]
        command(config,argv+[merged],dry_run=dry_run);return
    if step=='scores':
        model=input_path(config,config['model']) if config.get('model') else root/'model/neutral.mod'
        conversion=[]
        for sequence in sequence_names(config,selected):
            maf=root/'maf'/(sequence+'.maf');folder=root/'tracks'/sequence;folder.mkdir(parents=True,exist_ok=True)
            if config.get('score_mode')=='windowed':
                from phenofuncrank_repro.resources.conservation_upstream.native.windowed_maf import split_maf
                chunks=folder/'chunks'
                if dry_run:
                    print('windowed source sanitization, window_bp='+str(config['window_bp']));paths=[chunks/'chunk_00000000.maf']
                else:
                    with maf.open() as fh:reference=next(line.split()[1] for line in fh if line.split() and line.split()[0]=='s')
                    split_maf(maf,chunks,config['window_bp'],reference);paths=[Path(line) for line in (chunks/'chunks.list').read_text().splitlines()]
            else:paths=[maf]
            if not dry_run:
                for track in ['phastCons','phyloP']:(folder/(track+'.wig')).write_text('')
            for index,path in enumerate(paths):
                pc=folder/f'pc_{index:08d}.raw.wig';pp=folder/f'pp_{index:08d}.raw.wig';params=config['phastcons'];p=config['phylop']
                command(config,['phastCons','--target-coverage',params['target_coverage'],'--expected-length',params['expected_length'],'--refidx',1,'--seqname',config['target'] if config.get('score_mode')=='windowed' else sequence,'--msa-format','MAF',path,model],stdout=pc,dry_run=dry_run)
                command(config,['phyloP','--method',p['method'],'--mode',p['mode'],'--wig-scores','--refidx',1,'--msa-format','MAF',model,path],stdout=pp,dry_run=dry_run)
                if not dry_run:
                    for source,track in [(pc,'phastCons'),(pp,'phyloP')]:
                        normalized=source.with_suffix('.normalized.wig');normalize_wig(source,normalized,sequence)
                        with (folder/(track+'.wig')).open('a') as fo,normalized.open() as fi:
                            for line in fi:fo.write(line)
            if not dry_run:
                with maf.open() as handle:
                    first=next(line.split() for line in handle if line.split() and line.split()[0]=='s')
                for track in ['phastCons','phyloP']:
                    conversion.append({'chromosome':sequence,'track':track,'length':int(first[5]),'source_relative':str((folder/(track+'.wig')).relative_to(root)),'output_filename':sequence+'.'+track+'.f32'})
        if not dry_run:
            (root/'profile_used.json').write_text(json.dumps({k:v for k,v in config.items() if not k.startswith('_')},indent=2)+'\n')
            (root/'dense_conversion_profile.json').write_text(json.dumps({'profile':'raw_producer_WIG_to_dense','coordinate_convention':'WIG 1-based; dense indices 0-based; chromosome length from target MAF srcSize','missing_value':'NaN','tracks':conversion},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--step',choices=STEPS+['all'],default='all');p.add_argument('--sequence',action='append');p.add_argument('--threads',type=int);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();config=configuration(a.config,a.output_dir)
    if a.threads:config['threads']=a.threads;config['maf_threads']=a.threads
    for step in STEPS if a.step=='all' else [a.step]:run(config,step,selected=a.sequence,dry_run=a.dry_run)


if __name__=='__main__':main()
