"""Offline AAP sequence, cached annotation, alignment and processed-expression methods."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
from Bio.Seq import Seq


def fasta_sequence(path):
    records=[];current=[]
    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            if current:records.append(''.join(current));current=[]
        elif line.strip():current.append(line.strip())
    if current:records.append(''.join(current))
    if len(records)!=1:raise ValueError('Provide one verified focal reference sequence')
    return records[0]


def sequence_consequence(region, position, genomic_ref='G', genomic_alt='A'):
    """Native descending minus-strand CDS construction for the supplied AAP gene."""
    s=region['gene_reference_sequence'];lo=region['gene_reference_start']
    cds=sorted([v for v in region['gene_features'] if v['type']=='CDS'],key=lambda v:v['start'],reverse=True)
    if not cds or any(v['strand']!='-' for v in cds):raise ValueError('This source route requires the supplied minus-strand AAP model')
    complement=str.maketrans('ACGT','TGCA')
    seq=''.join(s[v['start']-lo:v['end']-lo+1][::-1].translate(complement) for v in cds)
    positions=[]
    for v in cds:positions.extend(range(v['end'],v['start']-1,-1))
    if s[position-lo]!=genomic_ref:raise ValueError('Genomic REF does not match the supplied reference fragment')
    index=positions.index(position)
    ref=genomic_ref.translate(complement);alt=genomic_alt.translate(complement)
    if seq[index]!=ref:raise ValueError('Transcript-oriented REF mismatch')
    altered=seq[:index]+alt+seq[index+1:]
    protein=str(Seq(seq).translate(table=1));alternative=str(Seq(altered).translate(table=1))
    protein=protein[:-1] if protein.endswith('*') else protein
    alternative=alternative[:-1] if alternative.endswith('*') else alternative
    codon_start=index//3*3
    return {'CDS':seq,'reference_protein':protein,'alternative_protein':alternative,
        'coding_position_1based':index+1,'coding_ref':ref,'coding_alt':alt,
        'ref_codon':seq[codon_start:codon_start+3],'alt_codon':altered[codon_start:codon_start+3],
        'aa_position_1based':index//3+1,'ref_aa':protein[index//3],'alt_aa':alternative[index//3],
        'CDS_source_rule':'Native complete CDS concatenation in descending genomic order; no per-exon phase trimming',
        'translation_table':1,'protein_terminal_stop_removed':True}


def splice_context(region, position, coding_alt='T'):
    s=region['gene_reference_sequence'];lo=region['gene_reference_start']
    cds=sorted([v for v in region['gene_features'] if v['type']=='CDS'],key=lambda v:v['start'],reverse=True)
    exon=next(v for v in cds if v['start']<=position<=v['end']);donor=exon['start']-1
    c=str.maketrans('ACGT','TGCA')
    window=s[donor-lo-15:donor-lo+15][::-1].translate(c)
    altered=list(window);offset=[donor+14-i for i in range(30)].index(position);altered[offset]=coding_alt;altered=''.join(altered)
    cryptic=[]
    for i in range(3,len(altered)-6):
        if altered[i:i+2]=='GT' and window[i:i+2]!='GT':
            cryptic.append({'offset_from_canonical_donor_nt':i-14,'new_9mer':altered[i-3:i]+'|'+altered[i:i+6],'ref_9mer':window[i-3:i]+'|'+window[i:i+6]})
    return {'reference_exon_intron':window[:14]+'|'+window[14:],'variant_exon_intron':altered[:14]+'|'+altered[14:],
        'canonical_donor_9mer_ref':window[11:14]+'|'+window[14:20],
        'canonical_donor_9mer_alt':altered[11:14]+'|'+altered[14:20],'new_GT_dinucleotides':cryptic,
        'interpretation':'Sequence context only. A new GT dinucleotide does not establish functional cryptic splicing.'}


def annotation_rows(reference_protein, uniprot, interpro, focal=323):
    U=uniprot['results'][0]
    if reference_protein!=U['sequence']['value']:raise ValueError('UniProt record is not the verified exact reference protein')
    rows=[]
    for r in interpro['results']:
        for p in r['proteins']:
            for loc in p['entry_protein_locations']:
                for frag in loc['fragments']:
                    rows.append({'source':r['metadata']['source_database'],'accession':r['metadata']['accession'],'description':r['metadata']['name'],
                        'start_aa':frag['start'],'end_aa':frag['end'],'method':'Database precomputed signature match',
                        'contains_A323':frag['start']<=focal<=frag['end'],'reported_score':loc.get('score')})
    for f in U.get('features',[]):
        if f['type']=='Transmembrane':
            a=f['location']['start']['value'];b=f['location']['end']['value']
            rows.append({'source':'UniProt','accession':U.get('primaryAccession','A0A1S2Y8P2'),'description':'Predicted transmembrane helix','start_aa':a,'end_aa':b,
                'method':'Phobius; ECO:0000256 (computational prediction)','contains_A323':a<=focal<=b,'reported_score':None})
    return rows


def aligned_focal(qseq, sseq, qstart, sstart, focal=323, half_window=10):
    if len(qseq)!=len(sseq):raise ValueError('Aligned query and subject must have equal column counts')
    q=qstart-1;s=sstart-1
    for col,(qa,sa) in enumerate(zip(qseq,sseq)):
        if qa!='-':q+=1
        if sa!='-':s+=1
        if qa!='-' and q==focal:
            return {'subject_position':s if sa!='-' else None,'query_residue':qa,'subject_residue':sa,
                'query_aligned_window':qseq[max(0,col-half_window):col+half_window+1],
                'subject_aligned_window':sseq[max(0,col-half_window):col+half_window+1]}
    return None


def parse_blast_alignments(path, focal=323):
    result=[]
    with Path(path).open() as f:
        for fields in csv.reader(f,delimiter='\t'):
            if not fields:continue
            accession=fields[1]
            if accession.startswith('sp|'):accession=accession.split('|')[1].split('.')[0]
            row={'accession':accession,'identity_percent':float(fields[2]),'aligned_length':int(fields[3]),'query_length':int(fields[4]),'subject_length':int(fields[5]),
                'evalue':float(fields[6]),'bitscore':float(fields[7]),'query_start':int(fields[8]),'query_end':int(fields[9]),'subject_start':int(fields[10]),'subject_end':int(fields[11]),
                'focal_mapping':aligned_focal(fields[12],fields[13],int(fields[8]),int(fields[10]),focal)}
            if len(fields)>14:row['title']=fields[14]
            result.append(row)
    return result


def expression_rows(archive, gene='Ca_TC15673'):
    """Stream one exact gene from the deposited processed FPKM tables."""
    result=[]
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith('_FPKM.txt.gz'):continue
            stem=Path(member.name).name[:-len('_FPKM.txt.gz')]
            sample,tissue=stem.split('_',1)
            with gzip.GzipFile(fileobj=tar.extractfile(member)) as compressed:
                text=io.TextIOWrapper(compressed,encoding='utf-8')
                found=[row for row in csv.DictReader(text,delimiter='\t') if row['Gene_ID']==gene]
            if len(found)!=1:raise ValueError(f'{sample}: target gene must occur exactly once')
            result.append({'GEO_sample':sample,'tissue_code':tissue,'gene':gene,'FPKM':float(found[0]['FPKM'])})
    return result


def verify_expression_mapping(path, reference_CDS, atlas_transcript, isoform='Ca_TC15673.1'):
    """Verify the archived full-CDS match against the actual atlas transcript."""
    candidates=[]
    with Path(path).open() as f:
        for r in csv.reader(f,delimiter='\t'):
            if r and r[1]==isoform and float(r[2])==100.0 and int(r[3])==len(reference_CDS) and int(r[4])==0 and int(r[5])==0 and int(r[6])==1 and int(r[7])==len(reference_CDS):
                candidates.append(r)
    if not candidates:raise ValueError('No archived complete, ungapped exact-CDS match for the selected expression isoform')
    row=candidates[0]
    if row[14]!=reference_CDS or row[15]!=reference_CDS:raise ValueError('Stored alignment strings do not match the reference CDS')
    start,end=int(row[8]),int(row[9])
    if atlas_transcript[start-1:end]!=reference_CDS:raise ValueError('Actual atlas transcript does not contain the claimed exact CDS segment')
    return {'atlas_isoform':isoform,'expression_gene':isoform.rsplit('.',1)[0],'CDS_length':len(reference_CDS),'identity_percent':100.,
        'mismatches':0,'gaps':0,'subject_start_1based':start,'subject_end_1based':end,'atlas_transcript_length':len(atlas_transcript),
        'archived_evalue':float(row[12]),'archived_bitscore':float(row[13]),'actual_transcript_sequence_verified':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['region-metadata','reference-protein','uniprot','interpro','swiss-alignments','pea-alignment','expression-archive','atlas-alignments','atlas-transcript']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--position',type=int,default=33198529)
    parser.add_argument('--focal-aa',type=int,default=323)
    parser.add_argument('--expression-gene',default='Ca_TC15673')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    region=load(args.region_metadata);consequence=sequence_consequence(region,args.position)
    protein=fasta_sequence(args.reference_protein)
    if consequence['reference_protein']!=protein:raise ValueError('CDS translation does not match the verified RefSeq sequence')
    output={'sequence_consequence':consequence,'splice_context':splice_context(region,args.position),
        'protein_annotations':annotation_rows(protein,load(args.uniprot),load(args.interpro),args.focal_aa),
        'swiss_alignments':parse_blast_alignments(args.swiss_alignments,args.focal_aa),
        'pea_alignment':parse_blast_alignments(args.pea_alignment,args.focal_aa),
        'processed_expression':expression_rows(args.expression_archive,args.expression_gene),
        'expression_gene_mapping':verify_expression_mapping(args.atlas_alignments,consequence['CDS'],fasta_sequence(args.atlas_transcript)),
        'scope':'Offline sequence and saved-annotation/BLAST/processed-FPKM replay. No new InterProScan, Phobius, BLAST database search or biological experiment.'}
    if output['expression_gene_mapping']['expression_gene']!=args.expression_gene:raise ValueError('Expression gene differs from the sequence-verified atlas gene')
    output['input_sha256']={name:hashlib.sha256(getattr(args,name.replace('-','_')).read_bytes()).hexdigest() for name in ['region-metadata','reference-protein','uniprot','interpro','swiss-alignments','pea-alignment','expression-archive','atlas-alignments','atlas-transcript']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
