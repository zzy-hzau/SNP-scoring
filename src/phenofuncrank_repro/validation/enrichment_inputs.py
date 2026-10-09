"full-family SNP-background preparation from supplied publication resources."
from pathlib import Path
import gzip,zipfile,json
import numpy as np
import pandas as pd
TRAITS={'Rice_PH':'Rice plant height','Cotton_FS':'Cotton fibre strength','Chickpea_HSW':'Chickpea hundred-seed weight','Chickpea_PW':'Chickpea plant width'}

def chromosome(value):
 text=str(value).replace('chr','').replace('Chr','')
 if text.isdigit():return str(int(text))
 return text

def public_inputs(publication_root,trait,spec,term_catalog_path,term_metadata,*,frozen_endpoints=False):
 root=Path(publication_root)
 species={'Rice_PH':'rice','Cotton_FS':'cotton','Chickpea_HSW':'chickpea','Chickpea_PW':'chickpea'}[trait]
 public_trait=TRAITS.get(trait,trait)
 c=pd.DataFrame([r for r in spec['ranked_variants'] if r['Trait']==public_trait]).rename(columns={'Trait':'trait','Rank':'rank','Chromosome':'chrom','Position (bp)':'pos'})
 c['trait']=trait
 c['chrom']=c.chrom.map(chromosome)
 m=pd.DataFrame([r for r in spec['matches'] if r['Trait']==public_trait]).rename(columns={'Trait':'trait','Candidate rank':'rank','First background row':'first_background_row','Last background row':'last_background_row'})
 m['trait']=trait
 chunks=[];intervals=[]
 for archive in sorted(root.glob('Supplementary_Data_[2-5].zip')):
  with zipfile.ZipFile(archive) as z:
   for name in z.namelist():
    if name.endswith(f'{species}_gene_intervals.tsv.gz'):
     intervals.append(pd.read_csv(gzip.open(z.open(name),'rt'),sep='\t'))
    elif Path(name).name.startswith(species+'_') and name.endswith('_background.tsv.gz'):
     d=pd.read_csv(gzip.open(z.open(name),'rt'),sep='\t',dtype={'Background row':'int32','Position (bp)':'int32','Chromosome':'category'})
     chunks.append(d.rename(columns={'Background row':'background_row','Chromosome':'chrom','Position (bp)':'pos'}))
 if not chunks or not intervals:raise FileNotFoundError('Public background/annotation archives missing')
 bg=pd.concat(chunks,ignore_index=True);bg['chrom']=bg.chrom.astype(str).map(chromosome).astype('category')
 genes=pd.concat(intervals,ignore_index=True).drop_duplicates('Gene ID').rename(columns={'Gene ID':'gene_id','Chromosome':'chrom','Annotation start (0-based)':'start','Annotation end (0-based exclusive)':'end'})
 genes['chrom']=genes.chrom.map(chromosome)
 if not frozen_endpoints:genes['start']=genes.start+1
 catalog_path=Path(term_catalog_path)
 if not catalog_path.is_file():raise FileNotFoundError('Complete GO/KEGG term catalogue missing; foreground hit genes cannot define the null')
 records=pd.read_csv(catalog_path,sep='\t',dtype={'term_id':str})
 records=records[records.trait==trait]
 terms=pd.DataFrame(term_metadata['tables'][0]);terms=terms[terms.Trait==public_trait].rename(columns={'Trait':'trait','Database':'database','Term ID':'term_id','Term':'term'})
 terms['trait']=trait
 catalog=records.merge(terms[['trait','database','term_id','term']],on=['trait','database','term_id'],validate='many_to_one').merge(genes,on='gene_id',validate='many_to_one')
 if catalog.gene_id.nunique()!=records.gene_id.nunique():raise ValueError('Term gene coordinates are incomplete')
 return c,catalog,bg,m,terms
