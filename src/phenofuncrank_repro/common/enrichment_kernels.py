"Common source computations; profile dispatch lives in leaf modules."
from __future__ import annotations
import numpy as np
import pandas as pd

def bh_adjust(pvalues):
 p=np.asarray(pvalues,float)
 if not np.isfinite(p).all():raise ValueError('All tested terms must have valid P values')
 order=np.argsort(p,kind='stable');n=len(p);q=np.empty(n)
 q[order]=np.minimum.accumulate((p[order]*n/np.arange(1,n+1))[::-1])[::-1]
 return np.clip(q,0,1)

def interval_membership(chromosomes,positions,intervals,window=100000):
 """Intervals are 1-based inclusive; each SNP is counted once per term."""
 chrom=np.asarray(chromosomes,str);pos=np.asarray(positions,int);out=np.zeros(len(pos),bool)
 for chromosome,group in intervals.groupby('chrom',sort=False):
  bounds=sorted((max(1,int(a)-window),int(b)+window) for a,b in zip(group.start,group.end))
  merged=[]
  for a,b in bounds:
   if merged and a<=merged[-1][1]+1:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
   else:merged.append((a,b))
  if not merged:continue
  starts=np.array([a for a,b in merged]);ends=np.array([b for a,b in merged])
  idx=np.flatnonzero(chrom==str(chromosome));j=np.searchsorted(starts,pos[idx],side='right')-1
  out[idx]=(j>=0)&(pos[idx]<=ends[np.maximum(j,0)])
 return out

def matched_snp_enrichment(candidates,term_intervals,backgrounds,matches,*,window=100000):
 """Use explicit per-candidate matched row ranges; no gene-universe shortcut.

 candidates: trait, rank, chrom, pos
 term_intervals: trait, database, term_id, term, gene_id, chrom, start, end
 backgrounds[trait]: background_row, chrom, pos (sorted by background_row)
 matches: trait, rank, first_background_row, last_background_row
 """
 results=[]
 for trait,c in candidates.groupby('trait',sort=False):
  c=c.sort_values('rank');m=matches[matches.trait==trait].set_index('rank').loc[c['rank']]
  if m.index.duplicated().any():raise ValueError('Duplicate candidate/background pairing')
  bg=backgrounds[trait].sort_values('background_row').reset_index(drop=True)
  bg_rows=bg.background_row.to_numpy(int)
  if len(bg_rows)!=len(np.unique(bg_rows)):raise ValueError('Duplicate background row IDs')
  first=np.searchsorted(bg_rows,m.first_background_row.to_numpy(int),side='left')
  last=np.searchsorted(bg_rows,m.last_background_row.to_numpy(int),side='right')
  if (last<=first).any():raise ValueError('Empty matched background pool')

  chrom=bg.chrom.astype(str).to_numpy()
  changes=np.r_[0,np.cumsum(chrom[1:]!=chrom[:-1])]
  for row,a,b in zip(c.itertuples(index=False),first,last):
   if chrom[a]!=str(row.chrom) or changes[b-1]!=changes[a]:raise ValueError('Candidate/background chromosome mismatch')
  position_index={}
  for chromosome,indices in bg.groupby('chrom',observed=True).groups.items():
   indices=np.asarray(indices,int);positions=bg.pos.to_numpy()[indices];order=np.argsort(positions,kind='stable')
   position_index[str(chromosome)]=(positions[order],bg_rows[indices[order]])
  catalog=term_intervals[term_intervals.trait==trait]
  for (database,term_id),genes in catalog.groupby(['database','term_id'],sort=False):
   foreground=interval_membership(c.chrom,c.pos,genes,window)
   chunks=[]
   for chromosome,group in genes.groupby('chrom'):
    if str(chromosome) not in position_index:continue
    positions,rows=position_index[str(chromosome)]
    bounds=sorted((int(a)-window,int(b)+window) for a,b in zip(group.start,group.end));merged=[]
    for a,b in bounds:
     if merged and a<=merged[-1][1]+1:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
     else:merged.append((a,b))
    for a,b in merged:
     lo=np.searchsorted(positions,a,'left');hi=np.searchsorted(positions,b,'right')
     chunks.append(rows[lo:hi])
   hit_rows=np.sort(np.concatenate(chunks)) if chunks else np.array([],int)
   probabilities=(np.searchsorted(hit_rows,m.last_background_row.to_numpy(int),'right')-np.searchsorted(hit_rows,m.first_background_row.to_numpy(int),'left'))/(last-first)
   observed=int(foreground.sum());expected=float(probabilities.sum())
   results.append({'trait':trait,'database':database,'term_id':term_id,'term':genes.term.iloc[0],
    'term_genes':genes.gene_id.nunique(),'observed':observed,'expected':expected,
    'fold_enrichment':observed/expected if expected else None,
    'pvalue':poisson_binomial_tail(probabilities,observed)})
 out=pd.DataFrame(results)
 if not out.empty:
  out['qvalue']=np.nan
  for _,indices in out.groupby(['trait','database']).groups.items():
   out.loc[indices,'qvalue']=bh_adjust(out.loc[indices,'pvalue'])
 return out

def poisson_binomial_tail(probabilities,observed):
 p=np.asarray(probabilities,float)
 if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid matched hit probabilities')
 k=int(observed)
 if k<=0:return 1.
 if k>len(p):return 0.

 mass=np.zeros(k+1);mass[0]=1
 for probability in p:
  previous=mass.copy()
  mass[0]=previous[0]*(1-probability)
  mass[1:k]=previous[1:k]*(1-probability)+previous[:k-1]*probability
  mass[k]=previous[k]+previous[k-1]*probability
 return float(np.clip(mass[k],0,1))
