"Run the gap-driven retrieval engine on caller data."
from __future__ import annotations
import argparse,json,os
from pathlib import Path

def run_retrieval(payload,output_dir:Path,*,mode='replay'):
    from phenofuncrank_repro.interpretation import literature_retrieval as engine
    from phenofuncrank_repro.interpretation.evidence_store import EvidenceStore, ArtifactStore
    from phenofuncrank_repro.interpretation.schemas.literature import CandidateContext, EvidenceGapIntent, RetrievalCandidate, RetrievalBudget
    output_dir.mkdir(parents=True,exist_ok=True)
    store=EvidenceStore(output_dir/'jobs.sqlite3',ArtifactStore(output_dir/'artifacts'))
    context=CandidateContext(**payload['candidate_context'])
    gaps=[EvidenceGapIntent(**row) for row in payload['gap_intents']]
    candidates=[RetrievalCandidate(**row) for row in payload.get('article_candidates',[])]
    names=('europepmc_core_abstract','epmc_references','epmc_citations','openalex_references','openalex_cited_by','semantic_scholar_citations','complete_json','_http_get_json','_http_get_text','openalex_work_by_doi')
    originals={name:getattr(engine,name) for name in names}
    if mode=='replay':
        by_pmid={c.pmid:c for c in candidates if c.pmid}
        def core(pmid):
            c=by_pmid.get(str(pmid))
            return {'pmid':str(pmid),'title':c.title,'abstract':c.abstract_snippet,'source':c.source} if c else {}
        engine.europepmc_core_abstract=core
        for name in ('epmc_references','epmc_citations','openalex_references','openalex_cited_by','semantic_scholar_citations'):
            setattr(engine,name,lambda *args,**kwargs:[])
        for name in ('complete_json','_http_get_json','_http_get_text','openalex_work_by_doi'):
            setattr(engine,name,lambda *args,**kwargs:None)
    try:
        result=engine.retrieve_for_gaps(gaps,context,store=store,
            budget=RetrievalBudget(**payload['budget']) if payload.get('budget') else None,
            llm_rewrite=mode=='live',candidates_override=candidates if mode=='replay' else None)
    finally:
        for name,func in originals.items():setattr(engine,name,func)
    output={'mode':mode,'source_result':result.model_dump(mode='json'),
        'scope':'Original source retrieval on supplied input;not historical benchmark performance recomputation',
        'replay_transport_note':'Metadata uses supplied article fields;external citation lists/unprovidedHTTP are not evaluated in replay.' if mode=='replay' else None}
    (output_dir/'retrieval_result.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return output

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--mode',choices=('replay','live'),default='replay')
    args=parser.parse_args()
    run_retrieval(json.loads(args.input.read_text(encoding='utf-8')),args.output_dir,mode=args.mode)

if __name__=='__main__':main()
