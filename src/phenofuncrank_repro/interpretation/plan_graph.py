"Run the graph-guided planner on a small caller-supplied snapshot."
from __future__ import annotations
import argparse,json,os,sqlite3
from pathlib import Path

def plan_snapshot(snapshot: dict, output_dir: Path):
    from phenofuncrank_repro.interpretation.evidence_store import EvidenceStore, ArtifactStore
    from phenofuncrank_repro.interpretation.graph_rag import GraphRAGPlanner
    output_dir.mkdir(parents=True,exist_ok=True)
    root=output_dir/'working_store';root.mkdir(exist_ok=True)
    store=EvidenceStore(root/'jobs.sqlite3',ArtifactStore(output_dir/'artifacts'))
    with store.connect():pass
    def insert_rows(con,table,rows):
        if not rows:return
        columns=list(rows[0])
        con.executemany('INSERT OR REPLACE INTO '+table+' ('+','.join(columns)+') VALUES ('+','.join('?' for _ in columns)+')',[[row.get(name) for name in columns] for row in rows])
    with store.connect() as con:
        for table in ('snps','tool_runs','evidence_records'):
            insert_rows(con,table,snapshot['tables'].get(table,[]))
    for species,graph in snapshot['graphs'].items():
        with store.connect_graph(species) as con:
            insert_rows(con,'graph_nodes',graph['graph_nodes'])
            insert_rows(con,'graph_edges',graph['graph_edges'])
    result={query['snp_key']:GraphRAGPlanner(store).plan(query['snp_key'],trait=query.get('trait',''),hops=query.get('hops',2)) for query in snapshot['queries']}
    (output_dir/'graph_plans.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    plan_snapshot(json.loads(args.input.read_text(encoding='utf-8')),args.output_dir)

if __name__=='__main__':main()
