"Source-Figure6b aggregation and separate composite sensitivity metrics."

from __future__ import annotations

import argparse,json,hashlib

from pathlib import Path

import numpy as np

import pandas as pd

SEED = 20260812

BOOT = 5000

def bootstrap(values: list[bool]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    samples = rng.choice(arr, size=(BOOT, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))

def b(v) -> bool | None:
    if pd.isna(v) or v == "":
        return None
    if isinstance(v, bool):
        return v
    if str(v).lower() == "true":
        return True
    if str(v).lower() == "false":
        return False
    return None

def rows_for(card_dir: Path, answers: dict[str, dict], scorecard_name: str) -> pd.DataFrame:
    df = pd.read_csv(card_dir / scorecard_name, sep="\t", dtype=str).fillna("")
    df["gene_ok"] = df["gene_match"].map(b)
    df["lit_ok"] = df["lit_recall"].map(lambda x: None if x == "" else float(x) >= 1.0)
    df["label_ok"] = df["label_match"].map(b)
    df["negative_ok_b"] = df["negative_ok"].map(b)
    df["direction_b"] = df["direction_ok"].map(b)
    def max_def(row):
        t = answers[row.task_id].get("task_type")
        if t == "negative_control":
            return row.negative_ok_b
        if t == "claim_verification":
            return bool(row.gene_ok and row.lit_ok and row.label_ok and (row.direction_b is not False))
        return bool(row.gene_ok and row.lit_ok and (row.direction_b is not False))
    df["max_defensible"] = df.apply(max_def, axis=1)
    def abstain(row):
        a = answers[row.task_id]
        if a.get("task_type") == "negative_control":
            return row.negative_ok_b
        if a.get("task_type") == "claim_verification" and (a.get("gold") or {}).get("gold_label") == "NEI":
            return row.label_ok
        return None
    df["correct_abstention"] = df.apply(abstain, axis=1)

    
    cards = {
        c["task_id"]: c
        for c in [json.loads(x) for x in (card_dir / "answer_cards.jsonl").read_text().splitlines() if x.strip()]
    }
    grounded, id_error = [], []
    for _, row in df.iterrows():
        card = cards.get(row.task_id, {})
        ev_ids = {str(e.get("evidence_id")) for e in card.get("evidence", [])}
        claims = card.get("claims", [])
        if not claims:
            continue
        for claim in claims:
            ids = [str(x) for x in (claim.get("supporting_evidence_ids") or [])]
            if not ids:
                continue
            ok = all(x in ev_ids for x in ids)
            grounded.append(ok)
            id_error.append(not ok)
    df.attrs["grounded_claims"] = grounded
    df.attrs["evidence_id_errors"] = id_error
    return df

def add_metric(out: list[dict], condition: str, metric: str, values: list[bool], note: str = "", source: str = "hard_rule"):
    rate, lo, hi = bootstrap(values)
    out.append({
        "condition": condition,
        "metric": metric,
        "numerator": int(sum(values)) if values else 0,
        "denominator": len(values),
        "rate": rate,
        "percentage": None if rate is None else round(100 * rate, 2),
        "ci95_low": lo,
        "ci95_high": hi,
        "status": "EVALUATED" if values else "NOT_EVALUABLE",
        "source": source,
        "note": note,
    })

def paired(a: pd.DataFrame, f: pd.DataFrame, metric: str, bname: str, note: str = "") -> dict:
    merged = a[["task_id", metric]].merge(f[["task_id", metric]], on="task_id", suffixes=("_baseline", "_full"))
    vals = [(bool(x), bool(y)) for x, y in zip(merged[f"{metric}_baseline"], merged[f"{metric}_full"]) if x in (True, False) and y in (True, False)]
    diffs = [int(y) - int(x) for x, y in vals]
    if diffs:
        rng = np.random.default_rng(SEED)
        samples = rng.choice(diffs, size=(BOOT, len(diffs)), replace=True).mean(axis=1)
        ci = [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
        b01 = sum(x and not y for x, y in vals)
        b10 = sum((not x) and y for x, y in vals)
        try:
            from scipy.stats import binomtest
            p = float(binomtest(min(b01, b10), n=b01 + b10, p=0.5).pvalue) if b01 + b10 else 1.0
        except Exception:
            p = None
    else:
        ci, b01, b10, p = [None, None], 0, 0, None
    return {
        "comparison": f"{bname} vs FULL",
        "metric": metric,
        "paired_n": len(vals),
        "baseline_to_full_gain_n": b10,
        "full_to_baseline_loss_n": b01,
        "mean_full_minus_baseline": (sum(diffs) / len(diffs)) if diffs else None,
        "paired_bootstrap_ci95": ci,
        "mcnemar_p": p,
        "note": note,
    }


def aggregate(root, answers_path):
    answers={r['task_id']:r for r in [json.loads(line) for line in Path(answers_path).read_text(encoding='utf-8').splitlines() if line.strip()]}
    frames={name:rows_for(Path(root)/(name+'_30'),answers,name+'_scorecard.tsv') for name in ('b0','b1','full')}
    primary=[];extended=[]
    columns=[('gene_ok','Gene identification'),('lit_recall','Literature recall'),('ev_type_coverage','Evidence coverage'),('direction_b','Direction correct'),('label_ok','Claim decision correct'),('negative_ok_b','Negative control correct')]
    for name,df in frames.items():
        for col,label in columns:
            values=[]
            for value in df[col].tolist():
                if value is None or value=='':continue
                if isinstance(value,(bool,np.bool_)):values.append(float(value))
                elif col in ('lit_recall','ev_type_coverage'):
                    value=float(value)
                    if np.isfinite(value):values.append(value)
            mean,low,high=bootstrap(values)
            primary.append({'condition':name.upper(),'metric':label,'sum_scores':float(sum(values)),'n_applicable':len(values),'mean_score':mean,'native_case_bootstrap_CI95':[low,high],
              'scope':'Primary published per-case score mean; fractional coverage preserved; null values omitted for this metric only'})
        for col,label in [('gene_ok','gene attribution correctness'),('lit_ok','literature target coverage'),('label_ok','claim label correctness'),('negative_ok_b','negative-control safety'),('max_defensible','maximum-defensible-conclusion correctness'),('correct_abstention','correct abstention / HUMAN_REVIEW proxy')]:
            sub=df
            if col=='gene_ok':sub=df[df.task_type!='negative_control']
            elif col=='lit_ok':sub=df[df.lit_ok.notna()]
            elif col=='label_ok':sub=df[df.task_type=='claim_verification']
            elif col=='negative_ok_b':sub=df[df.task_type=='negative_control']
            add_metric(extended,name.upper(),label,[x for x in sub[col].tolist() if x in (True,False)])
        add_metric(extended,name.upper(),'claim grounding precision (provenance contract)',df.attrs['grounded_claims'])
        add_metric(extended,name.upper(),'evidence-ID / provenance error rate',df.attrs['evidence_id_errors'])
    comparisons=[paired(frames[base],frames['full'],field,base.upper()) for field in ['max_defensible','gene_ok','label_ok'] for base in ['b0','b1']]
    return {'protocol_seed':SEED,'bootstrap_resamples':BOOT,'primary_score_means':primary,'separate_native_composite_metrics':extended,'separate_native_paired_comparisons':comparisons,
      'scope':'Frozen scorecard and citation-ID replay. Primary coverage means, full-recall binary composites and semantic judgement are distinct.',
      'source_file_hashes':{name+'_scorecard.tsv':hashlib.sha256((Path(root)/(name+'_30')/(name+'_scorecard.tsv')).read_bytes()).hexdigest() for name in frames}}


def summarize_published_workbook(path):
    from phenofuncrank_repro.interpretation.evaluation.extract_frozen_figure6 import read_tables
    rows=read_tables(path,'Figure 6')['BenchmarkCleanSource_1']
    mapping={'LLM only':'B0','Tool-assisted LLM':'B1','Full workflow':'FULL'}
    output=[]
    for condition in mapping:
        selected=[r for r in rows if r['Condition']==condition]
        for field in ['Gene identification','Literature recall','Evidence coverage','Direction correct','Claim decision correct','Negative control correct']:
            values=[float(r[field]) for r in selected if isinstance(r.get(field),(int,float,bool))]
            mean,lo,hi=bootstrap(values)
            output.append({'condition':mapping[condition],'metric':field,'sum_scores':float(sum(values)),'n_applicable':len(values),'mean_score':mean,'native_case_bootstrap_CI95':[lo,hi]})
    return {'primary_score_means':output,'conditions_case_counts':{mapping[c]:sum(r['Condition']==c for r in rows) for c in mapping},
      'protocol_seed':SEED,'bootstrap_resamples':BOOT,'source_workbook_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
      'scope':'Published per-case score means only; fractional coverage retained. Added native bootstrap is a separate uncertainty view. Raw-card scoring and provenance composites use the supplied raw-record mode.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--root',type=Path,help='Supplied b0_30,b1_30,full_30 folders; no fixed server path')
    group.add_argument('--source-data',type=Path,help='Published workbook mode: primary per-case means')
    parser.add_argument('--answers',type=Path,help='Required in supplied raw-record directory mode')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.root and not args.answers:parser.error('--answers is required with --root')
    result=summarize_published_workbook(args.source_data) if args.source_data else aggregate(args.root,args.answers)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
