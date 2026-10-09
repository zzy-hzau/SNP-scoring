"source keyword and evidence-set scoring for Figure6b cards."

from __future__ import annotations

import argparse,json,re

from pathlib import Path

import pandas as pd

PMID_RE = re.compile(r"(?:PMID\s*[:#]?\s*)(\d{5,9})", re.I)

SRID_DIGITS_RE = re.compile(r"(?<![A-Za-z0-9])(\d{5,9})(?![A-Za-z0-9])")

DOI_RE = re.compile(r"(?:10\.\d{4,9}/[^\s;,\"']+)", re.I)

def extract_pmids(evidence_items: list[dict]) -> set[str]:
    out: set[str] = set()
    for ev in evidence_items:
        srid = str(ev.get("source_record_id") or "")
        statement = str(ev.get("statement") or "")
        src = str(ev.get("source_name") or "").lower()
        for m in PMID_RE.finditer(srid + " " + statement):
            out.add(m.group(1))
        if ("literature" in src or "pubmed" in src or "europepmc" in src or "openalex" in src
                or "pmid" in srid.lower()) and not DOI_RE.search(srid):

            
            for m in SRID_DIGITS_RE.finditer(srid):
                out.add(m.group(1))
        payload = ev.get("payload") or {}
        for key in ("pmid", "pmids", "PMID", "PMIDs"):
            val = payload.get(key)
            if val:
                for token in re.findall(r"\d{5,9}", str(val)):
                    out.add(token)
    return out

def extract_dois(evidence_items: list[dict]) -> set[str]:
    out: set[str] = set()
    for ev in evidence_items:
        srid = str(ev.get("source_record_id") or "")
        for m in DOI_RE.finditer(srid):
            out.add(m.group(0).rstrip("."))
        payload = ev.get("payload") or {}
        for key in ("doi", "DOI", "provider_id"):
            val = str(payload.get(key) or "")
            for m in DOI_RE.finditer(val):
                out.add(m.group(0).rstrip("."))
    return out

def all_text(card: dict) -> str:
    parts = []
    for ev in card.get("evidence", []):
        parts.extend([
            str(ev.get("subject_id", "")),
            str(ev.get("statement", "")),
            str(ev.get("source_record_id", "")),
        ])
    for c in card.get("claims", []):
        parts.append(str(c.get("proposed_mechanism", "")))
        parts.extend(str(x) for x in c.get("alternative_hypotheses", []))
    parts.extend(str(x) for x in card.get("report_facts", []))
    parts.extend(str(x) for x in card.get("report_hypotheses", []))
    return " ".join(parts)

def gene_match(card: dict, gold_gene_id: str, aliases_by_gene: dict) -> bool:
    if not gold_gene_id:
        return None
    text = all_text(card)
    if gold_gene_id in text:
        return True
    aliases = aliases_by_gene.get(gold_gene_id, set())
    for alias in aliases:
        if len(alias) >= 5 and alias.lower() in text.lower():
            return True
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return True
    return False

DIRECTION = {
    "loss_of_function_reduces_height": (
        ["dwarf", "semidwarf", "reduc", "decreas", "矮", "半矮", "short stature"],
        ["increas.*height", "taller", "增高", "株高增加"],
    ),
    "loss_of_function_increases_tillering": (
        ["tiller", "branch", "outgrowth", "分蘖", "分枝"],
        ["inhibit.*tiller", "suppress.*tiller", "抑制分蘖", "减少分蘖"],
    ),
    "loss_of_function_increases_elongation": (
        ["elongat", "slender", "伸长", "细长", "tall"],
        ["dwarf", "矮"],
    ),
    "loss_of_function_increases_internode_elongation": (
        ["internode", "节间", "elongat", "伸长", "tall"],
        ["dwarf", "矮"],
    ),
}

def direction_ok(card: dict, known_direction: str) -> bool | None:
    if not known_direction or known_direction == "not_established":
        return None
    text = all_text(card).lower()
    if not text:
        return None
    consistent, contradictory = DIRECTION[known_direction]
    has_consistent = any(re.search(kw, text) for kw in consistent)
    has_contradictory = any(re.search(kw, text) for kw in contradictory)
    if has_contradictory and not has_consistent:
        return False
    return None if not has_consistent else True

CLAIM_FUNCTION_KW = {
    "cf_rice_sd1_support": ["GA20", "20-oxidase", "20-氧化酶", "gibberellin", "赤霉素", "GA biosynthesis"],
    "cf_rice_d61_support": ["brassinosteroid", "油菜素内酯", "BRI1", "受体"],
    "cf_rice_d2_support": ["CYP90D2", "brassinosteroid", "油菜素内酯", "BR biosynthesis"],
    "cf_rice_d14_support": ["strigolactone", "独脚金内酯", "tillering", "分蘖", "shoot branching"],
    "cf_rice_slr1_support": ["DELLA", "gibberellin", "赤霉素", "GA signal"],
}

CLAIM_REFUTE_ASSERT = {
    "cf_rice_sd1_refute": ["DELLA"],
    "cf_rice_d61_refute": ["gibberellin", "GA receptor", "赤霉素受体"],
    "cf_rice_d2_refute": ["strigolactone", "独脚金内酯", "SL biosynthesis"],
}

CLAIM_REFUTE_CORRECT_EVIDENCE = {
    "cf_rice_sd1_refute": ["GA20", "20-oxidase", "20-氧化酶"],
    "cf_rice_d61_refute": ["brassinosteroid", "油菜素内酯", "BRI1"],
    "cf_rice_d2_refute": ["brassinosteroid", "油菜素内酯", "CYP90D2"],
}

CLAIM_NEI_ASSERT = {
    "cf_rice_sd1_nei": ["causal", "因果", "IR8"],
    "cf_rice_d2_nei": ["increase height", "increases height", "增高", "株高增加", "causal", "因果"],
}

def label_match(card: dict, task_id: str, gold_label: str) -> tuple[bool | None, str]:
    claims_text = " ".join(
        str(c.get("proposed_mechanism", "")) for c in card.get("claims", [])
    ).lower()
    facts_text = " ".join(
        [str(x).lower() for x in card.get("report_facts", [])]
        + [str(x).lower() for x in card.get("report_hypotheses", [])]
    )
    text = claims_text + " " + facts_text

    if gold_label == "SUPPORT":
        kws = CLAIM_FUNCTION_KW.get(task_id, [])
        if not kws:
            return None, "no keyword table"
        hit = any(kw.lower() in text for kw in kws)
        return bool(hit), "function keyword in claims/facts"
    if gold_label == "REFUTE":
        kws = CLAIM_REFUTE_ASSERT.get(task_id, [])

        
        asserted = any(kw.lower() in claims_text for kw in kws)
        ev_text = all_text(card).lower()
        corrected = any(kw.lower() in ev_text for kw in CLAIM_REFUTE_CORRECT_EVIDENCE.get(task_id, []))
        return (not asserted), f"asserted={asserted} corrected_evidence={corrected}"
    if gold_label == "NEI":
        kws = CLAIM_NEI_ASSERT.get(task_id, [])
        asserted = any(kw.lower() in claims_text for kw in kws)
        return (not asserted), f"asserted={asserted}"
    return None, "unknown label"

GENE_BODY_RE = re.compile(
    r"(位于|within|in the)\s*.{0,15}(基因体|gene body|CDS|外显子|内含子|intron|exon)", re.I
)

def negative_ok(card: dict) -> tuple[bool, list[str]]:
    problems = []
    for ev in card.get("evidence", []):
        stmt = str(ev.get("statement", ""))
        if ev.get("evidence_type") == "gene_context" and GENE_BODY_RE.search(stmt):
            problems.append(f"gene_body_claim: {stmt[:120]}")
    for c in card.get("claims", []):
        if c.get("causal_claim_allowed"):
            problems.append(f"causal_claim_allowed: {c.get('proposed_mechanism', '')[:120]}")
    if problems:
        return False, problems
    return True, []


def load_aliases(path):
    genes=pd.read_csv(path,sep='\t',low_memory=False,dtype=str).fillna('')
    universe={}
    for _,r in genes.iterrows():
        aliases={r['gene_id_irgsp1']}
        for col in ('display_symbol','all_symbol_aliases','msu_locus_aliases'):
            for chunk in str(r[col]).replace(';','|').split('|'):
                for token in chunk.split(','):
                    token=token.strip()
                    if token:aliases.add(token)
        universe[r['gene_id_irgsp1']]=aliases
    return universe


def score_cards(cards, answers, aliases_by_gene, literature, exclude_curated=False, existence_cache=None):
    golds={r['task_id']:r for r in answers};cards={r['task_id']:r for r in cards}
    gold_doi_by_pmid={str(r['pmid']):str(r['doi']).strip().rstrip('.') for r in literature if str(r.get('doi') or '').strip()}
    existence_cache=existence_cache or {}
    rows=[]
    for task_id,gold in sorted(golds.items()):
        card=cards.get(task_id);g=gold['gold'];ttype=gold['task_type']
        if card is None:
            rows.append({'task_id':task_id,'task_type':ttype,'status_a2':'missing','gene_match':None,'lit_recall':None,'ev_type_coverage':None,'direction_ok':None,'label_match':None,'negative_ok':None})
            continue
        if exclude_curated:
            card={**card,'evidence':[ev for ev in card.get('evidence',[]) if 'curated' not in str(ev.get('source_name') or '').lower()]}
        extracted=extract_pmids(card.get('evidence',[]));extracted_dois=extract_dois(card.get('evidence',[]))
        expected_types=set(g.get('expected_evidence_types',[]));have_types={ev.get('evidence_type') for ev in card.get('evidence',[])}
        coverage=len(expected_types & have_types)/len(expected_types) if expected_types else None
        expected_pmids=set(g.get('expected_pmids',[]));recall=None
        if expected_pmids:
            covered=sum(1 for pid in expected_pmids if pid in extracted or (gold_doi_by_pmid.get(pid) and gold_doi_by_pmid[pid] in extracted_dois))
            recall=covered/len(expected_pmids)
        label,note=(None,'')
        if ttype=='claim_verification':label,note=label_match(card,task_id,g.get('gold_label'))
        negative,problems=(None,[])
        if ttype=='negative_control':negative,problems=negative_ok(card)
        rows.append({'task_id':task_id,'task_type':ttype,'status_a2':(card.get('status') or {}).get('a2'),
          'status_a4':(card.get('status') or {}).get('a4'),'n_evidence':len(card.get('evidence',[])),'n_claims':len(card.get('claims',[])),
          'gene_match':gene_match(card,g.get('gene_id',''),aliases_by_gene),'lit_recall':recall,'ev_type_coverage':coverage,
          'direction_ok':direction_ok(card,g.get('known_direction','')),'label_match':label,'negative_ok':negative,'label_note':note,
          'external_PMID_diagnostic_mode':'supplied frozen existence cache; no network request',
          'cached_invalid_pmids':sorted(p for p in extracted if existence_cache.get(p) is False),
          'not_checked_in_existence_cache':sorted(p for p in extracted if p not in existence_cache)})
    return rows


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cards',type=Path,required=True)
    parser.add_argument('--answers',type=Path,required=True)
    parser.add_argument('--gene-aliases',type=Path,required=True)
    parser.add_argument('--literature',type=Path,required=True,help='TSV containing pmid and doi columns')
    parser.add_argument('--exclude-curated',action='store_true',help='Separate original autonomous-retrieval diagnostic profile')
    parser.add_argument('--existence-cache',type=Path,help='Optional frozen PMID existence dictionary; never queried live')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cache=json.loads(args.existence_cache.read_text()) if args.existence_cache else None
    literature=pd.read_csv(args.literature,sep='\t',dtype=str).fillna('').to_dict('records')
    rows=score_cards(read_jsonl(args.cards),read_jsonl(args.answers),load_aliases(args.gene_aliases),literature,args.exclude_curated,cache)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output,sep='\t',index=False)


if __name__=='__main__':
    main()
