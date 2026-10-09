"""Frozen paper node-QC candidate-pool construction, with explicit source layout."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from phenofuncrank_repro.prioritization.paper_panel_20260917 import Panel, log
CR_MIN, MAC_MIN = 0.90, 10
CORE = OUT = None
TRAITS = {}

def main() -> None:
    rows = []
    for tag, (panel_dir, species) in TRAITS.items():
        t0 = time.time()
        log(f"=== {tag} ===")
        tdir = OUT / tag
        tdir.mkdir(parents=True, exist_ok=True)
        panel = Panel(panel_dir, species)


        core = pd.read_csv(CORE / tag / "core_real_direction_top5.tsv", sep="\t",
                           usecols=["snp_id", "pool_id", "chrom", "pos"])
        core_ids = core["pool_id"].astype(str).tolist()
        cvi, cfd = panel.indices(core_ids)
        ccr, cmac, cmaf = panel.qc(cvi, cfd)
        cpass_arr = (ccr >= CR_MIN) & (cmac >= MAC_MIN) & cfd
        core["call_rate"] = ccr
        core["mac"] = cmac
        core["maf"] = cmaf
        core["pass_qc"] = cpass_arr
        core["status"] = np.where(cpass_arr, "qualified", "demoted")
        core.to_csv(tdir / "core_qc.tsv", sep="\t", index=False)
        core[~cpass_arr].to_csv(tdir / "core_demoted.tsv", sep="\t", index=False)
        cpass = dict(zip(core_ids, cpass_arr.tolist()))
        core_set = set(core_ids)
        log(f"  (1) Core n={len(core_ids)} qualified={int(cpass_arr.sum())} demoted={int((~cpass_arr).sum())}")


        ed = pd.read_csv(CORE / tag / "ld_edges_r2ge06_revised.tsv", sep="\t")
        a = ed["snp_a"].astype(str)
        b = ed["snp_b"].astype(str)
        a_core = a.isin(core_set).to_numpy()
        b_core = b.isin(core_set).to_numpy()
        qualified_ids = {k for k, v in cpass.items() if v}
        a_ok = a.isin(qualified_ids).to_numpy()
        b_ok = b.isin(qualified_ids).to_numpy()
        edge_valid = (a_ok | ~a_core) & (b_ok | ~b_core)
        edv = ed.loc[edge_valid].copy()
        edv.to_csv(tdir / "ld_edges_r2ge06_qc_filtered.tsv.gz", sep="\t",
                   index=False, compression="gzip")
        log(f"  (2) edges {len(ed)} -> {len(edv)} kept (dropped {int((~edge_valid).sum())} "
            f"= {(~edge_valid).mean()*100:.1f}% by unqualified core endpoint)")


        av = edv["snp_a"].astype(str).to_numpy()
        bv = edv["snp_b"].astype(str).to_numpy()
        av_core = np.isin(av, list(core_set))
        prox = np.where(av_core, bv, av)
        core_side = np.where(av_core, av, bv)
        r2v = pd.to_numeric(edv["r2"], errors="coerce").to_numpy(float)
        is_proxy = ~pd.Series(prox).isin(core_set).to_numpy()
        agg = pd.DataFrame({"proxy": prox[is_proxy], "core": core_side[is_proxy], "r2": r2v[is_proxy]})
        agg = agg.sort_values(["proxy", "r2"], kind="stable")
        best = agg.groupby("proxy", as_index=False).tail(1).set_index("proxy")
        n_edges = agg.groupby("proxy").size()
        log(f"  (2b) distinct proxy endpoints with >=1 qualified core edge = {len(best)}")


        pool = pd.read_csv(CORE / tag / "candidate_pool.tsv", sep="\t")
        pids = pool["panel_id"].astype(str).tolist()
        pvi, pfd = panel.indices(pids)
        pcr, pmac, pmaf = panel.qc(pvi, pfd)
        own = (pcr >= CR_MIN) & (pmac >= MAC_MIN) & pfd
        role = pool["role"].to_numpy()
        has_good_core = pool["panel_id"].astype(str).isin(best.index).to_numpy()
        keep = np.where(role == "core", own, own & has_good_core)

        pool["call_rate"] = pcr
        pool["mac"] = pmac
        pool["maf"] = pmaf
        pool["pass_qc_self"] = own
        pool["n_qualified_core_edges"] = pool["panel_id"].astype(str).map(n_edges).fillna(0).astype(int).to_numpy()
        pool["best_qualified_core"] = pool["panel_id"].astype(str).map(best["core"]).to_numpy()
        pool["best_qualified_r2"] = pool["panel_id"].astype(str).map(best["r2"]).to_numpy()

        proxy_mask = role == "proxy"
        pool.loc[proxy_mask, "source_core"] = pool.loc[proxy_mask, "best_qualified_core"]
        pool.loc[proxy_mask, "r2_to_core"] = pool.loc[proxy_mask, "best_qualified_r2"]

        usable = pool.loc[keep].drop(columns=["best_qualified_core", "best_qualified_r2"])
        usable.to_csv(tdir / "candidate_pool.tsv", sep="\t", index=False)

        reason = np.where(
            role == "core",
            np.where(own, "kept", "core_failed_self_qc"),
            np.where(~own, "proxy_failed_self_qc",
                     np.where(has_good_core, "kept", "proxy_no_qualified_core_edge")),
        )
        pd.DataFrame({"panel_id": pids, "role": role, "reason": reason}) \
            .query("reason != 'kept'") \
            .to_csv(tdir / "dropped_proxy_reasons.tsv", sep="\t", index=False)
        log(f"  (3)(4) usable Core+LD set = {int(keep.sum())} / {len(pool)}  ({keep.mean()*100:.2f}%)")

        row = {
            "trait": tag, "samples": panel.n_samples,
            "core_definition_n": len(core_ids), "core_qualified": int(cpass_arr.sum()),
            "core_demoted": int((~cpass_arr).sum()),
            "edges_in": int(len(ed)), "edges_kept": int(len(edv)),
            "edges_dropped": int((~edge_valid).sum()),
            "pool_in": int(len(pool)),
            "usable_core": int(((role == "core") & keep).sum()),
            "usable_proxy": int(((role == "proxy") & keep).sum()),
            "usable_total": int(keep.sum()),
            "usable_pct": round(float(keep.mean()) * 100, 2),
            "proxy_self_qc_pass": int((own & proxy_mask).sum()),
            "proxy_with_qualified_core": int((has_good_core & proxy_mask).sum()),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "build_summary.tsv", sep="\t", index=False)
        with open(OUT / "build_summary.json", "w") as fh:
            json.dump(rows, fh, indent=2)
        log("  " + json.dumps(row))
        log(f"  ({time.time()-t0:.1f}s)")
        del panel, pool, ed, edv, agg, best

    log("done -> " + str(OUT))


def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    base=args.config.resolve().parent
    def locate(value):
        path=Path(value)
        return path if path.is_absolute() else base/path
    global CORE, OUT, TRAITS
    CORE=locate(cfg['core_source_root'])
    OUT=locate(cfg['output_root'])
    OUT.mkdir(parents=True,exist_ok=True)
    TRAITS={tag:(locate(row['panel_directory']),row['species']) for tag,row in cfg['traits'].items()}
    main()

if __name__=='__main__':
    cli()
