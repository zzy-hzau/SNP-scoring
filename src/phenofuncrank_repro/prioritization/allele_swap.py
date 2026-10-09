"genotype layout and allele-swap math with model loading."
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from phenofuncrank_repro.models.dpcformer.network import PhenotypePredictor_1v

_BASE_TO_VEC={
    "A":np.array([1,0,0,0],dtype=np.float32),
    "C":np.array([0,1,0,0],dtype=np.float32),
    "G":np.array([0,0,1,0],dtype=np.float32),
    "T":np.array([0,0,0,1],dtype=np.float32),
}

def _import_model_class():
    return PhenotypePredictor_1v

def _fold_artifact(model_dir: Path, fold: int, suffix: str) -> Path | None:
    folder = model_dir / f"fold_{fold:02d}"
    matches = sorted(folder.glob(f"*{suffix}"))
    return matches[0] if matches else None


def _codes_to_onehot(codes_snp_by_sample: np.ndarray, snp_refs: np.ndarray, snp_alts: np.ndarray, selected_snp_count: int) -> np.ndarray:
    n_selected, n_samples = codes_snp_by_sample.shape
    out = np.zeros((n_samples, selected_snp_count, 8), dtype=np.float32)
    for j in range(min(n_selected, selected_snp_count)):
        ref = snp_refs[j]
        alt = snp_alts[j]
        if ref not in _BASE_TO_VEC or alt not in _BASE_TO_VEC:
            continue
        rv = _BASE_TO_VEC[ref]
        av = _BASE_TO_VEC[alt]
        c = codes_snp_by_sample[j, :]
        m0 = c == 1
        out[m0, j, 0:4] = rv
        out[m0, j, 4:8] = rv
        m2 = c == 2
        out[m2, j, 0:4] = rv
        out[m2, j, 4:8] = av
        m3 = c == 3
        out[m3, j, 0:4] = av
        out[m3, j, 4:8] = av
    return out


def _set_homozygous(x: "torch.Tensor", group_idx: int, rank_idx: int, base_vec: np.ndarray):
    import torch
    y = x.clone()
    y[:, group_idx, rank_idx, 0:4] = torch.tensor(base_vec, dtype=torch.float32, device=y.device)
    y[:, group_idx, rank_idx, 4:8] = torch.tensor(base_vec, dtype=torch.float32, device=y.device)
    return y


def _build_input_tensor(geno: np.ndarray, selected_map: pd.DataFrame, sample_indices: np.ndarray, selected_snp_count: int, n_groups: int):
    import torch
    grouped = []
    for group_id in range(1, n_groups + 1):
        gm = selected_map[selected_map["group_id"].eq(group_id)].sort_values("selected_rank").reset_index(drop=True)
        if len(gm) == 0:
            raise RuntimeError(f"No selected SNPs for group {group_id}")
        if len(gm) > selected_snp_count:
            gm = gm.iloc[:selected_snp_count].copy()
        variant_idx = gm["variant_index"].to_numpy(dtype=np.int64)
        codes = np.asarray(geno[variant_idx, :][:, sample_indices], dtype=np.uint8)
        encoded = _codes_to_onehot(codes, gm["Ref"].str.upper().values, gm["Alt"].str.upper().values, len(gm))
        ranks = gm["selected_rank"].to_numpy(dtype=np.int64) - 1
        if (ranks < 0).any() or (ranks >= selected_snp_count).any() or len(np.unique(ranks)) != len(ranks):
            raise ValueError("Selected SNP ranks do not match the trained input layout")
        onehot = np.zeros((len(sample_indices), selected_snp_count, 8), dtype=np.float32)
        onehot[:, ranks, :] = encoded
        grouped.append(onehot)
    x = np.stack(grouped, axis=1)
    return torch.tensor(x, dtype=torch.float32)


def torch_no_grad():
    """Context-manager decorator that uses torch.inference_mode or torch.no_grad."""
    try:
        import torch
        if hasattr(torch, "inference_mode"):
            return torch.inference_mode()
        return torch.no_grad()
    except ImportError:
        class _DummyCtx:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __call__(self, fn):
                def wrapper(*args, **kwargs):
                    return fn(*args, **kwargs)
                return wrapper
        return _DummyCtx()


@torch_no_grad()
def _predict_batches(model, x, batch_size: int, device):
    import torch
    model.eval()
    preds = []
    n = x.shape[0]
    for i in range(0, n, batch_size):
        xb = x[i:i + batch_size].to(device, non_blocking=True)
        y = model(xb).cpu().numpy().reshape(-1)
        preds.append(y)
    return np.concatenate(preds)


def _load_fold_splits(model_dir: Path, trait: str, fold: int) -> np.ndarray | None:
    del trait
    path = _fold_artifact(model_dir, fold, ".splits.npz")
    if path is None:
        return None
    data = np.load(path)
    return data["test_memmap_cols"].astype(int)


def _load_fold_scaler(model_dir: Path, trait: str, fold: int) -> tuple[float, float] | None:
    del trait
    path = _fold_artifact(model_dir, fold, ".phenotype_scaler.npz")
    if path is None:
        return None
    data = np.load(path)
    return float(data["mean_"].item()), float(data["scale_"].item())


def _load_fold_selected_map(model_dir: Path, trait: str, fold: int) -> pd.DataFrame | None:
    del trait
    path = _fold_artifact(model_dir, fold, ".selected_snp_map.tsv")
    if path is None:
        return None
    sm = pd.read_csv(path, sep="\t")
    sm["chromosome"] = sm["chromosome"].astype(str)

    if "selected_rank" not in sm.columns and "selected_rank_physical" in sm.columns:
        sm["selected_rank"] = sm["selected_rank_physical"]
    return sm


def _load_fold_model(model_dir: Path, trait: str, fold: int, n_groups: int, selected_snp_count: int, device):
    import torch
    PhenotypePredictor_1v = _import_model_class()
    preferred = model_dir / "saved_models" / f"{trait}.best_model_fold_{fold}.pth"
    candidates = sorted((model_dir / "saved_models").glob(f"*.best_model_fold_{fold}.pth"))
    model_path = preferred if preferred.exists() else (candidates[0] if candidates else None)
    if model_path is None:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model = PhenotypePredictor_1v(n_groups, [selected_snp_count] * n_groups, 8).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _permutation_signflip_pvalues(effect_records: list[dict], n_perm: int = 1000) -> dict[str, float]:
    if len(effect_records) == 0:
        return {}
    df = pd.DataFrame(effect_records)
    pvals: dict[str, float] = {}
    rng = np.random.RandomState(42)
    for snp_id, sub in df.groupby("snp_id"):
        deltas = sub["delta_y"].values.astype(float)
        obs = np.mean(deltas)
        if len(deltas) < 2:
            pvals[snp_id] = 1.0
            continue
        perm_means = np.empty(n_perm)
        for i in range(n_perm):
            flipped = deltas * np.where(rng.rand(len(deltas)) < 0.5, -1.0, 1.0)
            perm_means[i] = np.mean(flipped)
        pvals[snp_id] = float((np.sum(np.abs(perm_means) >= np.abs(obs)) + 1) / (n_perm + 1))
    return pvals


def _bh_fdr(pvalues: dict[str, float]) -> dict[str, float]:
    if not pvalues:
        return {}
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    n = len(items)
    qvals: dict[str, float] = {}
    for rank, (key, p) in enumerate(items, start=1):
        q = min(p * n / rank, 1.0)
        qvals[key] = q
    for i in range(len(items) - 2, -1, -1):
        prev_key = items[i][0]
        next_key = items[i + 1][0]
        qvals[prev_key] = min(qvals[prev_key], qvals[next_key])
    return qvals


def _set_group(x_group, r0, vec):
    """Return a copy of (n_sam, n_sel, 8) with position r0 set to `vec`."""
    y = x_group.clone()
    t = torch.tensor(vec, dtype=torch.float32, device=y.device)
    y[:, r0, 0:4] = t
    y[:, r0, 4:8] = t
    return y


@torch.no_grad()
def group_cached_deltas(model, xb, items, base_vec, device,
                        snp_chunk: int = 128, sample_batch: int = 4096):
    """items: list of (group0, rank0, ref_base, alt_base).

    Returns `mean(pred_alt) - mean(pred_ref)` for each item, in order.
    """
    n_sam = xb.shape[0]
    model.eval()
    base_emb = [cnn(xb[:, i, :, :].permute(0, 2, 1))
                for i, cnn in enumerate(model.chromosome_cnns)]
    n_grp = len(base_emb)

    by_group: dict[int, list[int]] = defaultdict(list)
    for k, it in enumerate(items):
        by_group[it[0]].append(k)

    preds = {}          
    for g0, idxs in by_group.items():
        cnn = model.chromosome_cnns[g0]
        xg_base = xb[:, g0, :, :]
        for state in (2, 3):        
            for s in range(0, len(idxs), snp_chunk):
                ks = idxs[s:s + snp_chunk]
                mods = [_set_group(xg_base, items[k][1], base_vec[items[k][state]])
                        for k in ks]
                mod = torch.stack(mods, dim=0)                     
                B = mod.shape[0]
                flat = mod.reshape(B * n_sam, mod.shape[2], mod.shape[3]).permute(0, 2, 1)
                emb_flat = cnn(flat)
                emb = emb_flat.reshape(B, n_sam, *emb_flat.shape[1:])

                seqs = []
                for bi in range(B):
                    outs = list(base_emb)
                    outs[g0] = emb[bi]
                    seqs.append(torch.cat(outs, dim=2))
                combined = torch.stack(seqs, dim=0).permute(0, 1, 3, 2)   
                combined = combined.reshape(B * n_sam, *combined.shape[2:])
                h = model.transformer_decoder(combined)
                p = model.mlp(torch.flatten(h, 1)).reshape(B, n_sam).cpu().numpy()
                for bi, k in enumerate(ks):
                    slot = preds.setdefault(k, [None, None])
                    slot[0 if state == 2 else 1] = p[bi]
                del mods, mod, flat, emb, seqs, combined, h, p

    return [(preds[k][1].mean() - preds[k][0].mean()) for k in range(len(items))]


