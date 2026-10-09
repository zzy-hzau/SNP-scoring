"""Utilities for DPCformer fold training."""
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def safe_corr(fn, x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return float("nan")
    return float(fn(x[ok], y[ok])[0])

def available_memory_gb() -> float:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return float("inf")
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    return float("inf")

def enforce_memory_reserve(min_available_gb: float, context: str) -> None:
    if min_available_gb <= 0:
        return
    available_gb = available_memory_gb()
    if available_gb < min_available_gb:
        raise RuntimeError(
            f"Refusing to start {context}: MemAvailable={available_gb:.1f} GiB "
            f"is below required reserve {min_available_gb:.1f} GiB"
        )

def first_present(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None

def resolve_path(inputs_dir: Path, value, fallbacks):
    candidates = []
    if value and not Path(str(value)).is_absolute():
        candidates.append(inputs_dir / str(value))
    candidates.extend(inputs_dir / x for x in fallbacks)
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("Required supplied input resource is missing; use relative metadata paths")

def select_top(df, scores, k, score_col):
    if len(df) != len(scores):
        raise RuntimeError("Feature-score length mismatch")
    work = df.copy()
    work[score_col] = scores
    work = work.sort_values(
        [score_col, "Chr", "Pos", "Ref", "Alt", "snp_id"],
        ascending=[False, True, True, True, True, True],
        kind="mergesort",
    )
    return work.head(min(int(k), len(work))).copy()

def make_loader(x, y, indices, batch_size, shuffle, seed):
    idx = torch.as_tensor(indices, dtype=torch.long)
    dataset = TensorDataset(x.index_select(0, idx), y.index_select(0, idx))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )

def evaluate(model, loader, criterion, device):
    model.eval()
    losses = 0.0
    n = 0
    preds = []
    obs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(xb)
            loss = criterion(out, yb)
            losses += float(loss.item()) * xb.size(0)
            n += xb.size(0)
            preds.extend(out.detach().cpu().numpy().reshape(-1))
            obs.extend(yb.detach().cpu().numpy().reshape(-1))
    return {
        "loss": losses / max(n, 1),
        "pearson": safe_corr(pearsonr, preds, obs),
        "spearman": safe_corr(spearmanr, preds, obs),
        "preds": np.asarray(preds, dtype=float),
        "obs": np.asarray(obs, dtype=float),
    }
