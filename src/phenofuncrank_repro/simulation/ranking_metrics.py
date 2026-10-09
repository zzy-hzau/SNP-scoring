"""Original paired QTN ranking metric calculations, with caller-supplied inputs."""
import numpy as np
import pandas as pd
K_VALUES=(20,50,100)
LUT=np.array([np.nan,0.0,1.0,2.0])

def same_chrom_r2(geno, truth_index: int, selected_indices: np.ndarray):
    a = np.asarray(geno[truth_index], dtype=np.uint8)
    x = LUT[a]
    valid = a != 0
    x = np.where(valid, x, np.nan)
    xc = x - np.nanmean(x)
    denx = np.nansum(xc * xc)
    rows = np.asarray(geno[selected_indices], dtype=np.uint8)
    y = LUT[rows]
    y = np.where(valid[None, :] & (rows != 0), y, np.nan)
    yc = y - np.nanmean(y, axis=1, keepdims=True)
    num = np.nansum(yc * xc[None, :], axis=1)
    den = np.sqrt(np.nansum(yc * yc, axis=1) * denx)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0) ** 2


def norm_chr(value):
    raw = str(value).lower().replace('chr', '')
    return str(int(raw)) if raw.isdigit() else raw.upper()


def allele_key(chrom, pos, ref, alt):
    return f'{norm_chr(chrom)}:{int(pos)}_{str(ref).upper()}_{str(alt).upper()}'


def source_vector(ctx, idx: int):
    start, end = int(ctx['cand_ptr'][idx]), int(ctx['cand_ptr'][idx + 1])
    return dict(zip(ctx['cand_src'][start:end], ctx['cand_p'][start:end]))


def summarize_one_ranking(tag, rep_no, method, order, ctx, z, truth, truth_map):
    order = np.asarray(order[:100], dtype=np.int64)
    assert len(order) == 100 and len(set(order)) == 100
    selected_keys = [allele_key(ctx['chr'][i], ctx['pos'][i], ctx['ref'][i], ctx['alt'][i]) for i in order]
    truth_keys = [allele_key(x.chrom, x.pos, x.snp_id.split('_')[-2], x.snp_id.split('_')[-1])
                  for x in truth.itertuples(index=False)]
    truth_ids = truth.snp_id.astype(str).tolist()
    chrom = np.array([norm_chr(ctx['chr'][i]) for i in order], dtype=object)
    geno_idx = np.asarray(ctx['geno_idx'][order], dtype=np.int64)
    ld_matrix = np.vstack([
        np.where(chrom == norm_chr(truth_map[q]['chrom']),
                 same_chrom_r2(z.geno, truth_map[q]['index'], geno_idx), 0.0)
        for q in truth_ids
    ])
    Z = np.asarray(z[order], dtype=np.float32)
    pair = np.square(Z @ Z.T).astype(np.float64)
    pair = np.clip(pair, 0.0, 1.0)
    source_maps = [source_vector(ctx, int(i)) for i in order]
    jaccard = np.zeros((100, 100), dtype=np.float32)
    for i in range(100):
        left = source_maps[i]
        for j in range(i + 1, 100):
            right = source_maps[j]
            if len(left) > len(right):
                left, right = right, left
            overlap = sum(min(float(v), float(right.get(s, 0.0))) for s, v in left.items())
            jaccard[i, j] = overlap / (2.0 - overlap) if overlap > 0 else 0.0
            left = source_maps[i]
    rows = []
    for k in K_VALUES:
        truth_exact = sum(q in set(selected_keys[:k]) for q in truth_keys)
        truth_ld = int(np.any(ld_matrix[:, :k] >= 0.60, axis=1).sum())
        upper = np.triu_indices(k, 1)
        same_chr = chrom[upper[0]] == chrom[upper[1]]
        r2_vals = pair[:k, :k][upper][same_chr]
        j_vals = jaccard[:k, :k][upper]
        rows.append({
            'trait': tag, 'rep': rep_no, 'method': method, 'K': k, 'n_qtn': len(truth),
            'exact_qtn_hits': int(truth_exact), 'samechrom_ld_qtn_covered': truth_ld,
            'exact_recall': truth_exact / len(truth), 'samechrom_ld_recall': truth_ld / len(truth),
            'distinct_ld_components': len(set(ctx['ldcc_id'][order[:k]])),
            'max_samechrom_pair_r2': float(np.max(r2_vals)) if len(r2_vals) else 0.0,
            'mean_samechrom_pair_r2': float(np.mean(r2_vals)) if len(r2_vals) else 0.0,
            'samechrom_pairs_r2ge06': int((r2_vals >= 0.60).sum()),
            'mean_pair_source_jaccard': float(np.mean(j_vals)) if len(j_vals) else 0.0,
            'pairs_sharing_source': int((j_vals > 0).sum()),
        })
    ranks = pd.DataFrame({
        'trait': tag, 'rep': rep_no, 'method': method,
        'rank': np.arange(1, 101), 'panel_id': np.asarray(ctx['snp_id'])[order],
        'T': np.asarray(ctx['T'])[order], 'F': np.asarray(ctx['F'])[order],
        'B': np.asarray(ctx['B'])[order], 'ldcc_id': np.asarray(ctx['ldcc_id'])[order],
    })
    return rows, ranks


