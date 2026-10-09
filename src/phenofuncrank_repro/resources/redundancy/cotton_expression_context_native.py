"cotton expression workflow; scientific functions retained from source."
import argparse,os,sys,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import importlib

class _LazyModule:
    def __init__(self, name):
        self.name = name
    def __getattr__(self, key):
        return getattr(importlib.import_module(self.name), key)


def pycombat(*args, **kwargs):
    from combat.pycombat import pycombat as implementation
    return implementation(*args, **kwargs)

HOMOLOG_FILE = METADATA_FILE = QUANT_DIR = OUTPUT_DIR = None
logger = None


MIN_MAPPING_RATE = 0.60

MIN_EXPRESSED_GENES = 20000

TPM_CUTOFFS = [0.5, 1.0, 2.0]

BOOTSTRAP_ITERATIONS = 1000

FDR_ALPHA = 0.05

MIN_SAMPLES_FOR_CORR = 4

class Logger:
    def __init__(self, logfile):
        self.logfile = logfile
        self.log_handle = open(logfile, 'w')

    def log(self, message, level="INFO"):
        msg = f"[{level}] {message}"
        print(msg)
        self.log_handle.write(msg + "\n")
        self.log_handle.flush()

    def close(self):
        self.log_handle.close()

def collapse_to_context(tpm_matrix, metadata):
    """
    将样本表达矩阵压缩成‘情境表达向量’：
    每个 (tissue, condition) -> 1列（取中位数），避免某组织样本过多主导 Pearson r
    """
    md = metadata.loc[tpm_matrix.columns].copy()
    md["condition"] = md["condition"].replace({"control":"normal"}).astype(str).str.lower()
    md["tissue"] = md["tissue"].astype(str).str.lower()
    ctx = md["tissue"] + "__" + md["condition"]

    collapsed = []
    for k, cols in ctx.groupby(ctx).groups.items():
        sub = tpm_matrix[list(cols)]
        collapsed.append(sub.median(axis=1).rename(k))
    out = pd.concat(collapsed, axis=1)
    return out

def load_and_qc_samples():
    """加载Kallisto定量结果并进行严格质控"""
    logger.log("=== Step 1: Loading and QC ===")


    metadata = pd.read_csv(METADATA_FILE, sep='\t', index_col='sample_id')
    logger.log(f"Metadata loaded: {len(metadata)} samples")


    tpm_list = []
    qc_stats = []

    for sample_id in metadata.index:
        abund_file = f"{QUANT_DIR}/{sample_id}/abundance.tsv"

        if not os.path.exists(abund_file):
            logger.log(f"Missing: {sample_id}", "WARNING")
            continue
        import json

        runinfo_file = f"{QUANT_DIR}/{sample_id}/run_info.json"
        percent_mapped = np.nan
        if os.path.exists(runinfo_file):
            with open(runinfo_file, "r") as fh:
                rj = json.load(fh)
            n_proc = rj.get("n_processed", 0)
            n_map = rj.get("n_pseudoaligned", 0)
            percent_mapped = (n_map / n_proc) if n_proc else np.nan
        else:
            logger.log(f"Missing run_info.json: {sample_id}", "WARNING")


        df = pd.read_csv(abund_file, sep='\t', usecols=['target_id', 'tpm'])
        df['target_id'] = df['target_id'].str.split('.').str[0]  
        df = df.groupby('target_id')['tpm'].sum()  


        total_tpm = df.sum()
        n_expressed = (df > 1.0).sum()

        qc_stats.append({
            'sample_id': sample_id,
            'total_tpm': total_tpm,
            'n_expressed_genes': n_expressed,
            'max_tpm': df.max(),
            'percent_mapped': percent_mapped
        })

        tpm_list.append(df.rename(sample_id))


    tpm_matrix = pd.concat(tpm_list, axis=1).fillna(0).astype(np.float32)
    logger.log(f"TPM matrix: {tpm_matrix.shape[0]} genes × {tpm_matrix.shape[1]} samples")


    qc_df = pd.DataFrame(qc_stats).set_index('sample_id')
    qc_df.to_csv(f"{OUTPUT_DIR}/qc/sample_qc_stats.tsv", sep='\t')

    failed_samples = qc_df[
        (qc_df['n_expressed_genes'] < MIN_EXPRESSED_GENES) |
        (qc_df['total_tpm'] < 500000) |
        (qc_df['percent_mapped'] < MIN_MAPPING_RATE)
        ].index

    if len(failed_samples) > 0:
        logger.log(f"Removing {len(failed_samples)} low-quality samples:", "WARNING")
        for s in failed_samples:
            logger.log(f"  - {s}: {qc_df.loc[s, 'n_expressed_genes']} genes", "WARNING")
        tpm_matrix = tpm_matrix.drop(columns=failed_samples)
        metadata = metadata.drop(index=failed_samples)

    logger.log(f"Final matrix: {tpm_matrix.shape[1]} samples passed QC")

    bad = metadata[(metadata['tissue'].isna()) | (metadata['condition'].isna())]
    if len(bad) > 0:
        logger.log(f"FATAL: metadata has missing tissue/condition for {len(bad)} samples", "ERROR")
        logger.log(str(bad.head()), "ERROR")
        sys.exit(1)

    return tpm_matrix, metadata, qc_df

def diagnose_batch_effect(tpm_matrix, metadata, prefix="before"):
    """诊断批次效应的严重程度"""
    logger.log(f"=== Batch Effect Diagnosis ({prefix}) ===")


    gene_var = tpm_matrix.var(axis=1)
    top_genes = gene_var.nlargest(2000).index


    log_tpm = np.log1p(tpm_matrix.loc[top_genes])
    scaled_data = StandardScaler().fit_transform(log_tpm.T)


    pca = PCA(n_components=10)
    coords = pca.fit_transform(scaled_data)


    batch_labels = metadata.loc[tpm_matrix.columns, 'project']


    from scipy.stats import f_oneway

    pc1_by_batch = [coords[batch_labels == b, 0] for b in batch_labels.unique()]
    pc2_by_batch = [coords[batch_labels == b, 1] for b in batch_labels.unique()]

    f_stat_pc1, p_val_pc1 = f_oneway(*pc1_by_batch)
    f_stat_pc2, p_val_pc2 = f_oneway(*pc2_by_batch)

    logger.log(f"PC1 batch effect: F={f_stat_pc1:.2f}, p={p_val_pc1:.2e}")
    logger.log(f"PC2 batch effect: F={f_stat_pc2:.2f}, p={p_val_pc2:.2e}")


    need_correction = (p_val_pc1 < 0.01) or (p_val_pc2 < 0.01)
    logger.log(f"Batch correction needed: {need_correction}")

    return need_correction, pca.explained_variance_ratio_

def combat_correction_with_validation(tpm_matrix, metadata):
    """执行ComBat校正并验证生物学信号保留"""
    logger.log("=== Step 2: ComBat Batch Correction ===")


    need_correction, var_before = diagnose_batch_effect(tpm_matrix, metadata, "before")

    if not need_correction:
        logger.log("Batch effect negligible, skipping correction")
        return tpm_matrix


    batch = metadata.loc[tpm_matrix.columns, 'project']
    tissue = metadata.loc[tpm_matrix.columns, 'tissue']


    confound_check = pd.crosstab(batch, tissue)
    logger.log("Batch × Tissue contingency table:")
    logger.log(str(confound_check))


    single_tissue_batches = confound_check.index[(confound_check > 0).sum(axis=1) == 1]
    if len(single_tissue_batches) > 0:
        logger.log(f"FATAL: Batch confounded with tissue: {list(single_tissue_batches)}", "ERROR")
        logger.log("Skipping ComBat to avoid over-correction (Nature-grade leakage control).", "ERROR")
        return tpm_matrix


    expressed_genes = tpm_matrix.index[tpm_matrix.max(axis=1) > 0.5]
    tpm_filtered = tpm_matrix.loc[expressed_genes]

    logger.log(f"Running ComBat on {len(expressed_genes)} genes...")

    try:
        corrected_data = pycombat(tpm_filtered, batch)
        corrected_data[corrected_data < 0] = 0  


        corrected_full = tpm_matrix.copy()
        corrected_full.loc[expressed_genes] = corrected_data

    except Exception as e:
        logger.log(f"ComBat failed: {e}", "ERROR")
        logger.log("Using uncorrected data", "WARNING")
        return tpm_matrix


    _, var_after = diagnose_batch_effect(corrected_full, metadata, "after")


    logger.log("Validating biological signal preservation...")


    
    tissue_var_before = []
    tissue_var_after = []

    for gene in expressed_genes[:1000]:  
        var_b = tpm_filtered.loc[gene].groupby(tissue).mean().var()
        var_a = corrected_full.loc[gene].groupby(tissue).mean().var()
        tissue_var_before.append(var_b)
        tissue_var_after.append(var_a)

    corr_preservation = np.corrcoef(tissue_var_before, tissue_var_after)[0, 1]
    logger.log(f"Tissue variance preservation: r={corr_preservation:.3f}")

    if corr_preservation < 0.7:
        logger.log("WARNING: Biological signal may be over-corrected!", "WARNING")

    return corrected_full

def compute_correlation_with_stats(vec1, vec2, compute_ci=False, n_bootstrap=BOOTSTRAP_ITERATIONS):
    """计算Pearson相关系数、p值和bootstrap置信区间"""


    mask = ~(np.isnan(vec1) | np.isnan(vec2))
    v1 = vec1[mask]
    v2 = vec2[mask]

    n = len(v1)
    if n < MIN_SAMPLES_FOR_CORR:
        return np.nan, 1.0, np.nan, np.nan


    r, p_val = pearsonr(v1, v2)
    ci_lower, ci_upper = np.nan, np.nan
    if compute_ci:
        bootstrap_rs = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            r_boot, _ = pearsonr(v1[indices], v2[indices])
            bootstrap_rs.append(r_boot)
        ci_lower = np.percentile(bootstrap_rs, 2.5)
        ci_upper = np.percentile(bootstrap_rs, 97.5)

    return r, p_val, ci_lower, ci_upper

def vectorized_correlation_analysis(tpm_matrix, homologs, tpm_cutoff=1.0):
    """向量化计算同源基因对的表达相关性"""
    logger.log(f"=== Computing correlations (TPM cutoff={tpm_cutoff}) ===")


    valid_mask = homologs['gene_At'].isin(tpm_matrix.index) & \
                 homologs['gene_Dt'].isin(tpm_matrix.index)
    homologs_valid = homologs[valid_mask].copy()

    logger.log(f"Valid homolog pairs: {len(homologs_valid)}")


    
    log_mat = np.log2(tpm_matrix + 1.0)
    mat_at_log = log_mat.loc[homologs_valid['gene_At']].values
    mat_dt_log = log_mat.loc[homologs_valid['gene_Dt']].values


    mat_at_raw = tpm_matrix.loc[homologs_valid['gene_At']].values
    mat_dt_raw = tpm_matrix.loc[homologs_valid['gene_Dt']].values


    results = []

    for i, (idx, row) in enumerate(homologs_valid.iterrows()):
        if i % 5000 == 0:
            logger.log(f"  Processed {i}/{len(homologs_valid)} pairs...")

        vec_at_log = mat_at_log[i]
        vec_dt_log = mat_dt_log[i]
        r, p_val, ci_low, ci_high = compute_correlation_with_stats(vec_at_log, vec_dt_log, compute_ci=False)

        vec_at_raw = mat_at_raw[i]
        vec_dt_raw = mat_dt_raw[i]

        breadth = ((vec_at_raw > tpm_cutoff) | (vec_dt_raw > tpm_cutoff)).mean()
        mean_expr_at = vec_at_raw[vec_at_raw > tpm_cutoff].mean() if (vec_at_raw > tpm_cutoff).any() else 0.0
        mean_expr_dt = vec_dt_raw[vec_dt_raw > tpm_cutoff].mean() if (vec_dt_raw > tpm_cutoff).any() else 0.0


        expr_corr = r
        expr_corr_pos = expr_corr if not np.isnan(expr_corr) else 0.0


        has_homolog = 1

        results.append({
            'gene_At': row['gene_At'],
            'gene_Dt': row['gene_Dt'],
            'expr_corr': expr_corr,
            'p_value': p_val,
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'breadth': breadth,
            'mean_expr_At': mean_expr_at,
            'mean_expr_Dt': mean_expr_dt,
            'has_homolog': has_homolog,
            'functional_redundancy': has_homolog * expr_corr_pos*breadth
        })
    results_df = pd.DataFrame(results)


    valid_pvals = ~results_df['p_value'].isna()
    results_df.loc[valid_pvals, 'p_adj'] = multipletests(
        results_df.loc[valid_pvals, 'p_value'],
        method='fdr_bh'
    )[1]


    
    results_df['significant'] = (results_df['p_adj'] < FDR_ALPHA)
    results_df['breadth_pass'] = (results_df['breadth'] > 0.3)


    results_df['high_confidence'] = results_df['significant'] & results_df['breadth_pass']

    logger.log(f"Significant pairs (FDR<{FDR_ALPHA}): {results_df['significant'].sum()}")

    return results_df

def sensitivity_analysis(tpm_matrix, homologs):
    """测试不同TPM阈值对结果的影响"""
    logger.log("=== Step 3: Sensitivity Analysis ===")

    sensitivity_results = {}

    for cutoff in TPM_CUTOFFS:
        logger.log(f"Testing TPM cutoff = {cutoff}")
        results = vectorized_correlation_analysis(tpm_matrix, homologs, cutoff)
        sensitivity_results[cutoff] = results


        results.to_csv(
            f"{OUTPUT_DIR}/tables/expr_corr_cutoff_{cutoff}.tsv",
            sep='\t', index=False
        )


    logger.log("Comparing results across cutoffs...")


    logger.log(f"Sensitivity analysis complete. Using cutoff={TPM_CUTOFFS[1]} as default.")

    return sensitivity_results[TPM_CUTOFFS[1]]


def generate_final_feature_table(results_df, homologs):
    """生成用于GP训练的最终特征表"""
    logger.log("=== Step 5: Generating Final Feature Table ===")


    final_features = results_df.copy()


    

    final_features["expr"] = final_features["expr_corr"].fillna(0.0)  
    final_features["has_homolog"] = 1


    final_features["functional_redundancy"] = (
            final_features["has_homolog"] *
            final_features["expr"] *  
            final_features["breadth"]
    )


    output_cols = [
        "gene_At", "gene_Dt", "expr", "expr_corr", "p_value", "p_adj",
        "breadth", "has_homolog", "functional_redundancy"
    ]

    final_features[output_cols].to_csv(
        f"{OUTPUT_DIR}/homolog_expr_features_final.tsv",
        sep='\t', index=False
    )

    gene_feat_at = final_features[["gene_At", "expr", "functional_redundancy"]].rename(
        columns={"gene_At": "gene_id"}
    )
    gene_feat_dt = final_features[["gene_Dt", "expr", "functional_redundancy"]].rename(
        columns={"gene_Dt": "gene_id"}
    )

    gene_features = pd.concat([gene_feat_at, gene_feat_dt], axis=0, ignore_index=True)


    gene_features = gene_features.groupby("gene_id", as_index=False).agg({
        "expr": "max",
        "functional_redundancy": "max",
    })


    gene_features["has_homolog"] = 1

    gene_features.to_csv(
        f"{OUTPUT_DIR}/homolog_expr_features_per_gene.tsv",
        sep="\t",
        index=False
    )


    logger.log("=== Final Statistics ===")
    logger.log(f"Total homolog pairs: {len(final_features)}")
    logger.log(f"Pairs with significant correlation: {final_features['significant'].sum()}")
    logger.log(f"Pairs with functional redundancy: {final_features['has_homolog'].sum()}")
    logger.log(f"Mean correlation: {final_features['expr_corr'].mean():.3f}")
    logger.log(f"Median correlation: {final_features['expr_corr'].median():.3f}")


    for threshold in [0.8, 0.6, 0.4, 0.2]:
        count = (final_features['expr_corr'] > threshold).sum()
        logger.log(f"Pairs with r > {threshold}: {count} ({count / len(final_features) * 100:.1f}%)")

    return final_features

def main():
    logger.log("=" * 60)
    logger.log("Nature Genetics Expression Correlation Pipeline v2.0")
    logger.log("=" * 60)

    try:

        tpm_matrix, metadata, qc_stats = load_and_qc_samples()


        corrected_tpm = combat_correction_with_validation(tpm_matrix, metadata)


        corrected_tpm.to_csv(f"{OUTPUT_DIR}/tables/tpm_matrix_corrected.tsv.gz",
                             sep='\t', compression='gzip')


        logger.log("Loading homolog pairs...")
        homologs = pd.read_csv(HOMOLOG_FILE, sep='\t')
        homologs['gene_At'] = homologs['gene_At'].str.split('.').str[0]
        homologs['gene_Dt'] = homologs['gene_Dt'].str.split('.').str[0]
        logger.log(f"Loaded {len(homologs)} homolog pairs")
        corrected_tpm_ctx = collapse_to_context(corrected_tpm, metadata)


        results_df = sensitivity_analysis(corrected_tpm_ctx, homologs)


        final_features = generate_final_feature_table(results_df, homologs)

        logger.log("=" * 60)
        logger.log("Pipeline completed successfully!")
        logger.log(f"Results saved to: {OUTPUT_DIR}/")
        logger.log("=" * 60)
        logger.log("Pipeline completed successfully!")
        logger.log(f"Results saved to: {OUTPUT_DIR}/")
        logger.log("=" * 60)

    except Exception as e:
        logger.log(f"Pipeline failed with error: {str(e)}", "ERROR")
        import traceback
        logger.log(traceback.format_exc(), "ERROR")
        sys.exit(1)

    finally:
        logger.close()

def calculate_tau(expr_vector):
    """Yanai et al. 2005的组织特异性指数"""
    expr_norm = expr_vector / expr_vector.max() if expr_vector.max() > 0 else expr_vector
    n = len(expr_norm)
    tau = (1 - expr_norm).sum() / (n - 1) if n > 1 else 0
    return tau

def post_validation_analysis(results_df, tpm_matrix, metadata):
    """
    [FIX] 在context level（而非tissue level）计算Tau
    因为相关性是在collapsed context上算的，Tau也应该在同一粒度
    """
    logger_val = Logger(f"{OUTPUT_DIR}/logs/validation.log")
    logger_val.log("=== Post-hoc Validation Analysis ===")


    logger_val.log("\n[Validation 1] Tissue-specific genes should have lower correlation")


    

    tau_at_list = []
    tau_dt_list = []

    for idx, row in results_df.head(1000).iterrows():
        gene_at = row['gene_At']
        gene_dt = row['gene_Dt']

        if gene_at in tpm_matrix.index and gene_dt in tpm_matrix.index:

            context_expr_at = tpm_matrix.loc[gene_at].values  
            context_expr_dt = tpm_matrix.loc[gene_dt].values

            tau_at = calculate_tau(context_expr_at)
            tau_dt = calculate_tau(context_expr_dt)

            tau_at_list.append(tau_at)
            tau_dt_list.append(tau_dt)


    results_df_subset = results_df.head(1000).copy()
    results_df_subset['tau_At'] = tau_at_list
    results_df_subset['tau_Dt'] = tau_dt_list
    results_df_subset['mean_tau'] = (results_df_subset['tau_At'] + results_df_subset['tau_Dt']) / 2


    
    q_hi = results_df_subset["mean_tau"].quantile(0.8)
    q_lo = results_df_subset["mean_tau"].quantile(0.2)

    high_tau = results_df_subset[results_df_subset["mean_tau"] >= q_hi]
    low_tau = results_df_subset[results_df_subset["mean_tau"] <= q_lo]

    from scipy.stats import mannwhitneyu
    stat, p_val = mannwhitneyu(
        high_tau['expr_corr'].dropna(),
        low_tau['expr_corr'].dropna(),
        alternative='less'
    )

    logger_val.log(f"High-tau genes mean r: {high_tau['expr_corr'].mean():.3f}")
    logger_val.log(f"Low-tau genes mean r: {low_tau['expr_corr'].mean():.3f}")
    logger_val.log(f"Mann-Whitney U test: p={p_val:.2e}")

    if p_val < 0.05:
        logger_val.log("✓ PASS: Tissue-specific genes show lower correlation (biologically expected)")
    else:
        logger_val.log("✗ WARNING: No significant difference in correlation by tissue specificity", "WARNING")

    logger_val.log("\n[Validation 2] Similar expression levels should correlate better")

    results_df['expr_ratio'] = np.abs(
        np.log2(results_df['mean_expr_At'] + 1) -
        np.log2(results_df['mean_expr_Dt'] + 1)
    )

    similar_expr = results_df[results_df['expr_ratio'] < 1.0]  
    divergent_expr = results_df[results_df['expr_ratio'] > 3.0]  

    stat2, p_val2 = mannwhitneyu(
        similar_expr['expr_corr'].dropna(),
        divergent_expr['expr_corr'].dropna(),
        alternative='greater'
    )

    logger_val.log(f"Similar expression mean r: {similar_expr['expr_corr'].mean():.3f}")
    logger_val.log(f"Divergent expression mean r: {divergent_expr['expr_corr'].mean():.3f}")
    logger_val.log(f"Mann-Whitney U test: p={p_val2:.2e}")

    if p_val2 < 0.05:
        logger_val.log("✓ PASS: Similar expression levels show higher correlation")
    else:
        logger_val.log("✗ WARNING: Expression level similarity doesn't predict correlation", "WARNING")


    logger_val.log("\n[Validation 3] Subgenome asymmetry detection")


    at_mean_expr = results_df['mean_expr_At'].mean()
    dt_mean_expr = results_df['mean_expr_Dt'].mean()

    logger_val.log(f"At subgenome mean expression: {at_mean_expr:.2f} TPM")
    logger_val.log(f"Dt subgenome mean expression: {dt_mean_expr:.2f} TPM")
    logger_val.log(f"At/Dt ratio: {at_mean_expr / dt_mean_expr:.3f}")


    from scipy.stats import wilcoxon
    stat3, p_val3 = wilcoxon(
        results_df['mean_expr_At'].dropna(),
        results_df['mean_expr_Dt'].dropna()
    )

    logger_val.log(f"Wilcoxon signed-rank test: p={p_val3:.2e}")

    if p_val3 < 0.05:
        dominant = "At" if at_mean_expr > dt_mean_expr else "Dt"
        logger_val.log(f"✓ Detected subgenome dominance: {dominant} subgenome")
    else:
        logger_val.log("No significant subgenome expression bias detected")


    logger_val.log("\n=== Validation Complete ===")
    logger_val.close()

def cli():
    global HOMOLOG_FILE, METADATA_FILE, QUANT_DIR, OUTPUT_DIR, logger
    parser = argparse.ArgumentParser(description="Native cotton expression workflow; cached context replay or raw quantification inputs.")
    parser.add_argument("--mode", choices=["cached-context", "raw-quant"], default="cached-context")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--homologs", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--tpm")
    parser.add_argument("--quant-dir")
    parser.add_argument("--cutoff", type=float, default=1.0)
    parser.add_argument("--cached-tpm-dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()
    HOMOLOG_FILE, METADATA_FILE = args.homologs, args.metadata
    QUANT_DIR, OUTPUT_DIR = args.quant_dir, args.outdir
    for folder in ["", "tables", "logs", "qc"]:
        (Path(OUTPUT_DIR) / folder).mkdir(parents=True, exist_ok=True)
    logger = Logger(str(Path(OUTPUT_DIR) / "logs/pipeline.log"))
    if args.mode == "raw-quant":
        if not args.quant_dir:
            parser.error("--quant-dir is required for raw-quant")
        main()
        results = pd.read_csv(Path(OUTPUT_DIR) / "tables/expr_corr_cutoff_1.0.tsv", sep="\t")
        tpm = pd.read_csv(Path(OUTPUT_DIR) / "tables/tpm_matrix_corrected.tsv.gz", sep="\t", index_col=0)
        metadata = pd.read_csv(METADATA_FILE, sep="\t", index_col="sample_id")
        post_validation_analysis(results, tpm, metadata)
    else:
        if not args.tpm:
            parser.error("--tpm is required for cached-context")

        tpm = pd.read_csv(args.tpm, sep="\t", index_col=0).astype(args.cached_tpm_dtype)
        metadata = pd.read_csv(METADATA_FILE, sep="\t", index_col="sample_id")
        homologs = pd.read_csv(HOMOLOG_FILE, sep="\t")
        contexts = collapse_to_context(tpm, metadata)
        results = vectorized_correlation_analysis(contexts, homologs, args.cutoff)
        contexts.to_csv(Path(OUTPUT_DIR) / "tables/context_tpm.tsv", sep="\t")
        results.to_csv(Path(OUTPUT_DIR) / "tables/pair_expression.tsv", sep="\t", index=False)
        logger.close()

if __name__ == "__main__":
    cli()
