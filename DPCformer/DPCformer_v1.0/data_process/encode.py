import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_regression


def genotype_to_dataframe(genotype_series):
    """Convert genotype strings to a DataFrame of individual alleles.
    
    Supports two formats:
    1. Space-separated: "C C C C T A A ..." (rice, tomato, cotton)
    2. Concatenated: "GGCCTT..." (maize, wheat)
    
    Returns DataFrame where each column = one allele character.
    """
    first = str(genotype_series.iloc[0])
    if ' ' in first:
        genotype_list = genotype_series.str.split(' ')
    else:
        genotype_list = genotype_series.apply(lambda x: list(str(x).strip()))
    df = pd.DataFrame(genotype_list.tolist())
    # Ensure even number of columns (pairs of alleles)
    if df.shape[1] % 2 != 0:
        df = df.iloc[:, :-1]
    return df


def detect_separator(file_path):
    """Auto-detect CSV separator (tab or comma)."""
    with open(file_path, 'r') as f:
        first_line = f.readline()
    if '\t' in first_line:
        return '\t'
    elif ',' in first_line:
        return ','
    return '\t'


def base_one_hot_encoding_1v(seq_all_df):
    trans_code = {
        "AA": 0, "AT": 1, "TA": 1, "AC": 2, "CA": 2, "AG": 3, "GA": 3,
        "TT": 4, "TC": 5, "CT": 5, "TG": 6, "GT": 6, "CC": 7,
        "CG": 8, "GC": 8, "GG": 9,
        "00": -1, "A0": -1, "0A": -1, "T0": -1, "0T": -1,
        "C0": -1, "0C": -1, "G0": -1, "0G": -1
    }
    seq_all_np = seq_all_df.map(str).to_numpy()
    num_rows, num_cols = seq_all_np.shape
    output_cols = num_cols // 2
    code_arr = np.empty((num_rows, output_cols), dtype=int)
    for i in range(0, num_cols - 1, 2):
        joint_seq = seq_all_np[:, i] + seq_all_np[:, i + 1]
        code_arr[:, i // 2] = np.vectorize(trans_code.get)(joint_seq)
    return pd.DataFrame(code_arr)


def base_one_hot_encoding_8v_dif(seq_all_df):
    base_code = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
                 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    trans_code = {}
    bases = ['A', 'C', 'G', 'T']
    for base1 in bases:
        for base2 in bases:
            trans_code[base1 + base2] = base_code[base1] + base_code[base2]
    seq_all_np = seq_all_df.values
    n_cols = seq_all_np.shape[1]
    if n_cols % 2 != 0:
        n_cols -= 1
    one_hot_arr = np.zeros((seq_all_np.shape[0], n_cols // 2, 8), dtype=np.int32)
    for row in range(seq_all_np.shape[0]):
        for base in range(0, n_cols, 2):
            b1 = str(seq_all_np[row, base])
            b2 = str(seq_all_np[row, base + 1])
            joint_seq = b1 + b2
            code = trans_code.get(joint_seq, [0] * 8)
            one_hot_arr[row, base // 2] = code
    return pd.DataFrame(one_hot_arr.reshape(one_hot_arr.shape[0], -1))


class MICSelector(BaseEstimator, TransformerMixin):
    """Select top-k SNPs per chromosome/group via mutual information."""
    def __init__(self, k=10000):
        self.k = k
        self.mic_scores_ = None
        self.top_k_indices_ = None

    def fit(self, X, y):
        self.mic_scores_ = mutual_info_regression(X, y, discrete_features=True, n_neighbors=7)
        self.top_k_indices_ = np.argsort(self.mic_scores_)[-self.k:]
        return self

    def transform(self, X):
        return X[:, self.top_k_indices_]
