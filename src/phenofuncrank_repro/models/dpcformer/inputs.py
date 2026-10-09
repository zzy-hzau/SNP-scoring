"""DPCformer model input encoding from supplied genotype resources."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

BASE_TO_VEC = {
    "A": np.array([1, 0, 0, 0], dtype=np.float32),
    "C": np.array([0, 1, 0, 0], dtype=np.float32),
    "G": np.array([0, 0, 1, 0], dtype=np.float32),
    "T": np.array([0, 0, 0, 1], dtype=np.float32),
}


def encode_8d(selected_codes, selected_map):
    """Encode missing/REF-REF/REF-ALT/ALT-ALT as the frozen paired nucleotide vectors."""
    codes = np.asarray(selected_codes, dtype=np.uint8).T
    ref = np.stack([
        BASE_TO_VEC.get(str(row.Ref).upper(), np.zeros(4, dtype=np.float32))
        for row in selected_map.itertuples(index=False)
    ])[None, :, :]
    alt = np.stack([
        BASE_TO_VEC.get(str(row.Alt).upper(), np.zeros(4, dtype=np.float32))
        for row in selected_map.itertuples(index=False)
    ])[None, :, :]
    output = np.zeros((*codes.shape, 8), dtype=np.float32)
    is_ref, is_het, is_alt = [(codes == v)[:, :, None] for v in (1, 2, 3)]
    output[:, :, :4] = np.where(is_ref | is_het, ref, 0.0) + np.where(is_alt, alt, 0.0)
    output[:, :, 4:] = np.where(is_ref, ref, 0.0) + np.where(is_het | is_alt, alt, 0.0)
    return output


def read_genotype_memmap(inputs_dir):
    root = Path(inputs_dir)
    metadata = json.loads((root / "metadata.json").read_text())
    shape = tuple(metadata.get("shape") or (metadata["n_snps"], metadata["n_samples"]))

    
    path = root / "cultivar_genotype_codes.variant_by_sample.uint8.memmap"
    relative = metadata.get("genotype_memmap", "")
    if not path.exists() and relative and not Path(relative).is_absolute():
        path = root / relative
    if path.stat().st_size != int(np.prod(shape)):
        raise ValueError("Genotype byte size disagrees with the declared uint8 matrix")
    return np.memmap(path, mode="r", dtype=np.uint8, shape=shape), metadata


def build_tensor(genotype, selected_map, sample_columns):
    """Read only selected rows and supplied sample columns, in physical SNP order."""
    groups = []
    for number, block in selected_map.groupby("selected_group", sort=True):
        block = block.sort_values("selected_rank_physical")
        codes = np.asarray(
            genotype[block.variant_index.to_numpy(dtype=np.int64), :][:, sample_columns],
            dtype=np.uint8,
        )
        groups.append(encode_8d(codes, block))
    return torch.from_numpy(np.stack(groups, axis=1))
