"""Frozen paper lookup and genotype QC; caller supplies the resource directory."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd

def log(msg: str) -> None:
    print(msg, flush=True)

class Panel:
    """Chrom:pos -> variant_index lookup + chunked genotype QC."""

    def __init__(self, base: Path, species: str):
        t0 = time.time()
        self.species = species
        sm = pd.read_csv(base / "cultivar.snp_map.tsv.gz", sep="\t", low_memory=False,
                         usecols=["variant_index", "Chr", "Pos", "Ref", "Alt"],
                         dtype={"Chr": "category", "Ref": "category", "Alt": "category"})
        cats = sm["Chr"].cat.categories

        self.cat_of, self.cat_of_stripped = self._make_cat_map(cats)
        code = np.array([self.cat_of[str(c)] for c in cats], dtype=np.int64)[sm["Chr"].cat.codes.to_numpy()]
        self.vi = sm["variant_index"].to_numpy(np.int64)
        self.ref_codes = sm["Ref"].cat.codes.to_numpy()
        self.alt_codes = sm["Alt"].cat.codes.to_numpy()
        self.ref_cats = np.asarray(sm["Ref"].cat.categories, dtype=object)
        self.alt_cats = np.asarray(sm["Alt"].cat.categories, dtype=object)
        pos = sm["Pos"].to_numpy(np.int64)
        del sm
        self.key = code * 10**9 + pos
        self.order = np.argsort(self.key, kind="stable")
        self.key_sorted = self.key[self.order]
        self.n_snps = len(self.vi)
        self.n_samples = sum(1 for _ in open(base / "matched_samples.vcf_order.txt"))
        import os
        self.geno = np.memmap(base / "cultivar_genotype_codes.variant_by_sample.uint8.memmap",
                              dtype="uint8", mode="r", shape=(self.n_snps, self.n_samples))
        self.norm = max(1, self.n_samples)
        log(f"  [panel] snps={self.n_snps} samples={self.n_samples} load={time.time()-t0:.1f}s")

    @staticmethod
    def _make_cat_map(cats):
        """(original-label -> code, stripped-label -> code)."""
        stripped = {}
        for c in cats:
            s = str(c)
            if s.lower().startswith("chr"):
                s2 = s[3:]
            elif s.lower().startswith("ca"):
                s2 = s[2:]
            else:
                s2 = s
            try:
                s2 = str(int(float(s2)))
            except ValueError:
                pass
            stripped[str(c)] = s2
        uniq = sorted(set(stripped.values()), key=lambda x: (len(x), x))
        idx = {v: i for i, v in enumerate(uniq)}
        return ({k: idx[v] for k, v in stripped.items()}, dict(idx))

    def indices(self, ids) -> tuple[np.ndarray, np.ndarray]:
        """Return (variant_index, found_mask) for panel_id strings."""
        row, found = self._lookup(ids)
        vi = np.full(len(row), -1, dtype=np.int64)
        vi[found] = self.vi[row[found]]
        return vi, found

    def alleles(self, ids) -> tuple[np.ndarray, np.ndarray]:
        """Return the panel's (Ref, Alt) strings for panel_id strings."""
        row, found = self._lookup(ids)
        r = np.full(len(row), "", dtype=object)
        a = np.full(len(row), "", dtype=object)
        idx = row[found]
        r[found] = self.ref_cats[self.ref_codes[idx]]
        a[found] = self.alt_cats[self.alt_codes[idx]]
        return r, a

    def _lookup(self, ids) -> tuple[np.ndarray, np.ndarray]:
        """Map panel_id strings to snp_map row positions."""
        s = pd.Series(ids).astype(str)
        if self.species == "rice":
            head = s.str.split("_", n=1).str[0]
            chrom = head.str.split(":", n=1).str[0]
            pos = head.str.split(":", n=1).str[1].astype(np.int64)
        elif self.species == "chickpea":
            sp = s.str.split("_", n=2)
            chrom = sp.str[0].str.replace("Ca", "", regex=False)
            pos = sp.str[1].astype(np.int64)
        else:
            sp = s.str.split(":", n=1)
            chrom = sp.str[0]
            pos = sp.str[1].astype(np.int64)
        code = np.array([self.cat_of_stripped.get(c, -1) for c in chrom], dtype=np.int64)
        key = code * 10**9 + pos.to_numpy(np.int64)

        lo = np.searchsorted(self.key_sorted, key, "left")
        hi = np.searchsorted(self.key_sorted, key, "right")
        found = (lo < hi) & (code >= 0)
        row = np.full(len(ids), -1, dtype=np.int64)
        row[found] = self.order[lo[found]]
        return row, found

    def qc(self, vi: np.ndarray, found: np.ndarray, chunk: int = 250_000):
        n = len(vi)
        called = np.zeros(n, dtype=np.int64)
        n0 = np.zeros(n, dtype=np.int64)
        n1 = np.zeros(n, dtype=np.int64)
        n2 = np.zeros(n, dtype=np.int64)
        good = np.flatnonzero(found)

        srt = good[np.argsort(vi[good], kind="stable")]
        for st in range(0, len(srt), chunk):
            idx = srt[st:st + chunk]
            codes = np.asarray(self.geno[vi[idx]]).astype(np.int64)
            n0[idx] = (codes == 1).sum(axis=1)
            n1[idx] = (codes == 2).sum(axis=1)
            n2[idx] = (codes == 3).sum(axis=1)
            called[idx] = n0[idx] + n1[idx] + n2[idx]
            del codes
        cr = called / self.norm
        mac = np.minimum(2 * n0 + n1, n1 + 2 * n2)
        denom = 2 * called
        af = np.where(denom > 0, (n1 + 2 * n2) / np.where(denom > 0, denom, 1), np.nan)
        maf = np.minimum(af, 1.0 - af)
        return cr, mac, maf
