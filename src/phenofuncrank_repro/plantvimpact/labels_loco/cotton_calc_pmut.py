#!/usr/bin/env python3
import sys
import gzip
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import multiprocessing as mp
from functools import partial
import logging
import re  
from typing import Optional
from collections import defaultdict

def _alt_counter():
    return defaultdict(int)

def _ctx_to_alt():
    return defaultdict(_alt_counter)

def _sg_to_ctx():
    return {"At": _ctx_to_alt(), "Dt": _ctx_to_alt()}
CONFIG = {
    "fasta_path": None,
    "vcf_path": None,


    "sites_4d_path": "4d_sites.bed",  

    "input_tsv": "trackA_training_labels.tsv",
    "output_tsv": "trackA_training_labels_with_pmut.tsv",
    "n_cores": 32,
    "chunk_size": 1000000,
    "smoothing": 1e-9,
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GLOBAL_GENOME = {}


def get_subgenome(chrom: str) -> str:
    """
    解析染色体名判断At/Dt亚基因组。
    未组装的Contig会被返回'Unknown'，使用默认smoothing值。
    """
    if chrom.startswith('Contig'):
        return 'Unknown'  

    m = re.search(r'([AD])\d+', chrom.upper())
    if m:
        return 'At' if m.group(1) == 'A' else 'Dt'


    if not chrom.startswith('Contig'):
        logger.warning(f"无法判断亚基因组: {chrom}")
    return 'Unknown'

def load_4d_sites_and_count_kmers(
    sites_4d_path: str,
    genome_dict: dict,
    exclude_chr: Optional[int] = None
) -> dict:
    logger.info(f"从4D位点文件加载并统计3-mer背景: {sites_4d_path}")


    df = pd.read_csv(
        sites_4d_path, sep='\t',
        usecols=['Chr', 'Start_0based', 'RefBase', 'SubGenome'],
        dtype={'Chr': 'category', 'Start_0based': 'int32',
               'RefBase': 'category', 'SubGenome': 'category'}
    )
    if exclude_chr is not None:

        chr_num = df['Chr'].astype(str).str.extract(r'(\d+)')[0].astype(float)
        df = df[chr_num != float(exclude_chr)]


    df = df[df['SubGenome'].isin(['At', 'Dt'])]
    df = df[df['Chr'].isin(genome_dict.keys())]
    df = df[(df['Start_0based'] >= 1)]  

    counts = {"At": Counter(), "Dt": Counter()}


    for chrom, grp in df.groupby('Chr', observed=True):
        seq = genome_dict[chrom]
        seq_len = len(seq)

        pos_arr = grp['Start_0based'].values  


        valid_mask = (pos_arr >= 1) & (pos_arr + 1 < seq_len)
        pos_arr = pos_arr[valid_mask]
        sg_arr  = grp['SubGenome'].values[valid_mask]
        ref_arr = grp['RefBase'].values[valid_mask]

        if len(pos_arr) == 0:
            continue


        
        seq_bytes = np.frombuffer(seq.encode('ascii'), dtype='S1')

        prev_bases = seq_bytes[pos_arr - 1]   
        curr_bases = seq_bytes[pos_arr]        
        next_bases = seq_bytes[pos_arr + 1]    


        N_byte = b'N'
        no_N_mask = (prev_bases != N_byte) & (curr_bases != N_byte) & (next_bases != N_byte)


        ref_bytes = np.array([r.encode('ascii') if isinstance(r, str)
                              else r for r in ref_arr], dtype='S1')
        consistent_mask = (curr_bases == ref_bytes)

        final_mask = no_N_mask & consistent_mask
        pos_arr   = pos_arr[final_mask]
        sg_arr    = sg_arr[final_mask]
        prev_bases = prev_bases[final_mask]
        curr_bases = curr_bases[final_mask]
        next_bases = next_bases[final_mask]

        if len(pos_arr) == 0:
            continue


        
        contexts = np.char.add(np.char.add(prev_bases, curr_bases), next_bases)
        contexts_str = contexts.astype(str)

        for sg_val in ('At', 'Dt'):
            sg_mask = (sg_arr == sg_val)
            if not np.any(sg_mask):
                continue
            ctx_sub = contexts_str[sg_mask]
            counts[sg_val].update(ctx_sub.tolist())

    n_total = sum(sum(c.values()) for c in counts.values())
    logger.info(f"4D位点3-mer统计完成: {n_total:,} 个位点纳入统计")
    for sg in ('At', 'Dt'):
        logger.info(f"  {sg}: {sum(counts[sg].values()):,} 个4D位点，{len(counts[sg])} 种3-mer类型")

    return counts


def count_vcf_mutations(
    vcf_path: str,
    genome_dict: dict,
    sites_4d_set: set,
    exclude_chr: Optional[int] = None   
) -> dict:
    """流式处理VCF,边读边过滤4D位点（支持排除指定染色体，防止LOCO泄露）"""
    logger.info("统计VCF中4D位点上的突变（流式版）...")

    mutations = {
        "At": defaultdict(lambda: defaultdict(int)),
        "Dt": defaultdict(lambda: defaultdict(int))
    }


    chrom_cache = {}  

    def get_chrom_info(chrom):
        if chrom not in chrom_cache:
            if chrom not in genome_dict:
                chrom_cache[chrom] = None
                return None
            sg = get_subgenome(chrom)
            if sg == 'Unknown':
                chrom_cache[chrom] = None
                return None
            seq = genome_dict[chrom]
            seq_bytes = np.frombuffer(seq.encode('ascii'), dtype='S1')
            chrom_cache[chrom] = (sg, seq_bytes, len(seq))
        return chrom_cache[chrom]

    open_func = gzip.open if vcf_path.endswith('.gz') else open
    total_vcf = 0
    valid_4d = 0

    with open_func(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue

            total_vcf += 1
            if total_vcf % 10000000 == 0:
                logger.info(f"  已处理 {total_vcf:,} 行VCF, 匹配到4D位点 {valid_4d:,} 个")

            parts = line.split('\t', 6)
            if len(parts) < 5:
                continue

            chrom = parts[0]


            

            if exclude_chr is not None:
                m = re.search(r'(\d+)', chrom)
                if m:
                    chr_num = int(m.group(1))
                    if chr_num == exclude_chr:
                        continue


            pos_1based = int(parts[1])
            ref = parts[3].strip().upper()
            alt = parts[4].strip().upper()


            if len(ref) != 1 or len(alt) != 1 or ',' in alt:
                continue
            if ref not in 'ATCG' or alt not in 'ATCG':
                continue

            pos_0based = pos_1based - 1

            if (chrom, pos_0based) not in sites_4d_set:
                continue
            valid_4d += 1


            info = get_chrom_info(chrom)
            if info is None:
                continue

            sg, seq_bytes, seq_len = info


            if pos_0based < 1 or pos_0based + 1 >= seq_len:
                continue


            prev_b = seq_bytes[pos_0based - 1]
            curr_b = seq_bytes[pos_0based]
            next_b = seq_bytes[pos_0based + 1]


            N_byte = b'N'
            ref_byte = ref.encode('ascii')
            if prev_b == N_byte or curr_b == N_byte or next_b == N_byte:
                continue
            if curr_b != ref_byte:
                continue


            context = (prev_b + curr_b + next_b).decode('ascii')
            mutations[sg][context][alt] += 1

    logger.info(f"VCF处理完成: 总行数={total_vcf:,}, 4D位点SNP={valid_4d:,}")
    if exclude_chr is not None:
        logger.info(f"已排除染色体编号: {exclude_chr}")

    return mutations
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import gzip
import re
from collections import defaultdict

def count_vcf_mutations_total_and_by_chrnum(
    vcf_path: str,
    genome_dict: dict,
    sites_4d_set: set
):
    """
    一次性扫VCF，统计4D位点SNP突变计数
    支持多ALT位点：ALT按逗号拆分后分别统计
    会跳过非单碱基ALT、'*'、以及非ATCG字符
    """
    total_mut = _sg_to_ctx()
    by_chrnum = defaultdict(_sg_to_ctx)
    chrom_cache = {}

    def get_chrom_info(chrom):
        if chrom not in chrom_cache:
            if chrom not in genome_dict:
                chrom_cache[chrom] = None
                return None
            sg = get_subgenome(chrom)
            if sg == 'Unknown':
                chrom_cache[chrom] = None
                return None
            seq = genome_dict[chrom]
            seq_bytes = np.frombuffer(seq.encode('ascii'), dtype='S1')
            chrom_cache[chrom] = (sg, seq_bytes, len(seq))
        return chrom_cache[chrom]

    open_func = gzip.open if vcf_path.endswith('.gz') else open

    with open_func(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.split('\t', 6)
            if len(parts) < 5:
                continue

            chrom = parts[0]
            chr_num = get_chr_num(chrom)
            if chr_num < 0:
                continue

            pos_1based = int(parts[1])
            ref = parts[3].strip().upper()
            alt_field = parts[4].strip().upper()


            if len(ref) != 1 or ref not in 'ATCG':
                continue

            pos_0based = pos_1based - 1
            if (chrom, pos_0based) not in sites_4d_set:
                continue

            info = get_chrom_info(chrom)
            if info is None:
                continue

            sg, seq_bytes, seq_len = info

            if pos_0based < 1 or pos_0based + 1 >= seq_len:
                continue

            prev_b = seq_bytes[pos_0based - 1]
            curr_b = seq_bytes[pos_0based]
            next_b = seq_bytes[pos_0based + 1]

            N_byte = b'N'
            ref_byte = ref.encode('ascii')
            if prev_b == N_byte or curr_b == N_byte or next_b == N_byte:
                continue
            if curr_b != ref_byte:
                continue

            context = (prev_b + curr_b + next_b).decode('ascii')


            alts = [a.strip().upper() for a in alt_field.split(',')]
            for alt in alts:

                if len(alt) != 1:
                    continue
                if alt not in 'ATCG':
                    continue
                if alt == ref:
                    continue

                total_mut[sg][context][alt] += 1
                by_chrnum[chr_num][sg][context][alt] += 1

    return total_mut, by_chrnum
def load_4d_counts_total_and_by_chrnum(
        sites_4d_path: str,
        genome_dict: dict
):
    """
    一次性统计4D位点3-mer背景：
    - total_counts[sg]: Counter(context->count)
    - by_chrnum[chr_num][sg]: Counter(context->count)
    """
    df = pd.read_csv(
        sites_4d_path, sep='\t',
        usecols=['Chr', 'Start_0based', 'RefBase', 'SubGenome'],
        dtype={'Chr': 'category', 'Start_0based': 'int32',
               'RefBase': 'category', 'SubGenome': 'category'}
    )

    df = df[df['SubGenome'].isin(['At', 'Dt'])]
    df = df[df['Chr'].isin(genome_dict.keys())]
    df = df[df['Start_0based'] >= 1]

    total_counts = {'At': Counter(), 'Dt': Counter()}
    by_chrnum = defaultdict(lambda: {'At': Counter(), 'Dt': Counter()})

    for chrom, grp in df.groupby('Chr', observed=True):
        chr_num = get_chr_num(str(chrom))
        if chr_num < 0:
            continue

        seq = genome_dict[str(chrom)]
        seq_len = len(seq)
        pos_arr = grp['Start_0based'].values

        valid_mask = (pos_arr >= 1) & (pos_arr + 1 < seq_len)
        pos_arr = pos_arr[valid_mask]
        sg_arr = grp['SubGenome'].values[valid_mask]
        ref_arr = grp['RefBase'].values[valid_mask]

        if len(pos_arr) == 0:
            continue

        seq_bytes = np.frombuffer(seq.encode('ascii'), dtype='S1')
        prev_bases = seq_bytes[pos_arr - 1]
        curr_bases = seq_bytes[pos_arr]
        next_bases = seq_bytes[pos_arr + 1]

        N_byte = b'N'
        no_N_mask = (prev_bases != N_byte) & (curr_bases != N_byte) & (next_bases != N_byte)

        ref_bytes = np.array([r.encode('ascii') if isinstance(r, str) else r for r in ref_arr], dtype='S1')
        consistent_mask = (curr_bases == ref_bytes)

        final_mask = no_N_mask & consistent_mask
        prev_bases = prev_bases[final_mask]
        curr_bases = curr_bases[final_mask]
        next_bases = next_bases[final_mask]
        sg_arr = sg_arr[final_mask]

        if len(sg_arr) == 0:
            continue

        contexts = np.char.add(np.char.add(prev_bases, curr_bases), next_bases).astype(str)

        for sg_val in ('At', 'Dt'):
            sg_mask = (sg_arr == sg_val)
            if not np.any(sg_mask):
                continue
            ctx_list = contexts[sg_mask].tolist()
            total_counts[sg_val].update(ctx_list)
            by_chrnum[chr_num][sg_val].update(ctx_list)


    by_chrnum = dict(by_chrnum)  
    for chr_num in by_chrnum:
        by_chrnum[chr_num] = dict(by_chrnum[chr_num])  

    return total_counts, by_chrnum
GLOBAL_RATE_SERIES: Optional[pd.Series] = None


def build_all_fold_rate_tables_fast(
        fasta_path: str,
        vcf_path: str,
        sites_4d_path: str,
        n_chr: int = 13
) -> tuple:  

    genome = {}
    with open(fasta_path, 'r') as f:
        chrom, seqs = "", []
        for line in f:
            if line.startswith('>'):
                if chrom:
                    genome[chrom] = "".join(seqs)
                chrom = line.strip()[1:].split()[0]
                seqs = []
            else:
                seqs.append(line.strip().upper())
        if chrom:
            genome[chrom] = "".join(seqs)


    sites_4d_set = set()
    with open(sites_4d_path, 'r') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            chrom_4d = parts[0]
            pos_0based = int(parts[1])
            sites_4d_set.add((chrom_4d, pos_0based))


    genome_total, genome_by_chrnum = load_4d_counts_total_and_by_chrnum(
        sites_4d_path, genome
    )


    mut_total, mut_by_chrnum = count_vcf_mutations_total_and_by_chrnum(
        vcf_path, genome, sites_4d_set
    )


    rate_tables = {}
    for chr_num in range(1, n_chr + 1):
        rate_tables[chr_num] = build_rate_table_excluding_chrnums(
            genome_total,
            genome_by_chrnum,
            mut_total,
            mut_by_chrnum,
            exclude_chrs=[chr_num]
        )

    return rate_tables, genome_total, genome_by_chrnum, mut_total, mut_by_chrnum


def build_fold_specific_statistics(
        fasta_path: str,
        vcf_path: str,
        sites_4d_path: str,
        exclude_chr: int
) -> tuple:

    genome = {}
    with open(fasta_path, 'r') as f:
        chrom, seqs = "", []
        for line in f:
            if line.startswith('>'):
                if chrom:
                    genome[chrom] = "".join(seqs)
                chrom = line.strip()[1:].split()[0]
                seqs = []
            else:
                seqs.append(line.strip().upper())
        if chrom:
            genome[chrom] = "".join(seqs)


    sites_4d_set = set()
    with open(sites_4d_path, 'r') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            chrom_4d = parts[0]
            pos_0based = int(parts[1])
            sites_4d_set.add((chrom_4d, pos_0based))


    genome_total_raw, genome_by_chrnum_raw = load_4d_counts_total_and_by_chrnum(
        sites_4d_path, genome
    )


    mut_total_raw, mut_by_chrnum_raw = count_vcf_mutations_total_and_by_chrnum(
        vcf_path, genome, sites_4d_set
    )


    genome_total = {'At': Counter(), 'Dt': Counter()}
    genome_by_chrnum = {}
    mut_total = {'At': defaultdict(lambda: defaultdict(int)),
                 'Dt': defaultdict(lambda: defaultdict(int))}
    mut_by_chrnum = {}

    for chr_num in range(1, 14):
        if chr_num == exclude_chr:
            continue


        if chr_num in genome_by_chrnum_raw:
            genome_by_chrnum[chr_num] = genome_by_chrnum_raw[chr_num]
            for sg in ('At', 'Dt'):
                genome_total[sg].update(genome_by_chrnum_raw[chr_num][sg])


        if chr_num in mut_by_chrnum_raw:
            mut_by_chrnum[chr_num] = mut_by_chrnum_raw[chr_num]
            for sg in ('At', 'Dt'):
                for context, alt_dict in mut_by_chrnum_raw[chr_num][sg].items():
                    for alt, count in alt_dict.items():
                        mut_total[sg][context][alt] += count

    logger.info(f"✅ Fold 专属统计构建完成(排除 chr{exclude_chr})")


    total_mut_at = sum(sum(alt_dict.values()) for alt_dict in mut_total['At'].values())
    total_mut_dt = sum(sum(alt_dict.values()) for alt_dict in mut_total['Dt'].values())

    logger.info(f"   4D 位点: At={sum(genome_total['At'].values()):,}, Dt={sum(genome_total['Dt'].values()):,}")
    logger.info(f"   突变数: At={total_mut_at:,}, Dt={total_mut_dt:,}")

    return genome_total, genome_by_chrnum, mut_total, mut_by_chrnum

def build_rate_table_excluding_chrnums(
    genome_counts_total: dict,
    genome_counts_by_chrnum: dict,
    mutation_counts_total: dict,
    mutation_counts_by_chrnum: dict,
    exclude_chrs
) -> dict:
    MU_BASE = 4.56e-9
    rate_table = {}
    bases = ['A', 'C', 'G', 'T']
    pseudo_count = 0.5

    import copy
    cur_gen = copy.deepcopy(genome_counts_total)
    cur_mut = copy.deepcopy(mutation_counts_total)

    exclude_chrs = set(int(x) for x in exclude_chrs)

    for chr_num in exclude_chrs:
        if chr_num in genome_counts_by_chrnum:
            for sg in ('At', 'Dt'):
                for ctx, count in genome_counts_by_chrnum.get(chr_num, {}).get(sg, {}).items():
                    cur_gen[sg][ctx] -= count

        if chr_num in mutation_counts_by_chrnum:
            for sg in ('At', 'Dt'):
                for context, alt_dict in mutation_counts_by_chrnum[chr_num][sg].items():
                    for alt, count in alt_dict.items():
                        cur_mut[sg][context][alt] -= count

    sum_gen = sum(sum(c.values()) for c in cur_gen.values())
    sum_mut = 0
    for sg in ('At', 'Dt'):
        for ctx_dict in cur_mut[sg].values():
            sum_mut += sum(ctx_dict.values())

    avg_obs_rate = sum_mut / sum_gen if sum_gen > 0 else 1e-10

    for sg in ('At', 'Dt'):
        for context, count in cur_gen[sg].items():
            if count <= 0:
                continue
            for alt in bases:
                if alt == context[1]:
                    continue
                mut_count = cur_mut[sg][context].get(alt, 0)
                obs_rate = (mut_count + pseudo_count) / (count + pseudo_count * 3)
                relative_rate = obs_rate / avg_obs_rate
                key = f"{sg}_{context}_{alt}"
                rate_table[key] = MU_BASE * relative_rate

    return rate_table
def get_chr_num(chrom: str) -> int:
    """
    从染色体名解析数字编号

    例如:
    Ghir_A01 -> 1
    Ghir_D12 -> 12
    A05 -> 5
    D13 -> 13

    无法解析返回 -1
    """
    m = re.search(r'(\d+)', chrom)
    if m:
        return int(m.group(1))
    return -1
def process_chunk(chunk_df, rate_table):
    local_genome = GLOBAL_GENOME
    smoothing_val = CONFIG.get('fallback_pmut', CONFIG['smoothing'])
    local_rate = GLOBAL_RATE_SERIES if GLOBAL_RATE_SERIES is not None else None

    n = len(chunk_df)
    pmut_arr = np.full(n, smoothing_val, dtype=np.float64)

    chunk_df = chunk_df.reset_index(drop=True)

    for chrom, grp in chunk_df.groupby('Chr', sort=False):
        if chrom not in local_genome:
            continue

        sg = get_subgenome(chrom)
        if sg == 'Unknown':
            continue

        seq = local_genome[chrom]
        seq_len = len(seq)
        seq_bytes = np.frombuffer(seq.encode('ascii'), dtype='S1')

        idx = grp.index.values
        pos_arr = grp['Pos'].values - 1


        ref_arr = grp['Ref'].str.upper().astype(str).values  
        alt_arr = grp['Alt'].str.upper().astype(str).values


        valid = (pos_arr >= 1) & (pos_arr + 1 < seq_len)
        idx = idx[valid]
        pos_arr = pos_arr[valid]
        ref_arr = ref_arr[valid]
        alt_arr = alt_arr[valid]

        if len(pos_arr) == 0:
            continue

        prev_b = seq_bytes[pos_arr - 1]
        curr_b = seq_bytes[pos_arr]
        next_b = seq_bytes[pos_arr + 1]


        ref_bytes = np.array([r.encode('ascii') for r in ref_arr], dtype='S1')
        N_byte = b'N'
        ok = (curr_b == ref_bytes) & (prev_b != N_byte) & (curr_b != N_byte) & (next_b != N_byte)

        idx = idx[ok]
        alt_arr = alt_arr[ok]
        prev_b = prev_b[ok]
        curr_b = curr_b[ok]
        next_b = next_b[ok]

        if len(idx) == 0:
            continue


        contexts = np.char.add(np.char.add(prev_b, curr_b), next_b).astype(str)


        contexts = np.char.add(np.char.add(prev_b, curr_b), next_b).astype(str)


        keys = np.array([f"{sg}_{ctx}_{alt}" for ctx, alt in zip(contexts, alt_arr)], dtype=object)


        if local_rate is not None:
            vals = local_rate.reindex(keys.tolist(), fill_value=smoothing_val).values
        else:
            vals = np.array([rate_table.get(k, smoothing_val) for k in keys], dtype=np.float64)
        pmut_arr[idx] = vals

    chunk_df['P_mut'] = pmut_arr
    return chunk_df


def main():
    logger.info("加载基因组到内存...")
    global GLOBAL_GENOME
    with open(CONFIG['fasta_path'], 'r') as f:
        chrom, seqs = "", []
        for line in f:
            if line.startswith('>'):
                if chrom:
                    GLOBAL_GENOME[chrom] = "".join(seqs)
                chrom = line.strip()[1:].split()[0]
                seqs = []
            else:
                seqs.append(line.strip().upper())
        if chrom:
            GLOBAL_GENOME[chrom] = "".join(seqs)
    logger.info(f"基因组加载完成: {len(GLOBAL_GENOME)} 条染色体")


    

    logger.info(f"加载4D位点集合: {CONFIG['sites_4d_path']}")

    sites_4d_set = set()

    with open(CONFIG['sites_4d_path'], 'r') as f:
        next(f)  
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 5:
                continue
            chrom_4d = parts[0]
            pos_0based = int(parts[1])
            sites_4d_set.add((chrom_4d, pos_0based))

    

    genome_counts_4d = load_4d_sites_and_count_kmers(
        CONFIG['sites_4d_path'],
        GLOBAL_GENOME
    )

    total_4d_at = sum(genome_counts_4d['At'].values())
    total_4d_dt = sum(genome_counts_4d['Dt'].values())
    logger.info(f"✅ 4D位点3-mer统计完成（用作P_mut分母）:")
    logger.info(f"  At: {total_4d_at:,} 个4D位点, {len(genome_counts_4d['At'])} 种3-mer")
    logger.info(f"  Dt: {total_4d_dt:,} 个4D位点, {len(genome_counts_4d['Dt'])} 种3-mer")


    

    mutation_counts = count_vcf_mutations(
        CONFIG['vcf_path'],
        GLOBAL_GENOME,
        sites_4d_set

    )

    total_mut_at = sum(sum(c.values()) for c in mutation_counts['At'].values())
    total_mut_dt = sum(sum(c.values()) for c in mutation_counts['Dt'].values())
    logger.info(f"✅ VCF突变统计完成（用作P_mut分子）:")
    logger.info(f"  At: {total_mut_at:,} 个观测突变")
    logger.info(f"  Dt: {total_mut_dt:,} 个观测突变")


    

    logger.info("构建P_mut查找表（分子分母均为4D位点数据）...")
    rate_table = build_rate_table(genome_counts_4d, mutation_counts)
    logger.info(f"✅ Rate Table构建完成: {len(rate_table)} 个条目")
    import json
    with open('rate_table.json', 'w') as f:
        json.dump(rate_table, f)
    logger.info("✅ rate_table.json 已保存，供 validate_pmut.py 使用")
    logger.info(f"   示例条目: {list(rate_table.items())[0]}")


    min_rate = min(rate_table.values())
    CONFIG['fallback_pmut'] = min_rate
    logger.info(f"✅ P_mut fallback值已设定（用于Contig/边界位点）: {min_rate:.3e}")
    logger.info(f"   （原smoothing={CONFIG['smoothing']:.0e} 已弃用，避免量级跳跃3个数量级）")


    

    logger.info("\n" + "=" * 60)
    logger.info("【诊断】P_mut 分布检查")
    logger.info("=" * 60)

    at_rates = [v for k, v in rate_table.items() if k.startswith('At_')]
    dt_rates = [v for k, v in rate_table.items() if k.startswith('Dt_')]

    logger.info(f"At 亚基因组 P_mut:")
    logger.info(f"  均值   = {np.mean(at_rates):.3e}")
    logger.info(f"  中位数 = {np.median(at_rates):.3e}")
    logger.info(f"  最小值 = {np.min(at_rates):.3e} (Jeffreys先验下理论最小值约为 {0.5 / (total_4d_at + 1.5):.3e})")
    logger.info(f"  最大值 = {np.max(at_rates):.3e}")

    logger.info(f"Dt 亚基因组 P_mut:")
    logger.info(f"  均值   = {np.mean(dt_rates):.3e}")
    logger.info(f"  中位数 = {np.median(dt_rates):.3e}")
    logger.info(f"  最小值 = {np.min(dt_rates):.3e}")
    logger.info(f"  最大值 = {np.max(dt_rates):.3e}")


    fallback = CONFIG['fallback_pmut']

    n_at_fallback = np.isclose(np.array(at_rates), fallback, rtol=0, atol=1e-15).sum()
    n_dt_fallback = np.isclose(np.array(dt_rates), fallback, rtol=0, atol=1e-15).sum()
    pct_at_fallback = n_at_fallback / len(at_rates) * 100
    pct_dt_fallback = n_dt_fallback / len(dt_rates) * 100

    logger.info(f"\n等于 fallback_pmut 的条目占比:")
    logger.info(f"  At: {pct_at_fallback:.1f}% ({n_at_fallback}/{len(at_rates)})")
    logger.info(f"  Dt: {pct_dt_fallback:.1f}% ({n_dt_fallback}/{len(dt_rates)})")
    logger.info(f"  ✓ 预期值: >50% (因为大部分理论突变未被观察到)")


    ratio_median = np.median(at_rates) / np.median(dt_rates)
    logger.info(f"\nAt/Dt 中位数比例: {ratio_median:.2f}")
    if 0.5 < ratio_median < 2.0:
        logger.info(f"  ✓ 正常范围 (0.5 ~ 2.0)")
    else:
        logger.warning(f"  ⚠️ 异常！可能存在亚基因组判断错误")

    logger.info("=" * 60 + "\n")


    

    global GLOBAL_RATE_SERIES
    GLOBAL_RATE_SERIES = pd.Series(rate_table, dtype=np.float64)
    logger.info(f"✅ Rate Series 就绪: {len(GLOBAL_RATE_SERIES)} 个条目")


    

    logger.info(f"开始处理输入文件: {CONFIG['input_tsv']}")
    logger.info(f"  输出路径: {CONFIG['output_tsv']}")
    logger.info(f"  并行度: {CONFIG['n_cores']} 核")
    logger.info(f"  Chunk大小: {CONFIG['chunk_size']:,} 行\n")

    reader = pd.read_csv(
        CONFIG['input_tsv'],
        sep='\t',
        chunksize=CONFIG['chunk_size']
    )
    first_chunk = True

    with mp.Pool(CONFIG['n_cores']) as pool:
        worker = partial(process_chunk, rate_table=rate_table)
        for result_df in pool.imap(worker, reader):
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            result_df.to_csv(
                CONFIG['output_tsv'],
                sep='\t',
                index=False,
                mode=mode,
                header=header
            )
            first_chunk = False
            sys.stdout.write(".")
            sys.stdout.flush()

    logger.info(f"\n✅ 处理完成！输出已保存至: {CONFIG['output_tsv']}")


    

    logger.info("\n" + "=" * 60)
    logger.info("【最终验证】输出文件 P_mut 分布")
    logger.info("=" * 60)


    sample_df = pd.read_csv(CONFIG['output_tsv'], sep='\t', nrows=1_000_000)

    logger.info(f"样本大小: {len(sample_df):,} 行")
    logger.info(f"P_mut 统计摘要:")
    logger.info(f"  均值   = {sample_df['P_mut'].mean():.3e}")
    logger.info(f"  中位数 = {sample_df['P_mut'].median():.3e}")
    logger.info(f"  最小值 = {sample_df['P_mut'].min():.3e}")
    logger.info(f"  最大值 = {sample_df['P_mut'].max():.3e}")

    fallback = CONFIG['fallback_pmut']

    n_equal_fallback = np.isclose(sample_df['P_mut'].values, fallback, rtol=0, atol=1e-15).sum()
    pct_fallback = n_equal_fallback / len(sample_df) * 100
    logger.info(f"\n等于 fallback_pmut 的位点: {pct_fallback:.2f}% ({n_equal_fallback:,}/{len(sample_df):,})")

    logger.info(f"  ✓ 预期值: <15% (Contig/边界位点使用fallback值，属正常现象)")


    for sg in ['A', 'D']:
        sg_mask = sample_df['Chr'].str.match(f'^{sg}', na=False)
        sg_df = sample_df[sg_mask]
        if len(sg_df) > 0:
            logger.info(f"\n{sg}t 亚基因组 ({len(sg_df):,} 个位点):")
            logger.info(f"  P_mut 中位数 = {sg_df['P_mut'].median():.3e}")

    logger.info("=" * 60)
    logger.info("✅ 全部流程完成！")
    logger.info("=" * 60)
if __name__ == "__main__":
    main()
