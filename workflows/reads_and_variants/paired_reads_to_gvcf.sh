#!/usr/bin/env bash
# Portable entry for the recovered paired-read rice workflow.
set -euo pipefail

SAMPLE_ID=$1
R1=$2
R2=$3
REF_FA=$4
OUT_DIR=$5
THREADS=${THREADS:-8}
JAVA_HEAP=${JAVA_HEAP:-48g}
SORT_MEM=${SORT_MEM:-2G}

WORK_TMP="${OUT_DIR}/tmp/${SAMPLE_ID}"
mkdir -p "${WORK_TMP}" "${OUT_DIR}/fastp" "${OUT_DIR}/bam" "${OUT_DIR}/gvcf" "${OUT_DIR}/qc"
RG="@RG\tID:${SAMPLE_ID}\tSM:${SAMPLE_ID}\tPL:ILLUMINA\tLB:lib1\tPU:${SAMPLE_ID}"
CLEAN_R1="${OUT_DIR}/fastp/${SAMPLE_ID}_R1.clean.fastq.gz"
CLEAN_R2="${OUT_DIR}/fastp/${SAMPLE_ID}_R2.clean.fastq.gz"
NAME_BAM="${WORK_TMP}/${SAMPLE_ID}.name.bam"
FIX_BAM="${WORK_TMP}/${SAMPLE_ID}.fixmate.bam"
POS_BAM="${WORK_TMP}/${SAMPLE_ID}.pos.bam"
DEDUP_BAM="${OUT_DIR}/bam/${SAMPLE_ID}.dedup.bam"
GVCF="${OUT_DIR}/gvcf/${SAMPLE_ID}.g.vcf.gz"

fastp -i "${R1}" -I "${R2}" -o "${CLEAN_R1}" -O "${CLEAN_R2}" \
  -w "${THREADS}" -h "${OUT_DIR}/fastp/${SAMPLE_ID}.fastp.html" \
  -j "${OUT_DIR}/fastp/${SAMPLE_ID}.fastp.json"
bwa-mem2 mem -t "${THREADS}" -R "${RG}" "${REF_FA}" "${CLEAN_R1}" "${CLEAN_R2}" | \
  samtools sort -n -@ "${THREADS}" -m "${SORT_MEM}" -o "${NAME_BAM}" -
samtools fixmate -m -@ "${THREADS}" "${NAME_BAM}" "${FIX_BAM}"
samtools sort -@ "${THREADS}" -m "${SORT_MEM}" -o "${POS_BAM}" "${FIX_BAM}"
samtools markdup -@ "${THREADS}" -s "${POS_BAM}" "${DEDUP_BAM}"
samtools index -@ "${THREADS}" "${DEDUP_BAM}"
samtools flagstat "${DEDUP_BAM}" > "${OUT_DIR}/qc/${SAMPLE_ID}.flagstat.txt"
samtools stats "${DEDUP_BAM}" > "${OUT_DIR}/qc/${SAMPLE_ID}.stats.txt"
gatk --java-options "-Xmx${JAVA_HEAP} -Djava.io.tmpdir=${WORK_TMP}" HaplotypeCaller \
  -R "${REF_FA}" -I "${DEDUP_BAM}" -O "${GVCF}" -ERC GVCF \
  --native-pair-hmm-threads "${THREADS}" -ploidy 2
gatk IndexFeatureFile -I "${GVCF}" || true
rm -f "${NAME_BAM}" "${FIX_BAM}" "${POS_BAM}" "${CLEAN_R1}" "${CLEAN_R2}"
