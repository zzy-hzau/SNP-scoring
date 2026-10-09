"""Typed failures, retry policy and breeder-facing recovery guidance."""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, TypeVar


T = TypeVar("T")


class NoRetryError(Exception):
    pass


INPUT_ISSUE_DETAILS = {
    "missing_columns": "The table is missing one or more required chrom, pos, ref or alt columns.",
    "empty_input": "The file contains no supported variant rows.",
    "identical_alleles": "REF and ALT are identical; a variant must describe an actual allele change.",
    "non_biallelic_variant": "Only one-base, biallelic A/C/G/T variants are accepted.",
    "unsupported_chromosome": "The chromosome label is not valid for the selected species/reference.",
    "invalid_coordinate": "The position column contains a non-integer or otherwise invalid coordinate.",
    "ambiguous_snp_id": "Each snp_id must identify exactly one chromosome, position and allele pair.",
    "no_supported_vcf_records": "The VCF contains no supported biallelic SNP records.",
    "invalid_file_encoding": "The uploaded file cannot be parsed as the format indicated by its extension.",
    "input_invalid": "The genotype input does not satisfy the required variant schema.",
}


def classify_input_issue(exc: BaseException) -> str:
    text = str(exc).lower()
    if "missing required columns" in text or "missing required" in text:
        return "missing_columns"
    if "no variant rows" in text or "empty data" in text:
        return "empty_input"
    if "ref and alt must differ" in text:
        return "identical_alleles"
    if "vcf contains no supported" in text:
        return "no_supported_vcf_records"
    if "biallelic" in text or "single-nucleotide" in text:
        return "non_biallelic_variant"
    if "unsupported" in text and "chromosome" in text:
        return "unsupported_chromosome"
    if any(token in text for token in ("position", "pos column", "unable to parse string", "invalid literal for int")):
        return "invalid_coordinate"
    if "snp_id maps to multiple" in text or "snp_id cannot be empty" in text:
        return "ambiguous_snp_id"
    if any(token in text for token in ("no columns to parse", "excel file format", "utf-8", "decode")):
        return "invalid_file_encoding"
    return "input_invalid"


@dataclass(frozen=True)
class ErrorInfo:
    error_class: str
    retryable: bool
    user_message: str
    suggested_action: str


def classify_exception(exc: BaseException) -> ErrorInfo:
    if isinstance(exc, NoRetryError):
        return ErrorInfo("rate_limited", False, "外部服务限流。", "稍后再试或降低请求频率。")
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(token in text for token in ("timeout", "timed out", "503", "502", "429", "connection reset", "temporary", "temporarily")):
        return ErrorInfo("external_transient", True, "外部服务暂时不可用。", "系统会自动重试；若持续失败，可跳过该可选证据后继续。")
    if any(token in text for token in ("database is locked", "locked", "busy")):
        return ErrorInfo("database_contention", True, "证据库正忙。", "稍后重试；生产环境应使用外部数据库。")
    if any(token in text for token in ("vcf", "tsv", "csv", "format", "parse", "allele", "biallelic", "column", "variant", "chromosome", "position", "ref and alt", "snp_id")):
        return ErrorInfo("input_invalid", False, "输入文件或等位基因格式无法解析。", "检查参考版本以及 chrom、pos、ref、alt 列，或上传标准 VCF。")
    if any(token in text for token in ("model", "weight", "resource", "reference fasta", "not found", "no such file")):
        return ErrorInfo("resource_missing", False, "分析资源未安装或版本不匹配。", "联系管理员检查模型、参考基因组和资源清单。")
    if any(token in text for token in ("json", "schema", "validation")):
        return ErrorInfo("schema_invalid", False, "工具返回结果未通过结构校验。", "保留原始产物并联系管理员检查工具版本。")
    return ErrorInfo("tool_runtime", False, "分析工具未能完成。", "查看该工具的审计记录；可选工具失败时可使用降级报告。")


def error_payload(exc: BaseException, *, include_trace: bool = True) -> dict[str, Any]:
    info = classify_exception(exc)
    payload: dict[str, Any] = {
        "status": "error", "error_class": info.error_class, "retryable": info.retryable,
        "message": info.user_message, "user_message": info.user_message,
        "suggested_action": info.suggested_action,
    }
    if info.error_class == "input_invalid":
        issue = classify_input_issue(exc)
        payload["input_issue"] = issue
        payload["detail"] = INPUT_ISSUE_DETAILS[issue]
    if include_trace:
        payload["technical_message"] = str(exc)
        payload["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-12000:]
    return payload


def with_retry(action: Callable[[], T], *, attempts: int = 3, base_delay: float = 0.25, on_retry: Callable[[BaseException, int], None] | None = None) -> T:
    last: BaseException | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return action()
        except Exception as exc:
            last = exc
            info = classify_exception(exc)
            if not info.retryable or attempt >= attempts:
                raise
            if on_retry:
                on_retry(exc, attempt)
            time.sleep(min(4.0, base_delay * (2 ** (attempt - 1))))
    assert last is not None
    raise last
