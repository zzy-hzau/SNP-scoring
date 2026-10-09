"""Original literature-provider dependency closure; unrelated sequence/annotation APIs excluded."""
from __future__ import annotations

import hashlib

import json

import os

import re

import threading

import time

import urllib.parse

import urllib.request

from datetime import datetime, timezone

from functools import lru_cache

from pathlib import Path

from typing import Any

import requests

from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Callable, TypeVar

from phenofuncrank_repro.interpretation.errors import with_retry

from phenofuncrank_repro.interpretation.logging_utils import get_logger

from phenofuncrank_repro.interpretation.settings import settings

T = TypeVar("T")

logger = get_logger(__name__)

REQUEST_TIMEOUT = 10  

DEFAULT_MAX_WORKERS = int(os.environ.get("RICE_PH_AGENT_CONCURRENCY", "8"))

CACHE_TTL_LITERATURE = 7 * 24 * 3600   

EUROPEPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

OPENALEX_URL = "https://api.openalex.org/works"

_LIT_QUERY_VERSION = "v7-direct-transport"

_LOCUS_ID_RE = re.compile(
    r"^(os\d{2}g\d+|grmzm\d*g\d+|solyc\d+g\d+|loc\d+|gh_[adt]\d+g\d+|ca\d+)",
    re.I,
)

_API_FAILURE_LOG: dict[str, int] = {}

_API_FAILURE_LOCK = threading.Lock()

_API_CALL_LOG: dict[str, int] = {}

_API_CALL_LOCK = threading.Lock()

def _record_api_call(source: str) -> None:
    with _API_CALL_LOCK:
        _API_CALL_LOG[source] = _API_CALL_LOG.get(source, 0) + 1

def _record_api_failure(source: str, message: str = "") -> None:
    with _API_FAILURE_LOCK:
        _API_FAILURE_LOG[source] = _API_FAILURE_LOG.get(source, 0) + 1
    logger.warning("[api-failure] %s: %s", source, message[:200])

def _api_failure_snapshot() -> dict[str, int]:
    with _API_FAILURE_LOCK:
        return dict(_API_FAILURE_LOG)

def _api_failure_delta(before: dict[str, int]) -> dict[str, int]:
    after = _api_failure_snapshot()
    return {
        k: after.get(k, 0) - before.get(k, 0)
        for k in set(after) | set(before)
        if after.get(k, 0) - before.get(k, 0) > 0
    }

def _proxy() -> dict[str, str] | None:
    for env_var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        val = os.environ.get(env_var, "")
        if val:
            return {"http": val, "https": val}

    global _CLASH_OK
    if _CLASH_OK is None:
        import socket
        try:
            with socket.create_connection(("127.0.0.1", 7890), timeout=1):
                _CLASH_OK = True
        except OSError:
            _CLASH_OK = False
    if _CLASH_OK:
        return {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
    return None

_CLASH_OK: bool | None = None

def _cache_dir() -> Path:
    p = settings.root / "_api_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _cache_get(cache_key: str, ttl: int) -> dict[str, Any] | list[dict] | None:
    path = _cache_dir() / f"{cache_key}.json"
    if not path.is_file():
        return None
    age = time.time() - path.stat().st_mtime
    if age > ttl:
        path.unlink(missing_ok=True)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

def _cache_set(cache_key: str, data: Any) -> None:
    path = _cache_dir() / f"{cache_key}.json"
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
    except OSError:
        logger.warning("Failed to write cache entry: %s", path)

def _concurrent_run(
    funcs: list[Callable[[], T]],
    *,
    max_workers: int | None = None,
) -> list[T]:
    if not funcs:
        return []
    workers = min(len(funcs), max_workers or DEFAULT_MAX_WORKERS)
    workers = max(1, workers)
    results: list[T] = [None] * len(funcs)  
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ccz-io") as ex:
        futures = {
            ex.submit(_safe_call, fn): idx
            for idx, fn in enumerate(funcs)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
    return results

def _safe_call(fn: Callable[[], T]) -> T:
    """Wrapper that turns any exception into ``[]``/``{}``/None depending on type.

    For API helpers that should fail-soft, we use a default fallback of
    an empty list.  Functions returning dicts/None can wrap themselves.
    """
    try:
        return fn()
    except Exception as exc:
        _record_api_failure(getattr(fn, "__name__", "unknown_api"), str(exc))
        logger.warning("concurrent task failed: %s", exc)
        return []  

def _requests_get(url, **kwargs):
    if kwargs.get("proxies") is None:
        kwargs["proxies"] = {"http": "", "https": "", "all": ""}
    return requests.get(url, **kwargs)

def _source_from_url(url: str) -> str:
    """Map an endpoint URL to the source label used in failure reporting."""
    low = url.lower()
    if "eutils" in low or "ncbi" in low:
        return "pubmed"
    if "europepmc" in low:
        return "europepmc"
    if "openalex" in low:
        return "openalex"
    if "semanticscholar" in low:
        return "semantic_scholar"
    if "interpro" in low or "ebi.ac.uk" in low:
        return "ebi"
    if "plantregmap" in low:
        return "plantregmap"
    if "jaspar" in low:
        return "jaspar"
    return "http"

def _http_get_json(
    url: str, *, timeout: float = REQUEST_TIMEOUT, headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """GET a JSON endpoint — try direct first, then proxy fallback.

    Many EBI-hosted services (EuropePMC, InterPro, EBI Proteins) fail through
    the Clash proxy because the TLS handshake times out after CONNECT, while
    direct connections succeed.  OpenAlex needs the proxy.  This two-phase
    strategy lets both classes of service work without per-call-site tuning.
    """
    proxies = _proxy()
    transient_status = {429, 500, 502, 503, 504}

    def _request(proxies_override: dict[str, str] | None) -> dict[str, Any]:
        last_exc: BaseException | None = None
        for attempt in range(3):
            _record_api_call(_source_from_url(url))
            try:
                resp = _requests_get(url, timeout=timeout, proxies=proxies_override, headers=headers)
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(min(2.0, 0.5 * (2 ** attempt)))
                    continue
                raise
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After") or 1.0)
                time.sleep(min(max(wait, 1.0), 5.0))
                continue
            if resp.status_code in transient_status and attempt < 2:
                time.sleep(min(2.0, 0.5 * (2 ** attempt)))
                continue
            if resp.status_code in {204, 404}:
                return {}
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"transient HTTP failure for {url[:120]}: {last_exc}")


    try:
        return _request(None)
    except Exception:
        pass


    if proxies is not None:
        try:
            return _request(proxies)
        except Exception as exc:
            _record_api_failure(_source_from_url(url), str(exc))
            logger.warning("HTTP GET failed for %s: %s", url[:120], exc)
    else:
        _record_api_failure(_source_from_url(url), "no proxy available")
        logger.warning("HTTP GET failed for %s (no proxy available)", url[:120])
    return {}

def _http_get_text(
    url: str,
    *,
    timeout: float = REQUEST_TIMEOUT,
    **kwargs: Any,
) -> requests.Response:
    """GET an endpoint with two-phase strategy — direct first, then proxy fallback.

    Like :func:`_http_get_json` but returns the raw ``requests.Response``
    instead of calling ``.json()``.  Needed for endpoints that return TSV,
    plain text, or non-JSON responses (UniProt search, Ensembl xrefs, etc.).
    """
    kwargs.pop("proxies", None)
    proxies = _proxy()
    transient_status = {429, 500, 502, 503, 504}

    try:
        last_exc: BaseException | None = None
        for attempt in range(3):
            _record_api_call(_source_from_url(url))
            try:
                resp = _requests_get(url, timeout=timeout, proxies=None, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(min(2.0, 0.5 * (2 ** attempt)))
                    continue
                raise
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After") or 1.0)
                time.sleep(min(max(wait, 1.0), 5.0))
                continue
            if resp.status_code in transient_status and attempt < 2:
                time.sleep(min(2.0, 0.5 * (2 ** attempt)))
                continue
            return resp
        raise RuntimeError(f"transient HTTP failure for {url[:120]}: {last_exc}")
    except Exception:
        pass


    if proxies is not None:
        last_exc = None
        for attempt in range(3):
            _record_api_call(_source_from_url(url))
            try:
                resp = _requests_get(url, timeout=timeout, proxies=proxies, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(min(2.0, 0.5 * (2 ** attempt)))
                    continue
                break
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After") or 1.0)
                time.sleep(min(max(wait, 1.0), 5.0))
                continue
            if resp.status_code in transient_status and attempt < 2:
                time.sleep(min(2.0, 0.5 * (2 ** attempt)))
                continue
            return resp
        _record_api_failure(_source_from_url(url), str(last_exc) if last_exc else "transient 5xx")
        raise RuntimeError(f"transient HTTP failure via proxy for {url[:120]}: {last_exc}")


    raise RuntimeError(f"No proxy available and direct connection failed for {url[:120]}")

def openalex_search(
    query: str,
    filter_species: str | None = None,
    max_results: int = 10,
    *,
    sort_by: str = "cited_by_count:desc",
) -> list[dict[str, Any]]:
    """Search OpenAlex for scholarly works.

    Uses OpenAlex REST API:
      GET /works?search={query}&per_page={max_results}

    Returns dicts with: title, authors, year, doi, cited_by_count, abstract_snippet.

    When *filter_species* is given, client-side filters results whose title+abstract
    contain the species name.
    """
    if not query:
        return []


    

    
    auth_mode = "keyed" if os.environ.get("OPENALEX_API_KEY") else "anonymous"
    cache_suffix = (
        f"openalex_{_LIT_QUERY_VERSION}_{auth_mode}_"
        f"{hashlib.sha256(query.encode()).hexdigest()[:16]}_{max_results}_"
        f"{sort_by.replace(':', '_')}"
    )
    cached = _cache_get(cache_suffix, CACHE_TTL_LITERATURE)
    if cached is not None:
        results = cached if isinstance(cached, list) else []
        if filter_species:
            return _filter_by_species(results, filter_species)
        return results

    results: list[dict[str, Any]] = []
    params: dict[str, Any] = {
        "search": query,
        "per-page": max(10, max_results),
        "sort": sort_by,
        "mailto": os.environ.get("RICE_PH_AGENT_OPENALEX_MAILTO", "rice-ph-agent@example.com"),
    }
    if os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]

    try:
        url = OPENALEX_URL + "?" + urllib.parse.urlencode(params)
        data = _http_get_json(url)
        works = data.get("results", [])
    except Exception as exc:
        _record_api_failure("openalex", str(exc))
        logger.warning("OpenAlex search failed for query '%s': %s", query[:80], exc)
        return []

    for work in works:
        results.append(_openalex_work_to_article(work))

    _cache_set(cache_suffix, results)

    if filter_species:
        return _filter_by_species(results, filter_species)
    return results

def _openalex_work_to_article(work: dict[str, Any]) -> dict[str, Any]:
    """Normalize an OpenAlex work record into the shared article schema."""
    title = work.get("title", "") or ""
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    abstract = ""
    abstract_inverted = work.get("abstract_inverted_index")
    if isinstance(abstract_inverted, dict) and abstract_inverted:
        try:
            word_positions: list[tuple[int, str]] = []
            for word, positions in abstract_inverted.items():
                if isinstance(positions, list):
                    for pos in positions:
                        word_positions.append((pos, str(word)))
            word_positions.sort(key=lambda x: x[0])
            abstract = " ".join(w for _, w in word_positions)
        except Exception:
            abstract = ""
    authors_list = work.get("authorships", [])
    authors = ", ".join(
        a.get("author", {}).get("display_name", "")
        for a in authors_list[:5]
    )
    relevance = work.get("relevance_score")
    return {
        "title": title,
        "authors": authors,
        "year": work.get("publication_year", ""),
        "doi": doi,
        "cited_by_count": work.get("cited_by_count", 0),
        "abstract_snippet": abstract[:500] if abstract else "",
        "source": "OpenAlex",
        "source_id": work.get("id", ""),
        "relevance_score": float(relevance) if relevance is not None else 0.0,
    }

def openalex_work_by_doi(doi: str) -> dict[str, Any] | None:
    """Fetch one OpenAlex work by DOI, normalized to the shared article schema."""
    doi = (doi or "").strip().lower().replace("https://doi.org/", "")
    if not doi:
        return None
    try:
        mailto = os.environ.get("RICE_PH_AGENT_OPENALEX_MAILTO", "rice-ph-agent@example.com")
        url = f"{OPENALEX_URL}/https://doi.org/{urllib.parse.quote(doi)}?mailto={mailto}"
        work = _http_get_json(url)
        return _openalex_work_to_article(work) if isinstance(work, dict) and work else None
    except Exception as exc:
        _record_api_failure("openalex", str(exc))
        logger.warning("OpenAlex work-by-DOI failed for %s: %s", doi, exc)
        return None

def openalex_cited_by(doi: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Return works citing *doi* (OpenAlex), normalized to the article schema."""
    doi = (doi or "").strip().lower().replace("https://doi.org/", "")
    if not doi:
        return []
    cache_suffix = f"openalex_cited_{hashlib.sha256(doi.encode()).hexdigest()[:16]}_{max_results}"
    cached = _cache_get(cache_suffix, CACHE_TTL_LITERATURE)
    if cached is not None:
        return cached if isinstance(cached, list) else []
    try:
        work = _http_get_json(f"{OPENALEX_URL}/https://doi.org/{urllib.parse.quote(doi)}")
        work_id = work.get("id", "")
        if not work_id:
            return []
        params = {
            "filter": f"cites:{work_id}",
            "per-page": max(5, max_results),
            "sort": "cited_by_count:desc",
            "mailto": os.environ.get("RICE_PH_AGENT_OPENALEX_MAILTO", "rice-ph-agent@example.com"),
        }
        data = _http_get_json(OPENALEX_URL + "?" + urllib.parse.urlencode(params))
        results = [_openalex_work_to_article(w) for w in data.get("results", [])]
    except Exception as exc:
        logger.warning("OpenAlex cited-by failed for %s: %s", doi, exc)
        results = []
    _cache_set(cache_suffix, results)
    return results

def openalex_references(doi: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Return works referenced by *doi* (backward citation traversal).

    This mirrors PaperQA2's citation-traversal idea in the backward
    direction: starting from a highly relevant paper, follow its reference
    list to surface the classic primary sources that keyword search often
    misses.  Forward traversal is ``openalex_cited_by``.
    """
    doi = (doi or "").strip().lower().replace("https://doi.org/", "")
    if not doi:
        return []
    cache_suffix = f"openalex_refs_{hashlib.sha256(doi.encode()).hexdigest()[:16]}_{max_results}"
    cached = _cache_get(cache_suffix, CACHE_TTL_LITERATURE)
    if cached is not None:
        return cached if isinstance(cached, list) else []
    try:
        work = _http_get_json(f"{OPENALEX_URL}/https://doi.org/{urllib.parse.quote(doi)}")
        referenced_ids = [str(i) for i in (work.get("referenced_works") or [])][:max_results]
        if not referenced_ids:
            return []
        or_ids = "|".join(referenced_ids)
        params = {
            "filter": f"ids.openalex:{or_ids}",
            "per-page": max(5, max_results),
            "mailto": os.environ.get("RICE_PH_AGENT_OPENALEX_MAILTO", "rice-ph-agent@example.com"),
        }
        data = _http_get_json(OPENALEX_URL + "?" + urllib.parse.urlencode(params))
        results = [_openalex_work_to_article(w) for w in data.get("results", [])]
        results.sort(key=lambda a: float(a.get("cited_by_count") or 0), reverse=True)
        results = results[:max_results]
    except Exception as exc:
        logger.warning("OpenAlex references failed for %s: %s", doi, exc)
        results = []
    _cache_set(cache_suffix, results)
    return results

def semantic_scholar_citations(
    paper_id: str,
    direction: str = "citations",
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Semantic Scholar citation graph traversal (SOP §4.2.6).

    ``direction``: ``citations`` (forward citers) or ``references`` (backward
    refs).  Requires SEMANTIC_SCHOLAR_API_KEY or
    RICE_PH_AGENT_ENABLE_SEMANTIC_SCHOLAR=1; otherwise returns [] (fail-soft).
    """
    enabled = (
        bool(os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))
        or os.environ.get("RICE_PH_AGENT_ENABLE_SEMANTIC_SCHOLAR", "0").lower()
        in {"1", "true", "yes", "on"}
    )
    if not enabled or direction not in {"citations", "references"}:
        return []
    paper_id = str(paper_id or "").strip()
    if not paper_id:
        return []
    if not paper_id.startswith(("DOI:", "arXiv:", "ACL:", "CorpusId:", "PMID:")):
        paper_id = f"DOI:{paper_id}"
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/{direction}"
    params = {
        "fields": "title,year,authors,externalIds,citationCount,venue,url",
        "limit": max_results,
    }
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    try:
        data = _http_get_json(
            url + "?" + urllib.parse.urlencode(params),
            headers={"x-api-key": api_key} if api_key else None,
        )
        rows = data.get("data", [])
    except Exception as exc:
        _record_api_failure("semantic_scholar", str(exc))
        logger.warning("Semantic Scholar %s failed for %s: %s", direction, paper_id, exc)
        return []
    articles: list[dict[str, Any]] = []
    for row in rows:
        ext = row.get("externalIds") or {}
        articles.append({
            "title": row.get("title", "").strip(),
            "authors": ", ".join(a.get("name", "") for a in (row.get("authors") or [])[:5]),
            "year": str(row.get("year", "")),
            "doi": (ext.get("DOI") or "").replace("https://doi.org/", ""),
            "cited_by_count": row.get("citationCount"),
            "abstract_snippet": "",
            "source": "SemanticScholarTraversal",
            "source_id": row.get("paperId", ""),
            "relevance_score": 0.0,
        })
    return articles

def epmc_references(pmid: str, max_results: int = 8) -> list[dict[str, Any]]:
    """Backward citation traversal via EuropePMC references (SOP §4.2.6)."""
    pmid = re.sub(r"\D", "", str(pmid or ""))
    if not pmid:
        return []
    cache_suffix = f"epmc_refs_{pmid}_{max_results}"
    cached = _cache_get(cache_suffix, CACHE_TTL_LITERATURE)
    if cached is not None:
        return cached if isinstance(cached, list) else []
    try:
        url = (
            f"{EUROPEPMC_URL.replace('/search', '')}/MED/{pmid}/references"
            f"?format=json&pageSize={max(10, max_results)}"
        )
        data = _http_get_json(url)
        refs = ((data or {}).get("referenceList") or {}).get("reference", []) or []
        results: list[dict[str, Any]] = []
        for r in refs[:max_results]:
            ref_pmid = re.sub(r"\D", "", str(r.get("id") or ""))
            if not ref_pmid:
                continue
            results.append({
                "title": str(r.get("title") or "").strip(),
                "pmid": ref_pmid,
                "doi": str(r.get("doi") or ""),
                "year": str(r.get("pubYear") or ""),
                "journal": str(r.get("journalAbbreviation") or ""),
                "authors": "",
                "abstract_snippet": "",
                "source": "EPMCReference",
                "source_id": f"PMID:{ref_pmid}",
                "relevance_score": 0.0,
            })
    except Exception as exc:
        _record_api_failure("europepmc", str(exc))
        logger.warning("EPMC references failed for %s: %s", pmid, exc)
        results = []
    _cache_set(cache_suffix, results)
    return results

def epmc_citations(pmid: str, max_results: int = 8) -> list[dict[str, Any]]:
    """Forward citation traversal via EuropePMC citations."""
    pmid = re.sub(r"\D", "", str(pmid or ""))
    if not pmid:
        return []
    cache_suffix = f"epmc_cites_{pmid}_{max_results}"
    cached = _cache_get(cache_suffix, CACHE_TTL_LITERATURE)
    if cached is not None:
        return cached if isinstance(cached, list) else []
    try:
        url = (
            f"{EUROPEPMC_URL.replace('/search', '')}/MED/{pmid}/citations"
            f"?format=json&pageSize={max(10, max_results)}"
        )
        data = _http_get_json(url)
        cites = ((data or {}).get("citationList") or {}).get("citation", []) or []
        results: list[dict[str, Any]] = []
        for r in cites[:max_results]:
            cite_pmid = re.sub(r"\D", "", str(r.get("id") or ""))
            if not cite_pmid:
                continue
            results.append({
                "title": str(r.get("title") or "").strip(),
                "pmid": cite_pmid,
                "doi": str(r.get("doi") or ""),
                "year": str(r.get("pubYear") or ""),
                "journal": str(r.get("journalAbbreviation") or ""),
                "authors": "",
                "abstract_snippet": "",
                "source": "EPMCCitation",
                "source_id": f"PMID:{cite_pmid}",
                "relevance_score": 0.0,
            })
    except Exception as exc:
        _record_api_failure("europepmc", str(exc))
        logger.warning("EPMC citations failed for %s: %s", pmid, exc)
        results = []
    _cache_set(cache_suffix, results)
    return results

def _europepmc_raw(query: str, page_size: int = 10) -> list[dict[str, Any]]:
    """Raw EuropePMC search returning normalized dicts (same schema as OpenAlex)."""
    cache_suffix = f"epmc_raw_{hashlib.sha256((query or '').encode()).hexdigest()[:16]}_{page_size}"
    cached = _cache_get(cache_suffix, CACHE_TTL_LITERATURE)
    if cached is not None:
        return cached if isinstance(cached, list) else []
    params = {
        "query": query,
        "resultType": "lite",
        "pageSize": page_size,
        "format": "json",

    }
    success = False
    try:
        url = EUROPEPMC_URL + "?" + urllib.parse.urlencode(params)
        data = _http_get_json(url)
        rows = data.get("resultList", {}).get("result", [])
        success = True
    except Exception as exc:
        _record_api_failure("europepmc", str(exc))
        rows = []

    articles: list[dict[str, Any]] = []
    for row in rows:
        articles.append({
            "title": row.get("title", "").strip(),
            "pmid": str(row.get("pmid") or ""),
            "pmcid": str(row.get("pmcid") or ""),
            "authors": (row.get("authorString", "") or "")[:200],
            "year": row.get("pubYear", ""),
            "doi": row.get("doi", ""),
            "cited_by_count": row.get("citedByCount", 0),
            "abstract_snippet": (row.get("abstractText", "") or "")[:1500],
            "abstract_source": "europepmc_lite" if row.get("abstractText") else "",
            "source": "EuropePMC",
            "source_id": f"PMID:{row.get('pmid', '')}" if row.get("pmid") else row.get("id", ""),
        })
    if success:
        _cache_set(cache_suffix, articles)
    return articles

def europepmc_core_abstract(pmid: str) -> str:
    """Full abstractText from a EuropePMC core record (enrichment path)."""
    pmid = (pmid or "").strip()
    if not pmid:
        return ""
    cache_suffix = f"epmc_core_abs_{pmid}"
    cached = _cache_get(cache_suffix, CACHE_TTL_LITERATURE)
    if cached is not None:
        return str(cached)
    try:
        url = EUROPEPMC_URL + "?" + urllib.parse.urlencode({
            "query": f"EXT_ID:{pmid}",
            "resultType": "core",
            "format": "json",
        })
        data = _http_get_json(url)
        rows = (data.get("resultList") or {}).get("result") or []
        abstract = str((rows[0] or {}).get("abstractText") or "") if rows else ""
    except Exception as exc:
        _record_api_failure("europepmc", str(exc))
        logger.warning("EuropePMC core abstract failed for %s: %s", pmid, exc)
        abstract = ""
    _cache_set(cache_suffix, abstract)
    return abstract

def _pubmed_base_url() -> str:
    """NCBI EUtils is unreachable over HTTPS in some GFW/proxy setups;
    fall back to the plain-HTTP endpoint when a proxy is configured."""
    if os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"):
        return "http://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    return "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

_PRODUCTION_LITERATURE_PROVIDERS = frozenset({"europepmc", "openalex", "semantic_scholar"})

def _production_literature_providers(values: Any) -> tuple[str, ...]:
    """Normalize the provider allowlist and exclude PubMed unconditionally."""
    if isinstance(values, str):
        values = values.split(",")
    return tuple(dict.fromkeys(
        str(value).strip().lower()
        for value in (values or ())
        if str(value).strip().lower() in _PRODUCTION_LITERATURE_PROVIDERS
    ))

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

def _pubmed_raw(query: str, page_size: int = 10) -> list[dict[str, Any]]:
    """Search PubMed via EUtils and return normalized article dicts."""
    api_key = os.environ.get("NCBI_API_KEY", "")
    base_urls = [_pubmed_base_url()]
    alternate = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        if base_urls[0].startswith("http://")
        else "http://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    )
    base_urls.append(alternate)
    last_error = "empty/error response"
    summary: dict[str, Any] = {}
    ids: list[str] = []
    for base in base_urls:
        search_params = {
            "db": "pubmed", "retmode": "json", "term": query,
            "retmax": page_size,
        }
        if api_key:
            search_params["api_key"] = api_key
        try:
            data = _http_get_json(
                f"{base}/esearch.fcgi?" + urllib.parse.urlencode(search_params),
                timeout=5,
            )
            ids = (data.get("esearchresult") or {}).get("idlist", [])
            if not ids:
                last_error = f"empty esearch from {base}"
                continue
            summary_params = {
                "db": "pubmed", "retmode": "json", "id": ",".join(ids),
            }
            if api_key:
                summary_params["api_key"] = api_key
            summary = _http_get_json(
                f"{base}/esummary.fcgi?" + urllib.parse.urlencode(summary_params),
                timeout=5,
            ).get("result", {})
            if not summary:
                last_error = f"empty esummary from {base}"
                continue
            break
        except Exception as exc:
            last_error = str(exc)
    else:
        _record_api_failure("pubmed", last_error)
        logger.warning("PubMed search failed for query '%s': %s", query[:80], last_error)
        return []

    articles: list[dict[str, Any]] = []
    for pmid in ids:
        row = summary.get(str(pmid), {})
        if not row:
            continue
        article_ids = {item.get("idtype"): item.get("value") for item in row.get("articleids", [])}
        articles.append({
            "title": row.get("title", "").strip(),
            "authors": ", ".join(a.get("name", "") for a in row.get("authors", [])[:5]),
            "year": str(row.get("pubdate", ""))[:4],
            "doi": article_ids.get("doi", ""),
            "cited_by_count": None,
            "abstract_snippet": "",
            "source": "PubMed",
            "source_id": f"PMID:{pmid}",
            "relevance_score": 0.0,
        })
    return articles

def _semantic_scholar_raw(query: str, page_size: int = 10) -> list[dict[str, Any]]:
    """Search Semantic Scholar and return normalized article dicts."""
    enabled = (
        bool(os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))
        or os.environ.get("RICE_PH_AGENT_ENABLE_SEMANTIC_SCHOLAR", "0").lower()
        in {"1", "true", "yes", "on"}
    )
    if not enabled:
        return []
    try:
        params = {
            "query": query,
            "limit": page_size,
            "fields": "title,year,authors,abstract,externalIds,citationCount,venue,url",
        }
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        data = _http_get_json(
            SEMANTIC_SCHOLAR_URL + "?" + urllib.parse.urlencode(params),
            headers={"x-api-key": api_key} if api_key else None,
        )
        rows = data.get("data", [])
    except Exception as exc:
        _record_api_failure("semantic_scholar", str(exc))
        logger.warning("Semantic Scholar search failed for query '%s': %s", query[:80], exc)
        return []

    articles: list[dict[str, Any]] = []
    for row in rows:
        ext = row.get("externalIds") or {}
        articles.append({
            "title": row.get("title", "").strip(),
            "authors": ", ".join(a.get("name", "") for a in (row.get("authors") or [])[:5]),
            "year": str(row.get("year", "")),
            "doi": (ext.get("DOI") or "").replace("https://doi.org/", ""),
            "cited_by_count": row.get("citationCount"),
            "abstract_snippet": (row.get("abstract") or "")[:500],
            "source": "SemanticScholar",
            "source_id": row.get("paperId", ""),
            "relevance_score": 0.0,
        })
    return articles

def _filter_by_species(results: list[dict], species: str) -> list[dict]:
    """Client-side species filter on title+abstract."""
    if not species:
        return results
    species_lower = species.lower()
    filtered: list[dict] = []
    for r in results:
        text = (r.get("title", "") + " " + r.get("abstract_snippet", "")).lower()
        if species_lower in text:
            filtered.append(r)
    return filtered

def _deduplicate_by_doi(results: list[dict]) -> list[dict]:
    """Deduplicate results by DOI (lowercased), keeping the most informative entry."""
    seen: dict[str, dict] = {}
    for r in results:
        doi = (r.get("doi") or "").strip().lower()
        if not doi:

            title = _title_key(r.get("title", ""))
            doi = f"title:{hashlib.md5(title.encode()).hexdigest()[:12]}" if title else ""
        if not doi:
            continue
        if doi not in seen:
            seen[doi] = r
        else:

            existing = seen[doi]
            for field in (
                "abstract_snippet", "cited_by_count", "year", "authors",
                "pmid", "pmcid",
            ):
                if not existing.get(field) and r.get(field):
                    existing[field] = r[field]

            existing_sources = existing.get("sources", [existing.get("source", "")])
            if r.get("source") and r["source"] not in existing_sources:
                existing.setdefault("sources", existing_sources)
                existing["sources"].append(r["source"])
    return list(seen.values())

def _title_key(title: str) -> str:
    """Normalize a title for fuzzy dedup (lowercase, alnum only)."""
    return "".join(ch for ch in str(title or "").lower() if ch.isalnum())

def _source_record_key(article: dict[str, Any]) -> str:
    """Stable source-discovery identity, preferring DOI and PMID."""
    doi = str(article.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi.replace('https://doi.org/', '')}"
    pmid = str(article.get("pmid") or "").strip()
    if pmid:
        return f"pmid:{pmid}"
    title = _title_key(str(article.get("title") or ""))
    return f"title:{title}" if title else ""

def _stable_source_union(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate without allowing a later branch to reorder the first one."""
    seen: dict[str, dict[str, Any]] = {}
    for record in records:
        key = _source_record_key(record)
        if not key:
            continue
        if key not in seen:
            seen[key] = dict(record)
            seen[key].setdefault("sources", [])
            if record.get("source") and record["source"] not in seen[key]["sources"]:
                seen[key]["sources"].append(record["source"])
            continue
        existing = seen[key]
        for field in (
            "abstract_snippet", "cited_by_count", "year", "authors", "pmid",
            "pmcid", "doi", "journal",
        ):
            if not existing.get(field) and record.get(field):
                existing[field] = record[field]
        if record.get("source") and record["source"] not in existing.setdefault("sources", []):
            existing["sources"].append(record["source"])
        if record.get("discovery_provider") and record["discovery_provider"] not in existing.setdefault("discovery_providers", []):
            existing["discovery_providers"].append(record["discovery_provider"])
    return list(seen.values())

def source_discovery_union_search(
    *,
    canonical_terms: list[str],
    alias_terms: list[str] | None = None,
    gene_name: str = "",
    species: str = "",
    trait: str = "",
    variant_description: str = "",
    identifiers: list[str] | None = None,
    providers: tuple[str, ...] = ("europepmc", "openalex"),
    provider_top_k: int = 10,
    use_alias_branch: bool = False,
    document_normalization: bool = False,
) -> dict[str, Any]:
    from phenofuncrank_repro.interpretation.document_normalization import normalize_document_metadata

    provider_top_k = max(1, int(provider_top_k))
    providers = _production_literature_providers(providers)
    canonical = list(dict.fromkeys(str(t).strip() for t in canonical_terms if str(t).strip()))
    aliases = list(dict.fromkeys(str(t).strip() for t in (alias_terms or []) if str(t).strip()))
    if not use_alias_branch:
        aliases = []
    failures_before = _api_failure_snapshot()
    call_counts: dict[str, int] = {p: 0 for p in providers}

    def _provider_call(provider: str, query: str) -> list[dict[str, Any]]:
        call_counts[provider] = call_counts.get(provider, 0) + 1
        if provider == "europepmc":
            return _europepmc_raw(query, page_size=provider_top_k)
        if provider == "pubmed":
            return _pubmed_raw(query, page_size=provider_top_k)
        if provider == "openalex":
            return openalex_search(
                query, filter_species=None, max_results=provider_top_k,
                sort_by="relevance_score:desc",
            )
        logger.warning("Unknown source-discovery provider: %s", provider)
        return []

    def _run_branch(branch: str, terms: list[str]) -> list[dict[str, Any]]:
        jobs = [(term, provider) for term in terms for provider in providers]
        if not jobs:
            return []
        outputs = _concurrent_run([
            lambda term=term, provider=provider: _provider_call(provider, term)
            for term, provider in jobs
        ])
        ordered: list[dict[str, Any]] = []
        for (term, provider), part in zip(jobs, outputs):
            if not isinstance(part, list):
                continue
            for rank, article in enumerate(part[:provider_top_k], 1):
                item = dict(article)
                item["discovery_branch"] = branch
                item["discovery_provider"] = provider
                item["provider_rank"] = rank
                item["discovery_query"] = term
                if document_normalization:
                    item.update(normalize_document_metadata(
                    item, gene=gene_name, species=species, trait=trait,
                        variant_description=variant_description,
                    ))
                ordered.append(item)
        deduped = _stable_source_union(ordered)
        if document_normalization:
            for item in deduped:
                item.setdefault("normalized_metadata", "")
        return deduped

    canonical_pool = _run_branch("canonical", canonical)
    alias_pool = _run_branch("alias", aliases)


    

    
    identifiers = list(identifiers or [])
    canonical_ranked = _score_articles(
        canonical_pool, species, trait, identifiers or ([gene_name] if gene_name else []),
    )
    alias_ranked = _score_articles(
        alias_pool, species, trait, identifiers or ([gene_name] if gene_name else []),
    )
    canonical_keys = {_source_record_key(item) for item in canonical_ranked}
    alias_only = [item for item in alias_ranked if _source_record_key(item) not in canonical_keys]
    union = canonical_ranked + alias_only
    for rank, item in enumerate(union, 1):
        item["source_discovery_rank"] = rank

    failures = _api_failure_delta(failures_before)
    return {
        "_all_deduped": union,
        "_canonical_pool": canonical_ranked,
        "_alias_pool": alias_only,
        "_source_errors": failures,
        "_provider_calls": call_counts,
        "_query_count": len(canonical) + len(aliases),
        "_provider_top_k": provider_top_k,
        "_providers": list(providers),
        "_baseline_canonical_count": len(canonical_ranked),
        "_baseline_preserved_count": len(canonical_ranked),
    }

def _trait_terms(trait: str) -> list[str]:
    """Expand a trait id into search terms (id tokens + common synonyms)."""
    synonyms = {
        "Plant_height": ["plant height", "plant stature", "height"],
        "PH": ["plant height", "height"],
        "DTT": ["days to tassel", "tassel", "flowering time"],
        "EW": ["ear weight", "ear"],
        "FL": ["fiber length", "fibre length"],
        "FS": ["fiber strength", "fibre strength"],
        "FM": ["micronaire", "fiber fineness"],
        "FE": ["fiber elongation", "fibre elongation"],
        "Hundred_seed_weight": ["hundred seed weight", "seed weight", "100-seed weight"],
        "Yield_per_plant": ["yield per plant", "grain yield"],
        "Plant_width": ["plant width", "canopy width"],
    }
    terms = [t for t in synonyms.get(trait, [])]
    for token in re.split(r"[^A-Za-z0-9]+", str(trait).lower()):
        if len(token) >= 3:
            terms.append(token)
    return list(dict.fromkeys(terms))

def _score_articles(
    articles: list[dict],
    species: str,
    trait: str,
    identifiers: list[str],
) -> list[dict]:
    """Score articles by species/trait/gene-term hits + citations + source relevance."""
    import math
    species_l = (species or "").lower()
    trait_terms = _trait_terms(trait or "")
    id_terms = [str(i).lower() for i in (identifiers or []) if i]
    scored: list[dict] = []
    for art in articles:
        text = (
            str(art.get("title", "")) + " "
            + str(art.get("abstract_snippet", "")) + " "
            + str(art.get("normalized_metadata", ""))
        ).lower()
        score = 0.0
        if species_l and species_l in text:
            score += 2.0
        if any(term in text for term in trait_terms):
            score += 3.0
        if any(term in text for term in id_terms):
            score += 3.0
        rel = art.get("relevance_score")
        if isinstance(rel, (int, float)):
            score += min(5.0, float(rel) * 5.0)
        cited = art.get("cited_by_count")
        if isinstance(cited, (int, float)) and cited:
            score += min(5.0, math.log1p(float(cited)))
        art["relevance_score"] = round(score, 3)

        

        

        id_hit = any(
            re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text)
            for term in id_terms
            if len(term) >= 2
        )
        species_trait_hit = bool(
            (species_l and species_l in text)
            or any(term in text for term in trait_terms)
        )
        art["weak_relevance"] = not (id_hit and species_trait_hit)
        scored.append(art)
    scored.sort(key=lambda a: a.get("relevance_score", 0.0), reverse=True)
    return scored

def multi_query_literature_search(
    gene_name: str,
    species: str,
    trait: str,
    snp_id: str | None = None,
    aliases: list[str] | None = None,
    extra_terms: list[str] | None = None,
    *,
    citation_traversal: bool = True,
    source_discovery_only: bool = False,
    providers: tuple[str, ...] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Multi-source literature search with expanded query keys.

    Uses up to two identifiers (gene name + gene ID / alias) × trait/mutation/
    pathway queries, plus direct SNP / CRISPR / breeding queries, across the
    configured providers (all fail-soft).  Top
    articles are expanded through OpenAlex citation traversal (forward citers
    and, when ``citation_traversal`` is enabled, backward references) when a
    gene identifier is available, then relevance-scored.

    Returns dict[str, list[dict]] keyed by query label, each with relevance-scored
    articles, plus ``_source_errors`` (per-source failure counts for this call).
    Results are cached per (gene_name, aliases, extra_terms, species, trait, snp_id)
    for 7 days, so a critic-driven gap search (different extra_terms) bypasses the
    previous cache entry.
    """
    def _clean_ident(value: str) -> str:
        text = str(value or "").strip()
        low = text.lower()
        for prefix in ("gene:", "rna:", "transcript:"):
            if low.startswith(prefix):
                text = text[len(prefix):]
                break

        
        text = re.sub(r"\.\d+(\.\d+)?$", "", text)
        return text

    identifiers: list[str] = []
    for ident in [gene_name] + list(aliases or []):
        ident = _clean_ident(str(ident or ""))
        if ident and ident.lower() not in {i.lower() for i in identifiers}:
            identifiers.append(ident)
    alias_key = hashlib.md5(
        "|".join(sorted(i.lower() for i in identifiers)).encode()
    ).hexdigest()[:8]
    extra_terms = [
        str(t).strip() for t in (extra_terms or [])
        if str(t).strip() and str(t).strip().lower() != "nan"
    ]
    extra_key = hashlib.md5("|".join(sorted(t.lower() for t in extra_terms)).encode()).hexdigest()[:8]
    cache_suffix = (
        f"multi_lit_{gene_name}_{species}_{trait}_{snp_id or 'nosnp'}_{alias_key}"
        f"_{extra_key or 'base'}_{_LIT_QUERY_VERSION}"
        f"_{'source_only' if source_discovery_only else 'full'}"
    )
    cached = _cache_get(cache_suffix, CACHE_TTL_LITERATURE)
    if cached is not None:
        cached = dict(cached) if isinstance(cached, dict) else {}
        cached.pop("_source_errors", None)  
        return cached
    configured_providers = _production_literature_providers(
        providers or settings.source_discovery_providers
    )
    failures_before = _api_failure_snapshot()

    queries: dict[str, str] = {}
    if source_discovery_only:

        

        
        for i, term in enumerate(extra_terms[:8]):
            queries[f"source_term_{i}"] = term
    else:
        variant_id = snp_id or "variant"

        

        primary = next(
            (i for i in identifiers if not _LOCUS_ID_RE.match(i)),
            (identifiers[0] if identifiers else f"{species} {trait}"),
        )
        query_idents = list(dict.fromkeys(identifiers))[:4]
        for ident in query_idents:
            suffix = f"{species} {trait}".strip()
            tag = ident[:16]
            queries[f"gene_trait_{tag}"] = f"{ident} {suffix}".strip()
            queries[f"gene_mutation_{tag}"] = f"{ident} mutation {species}".strip()
            queries[f"gene_plain_{tag}"] = f"{ident} {species}".strip()
            if _LOCUS_ID_RE.match(ident):

                
                queries[f"gene_gwas_{tag}"] = f"{ident} GWAS {species}".strip()
                queries[f"gene_qtl_{tag}"] = f"{ident} QTL {species}".strip()
        queries["snp_association"] = f"{variant_id} {primary} {species}"
        queries["crispr_validation"] = f"{primary} CRISPR {species}"
        queries["breeding_application"] = f"{primary} breeding {trait} {species}".strip()
        for i, term in enumerate(extra_terms[:8]):
            queries[f"gap_term_{i}"] = term


    

    

    def _one_query(query_str: str) -> list[dict[str, Any]]:
        if source_discovery_only:

            

            
            parts = (_europepmc_raw(query_str, page_size=10),)
        else:
            provider_calls: dict[str, Callable[[], list[dict[str, Any]]]] = {
                "europepmc": lambda: _europepmc_raw(query_str, page_size=10),
                "openalex": lambda: openalex_search(query_str, filter_species=None, max_results=10),
                "pubmed": lambda: _pubmed_raw(query_str, page_size=10),
                "semantic_scholar": lambda: _semantic_scholar_raw(query_str, page_size=10),
            }
            jobs = [provider_calls[p] for p in configured_providers if p in provider_calls]
            outputs = _concurrent_run(jobs) if jobs else []
            parts = tuple(outputs)
        combined: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, list):
                combined.extend(part)
        return _deduplicate_by_doi(combined)

    pairs = list(queries.items())
    results_per_query = _concurrent_run(
        [lambda q=q[1]: _one_query(q) for q in pairs]
    )

    out: dict[str, list[dict[str, Any]]] = {}
    all_articles: list[dict[str, Any]] = []
    for (label, _), articles in zip(pairs, results_per_query):
        if not isinstance(articles, list):
            articles = []
        out[label] = articles
        all_articles.extend(articles)


    scored = _score_articles(_deduplicate_by_doi(all_articles), species, trait, identifiers)
    if identifiers and scored and not source_discovery_only:
        expand_dois = [
            a.get("doi") for a in scored[:3]
            if a.get("doi") and not a.get("weak_relevance")
        ][:2]
        if expand_dois:
            def _expand(doi: str) -> list[dict]:
                forward = openalex_cited_by(doi, max_results=5)
                if not citation_traversal:
                    return forward
                backward = openalex_references(doi, max_results=5)
                return forward + backward
            expanded = _concurrent_run([
                lambda d=d: _expand(d) for d in expand_dois
            ])
            extra: list[dict] = []
            for part in expanded:
                if isinstance(part, list):
                    extra.extend(part)
            if extra:
                scored = _score_articles(
                    _deduplicate_by_doi(scored + extra), species, trait, identifiers,
                )
    out["_all_deduped"] = scored[:25]
    out["_source_errors"] = _api_failure_delta(failures_before)

    cacheable = {k: v for k, v in out.items() if k != "_source_errors"}
    if cacheable.get("_all_deduped"):
        _cache_set(cache_suffix, cacheable)
    return out


CACHE_TTL_ANNOTATION = 30 * 24 * 3600

def _is_protein(seq: str) -> bool:
    """Return True if *seq* looks like a protein (mostly amino-acid codes)."""
    seq_clean = (seq or "").strip().upper()
    if not seq_clean:
        return False
    aa_chars = set("ACDEFGHIKLMNPQRSTVWY*XBZUO")
    letters = [c for c in seq_clean if c.isalpha()]
    if len(letters) < 10:
        return False
    return sum(1 for c in letters if c in aa_chars) / len(letters) >= 0.90

_LOCAL_BLAST_DB_DIR = os.environ.get("RICE_PH_AGENT_BLAST_DB_DIR", "inputs/blast")

_LOCAL_BLAST_DBS = {
    "swissprot": f"{_LOCAL_BLAST_DB_DIR}/swissprot",
    "nr": f"{_LOCAL_BLAST_DB_DIR}/swissprot",  
    "swissprot_protein": f"{_LOCAL_BLAST_DB_DIR}/swissprot",
    "refseq_protein": f"{_LOCAL_BLAST_DB_DIR}/refseq_protein",
}

def _local_blast(
    sequence: str,
    database: str = "swissprot",
    is_protein: bool = True,
    species: str | None = None,
    max_hits: int = 5,
) -> list[dict[str, Any]]:
    import subprocess

    db_path = _LOCAL_BLAST_DBS.get(database) or _LOCAL_BLAST_DBS.get("swissprot", "")
    if not db_path or not (
        os.path.exists(db_path + ".pin") or os.path.exists(db_path + ".pal")
    ):
        logger.warning(
            "Local BLAST: database %s not found at %s (pin=%s pal=%s)",
            database, db_path,
            os.path.exists(db_path + ".pin"), os.path.exists(db_path + ".pal"),
        )
        return []

    program = "blastp" if is_protein else "blastn"
    outfmt = "6 qseqid sseqid pident evalue bitscore stitle"
    cmd = [
        program,
        "-db", db_path,
        "-max_target_seqs", str(max(5, max_hits * 3)),
        "-outfmt", outfmt,
        "-num_threads", "4",
        "-evalue", "1e-3",
    ]
    try:

        clean_env = {
            k: v for k, v in os.environ.items()
            if not (k.upper().endswith("_PROXY") or k.lower().endswith("_proxy"))
        }
        result = subprocess.run(
            cmd,
            input=sequence.strip() + "\n",
            capture_output=True, text=True,
            timeout=120,
            env=clean_env,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Local BLAST timed out (%s db=%s)", program, database)
        return []
    except Exception as exc:
        logger.warning("Local BLAST subprocess failed: %s", exc)
        return []

    if result.returncode != 0:
        stderr = (result.stderr or "")[:300]
        logger.warning("Local BLAST exit=%s stderr=%s", result.returncode, stderr)
        return []

    hits: list[dict[str, Any]] = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) < 6:
            continue
        desc = parts[5] if len(parts) > 5 else ""

        org = ""
        m_org = re.search(r'\[([^\]]+)\]$', desc)
        if m_org:
            org = m_org.group(1)
        hit = {
            "protein_id": parts[1],
            "identity_pct": float(parts[2]),
            "e_value": float(parts[3]),
            "score_bits": float(parts[4]),
            "description": desc,
            "organism": org,
            "gene_name": desc.split("[")[0].strip().split()[0] if desc else parts[1],
            "source": "BLAST(local)",
        }
        hits.append(hit)


    if species and hits:
        sp = species.lower()
        hits = [h for h in hits if sp in (h.get("organism", "")).lower()]

    return hits[:max_hits]

def blast_search(
    sequence: str,
    database: str = "nr",
    species: str | None = None,
    max_hits: int = 5,
) -> list[dict[str, Any]]:
    """Run BLASTP / BLASTN.

    Primary: local BLAST+ against Swiss-Prot / RefSeq (fast, no network).
    Fallback: EBI mirror → NCBI URL API (only when local DB is missing).

    - ``sequence`` may be protein or nucleotide; the alphabet detector
      chooses blastp vs blastn.
    - ``database`` accepts NCBI names (``nr``, ``swissprot``) or EBI
      names (``uniprotkb_swissprot``, …).
    - ``species`` (if given) is applied client-side: a hit is kept only
      when its organism string contains the species name.
    """
    if not sequence or len(sequence.strip()) < 10:
        return []

    is_prot = _is_protein(sequence)


    try:
        hits = _local_blast(
            sequence, database=database, is_protein=is_prot,
            species=species, max_hits=max_hits,
        )
        if hits:
            return [_normalize_blast_hit(h) for h in hits]
    except Exception as exc:
        logger.warning("Local BLAST path raised: %s", exc)


    

    

    logger.debug(
        "Local BLAST returned no hits for sequence len=%s db=%s",
        len(sequence), database,
    )
    return []

def fetch_protein_sequence(uniprot_acc: str) -> str | None:
    if not uniprot_acc or len(uniprot_acc) < 5:
        return None
    cache_suffix = f"prot_seq_{uniprot_acc}"
    cached = _cache_get(cache_suffix, CACHE_TTL_ANNOTATION)
    if cached is not None:
        return cached if isinstance(cached, str) else None

    try:
        resp = _http_get_text(
            f"https://www.ebi.ac.uk/proteins/api/proteins/{urllib.parse.quote(uniprot_acc)}",
            timeout=15, headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("EBI proteins API failed for %s: %s", uniprot_acc, exc)
        return None


    sequence: str | None = None
    if isinstance(data, dict):

        for entry in data.get("entry", []) or []:
            for seq in entry.get("sequence", []) or []:
                if seq.get("length") and seq.get("sequence"):
                    sequence = seq["sequence"]
                    break
            if sequence:
                break
        if not sequence:

            seq_obj = data.get("sequence") or {}
            sequence = seq_obj.get("sequence") if isinstance(seq_obj, dict) else None
    elif isinstance(data, list) and data:
        first = data[0] if isinstance(data[0], dict) else {}
        seq_obj = first.get("sequence") or {}
        sequence = seq_obj.get("sequence") if isinstance(seq_obj, dict) else None

    if sequence:
        _cache_set(cache_suffix, sequence)
    return sequence

def _normalize_blast_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Map EBI / NCBI BLAST hit dicts to a single downstream schema.

    Both backends return slightly different field names; downstream
    code (e.g. EvidenceRetrievalAgent) reads only ``protein_id``,
    ``gene_name``, ``organism``, ``e_value``, ``identity_pct``.
    """
    return {
        "protein_id": hit.get("protein_id") or hit.get("accession") or "",
        "gene_name": hit.get("gene_name") or hit.get("description", "").split()[0:3] and " ".join(hit.get("description", "").split()[0:3]),
        "organism": hit.get("organism", ""),
        "e_value": hit.get("e_value"),
        "identity_pct": hit.get("identity_pct")
            if hit.get("identity_pct") is not None
            else hit.get("identities_pct"),
        "score_bits": hit.get("score_bits"),
        "description": hit.get("description", ""),
        "source": hit.get("source", "BLAST"),
    }

INTERPRO_UNIPROT_URL = "https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/UniProt"

INTERPRO_PFAM_URL = "https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/UniProt"

def _interpro_by_uniprot_id(protein_id: str) -> dict[str, Any]:
    """Fetch InterPro entries for a reviewed UniProt protein."""
    if not _looks_like_uniprot_acc(protein_id):
        return {}
    url = f"{INTERPRO_UNIPROT_URL}/{urllib.parse.quote(protein_id)}?page_size=50"
    try:
        resp = _http_get_text(url, timeout=REQUEST_TIMEOUT, headers={"Accept": "application/json"})
        resp.raise_for_status()
        content_type = str(resp.headers.get("content-type") or "").lower()
        body = resp.text.lstrip()
        if "json" not in content_type and not body.startswith(("{", "[")):

            

            logger.info(
                "InterPro UniProt lookup returned non-JSON for %s (content-type=%s)",
                protein_id,
                content_type or "unknown",
            )
            return {}
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.info("InterPro UniProt lookup unavailable for %s: %s", protein_id, exc)
        return {}

def _parse_interpro_entries(data: dict[str, Any]) -> dict[str, Any]:
    """Parse InterPro API response into domains, GO terms, pathways."""
    domains: list[dict[str, Any]] = []
    go_terms: list[dict[str, Any]] = []
    pathways: list[dict[str, Any]] = []

    results = data.get("results", [])
    for entry in results:
        metadata = entry.get("metadata", {})
        name = metadata.get("name", "")
        entry_type = metadata.get("type", "")
        accession = metadata.get("accession", "")
        source_db = metadata.get("source_database", "")

        domain_entry = {
            "name": name,
            "accession": accession,
            "type": entry_type,
            "source": source_db,
        }
        domains.append(domain_entry)


        go_annotations = metadata.get("go_terms", []) or entry.get("go_terms", [])
        for go in go_annotations:
            go_terms.append({
                "go_id": go.get("identifier", go.get("accession", "")),
                "go_name": go.get("name", ""),
                "go_category": go.get("category", {}).get("code", "") if isinstance(go.get("category"), dict) else "",
            })


        pw_annotations = metadata.get("pathways", []) or {}
        if isinstance(pw_annotations, dict):
            for pw_name, pw_info in pw_annotations.items():
                pathways.append({"name": pw_name, "id": str(pw_info) if pw_info else ""})
        elif isinstance(pw_annotations, list):
            for pw in pw_annotations:
                if isinstance(pw, dict):
                    pathways.append({"name": pw.get("name", ""), "id": pw.get("accession", pw.get("id", ""))})

    return {
        "protein_domains": domains,
        "go_terms": go_terms,
        "pathways": pathways,
    }

def interpro_scan(protein_sequence: str, protein_id: str | None = None) -> dict[str, Any]:
    """Retrieve InterPro domain/GO/pathway annotation.

    Strategy:
      1. If *protein_id* is given, query reviewed UniProt entries first.
      2. If that fails or no protein_id, attempt InterProScan REST submission.
      3. Parse unified result with protein_domains, go_terms, pathways.

    Cached per protein_id (or sequence hash) for 30 days.
    """
    result: dict[str, Any] = {"protein_domains": [], "go_terms": [], "pathways": []}

    cache_key_base = protein_id if protein_id else hashlib.sha256(protein_sequence.encode()).hexdigest()[:20]
    cache_suffix = f"interpro_{cache_key_base}"
    cached = _cache_get(cache_suffix, CACHE_TTL_ANNOTATION)
    if cached is not None:
        return cached

    if protein_id:
        data = _interpro_by_uniprot_id(protein_id)
        if data.get("results"):
            result = _parse_interpro_entries(data)
            if result["protein_domains"]:
                _cache_set(cache_suffix, result)
                return result


    if protein_sequence and len(protein_sequence) >= 10:
        try:
            proxies = _proxy()

            submit_url = "https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/unreviewed/"
            resp = _http_get_text(
                f"{submit_url}?search={urllib.parse.quote(protein_sequence[:200])}",
                timeout=REQUEST_TIMEOUT,
                headers={"Accept": "application/json"},
                proxies=None,
            )
            resp.raise_for_status()
            data = resp.json()
            parsed = _parse_interpro_entries(data)
            if parsed["protein_domains"]:
                result = parsed
        except Exception as exc:
            logger.warning("InterPro sequence search failed: %s", exc)

    _cache_set(cache_suffix, result)
    return result

PLANTREGMAP_URL = "http://plantregmap.gao-lab.org/"

def plantregmap_search(gene_id: str, species: str) -> list[dict[str, Any]]:
    """Search PlantRegMap for TF binding sites and regulatory elements.

    Uses the PlantRegMap API endpoint:
      http://plantregmap.gao-lab.org/api/

    Returns list of dicts with: tf_name, binding_motif, target_genes, regulatory_type.

    Note: PlantRegMap is a web service with limited programmatic API. This
    implementation uses the known REST endpoints where available and falls
    back to empty results gracefully.
    """
    if not gene_id:
        return []

    cache_suffix = f"plantregmap_{gene_id}_{species}"
    cached = _cache_get(cache_suffix, CACHE_TTL_ANNOTATION)
    if cached is not None:
        return cached if isinstance(cached, list) else []

    results: list[dict[str, Any]] = []


    
    species_short = species.lower().replace(" ", "_")[:20] if species else ""

    try:

        api_url = f"{PLANTREGMAP_URL}api/tfbs_search"
        params: dict[str, Any] = {
            "gene_id": gene_id,
            "species": species_short,
        }
        resp = _http_get_text(api_url, timeout=REQUEST_TIMEOUT, params=params)
        if resp.status_code == 200:
            try:
                data = resp.json()
                for item in data if isinstance(data, list) else data.get("results", data.get("data", [])):
                    if isinstance(item, dict):
                        results.append({
                            "tf_name": item.get("tf_name", item.get("TF", "")),
                            "binding_motif": item.get("motif", item.get("binding_motif", "")),
                            "target_genes": item.get("target_genes", item.get("targets", [])),
                            "regulatory_type": item.get("regulatory_type", item.get("type", "TFBS")),
                        })
            except json.JSONDecodeError:
                pass
    except Exception as exc:
        logger.warning("PlantRegMap TFBS search failed for %s/%s: %s", gene_id, species, exc)


    if not results:
        try:
            api_url = f"{PLANTREGMAP_URL}api/regulation"
            params = {
                "gene": gene_id,
                "species": species_short,
            }
            resp = _http_get_text(api_url, timeout=REQUEST_TIMEOUT, params=params)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    items = data if isinstance(data, list) else data.get("results", data.get("data", []))
                    for item in items:
                        if isinstance(item, dict):
                            results.append({
                                "tf_name": item.get("tf_name", item.get("name", "")),
                                "binding_motif": item.get("motif", item.get("binding_motif", "")),
                                "target_genes": item.get("target_genes", item.get("targets", [])),
                                "regulatory_type": item.get("regulatory_type", item.get("type", "regulation")),
                            })
                except json.JSONDecodeError:
                    pass
        except Exception as exc:
            logger.warning("PlantRegMap regulation search failed for %s/%s: %s", gene_id, species, exc)

    _cache_set(cache_suffix, results)
    return results

JASPAR_API_URL = "https://jaspar.elixir.no/api/v1"

JASPAR_SCAN_LIMIT = 20

JASPAR_REL_SCORE_MIN = 0.80

def _jaspar_tax_id(species_key: str) -> str:
    val = _SPECIES_TAX_ID.get(species_key, 0)
    return str(val) if val else ""

def _jaspar_matrix_list(tax_group: str, tax_id: str, page_size: int) -> tuple[str, ...]:
    """Return a tuple of plant-relevant JASPAR matrix IDs.

    Cached per (tax_group, tax_id, page_size) — the matrix set is stable
    across runs of a given species so this avoids re-hitting the list
    endpoint.
    """
    proxies = _proxy()
    params: dict[str, Any] = {
        "tax_group": tax_group,
        "page_size": str(page_size),
        "format": "json",
        "version": "latest",
    }
    if tax_id:
        params["tax_id"] = tax_id
    try:
        resp = _http_get_text(
            f"{JASPAR_API_URL}/matrix/",
            params=params,
            timeout=REQUEST_TIMEOUT,
            proxies=None,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("JASPAR matrix list failed: %s", exc)
        return ()

    raw = data.get("results", []) if isinstance(data, dict) else []
    if isinstance(data, list):
        raw = data
    matrix_ids: list[str] = []
    for entry in raw:
        mid = entry.get("matrix_id") or entry.get("id")
        if mid:
            matrix_ids.append(str(mid))
    return tuple(matrix_ids)

def _jaspar_pfm(matrix_id: str) -> dict[str, tuple[float, ...]] | None:
    """Fetch a JASPAR PFM and convert it to a {base: tuple} dict.

    Returns None on any failure (HTTP error, malformed response).  Cached
    per matrix_id so subsequent calls within a process are free.
    """
    proxies = _proxy()
    cache_key = f"jaspar_pfm_{matrix_id}"
    cached = _cache_get(cache_key, CACHE_TTL_ANNOTATION)
    if cached is not None:
        if isinstance(cached, dict):
            return {k: tuple(v) for k, v in cached.items()}
        return None
    try:
        resp = _http_get_text(
            f"{JASPAR_API_URL}/matrix/{urllib.parse.quote(matrix_id)}/",
            timeout=REQUEST_TIMEOUT,
            proxies=None,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("JASPAR PFM fetch failed for %s: %s", matrix_id, exc)
        return None
    pfm = data.get("pfm") if isinstance(data, dict) else None
    if not isinstance(pfm, dict):
        return None
    out: dict[str, tuple[float, ...]] = {}
    for base in ("A", "C", "G", "T"):
        values = pfm.get(base)
        if values is None:
            return None
        try:
            out[base] = tuple(float(v) for v in values)
        except (TypeError, ValueError):
            return None
    if not out or len(set(len(v) for v in out.values())) != 1:
        return None
    _cache_set(cache_key, {k: list(v) for k, v in out.items()})
    return out

def _pfm_to_pwm(
    pfm: dict[str, tuple[float, ...]],
    *,
    pseudocount: float = 0.8,
    background: float = 0.25,
) -> tuple[tuple[float, ...], ...] | None:
    """Convert a JASPAR PFM to a PWM (log2-odds) matrix.

    Returns a 4-tuple of (A, C, G, T) score rows.  Returns None if the
    PFM is empty or any row is non-positive (log undefined).
    """
    import math
    try:
        ncols = len(pfm["A"])
    except KeyError:
        return None
    if ncols == 0:
        return None
    out: list[tuple[float, ...]] = []
    ncols = len(pfm["A"])
    col_sums = [sum(float(pfm[b][i]) + pseudocount for b in ("A", "C", "G", "T"))
                for i in range(ncols)]
    for base in ("A", "C", "G", "T"):
        row = pfm[base]
        new_row: list[float] = []
        for i, v in enumerate(row):
            count = float(v) + pseudocount
            ppm = count / col_sums[i]
            if ppm <= 0 or background <= 0:
                return None
            new_row.append(math.log2(ppm / background))
        out.append(tuple(new_row))
    return tuple(out)

def _scan_pwm(
    sequence: str,
    pwm: tuple[tuple[float, ...], ...],
    *,
    rel_min: float = JASPAR_REL_SCORE_MIN,
) -> list[dict[str, Any]]:
    """Find motif hits in *sequence* above the relative-score threshold.

    Returns a list of dicts with keys: position, score, relative_score.
    Relative score is the per-position score divided by the max possible
    per-position score for the same PWM (0..1).
    """
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    ncols = len(pwm[0])
    if len(sequence) < ncols:
        return []
    seq = sequence.upper()
    max_score = sum(max(col) for col in zip(*pwm))
    if max_score <= 0:
        return []
    hits: list[dict[str, Any]] = []
    for start in range(0, len(seq) - ncols + 1):
        sub = seq[start:start + ncols]
        score = 0.0
        valid = True
        for offset, base in enumerate(sub):
            idx = base_to_idx.get(base)
            if idx is None:
                valid = False
                break
            score += pwm[idx][offset]
        if not valid:
            continue
        rel = score / max_score
        if rel >= rel_min:
            hits.append({
                "position": start,
                "score": round(score, 3),
                "relative_score": round(rel, 3),
            })
    return hits

def jaspar_motif_scan(sequence: str, species: str | None = None) -> list[dict[str, Any]]:
    """Scan *sequence* for plant TF binding motifs using JASPAR PFMs.

    Workflow:
      1. List plant-relevant JASPAR matrix IDs (cached per species).
      2. Fetch each PFM (cached per matrix_id) and convert to PWM.
      3. Slide the PWM across the sequence; report hits with relative
         score ≥ :data:`JASPAR_REL_SCORE_MIN`.

    Returns dicts with: ``tf_name``, ``motif_id``, ``score``,
    ``relative_score``, ``position``.

    All network calls are fail-soft — a single matrix failing is logged
    and skipped; a total outage produces an empty result.
    """
    if not sequence or len(sequence) < 6:
        return []

    species_key = (species or "any").lower().strip()
    tax_id = _jaspar_tax_id(species_key)
    tax_group = "plants"

    cache_suffix = (
        f"jaspar_{hashlib.sha256(sequence.encode()).hexdigest()[:16]}"
        f"_{species_key}_v2"
    )
    cached = _cache_get(cache_suffix, CACHE_TTL_ANNOTATION)
    if cached is not None:
        return cached if isinstance(cached, list) else []

    matrix_ids = _jaspar_matrix_list(tax_group, tax_id, JASPAR_SCAN_LIMIT)
    if not matrix_ids:
        return []

    seq_clean = sequence.upper()


    

    
    def _scan_one(mid: str) -> list[dict[str, Any]]:
        pfm = _jaspar_pfm(mid)
        if pfm is None:
            return []
        pwm = _pfm_to_pwm(pfm)
        if pwm is None:
            return []
        try:
            hits = _scan_pwm(seq_clean, pwm)
        except Exception:
            return []
        out: list[dict[str, Any]] = []
        for hit in hits[:3]:  
            out.append({
                "tf_name": mid,  
                "motif_id": mid,
                "score": hit["score"],
                "relative_score": hit["relative_score"],
                "position": hit["position"],
            })
        return out

    matrix_results = _concurrent_run(
        [lambda m=mid: _scan_one(m) for mid in matrix_ids],
        max_workers=min(8, len(matrix_ids)),
    )

    results: list[dict[str, Any]] = []
    for sub in matrix_results:
        if isinstance(sub, list):
            results.extend(sub)


    seen: set[tuple[str, int]] = set()
    deduped: list[dict[str, Any]] = []
    for r in results:
        key = (r["motif_id"], r["position"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    _cache_set(cache_suffix, deduped)
    return deduped

INTERPRO_PFAM_MAX_PAGES = 5

def pfam_domain_for_protein(uniprot_acc: str) -> list[dict[str, Any]]:
    acc = (uniprot_acc or "").strip()
    if not _looks_like_uniprot_acc(acc):
        return []
    if not acc:
        return []
    cache_suffix = f"pfam_{acc}"
    cached = _cache_get(cache_suffix, CACHE_TTL_ANNOTATION)
    if cached is not None:
        return cached if isinstance(cached, list) else []

    proxies = _proxy()
    domains: list[dict[str, Any]] = []
    url: str | None = f"{INTERPRO_PFAM_URL}/{urllib.parse.quote(acc)}/?page_size=200"
    pages = 0
    while url and pages < INTERPRO_PFAM_MAX_PAGES:
        pages += 1
        try:
            resp = _http_get_text(
                url,
                headers={"Accept": "application/json"},
                timeout=REQUEST_TIMEOUT,
                proxies=None,
            )
        except Exception as exc:
            logger.warning("InterPro Pfam request failed for %s: %s", acc, exc)
            break
        if resp.status_code == 204 or resp.status_code == 404:

            break
        if resp.status_code != 200:
            logger.warning("InterPro Pfam HTTP %s for %s", resp.status_code, acc)
            break
        try:
            data = resp.json()
        except json.JSONDecodeError:
            logger.warning("InterPro Pfam returned non-JSON for %s", acc)
            break
        for entry in data.get("results", []):
            md = entry.get("metadata", {}) if isinstance(entry, dict) else {}
            domains.append({
                "accession": md.get("accession", ""),
                "name": md.get("name", ""),
                "description": md.get("description", ""),
                "source_database": md.get("source_database", "pfam"),
                "type": md.get("type", "domain"),

                
                "interpro_accession": md.get("integrated") or "",
            })

        next_url = data.get("next") if isinstance(data, dict) else None
        url = next_url if next_url else None

    _cache_set(cache_suffix, domains)
    return domains

_SPECIES_TAX_ID: dict[str, int] = {
    "rice": 4530,
    "oryza sativa": 4530,
    "maize": 4577,
    "zea mays": 4577,
    "tomato": 4081,
    "solanum lycopersicum": 4081,
    "chickpea": 3827,
    "cicer arietinum": 3827,
    "cotton": 3635,
    "gossypium hirsutum": 3635,
    "arabidopsis": 3702,
    "arabidopsis thaliana": 3702,
}

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb"

ENSEMBL_XREFS_URL = "https://rest.ensembl.org/xrefs/id"

_UNIPROT_ACC_RE = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$")

def _looks_like_uniprot_acc(value: str) -> bool:
    return bool(_UNIPROT_ACC_RE.fullmatch(value or ""))

_UNIPROT_TREMBL_RE = re.compile(r"^[A-Z0-9]{10}$")

def _is_swissprot(acc: str) -> bool:
    """Return True for Swiss-Prot (reviewed) accessions, False for TrEMBL."""
    if not acc or not _UNIPROT_ACC_RE.match(acc):
        return False

    if len(acc) == 10 and acc[0].isdigit():
        return False
    if _UNIPROT_TREMBL_RE.match(acc):
        return False
    return True

def _ensembl_xrefs_lookup(gene_id: str) -> list[dict[str, Any]]:
    """Look up an external ID via Ensembl Plants xrefs.

    The Ensembl Plants REST API is cross-species — it accepts any supported
    plant ID (RAP-DB ``Os01g0883800`` for rice, Araport ``AT1G01010`` for
    arabidopsis, ``Zm00001eb*`` for maize) and returns all linked external
    references including UniProt.  Returns a list of xref dicts; the caller
    filters for ``dbname == "Uniprot_gn"`` and picks the primary accession.

    Retries transient failures (5xx, network) up to 2 times; persistent
    errors (400, 404) are treated as "unknown ID" and return [].
    """
    gene_id = (gene_id or "").strip()
    if not gene_id:
        return []

    

    
    last_exc: Exception | None = None
    resp = None
    for attempt in range(3):
        try:
            resp = _http_get_text(
                f"{ENSEMBL_XREFS_URL}/{urllib.parse.quote(gene_id)}",
                timeout=15, params={"external_db": "Uniprot_gn"},
                headers={"Accept": "application/json"},
            )
            break
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    if resp is None:
        logger.warning("Ensembl xrefs request failed for %s: %s", gene_id, last_exc)
        return []
    if resp.status_code == 400:

        logger.debug("Ensembl xrefs: unknown ID %s", gene_id)
        return []
    if resp.status_code != 200:
        logger.debug("Ensembl xrefs HTTP %s for %s", resp.status_code, gene_id)
        return []
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        logger.warning("Ensembl xrefs returned non-JSON for %s", gene_id)
        return []
    return payload if isinstance(payload, list) else []

def _uniprot_search_lookup(
    gene_id: str,
    tax_id: int,
) -> list[dict[str, Any]]:
    """Fallback UniProt search by gene name within a species.

    Tries a strict ``gene_exact`` query first, then a fuzzy ``gene`` query.
    Each returns up to 3 hits as TSV rows parsed into dicts.
    """
    gene_id = (gene_id or "").strip()
    if not gene_id:
        return []
    fields = "accession,gene_names,protein_name,organism_name,length,id"
    out: list[dict[str, Any]] = []
    queries: list[tuple[str, str, str]] = [
        ("gene_exact", f"gene_exact:{gene_id} AND taxonomy_id:{tax_id}", "primary"),
        ("gene", f"gene:{gene_id} AND taxonomy_id:{tax_id}", "primary"),
    ]
    for label, q, _ in queries:
        try:
            resp = _http_get_text(
                UNIPROT_SEARCH_URL,
                timeout=REQUEST_TIMEOUT,
                params={"query": q, "format": "tsv", "fields": fields, "size": 3},
            )
        except Exception as exc:
            logger.warning("UniProt search (%s) failed for %s: %s", label, gene_id, exc)
            continue
        if resp.status_code != 200:
            continue
        lines = resp.text.strip().split("\n")
        if len(lines) < 2:
            continue
        header = lines[0].split("\t")
        for row in lines[1:]:
            cells = row.split("\t")
            if len(cells) != len(header):
                continue
            entry: dict[str, Any] = dict(zip(header, cells))
            if entry.get("Organism", "").lower() != "":

                
                out.append({
                    "accession": entry.get("Entry", ""),
                    "gene_names": entry.get("Gene Names", ""),
                    "protein_name": entry.get("Protein names", ""),
                    "organism": entry.get("Organism", ""),
                    "length": int(entry.get("Length", "0") or 0),
                    "entry_name": entry.get("Entry Name", ""),
                })
        if out:
            break
    return out

def resolve_to_uniprot(
    gene_id: str,
    species: str | None = None,
) -> dict[str, Any] | None:
    """Resolve any plant gene identifier to a primary UniProt accession.

    Resolver strategy (in order):

    1. **Direct accession** — if the input matches a UniProt accession
       pattern (``Q0JH50``-style), try the entry endpoint.  Fast path for
       cases where the caller already has an accession.
    2. **Ensembl Plants xrefs** — preferred for plant gene IDs
       (RAP-DB ``Os01g0883800``, Araport ``AT1G01010``, Ensembl Plants
       ``Zm00001eb*``).  Returns the canonical UniProt accession plus
       gene-name synonyms and protein description.
    3. **UniProt search fallback** — when Ensembl doesn't recognise the
       ID, try ``gene_exact``, then ``gene``, then free-text with the
       species tax_id filter.  Cross-species name collisions are
       avoided by the ``organism_id`` clause.

    Returns a dict with keys ``accession``, ``gene_names``,
    ``protein_name``, ``organism``, ``entry_name``, ``resolver`` (which
    step won: ``direct`` / ``ensembl`` / ``uniprot``), or ``None`` if no
    match is found.  Cached on disk for 30 days.

    Designed to be called once per variant from EvidenceRetrievalAgent and then
    passed to Pfam / Reactome / BLAST as the canonical identifier.
    """
    gene_id = (gene_id or "").strip()
    if not gene_id:
        return None
    species_key = (species or "").lower().strip()
    tax_id = _SPECIES_TAX_ID.get(species_key, 0)
    cache_suffix = f"resolve_uniprot_taxonomy_v2_{hashlib.sha256((gene_id + '|' + str(tax_id)).encode()).hexdigest()[:16]}"
    cached = _cache_get(cache_suffix, CACHE_TTL_ANNOTATION)
    if cached is not None:
        if isinstance(cached, dict):
            return cached
        return None

    proxies = _proxy()


    if _UNIPROT_ACC_RE.match(gene_id):
        try:
            resp = _http_get_text(
                f"{UNIPROT_ENTRY_URL}/{urllib.parse.quote(gene_id)}",
                timeout=REQUEST_TIMEOUT,
                params={"format": "tsv", "fields": "accession,gene_names,protein_name,organism_name,length,id"},
            )
        except Exception as exc:
            logger.warning("UniProt entry lookup failed for %s: %s", gene_id, exc)
        else:
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                if len(lines) >= 2:
                    header = lines[0].split("\t")
                    cells = lines[1].split("\t")
                    if len(cells) == len(header):
                        e = dict(zip(header, cells))
                        result = {
                            "accession": e.get("Entry", gene_id),
                            "gene_names": e.get("Gene Names", ""),
                            "protein_name": e.get("Protein names", ""),
                            "organism": e.get("Organism", ""),
                            "entry_name": e.get("Entry Name", ""),
                            "length": int(e.get("Length", "0") or 0),
                            "resolver": "direct",
                        }
                        _cache_set(cache_suffix, result)
                        return result


    if tax_id:
        hits = _uniprot_search_lookup(gene_id, tax_id)
        exact = [hit for hit in hits if gene_id.lower() in str(hit.get("gene_names", "")).lower().split()]
        if exact:
            result = {**sorted(exact, key=lambda hit: (len(hit.get("accession", "")) == 10, hit.get("accession", "")))[0], "resolver": "uniprot"}
            _cache_set(cache_suffix, result)
            return result


    xrefs = _ensembl_xrefs_lookup(gene_id)

    

    uniprot_xrefs = [x for x in xrefs if x.get("dbname") in ("Uniprot_gn", "Uniprot_sptremb")]
    if uniprot_xrefs:

        def _rank(x: dict[str, Any]) -> tuple[int, str]:
            acc = x.get("primary_id", "")
            return (0 if _is_swissprot(acc) else 1, acc)
        uniprot_xrefs.sort(key=_rank)
        best = uniprot_xrefs[0]
        result = {
            "accession": best.get("primary_id", ""),
            "gene_names": best.get("display_id", "") + "".join(
                f" {s}" for s in (best.get("synonyms") or [])
            ),
            "protein_name": best.get("description", ""),
            "organism": "",  
            "entry_name": "",
            "length": 0,
            "resolver": "ensembl",
        }
        _cache_set(cache_suffix, result)
        return result


    if tax_id:
        hits = _uniprot_search_lookup(gene_id, tax_id)

        if hits and species_key:

            

            base_name = {
                "rice": "oryza sativa",
                "maize": "zea mays",
                "tomato": "solanum lycopersicum",
                "chickpea": "cicer arietinum",
                "cotton": "gossypium",
                "arabidopsis": "arabidopsis thaliana",
            }.get(species_key, "")
            if base_name:
                hits = [h for h in hits if base_name in h.get("organism", "").lower()]
        if hits:

            hits.sort(key=lambda h: (0 if _is_swissprot(h.get("accession", "")) else 1,
                                     h.get("accession", "")))
            best = hits[0]
            result = {
                "accession": best.get("accession", ""),
                "gene_names": best.get("gene_names", ""),
                "protein_name": best.get("protein_name", ""),
                "organism": best.get("organism", ""),
                "entry_name": best.get("entry_name", ""),
                "length": best.get("length", 0),
                "resolver": "uniprot",
            }
            _cache_set(cache_suffix, result)
            return result

    return None

def quick_blastn(
    dna_sequence: str,
    database: str = "nt",
    species: str | None = None,
    max_hits: int = 5,
) -> list[dict[str, Any]]:
    """BLASTN on flanking DNA — SKIPPED (no local nt database, remote always fails).

    The remote NCBI/EBI BLASTN path consistently fails in this environment
    (EBI budget exceeded, NCBI RID extraction broken).  There is no local
    nucleotide database available for blastn.  The function returns [] so
    the caller's circuit breaker activates and saves time.
    """
    return []
