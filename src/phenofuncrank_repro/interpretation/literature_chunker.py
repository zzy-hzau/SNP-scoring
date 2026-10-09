"Section-aware full-text chunking (SOP §4.2.4, test 3)."

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from phenofuncrank_repro.interpretation.settings import settings


_SECTION_RE = re.compile(
    r"(?im)^\s*(abstract|introduction|materials?\s+and\s+methods?|methods?|"
    r"results?|discussion|conclusions?|supplementary\s+(?:materials?\s+)?(?:and\s+methods?)?|"
    r"figure\s+legends?|figures?|author\s+contributions?|acknowledgements?|references?)\b[.:]?\s*$"
)


@dataclass
class Chunk:
    section: str
    start_char: int
    end_char: int
    content: str
    content_hash: str


@dataclass
class ChunkingResult:
    chunks: list[Chunk] = field(default_factory=list)
    chunks_truncated: int = 0
    sections_found: list[str] = field(default_factory=list)


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (section_label, section_body) using heading markers.

    Unstructured text returns a single ``(text, ...)`` section.
    """
    if not text:
        return []
    matches = list(_SECTION_RE.finditer(text))
    sections: list[tuple[str, str]] = []
    if not matches:
        return [("text", text)]
    for i, m in enumerate(matches):
        label = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((label, body))
    if not sections:
        return [("text", text)]
    return sections


def _paragraphs(body: str) -> list[tuple[int, int]]:
    """Paragraph spans (start, end) within *body* using blank-line breaks."""
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(body):
        m = re.search(r"\n\s*\n", body[start:])
        if m is None:
            spans.append((start, len(body)))
            break
        end = start + m.start()
        if body[start:end].strip():
            spans.append((start, end))
        start = end + m.end()
    return spans


def _chunk_paragraphs(
    body: str,
    paragraphs: list[tuple[int, int]],
    *,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Greedy paragraph packing with overlap; returns char spans in *body*."""
    packed: list[tuple[int, int]] = []
    i = 0
    while i < len(paragraphs):
        start = paragraphs[i][0]
        end = paragraphs[i][1]
        j = i + 1
        while j < len(paragraphs) and (paragraphs[j][1] - start) <= chunk_size:
            end = paragraphs[j][1]
            j += 1
        packed.append((start, end))
        next_i = j - 1 if j > i + 1 else j
        if overlap > 0 and next_i < len(paragraphs):

            backtrack = paragraphs[next_i][0]
            tail_start = max(start, end - overlap)
            if tail_start <= backtrack:
                next_i = i + 1
        i = next_i

    spans: list[tuple[int, int]] = []
    for (start, end) in packed:
        if end - start <= chunk_size:
            spans.append((start, end))
            continue
        pos = start
        while pos < end:
            stop = min(pos + chunk_size, end)
            spans.append((pos, stop))
            if stop >= end:
                break
            pos = max(pos + 1, stop - overlap)
    return spans


def chunk_text(
    text: str,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
    max_chunks: int | None = None,
) -> ChunkingResult:
    chunk_size = chunk_size or settings.chunk_size_chars
    overlap = overlap if overlap is not None else settings.chunk_overlap_chars
    max_chunks = max_chunks or 200
    result = ChunkingResult()
    text = text or ""
    if not text.strip():
        return result
    for section, body in split_sections(text):
        result.sections_found.append(section)
        paragraphs = _paragraphs(body)
        if not paragraphs:
            continue
        for (start, end) in _chunk_paragraphs(
            body, paragraphs, chunk_size=chunk_size, overlap=overlap,
        ):
            if len(result.chunks) >= max_chunks:
                result.chunks_truncated += 1
                continue
            content = body[start:end].strip()
            if not content:
                continue
            result.chunks.append(Chunk(
                section=section,
                start_char=start,
                end_char=end,
                content=content,
                content_hash=_hash(content),
            ))
    return result
