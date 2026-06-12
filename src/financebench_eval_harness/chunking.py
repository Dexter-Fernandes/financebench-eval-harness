from __future__ import annotations

from dataclasses import dataclass

from financebench_eval_harness.retrieval_types import Chunk, DocumentPage


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    strategy: str = "recursive_text"
    min_chunk_chars: int = 0


def chunk_page(
    page: DocumentPage,
    config: ChunkingConfig,
    chunk_offset: int = 0,
) -> list[Chunk]:
    """Split one page into Chunk objects using recursive text splitting."""
    text = page.text
    if not text or not text.strip():
        return []

    spans = _recursive_split(text, config.chunk_size, config.chunk_overlap)
    chunks: list[Chunk] = []
    for idx, (start, end) in enumerate(spans):
        chunk_text = text[start:end]
        if config.min_chunk_chars > 0 and len(chunk_text.strip()) < config.min_chunk_chars:
            continue
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(page.doc_id, page.page_num, chunk_offset + idx),
                doc_id=page.doc_id,
                doc_name=page.doc_name,
                page_num=page.page_num,
                text=chunk_text,
                start_char=start,
                end_char=end,
            )
        )
    return chunks


def chunk_pages(
    pages: list[DocumentPage],
    config: ChunkingConfig,
) -> list[Chunk]:
    """Chunk all pages, producing globally unique chunk IDs."""
    all_chunks: list[Chunk] = []
    for page in pages:
        page_chunks = chunk_page(page, config, chunk_offset=len(all_chunks))
        all_chunks.extend(page_chunks)
    return all_chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_SEPARATORS = ["\n\n", "\n", " ", ""]


def _recursive_split(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Return (start, end) character spans for chunks of `text`."""
    splits = _split_text(text, size, _SEPARATORS)
    return _merge_splits(text, splits, size, overlap)


def _split_text(text: str, size: int, separators: list[str]) -> list[str]:
    """Recursively split text using the first separator that produces pieces <= size."""
    separator = separators[-1]
    for sep in separators:
        if sep == "":
            break
        pieces = text.split(sep)
        if any(len(p) > size for p in pieces):
            continue
        separator = sep
        break

    raw_pieces = text.split(separator) if separator else list(text)
    result: list[str] = []
    for piece in raw_pieces:
        if len(piece) <= size:
            result.append(piece)
        else:
            remaining = [s for s in separators if s != separator]
            result.extend(_split_text(piece, size, remaining or [""]))
    return [p for p in result if p]


def _merge_splits(
    original: str,
    splits: list[str],
    size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Merge small splits into chunks of ~size with overlap, returning (start, end) spans.

    Caveat: uses original.index(split, cursor) to reconstruct character positions. This is
    correct as long as cursor advances past each token before the next search. Pages with
    repeated identical substrings (e.g. repeated table headers) could produce wrong
    start_char/end_char offsets if cursor tracking drifts. Watch for this in real financial PDFs.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    current_start = 0
    current_len = 0

    for split in splits:
        split_start = original.index(split, cursor)
        split_len = len(split)

        if current_len + split_len > size and current_len > 0:
            end = current_start + current_len
            spans.append((current_start, end))
            # Start next chunk overlap characters before current end
            overlap_start = max(current_start, end - overlap)
            current_start = overlap_start
            current_len = end - overlap_start

        if current_len == 0:
            current_start = split_start
        current_len = split_start + split_len - current_start
        cursor = split_start + split_len

    if current_len > 0:
        spans.append((current_start, len(original)))

    return spans


def _make_chunk_id(doc_id: str, page_num: int, chunk_idx: int) -> str:
    return f"{doc_id}_p{page_num:03d}_c{chunk_idx:03d}"
