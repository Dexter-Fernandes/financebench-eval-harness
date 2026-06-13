"""M5.3 — RAG prompt templates (TDD)."""
from __future__ import annotations

from pathlib import Path

from financebench_eval_harness.evaluation import EvaluationMode, load_evaluation_config, render_prompt
from financebench_eval_harness.rag_types import RAGContextChunk

RAG_CONFIG_PATH = Path("configs/evaluation/rag_modes.yaml")

# ---------------------------------------------------------------------------
# Cycles 1-2 — RAGContextChunk.format_for_prompt()
# ---------------------------------------------------------------------------


def test_format_for_prompt_contains_chunk_id() -> None:
    chunk = RAGContextChunk(
        rank=1,
        chunk_id="ACME_2022_p12_c0",
        doc_name="ACME_2022_10K.pdf",
        page_num=12,
        text="Revenue was $10 million.",
    )
    assert "ACME_2022_p12_c0" in chunk.format_for_prompt()


def test_format_for_prompt_contains_doc_name() -> None:
    chunk = RAGContextChunk(
        rank=1, chunk_id="c0", doc_name="ACME_2022_10K.pdf", page_num=12, text="t"
    )
    assert "ACME_2022_10K.pdf" in chunk.format_for_prompt()


def test_format_for_prompt_contains_page_num() -> None:
    chunk = RAGContextChunk(
        rank=1, chunk_id="c0", doc_name="doc.pdf", page_num=42, text="t"
    )
    assert "42" in chunk.format_for_prompt()


def test_format_for_prompt_contains_text() -> None:
    chunk = RAGContextChunk(
        rank=1, chunk_id="c0", doc_name="doc.pdf", page_num=1, text="Revenue was $10 million."
    )
    assert "Revenue was $10 million." in chunk.format_for_prompt()


def test_format_for_prompt_header_precedes_text() -> None:
    chunk = RAGContextChunk(
        rank=1,
        chunk_id="ACME_2022_p12_c0",
        doc_name="ACME_2022_10K.pdf",
        page_num=12,
        text="Revenue was $10 million.",
    )
    formatted = chunk.format_for_prompt()
    header_pos = formatted.index("ACME_2022_p12_c0")
    text_pos = formatted.index("Revenue was $10 million.")
    assert header_pos < text_pos


def test_format_for_prompt_exact_structure() -> None:
    chunk = RAGContextChunk(
        rank=1,
        chunk_id="ACME_2022_p12_c0",
        doc_name="ACME_2022_10K.pdf",
        page_num=12,
        text="Revenue was $10 million.",
    )
    expected = (
        "[chunk_id: ACME_2022_p12_c0 | doc: ACME_2022_10K.pdf | page: 12]\n"
        "Revenue was $10 million."
    )
    assert chunk.format_for_prompt() == expected


# ---------------------------------------------------------------------------
# Cycles 3-4 — rag_dense_v2.txt content
# ---------------------------------------------------------------------------

RAG_DENSE_V2_PATH = Path("prompts/rag/rag_dense_v2.txt")


def test_rag_dense_v2_template_file_exists() -> None:
    assert RAG_DENSE_V2_PATH.is_file(), f"Template not found: {RAG_DENSE_V2_PATH}"


def test_rag_dense_v2_template_is_non_empty() -> None:
    assert RAG_DENSE_V2_PATH.read_text(encoding="utf-8").strip()


def test_rag_dense_v2_template_contains_question_placeholder() -> None:
    text = RAG_DENSE_V2_PATH.read_text(encoding="utf-8")
    assert "{question}" in text


def test_rag_dense_v2_template_contains_evidence_text_placeholder() -> None:
    text = RAG_DENSE_V2_PATH.read_text(encoding="utf-8")
    assert "{evidence_text}" in text


def test_rag_dense_v2_template_instructs_do_not_know() -> None:
    text = RAG_DENSE_V2_PATH.read_text(encoding="utf-8").lower()
    assert "do not know" in text or "don't know" in text


def test_rag_dense_v2_template_instructs_no_inventing() -> None:
    text = RAG_DENSE_V2_PATH.read_text(encoding="utf-8").lower()
    assert "do not invent" in text or "don't invent" in text or "not invent" in text


def test_rag_dense_v2_template_requests_chunk_citations() -> None:
    text = RAG_DENSE_V2_PATH.read_text(encoding="utf-8").lower()
    assert "chunk" in text and ("cite" in text or "citation" in text or "id" in text)


# ---------------------------------------------------------------------------
# Cycle 5 — rag_modes.yaml points rag_dense to v2
# ---------------------------------------------------------------------------


def test_rag_dense_config_is_v2() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())
    template = config.prompt_for(EvaluationMode.RAG_DENSE)
    assert template.id == "rag_dense_v2"
    assert template.version == "v2"


def test_rag_dense_template_path_is_v2() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())
    template = config.prompt_for(EvaluationMode.RAG_DENSE)
    assert template.template_path == Path("prompts/rag/rag_dense_v2.txt")


# ---------------------------------------------------------------------------
# Cycles 6-8 — render_prompt integration with format_for_prompt()
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str, doc_name: str, page_num: int, text: str) -> RAGContextChunk:
    return RAGContextChunk(rank=1, chunk_id=chunk_id, doc_name=doc_name, page_num=page_num, text=text)


def test_render_rag_dense_v2_includes_chunk_id_header_in_output() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())
    chunk = _make_chunk("ACME_2022_p12_c0", "ACME_2022_10K.pdf", 12, "Revenue was $10 million.")

    rendered = render_prompt(
        config,
        EvaluationMode.RAG_DENSE,
        question="What was FY2022 revenue?",
        evidence_texts=[chunk.format_for_prompt()],
    )

    assert "ACME_2022_p12_c0" in rendered.text
    assert "ACME_2022_10K.pdf" in rendered.text
    assert "Revenue was $10 million." in rendered.text


def test_render_rag_dense_v2_contains_citation_instruction() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())
    chunk = _make_chunk("c0", "doc.pdf", 1, "some text")

    rendered = render_prompt(
        config,
        EvaluationMode.RAG_DENSE,
        question="Q?",
        evidence_texts=[chunk.format_for_prompt()],
    )

    text_lower = rendered.text.lower()
    assert "cite" in text_lower or "citation" in text_lower


def test_render_rag_dense_v2_contains_do_not_invent_instruction() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())
    chunk = _make_chunk("c0", "doc.pdf", 1, "some text")

    rendered = render_prompt(
        config,
        EvaluationMode.RAG_DENSE,
        question="Q?",
        evidence_texts=[chunk.format_for_prompt()],
    )

    text_lower = rendered.text.lower()
    assert "do not invent" in text_lower or "don't invent" in text_lower or "not invent" in text_lower


def test_render_rag_dense_v2_question_appears_in_output() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())
    chunk = _make_chunk("c0", "doc.pdf", 1, "some text")

    rendered = render_prompt(
        config,
        EvaluationMode.RAG_DENSE,
        question="What was FY2022 net income?",
        evidence_texts=[chunk.format_for_prompt()],
    )

    assert "What was FY2022 net income?" in rendered.text
