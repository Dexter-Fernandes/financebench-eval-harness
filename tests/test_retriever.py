from __future__ import annotations

import json
from pathlib import Path

import pytest

from financebench_eval_harness.embedding import EmbeddingConfig, MockEmbeddingClient
from financebench_eval_harness.index_builder import IndexMetadata
from financebench_eval_harness.retrieval_types import Chunk
from financebench_eval_harness.retriever import (
    Question,
    RetrievalRow,
    RetrievalRunResult,
    RetrievalRunMetadata,
    RetrievedChunk,
    next_run_dir,
    retrieve_for_questions,
    run_retrieval,
)
from financebench_eval_harness.vector_store import FaissVectorStore


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

DIM = 4
VECS = {
    "north": [0.0, 1.0, 0.0, 0.0],
    "south": [0.0, -1.0, 0.0, 0.0],
    "east":  [1.0, 0.0, 0.0, 0.0],
    "west":  [-1.0, 0.0, 0.0, 0.0],
}

EMBED_CFG = EmbeddingConfig(provider="mock", model_name="mock-embed")


def make_chunk(chunk_id: str, direction: str = "north") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="DOC_A",
        doc_name="DOC_A.pdf",
        page_num=1,
        text=f"Text for {chunk_id} pointing {direction}.",
        start_char=0,
        end_char=30,
    )


def make_store() -> FaissVectorStore:
    store = FaissVectorStore(dim=DIM)
    chunks = [make_chunk(k, k) for k in VECS]
    store.add(chunks, list(VECS.values()))
    return store


def make_client() -> MockEmbeddingClient:
    return MockEmbeddingClient(EMBED_CFG, embedding_dim=DIM)


QUESTIONS = [
    Question(question_id="q001", query="north query"),
    Question(question_id="q002", query="east query"),
]


# ---------------------------------------------------------------------------
# Question
# ---------------------------------------------------------------------------


class TestQuestion:
    def test_stores_question_id_and_query(self) -> None:
        q = Question(question_id="fb_001", query="What was the revenue?")
        assert q.question_id == "fb_001"
        assert q.query == "What was the revenue?"

    def test_is_immutable(self) -> None:
        q = Question(question_id="fb_001", query="What was the revenue?")
        with pytest.raises(Exception):
            q.question_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RetrievedChunk
# ---------------------------------------------------------------------------


class TestRetrievedChunk:
    def test_stores_all_fields(self) -> None:
        rc = RetrievedChunk(
            rank=1,
            chunk_id="c001",
            doc_name="DOC_A.pdf",
            page_num=3,
            score=0.85,
            text="Revenue was $123M.",
        )
        assert rc.rank == 1
        assert rc.chunk_id == "c001"
        assert rc.doc_name == "DOC_A.pdf"
        assert rc.page_num == 3
        assert rc.score == 0.85
        assert rc.text == "Revenue was $123M."

    def test_is_immutable(self) -> None:
        rc = RetrievedChunk(rank=1, chunk_id="c001", doc_name="d.pdf",
                            page_num=1, score=0.5, text="t")
        with pytest.raises(Exception):
            rc.rank = 2  # type: ignore[misc]

    def test_to_dict_is_json_serialisable(self) -> None:
        rc = RetrievedChunk(rank=1, chunk_id="c001", doc_name="d.pdf",
                            page_num=1, score=0.5, text="t")
        json.dumps(rc.to_dict())

    def test_to_dict_includes_all_fields(self) -> None:
        rc = RetrievedChunk(rank=2, chunk_id="c002", doc_name="ACME.pdf",
                            page_num=5, score=0.72, text="Net income $45M.")
        d = rc.to_dict()
        assert d["rank"] == 2
        assert d["chunk_id"] == "c002"
        assert d["doc_name"] == "ACME.pdf"
        assert d["page_num"] == 5
        assert d["score"] == 0.72
        assert d["text"] == "Net income $45M."


# ---------------------------------------------------------------------------
# RetrievalRow
# ---------------------------------------------------------------------------


class TestRetrievalRow:
    def test_stores_question_id_query_and_retrieved(self) -> None:
        row = RetrievalRow(
            question_id="q001",
            query="What was revenue?",
            retrieved=[],
        )
        assert row.question_id == "q001"
        assert row.query == "What was revenue?"
        assert row.retrieved == []

    def test_to_dict_includes_all_fields(self) -> None:
        rc = RetrievedChunk(rank=1, chunk_id="c001", doc_name="d.pdf",
                            page_num=1, score=0.9, text="text")
        row = RetrievalRow(question_id="q001", query="Revenue?", retrieved=[rc])
        d = row.to_dict()
        assert d["question_id"] == "q001"
        assert d["query"] == "Revenue?"
        assert len(d["retrieved"]) == 1
        assert d["retrieved"][0]["chunk_id"] == "c001"

    def test_to_dict_is_json_serialisable(self) -> None:
        row = RetrievalRow(question_id="q001", query="Revenue?", retrieved=[])
        json.dumps(row.to_dict())


# ---------------------------------------------------------------------------
# retrieve_for_questions
# ---------------------------------------------------------------------------


class TestRetrieveForQuestions:
    def test_returns_one_row_per_question(self) -> None:
        store = make_store()
        rows = retrieve_for_questions(QUESTIONS, store, make_client(), top_k=2)
        assert len(rows) == len(QUESTIONS)

    def test_each_row_has_correct_question_id(self) -> None:
        store = make_store()
        rows = retrieve_for_questions(QUESTIONS, store, make_client(), top_k=1)
        ids = [r.question_id for r in rows]
        assert ids == ["q001", "q002"]

    def test_each_row_has_query_text(self) -> None:
        store = make_store()
        rows = retrieve_for_questions(QUESTIONS, store, make_client(), top_k=1)
        assert rows[0].query == "north query"
        assert rows[1].query == "east query"

    def test_top_k_limits_retrieved_count(self) -> None:
        store = make_store()
        rows = retrieve_for_questions(QUESTIONS, store, make_client(), top_k=2)
        assert all(len(r.retrieved) == 2 for r in rows)

    def test_retrieved_chunks_have_chunk_ids(self) -> None:
        store = make_store()
        rows = retrieve_for_questions(QUESTIONS, store, make_client(), top_k=1)
        for row in rows:
            assert all(isinstance(rc.chunk_id, str) for rc in row.retrieved)

    def test_retrieved_chunks_have_doc_metadata(self) -> None:
        store = make_store()
        rows = retrieve_for_questions(QUESTIONS, store, make_client(), top_k=1)
        for row in rows:
            for rc in row.retrieved:
                assert rc.doc_name != ""
                assert isinstance(rc.page_num, int)
                assert rc.text != ""

    def test_ranks_start_at_one_and_are_sequential(self) -> None:
        store = make_store()
        rows = retrieve_for_questions(QUESTIONS, store, make_client(), top_k=3)
        for row in rows:
            assert [rc.rank for rc in row.retrieved] == list(range(1, len(row.retrieved) + 1))

    def test_scores_are_between_zero_and_one(self) -> None:
        store = make_store()
        rows = retrieve_for_questions(QUESTIONS, store, make_client(), top_k=4)
        for row in rows:
            assert all(0.0 < rc.score <= 1.0 for rc in row.retrieved)

    def test_scores_decrease_with_rank(self) -> None:
        store = make_store()
        rows = retrieve_for_questions(QUESTIONS, store, make_client(), top_k=4)
        for row in rows:
            scores = [rc.score for rc in row.retrieved]
            assert scores == sorted(scores, reverse=True)

    def test_empty_question_list_returns_empty(self) -> None:
        store = make_store()
        rows = retrieve_for_questions([], store, make_client(), top_k=5)
        assert rows == []

    def test_top_k_larger_than_corpus_returns_all(self) -> None:
        store = make_store()
        rows = retrieve_for_questions([QUESTIONS[0]], store, make_client(), top_k=999)
        assert len(rows[0].retrieved) == store.count


# ---------------------------------------------------------------------------
# run_retrieval
# ---------------------------------------------------------------------------


class TestRunRetrieval:
    def test_writes_jsonl_output(self, tmp_path: Path) -> None:
        output_path = tmp_path / "results.jsonl"
        run_retrieval(QUESTIONS, make_store(), make_client(), output_path, top_k=2)
        assert output_path.is_file()

    def test_output_has_one_line_per_question(self, tmp_path: Path) -> None:
        output_path = tmp_path / "results.jsonl"
        run_retrieval(QUESTIONS, make_store(), make_client(), output_path, top_k=2)
        lines = [l for l in output_path.read_text().splitlines() if l.strip()]
        assert len(lines) == len(QUESTIONS)

    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        output_path = tmp_path / "results.jsonl"
        run_retrieval(QUESTIONS, make_store(), make_client(), output_path, top_k=2)
        for line in output_path.read_text().splitlines():
            if line.strip():
                json.loads(line)

    def test_output_contains_question_ids(self, tmp_path: Path) -> None:
        output_path = tmp_path / "results.jsonl"
        run_retrieval(QUESTIONS, make_store(), make_client(), output_path, top_k=1)
        rows = [json.loads(l) for l in output_path.read_text().splitlines() if l.strip()]
        assert rows[0]["question_id"] == "q001"
        assert rows[1]["question_id"] == "q002"

    def test_output_contains_retrieved_list(self, tmp_path: Path) -> None:
        output_path = tmp_path / "results.jsonl"
        run_retrieval(QUESTIONS, make_store(), make_client(), output_path, top_k=2)
        rows = [json.loads(l) for l in output_path.read_text().splitlines() if l.strip()]
        assert all("retrieved" in row for row in rows)
        assert all(len(row["retrieved"]) == 2 for row in rows)

    def test_returns_result_with_question_count(self, tmp_path: Path) -> None:
        result = run_retrieval(QUESTIONS, make_store(), make_client(),
                               tmp_path / "r.jsonl", top_k=1)
        assert isinstance(result, RetrievalRunResult)
        assert result.question_count == len(QUESTIONS)

    def test_returns_result_with_output_path(self, tmp_path: Path) -> None:
        output_path = tmp_path / "r.jsonl"
        result = run_retrieval(QUESTIONS, make_store(), make_client(),
                               output_path, top_k=1)
        assert result.output_path == output_path

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        output_path = tmp_path / "runs" / "run_001" / "retrieval_results.jsonl"
        run_retrieval(QUESTIONS, make_store(), make_client(), output_path, top_k=1)
        assert output_path.is_file()


# ---------------------------------------------------------------------------
# Helpers shared by M3.10 tests
# ---------------------------------------------------------------------------

SAMPLE_INDEX_META = IndexMetadata(
    embedding_provider="mock",
    embedding_model="mock-embed",
    chunk_size=800,
    chunk_overlap=150,
    min_chunk_chars=0,
    corpus_hash="abc123",
    chunk_count=4,
    build_time_utc="2026-06-11T00:00:00+00:00",
)


def _run_meta_dict(tmp_path: Path, **kwargs) -> dict:
    output = tmp_path / "run" / "retrieval_results.jsonl"
    run_retrieval(QUESTIONS, make_store(), make_client(), output, top_k=1,
                  index_metadata=SAMPLE_INDEX_META, **kwargs)
    return json.loads((tmp_path / "run" / "retrieval_run_metadata.json").read_text())


# ---------------------------------------------------------------------------
# RetrievalRunMetadata
# ---------------------------------------------------------------------------


class TestRetrievalRunMetadata:
    def _make(self, **overrides) -> RetrievalRunMetadata:
        defaults = dict(
            run_id="test-run",
            dataset_path="data/q.jsonl",
            chunks_path="data/chunks.jsonl",
            embedding_model="mock-embed",
            vector_store="faiss",
            chunk_size=800,
            chunk_overlap=150,
            top_k=5,
        )
        return RetrievalRunMetadata(**{**defaults, **overrides})

    def test_stores_run_id(self) -> None:
        assert self._make().run_id == "test-run"

    def test_stores_dataset_path(self) -> None:
        assert self._make().dataset_path == "data/q.jsonl"

    def test_stores_chunks_path(self) -> None:
        assert self._make().chunks_path == "data/chunks.jsonl"

    def test_stores_embedding_model(self) -> None:
        assert self._make().embedding_model == "mock-embed"

    def test_stores_vector_store(self) -> None:
        assert self._make().vector_store == "faiss"

    def test_stores_chunk_size(self) -> None:
        assert self._make().chunk_size == 800

    def test_stores_chunk_overlap(self) -> None:
        assert self._make().chunk_overlap == 150

    def test_stores_top_k(self) -> None:
        assert self._make().top_k == 5

    def test_is_immutable(self) -> None:
        m = self._make()
        with pytest.raises(Exception):
            m.run_id = "other"  # type: ignore[misc]

    def test_to_dict_is_json_serialisable(self) -> None:
        json.dumps(self._make().to_dict())

    def test_to_dict_includes_all_fields(self) -> None:
        d = self._make(run_id="r1", dataset_path="dp", chunks_path="cp",
                       embedding_model="em", vector_store="faiss",
                       chunk_size=512, chunk_overlap=64, top_k=10).to_dict()
        assert d["run_id"] == "r1"
        assert d["dataset_path"] == "dp"
        assert d["chunks_path"] == "cp"
        assert d["embedding_model"] == "em"
        assert d["vector_store"] == "faiss"
        assert d["chunk_size"] == 512
        assert d["chunk_overlap"] == 64
        assert d["top_k"] == 10


# ---------------------------------------------------------------------------
# run_retrieval — metadata file writing
# ---------------------------------------------------------------------------


class TestRunRetrievalMetadata:
    def test_writes_run_metadata_json(self, tmp_path: Path) -> None:
        output = tmp_path / "run" / "retrieval_results.jsonl"
        run_retrieval(QUESTIONS, make_store(), make_client(), output,
                      top_k=1, run_id="r", index_metadata=SAMPLE_INDEX_META)
        assert (tmp_path / "run" / "retrieval_run_metadata.json").is_file()

    def test_metadata_file_is_valid_json(self, tmp_path: Path) -> None:
        d = _run_meta_dict(tmp_path, run_id="r")
        assert isinstance(d, dict)

    def test_metadata_contains_provided_run_id(self, tmp_path: Path) -> None:
        d = _run_meta_dict(tmp_path, run_id="my-custom-run")
        assert d["run_id"] == "my-custom-run"

    def test_auto_generates_run_id_when_not_provided(self, tmp_path: Path) -> None:
        d = _run_meta_dict(tmp_path)
        assert isinstance(d["run_id"], str) and d["run_id"]

    def test_metadata_contains_embedding_model_from_index(self, tmp_path: Path) -> None:
        d = _run_meta_dict(tmp_path, run_id="r")
        assert d["embedding_model"] == "mock-embed"

    def test_metadata_vector_store_is_faiss(self, tmp_path: Path) -> None:
        d = _run_meta_dict(tmp_path, run_id="r")
        assert d["vector_store"] == "faiss"

    def test_metadata_contains_top_k(self, tmp_path: Path) -> None:
        output = tmp_path / "r" / "results.jsonl"
        run_retrieval(QUESTIONS, make_store(), make_client(), output,
                      top_k=7, run_id="r", index_metadata=SAMPLE_INDEX_META)
        d = json.loads((tmp_path / "r" / "retrieval_run_metadata.json").read_text())
        assert d["top_k"] == 7

    def test_metadata_contains_chunk_size_from_index_metadata(self, tmp_path: Path) -> None:
        d = _run_meta_dict(tmp_path, run_id="r")
        assert d["chunk_size"] == 800

    def test_metadata_contains_chunk_overlap_from_index_metadata(self, tmp_path: Path) -> None:
        d = _run_meta_dict(tmp_path, run_id="r")
        assert d["chunk_overlap"] == 150

    def test_metadata_contains_dataset_path(self, tmp_path: Path) -> None:
        d = _run_meta_dict(tmp_path, run_id="r", dataset_path="data/q.jsonl")
        assert d["dataset_path"] == "data/q.jsonl"

    def test_metadata_contains_chunks_path(self, tmp_path: Path) -> None:
        d = _run_meta_dict(tmp_path, run_id="r", chunks_path="data/chunks.jsonl")
        assert d["chunks_path"] == "data/chunks.jsonl"

    def test_result_metadata_path_points_to_written_file(self, tmp_path: Path) -> None:
        output = tmp_path / "run" / "retrieval_results.jsonl"
        result = run_retrieval(QUESTIONS, make_store(), make_client(), output,
                               top_k=1, run_id="r", index_metadata=SAMPLE_INDEX_META)
        assert result.metadata_path is not None
        assert result.metadata_path.is_file()

    def test_result_metadata_path_is_none_without_index_metadata(self, tmp_path: Path) -> None:
        output = tmp_path / "r.jsonl"
        result = run_retrieval(QUESTIONS, make_store(), make_client(), output, top_k=1)
        assert result.metadata_path is None


# ---------------------------------------------------------------------------
# next_run_dir
# ---------------------------------------------------------------------------


class TestNextRunDir:
    def test_returns_run_001_when_runs_dir_empty(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        runs.mkdir()
        assert next_run_dir(runs) == runs / "run_001"

    def test_returns_run_001_when_runs_dir_does_not_exist(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        assert next_run_dir(runs) == runs / "run_001"

    def test_increments_after_run_001(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        (runs / "run_001").mkdir(parents=True)
        assert next_run_dir(runs) == runs / "run_002"

    def test_increments_after_highest_existing(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        for name in ("run_001", "run_002", "run_005"):
            (runs / name).mkdir(parents=True)
        assert next_run_dir(runs) == runs / "run_006"

    def test_ignores_non_run_directories(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        (runs / "misc").mkdir(parents=True)
        (runs / "run_003").mkdir()
        assert next_run_dir(runs) == runs / "run_004"

    def test_numbers_are_zero_padded_to_three_digits(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        (runs / "run_009").mkdir(parents=True)
        result = next_run_dir(runs)
        assert result.name == "run_010"

    def test_handles_triple_digits(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        (runs / "run_099").mkdir(parents=True)
        assert next_run_dir(runs).name == "run_100"
