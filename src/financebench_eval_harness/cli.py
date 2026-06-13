from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from dataclasses import replace
import sys
from pathlib import Path
from typing import Sequence

from financebench_eval_harness.chunking import chunk_pages
from financebench_eval_harness.inspection import InspectionError, format_inspection, load_inspection
from financebench_eval_harness.retriever import Question, next_run_dir, run_retrieval
from financebench_eval_harness.embedding import (
    EmbeddingConfig,
    EmbeddingConfigError,
    MockEmbeddingClient,
    OllamaEmbeddingClient,
    load_embedding_config,
)
from financebench_eval_harness.index_builder import IndexBuildError, build_index, load_index
from financebench_eval_harness.query_embedder import QueryEmbeddingError, embed_question
from financebench_eval_harness.pipeline_config import PipelineConfigError, load_pipeline_config
from financebench_eval_harness.retrieval_config import RetrievalConfigError, RetrievalConfig, load_retrieval_config
from financebench_eval_harness.retrieval_types import Chunk, DocumentPage as RetrievalDocumentPage
from financebench_eval_harness.config import (
    DEFAULT_DATASET_CONFIG_PATH,
    DatasetConfig,
    DatasetConfigError,
    load_dataset_config,
)
from financebench_eval_harness.data import (
    DocumentExtractionError,
    DocumentPageLoadError,
    EvidencePageCheck,
    FinanceBenchQuestionLoadError,
    MissingFinanceBenchDataError,
    build_processed_financebench_examples,
    extract_financebench_documents,
    load_financebench_examples,
    validate_financebench_evidence_pages,
    validate_financebench_document_registry,
    validate_financebench_dataset,
    validate_financebench_data_layout,
)
from financebench_eval_harness.llm import (
    LLMClient,
    LLMConfigError,
    MockLLMClient,
    OllamaLLMClient,
)
from financebench_eval_harness.run import EvaluationRunError, run_evaluation_from_config
from financebench_eval_harness.run_config import (
    DEFAULT_EVALUATION_RUN_CONFIG_PATH,
    EvaluationRunConfigError,
    load_evaluation_run_config,
)
from financebench_eval_harness.report import (
    DEFAULT_REPORT_OUTPUT_DIR,
    BaselineReportError,
    generate_baseline_report,
)


DEFAULT_BASELINE_RUN_CONFIG_PATH = Path("configs/baseline_closed_book.yaml")
DEFAULT_BASELINE_REPORT_OUTPUT_DIR = Path("reports")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="financebench-harness",
        description="Utilities for the FinanceBench evaluation harness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-data",
        help="Validate the expected local FinanceBench data layout.",
    )
    validate_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DATASET_CONFIG_PATH,
        help="Dataset config YAML path.",
    )
    validate_parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory containing questions.jsonl and documents/.",
    )

    dataset_parser = subparsers.add_parser(
        "validate-dataset",
        help="Validate the FinanceBench question schema.",
    )
    _add_dataset_path_arguments(dataset_parser)

    documents_parser = subparsers.add_parser(
        "validate-documents",
        help="Validate FinanceBench evidence document file coverage.",
    )
    _add_dataset_path_arguments(documents_parser)

    extract_parser = subparsers.add_parser(
        "extract-documents",
        help="Extract local FinanceBench PDF text page by page.",
    )
    _add_dataset_path_arguments(extract_parser)

    evidence_pages_parser = subparsers.add_parser(
        "validate-evidence-pages",
        help="Validate FinanceBench evidence text against extracted document pages.",
    )
    _add_dataset_path_arguments(evidence_pages_parser)

    build_examples_parser = subparsers.add_parser(
        "build-examples",
        help="Build canonical processed FinanceBench examples JSONL.",
    )
    _add_dataset_path_arguments(build_examples_parser)

    run_eval_parser = subparsers.add_parser(
        "run-eval",
        help="Run a configured baseline evaluation.",
    )
    run_eval_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EVALUATION_RUN_CONFIG_PATH,
        help="Evaluation run config YAML path.",
    )
    run_eval_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional deterministic run directory name.",
    )
    run_eval_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Override the configured example limit for smoke tests.",
    )

    run_baseline_parser = subparsers.add_parser(
        "run-baseline",
        help="Run a baseline evaluation and generate its Markdown report.",
    )
    run_baseline_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_BASELINE_RUN_CONFIG_PATH,
        help="Baseline run config YAML path.",
    )
    run_baseline_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional deterministic run directory name.",
    )
    run_baseline_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Override the configured example limit for smoke tests.",
    )
    run_baseline_parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_BASELINE_REPORT_OUTPUT_DIR,
        help="Directory where the generated Markdown report should be written.",
    )

    report_parser = subparsers.add_parser(
        "report-baseline",
        help="Generate a Markdown baseline report for an evaluation run.",
    )
    report_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Evaluation run directory containing run_metadata.json, predictions.jsonl, and scores.jsonl.",
    )
    report_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_OUTPUT_DIR,
        help="Directory where the generated Markdown report should be written.",
    )

    chunk_doc_parser = subparsers.add_parser(
        "chunk-documents",
        help="Chunk extracted pages into text chunks and write a chunks JSONL.",
    )
    chunk_doc_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Unified pipeline config YAML (configs/retrieval.yaml).",
    )

    build_index_parser = subparsers.add_parser(
        "build-index",
        help="Chunk extracted pages and build a FAISS vector index.",
    )
    build_index_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Unified pipeline config YAML; when given, reads pre-chunked data and overrides individual flags.",
    )
    build_index_parser.add_argument(
        "--retrieval-config",
        type=Path,
        default=Path("configs/retrieval/recursive_text_800.yaml"),
        help="Retrieval config YAML (chunking strategy and settings).",
    )
    build_index_parser.add_argument(
        "--embedding-config",
        type=Path,
        default=Path("configs/embedding/ollama_nomic.yaml"),
        help="Embedding config YAML (provider, model, batch size).",
    )
    build_index_parser.add_argument(
        "--pages",
        type=Path,
        default=None,
        help="JSONL file of extracted pages. Required unless --config is given.",
    )
    build_index_parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/indexes/financebench"),
        help="Directory where the index artefacts will be written.",
    )
    build_index_parser.add_argument(
        "--progress",
        action="store_true",
        default=False,
        help="Show a rich progress bar while embedding chunks (requires rich).",
    )
    build_index_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate inputs and report what would be built without writing any files.",
    )

    embed_question_parser = subparsers.add_parser(
        "embed-question",
        help="Embed a question text using the configured embedding model.",
    )
    embed_question_parser.add_argument(
        "--question",
        required=True,
        help="Question text to embed.",
    )
    embed_question_parser.add_argument(
        "--embedding-config",
        type=Path,
        default=Path("configs/embedding/ollama_nomic.yaml"),
        help="Embedding config YAML (provider, model, batch size).",
    )

    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Retrieve top-k chunks for each question and write JSONL results.",
    )
    retrieve_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Unified pipeline config YAML; when given, overrides individual flags.",
    )
    retrieve_parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/indexes/financebench"),
        help="Directory containing the built FAISS index.",
    )
    retrieve_parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="JSONL file of questions (question_id + question fields). Required unless --config is given.",
    )
    retrieve_parser.add_argument(
        "--embedding-config",
        type=Path,
        default=Path("configs/embedding/ollama_nomic.yaml"),
        help="Embedding config YAML (must match the model used to build the index).",
    )
    retrieve_parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory where numbered run subdirectories are created (default: runs/).",
    )
    retrieve_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output JSONL path; overrides --runs-dir auto-numbering.",
    )
    retrieve_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question.",
    )
    retrieve_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier written into run_metadata.json (auto-generated if omitted).",
    )
    retrieve_parser.add_argument(
        "--chunks-path",
        type=Path,
        default=None,
        help="Path to the chunks JSONL used to build the index (recorded in run metadata).",
    )

    inspect_parser = subparsers.add_parser(
        "inspect-retrieval",
        help="Inspect retrieved chunks for one question from a completed run.",
    )
    inspect_parser.add_argument(
        "--question-id",
        required=True,
        help="Question ID to inspect.",
    )
    inspect_parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Run directory containing retrieval_results.jsonl.",
    )
    inspect_parser.add_argument(
        "--examples",
        type=Path,
        default=None,
        help="Examples JSONL path (default: read dataset_path from run_metadata.json).",
    )
    inspect_parser.add_argument(
        "--preview-chars",
        type=int,
        default=300,
        help="Characters of chunk text to show per result (default: 300).",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-data":
        try:
            dataset_config = _resolve_dataset_config(args.config, args.data_root)
            layout = validate_financebench_data_layout(dataset_config)
            examples = load_financebench_examples(dataset_config)
        except (
            DatasetConfigError,
            FinanceBenchQuestionLoadError,
            MissingFinanceBenchDataError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"FinanceBench data layout is valid: {layout.root}")
        print(f"Loaded {len(examples)} FinanceBench examples.")
        return 0

    if args.command == "validate-dataset":
        try:
            dataset_config = _resolve_dataset_config(args.config, args.data_root)
            result = validate_financebench_dataset(dataset_config)
        except (DatasetConfigError, FinanceBenchQuestionLoadError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"Valid examples: {result.valid_count}")
        print(f"Invalid examples: {result.invalid_count}")
        if result.invalid_count:
            print("Dataset schema validation failed.")
            for issue in result.issues:
                print(issue.format())
            return 1

        print("Dataset schema validation passed.")
        return 0

    if args.command == "validate-documents":
        try:
            dataset_config = _resolve_dataset_config(args.config, args.data_root)
            result = validate_financebench_document_registry(dataset_config)
        except (DatasetConfigError, FinanceBenchQuestionLoadError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"Resolved documents: {len(result.resolved_documents)}")
        print(f"Missing documents: {len(result.missing_documents)}")
        print(f"Unused documents: {len(result.unused_documents)}")

        if result.missing_documents:
            print("Document registry validation failed.")
            for filename in result.missing_documents:
                print(f"missing {filename}")
            return 1

        if result.unused_documents:
            print("Unused local documents:")
            for filename in result.unused_documents:
                print(f"unused {filename}")

        print("Document registry validation passed.")
        return 0

    if args.command == "extract-documents":
        try:
            dataset_config = _resolve_dataset_config(args.config, args.data_root)
            result = extract_financebench_documents(
                dataset_config,
                on_document_start=lambda path: print(f"Extracting {path.name}"),
            )
        except (DatasetConfigError, DocumentExtractionError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"Extracted {result.document_count} documents.")
        print(f"Wrote {result.page_count} pages to {result.output_path}.")
        print(f"Extraction failures: {result.failure_count}")
        if result.failure_count:
            print(f"Failure details written to {result.failures_path}.")
        return 0

    if args.command == "validate-evidence-pages":
        try:
            dataset_config = _resolve_dataset_config(args.config, args.data_root)
            result = validate_financebench_evidence_pages(dataset_config)
        except (
            DatasetConfigError,
            DocumentPageLoadError,
            FinanceBenchQuestionLoadError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        for check in result.checks:
            print(_format_evidence_page_check(check))

        print(f"Total evidence checks: {result.total_count}")
        print(f"Matches: {result.matched_count}")
        print(f"Mismatches: {result.mismatch_count}")

        if result.is_valid:
            print("Evidence page validation passed.")
            return 0

        print("Evidence page validation failed.")
        return 1

    if args.command == "build-examples":
        try:
            dataset_config = _resolve_dataset_config(args.config, args.data_root)
            result = build_processed_financebench_examples(dataset_config)
        except (
            DatasetConfigError,
            DocumentPageLoadError,
            FinanceBenchQuestionLoadError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"Accepted examples: {result.accepted_count}")
        print(f"Rejected examples: {result.rejected_count}")
        for reason, count in result.skip_reason_counts.items():
            print(f"{reason}: {count}")
        print(f"Wrote accepted examples to {result.output_path.resolve()}.")
        print(f"Wrote rejected examples to {result.rejected_path.resolve()}.")
        return 0

    if args.command == "run-eval":
        try:
            result = _run_configured_evaluation(
                config_path=args.config,
                run_id=args.run_id,
                limit=args.limit,
            )
        except (
            EvaluationRunConfigError,
            EvaluationRunError,
            LLMConfigError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"Evaluation run output: {result.output_dir}")
        print(f"Wrote config snapshot to {result.config_path}")
        print(f"Wrote {result.example_count} predictions to {result.predictions_path}")
        print(f"Wrote {result.example_count} score rows to {result.scores_path}")
        print(f"Wrote run metadata to {result.run_metadata_path}")
        print(f"Attempted: {result.attempted_count}")
        print(f"Succeeded: {result.success_count}")
        print(f"Errors: {result.error_count}")
        run_metadata = json.loads(result.run_metadata_path.read_text(encoding="utf-8"))
        judge_summary = run_metadata["judge_summary"]
        print(f"Judge attempted: {judge_summary['attempted_count']}")
        print(f"Judge succeeded: {judge_summary['success_count']}")
        print(f"Judge errors: {judge_summary['error_count']}")
        return 0

    if args.command == "run-baseline":
        try:
            result = _run_configured_evaluation(
                config_path=args.config,
                run_id=args.run_id,
                limit=args.limit,
            )
            report = generate_baseline_report(
                result.output_dir,
                output_dir=args.report_dir,
                report_filename=f"baseline_{result.output_dir.name}.md",
            )
            _record_report_path(result.run_metadata_path, report.report_path)
            run_metadata = json.loads(
                result.run_metadata_path.read_text(encoding="utf-8")
            )
            judge_summary = run_metadata["judge_summary"]
        except (
            BaselineReportError,
            EvaluationRunConfigError,
            EvaluationRunError,
            LLMConfigError,
            OSError,
            KeyError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"Baseline run output: {result.output_dir}")
        print(f"Wrote config snapshot to {result.config_path}")
        print(f"Wrote {result.example_count} predictions to {result.predictions_path}")
        print(f"Wrote {result.example_count} score rows to {result.scores_path}")
        print(f"Wrote run metadata to {result.run_metadata_path}")
        print(f"Baseline report: {report.report_path}")
        print(f"Attempted: {result.attempted_count}")
        print(f"Succeeded: {result.success_count}")
        print(f"Errors: {result.error_count}")
        print(f"Correct: {judge_summary['correct_count']}")
        print(f"Partially correct: {judge_summary['partially_correct_count']}")
        print(f"Incorrect: {judge_summary['incorrect_count']}")
        return 0

    if args.command == "report-baseline":
        try:
            result = generate_baseline_report(
                args.run_dir,
                output_dir=args.output_dir,
            )
            run_metadata = json.loads(
                (args.run_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            judge_summary = run_metadata["judge_summary"]
        except (BaselineReportError, OSError, KeyError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"Baseline report: {result.report_path}")
        print(f"Questions evaluated: {result.evaluated_count}")
        print(f"Correct: {judge_summary['correct_count']}")
        print(f"Partially correct: {judge_summary['partially_correct_count']}")
        print(f"Incorrect: {judge_summary['incorrect_count']}")
        return 0

    if args.command == "chunk-documents":
        try:
            pipeline_cfg = load_pipeline_config(args.config)
        except PipelineConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        if not pipeline_cfg.pages_path.is_file():
            print(f"Pages file not found: {pipeline_cfg.pages_path}", file=sys.stderr)
            return 1

        raw_pages = _read_jsonl_file(pipeline_cfg.pages_path)
        pages = [
            RetrievalDocumentPage(
                doc_id=row["doc_name"].removesuffix(".pdf"),
                doc_name=row["doc_name"],
                page_num=row["page_num"],
                text=row["text"],
            )
            for row in raw_pages
        ]
        chunks = chunk_pages(pages, pipeline_cfg.chunking)

        if not chunks:
            print("No chunks produced — pages file may be empty.", file=sys.stderr)
            return 1

        pipeline_cfg.chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with pipeline_cfg.chunks_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(dataclasses.asdict(chunk), ensure_ascii=False) + "\n")

        print(f"Chunked {len(pages)} pages into {len(chunks)} chunks.")
        print(f"Chunks written to {pipeline_cfg.chunks_path}.")
        return 0

    if args.command == "build-index":
        if args.config is not None:
            try:
                pipeline_cfg = load_pipeline_config(args.config)
            except PipelineConfigError as exc:
                print(str(exc), file=sys.stderr)
                return 1

            if not pipeline_cfg.chunks_path.is_file():
                print(f"Chunks file not found: {pipeline_cfg.chunks_path}", file=sys.stderr)
                return 1

            raw_chunks = _read_jsonl_file(pipeline_cfg.chunks_path)
            chunks = [Chunk(**row) for row in raw_chunks]

            if not chunks:
                print("No chunks found in chunks file — run chunk-documents first.", file=sys.stderr)
                return 1

            embedding_config = pipeline_cfg.embedding
            retrieval_config = RetrievalConfig(chunking=pipeline_cfg.chunking, evidence_overlap_threshold=pipeline_cfg.evidence_overlap_threshold)
            embedding_client = _build_embedding_client(embedding_config)

            try:
                meta = _build_index_with_progress(
                    chunks, embedding_client, retrieval_config, pipeline_cfg.index_dir,
                    show_progress=args.progress,
                )
            except IndexBuildError as exc:
                print(str(exc), file=sys.stderr)
                return 1

            print(f"Indexed {meta.chunk_count} chunks.")
            print(f"Embedding model: {meta.embedding_provider}/{meta.embedding_model}")
            print(f"Corpus hash: {meta.corpus_hash}")
            print(f"Built index at {pipeline_cfg.index_dir}.")
            return 0

        if args.pages is None:
            print("error: --pages is required when --config is not given.", file=sys.stderr)
            return 1

        try:
            retrieval_config = load_retrieval_config(args.retrieval_config)
        except (RetrievalConfigError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        try:
            embedding_config = load_embedding_config(args.embedding_config)
        except EmbeddingConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        if not args.pages.is_file():
            print(f"Pages file not found: {args.pages}", file=sys.stderr)
            return 1

        raw_pages = _read_jsonl_file(args.pages)
        pages = [
            RetrievalDocumentPage(
                doc_id=row["doc_name"].removesuffix(".pdf"),
                doc_name=row["doc_name"],
                page_num=row["page_num"],
                text=row["text"],
            )
            for row in raw_pages
        ]
        chunks = chunk_pages(pages, retrieval_config.chunking)

        if not chunks:
            print("No chunks produced from pages file — file may be empty.", file=sys.stderr)
            return 1

        if args.dry_run:
            print(f"[dry-run] {len(pages)} pages → {len(chunks)} chunks")
            print(f"[dry-run] Embedding model: {embedding_config.provider}/{embedding_config.model_name}")
            print(f"[dry-run] Index would be written to: {args.index_dir}")
            print("[dry-run] No files written.")
            return 0

        embedding_client = _build_embedding_client(embedding_config)

        try:
            meta = _build_index_with_progress(
                chunks, embedding_client, retrieval_config, args.index_dir,
                show_progress=args.progress,
            )
        except IndexBuildError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"Chunked {len(pages)} pages into {meta.chunk_count} chunks.")
        print(f"Embedding model: {meta.embedding_provider}/{meta.embedding_model}")
        print(f"Corpus hash: {meta.corpus_hash}")
        print(f"Built index at {args.index_dir}.")
        return 0

    if args.command == "embed-question":
        try:
            embedding_config = load_embedding_config(args.embedding_config)
        except EmbeddingConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        embedding_client = _build_embedding_client(embedding_config)
        try:
            vec = embed_question(args.question, embedding_client)
        except QueryEmbeddingError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"Embedding model: {embedding_config.provider}/{embedding_config.model_name}")
        print(f"Embedding dimension: {len(vec)}")
        print(f"Vector preview: {[round(v, 4) for v in vec[:4]]}...")
        return 0

    if args.command == "retrieve":
        if args.config is not None:
            try:
                pipeline_cfg = load_pipeline_config(args.config)
            except PipelineConfigError as exc:
                print(str(exc), file=sys.stderr)
                return 1

            try:
                store, index_meta = load_index(pipeline_cfg.index_dir)
            except IndexBuildError as exc:
                print(str(exc), file=sys.stderr)
                return 1

            if not pipeline_cfg.questions_path.is_file():
                print(f"Questions file not found: {pipeline_cfg.questions_path}", file=sys.stderr)
                return 1

            raw = _read_jsonl_file(pipeline_cfg.questions_path)
            questions = [
                Question(
                    question_id=row["question_id"],
                    query=row.get("query") or row["question"],
                )
                for row in raw
            ]

            run_dir = next_run_dir(pipeline_cfg.runs_dir)
            output_path = run_dir / "retrieval_results.jsonl"
            embedding_client = _build_embedding_client(pipeline_cfg.embedding)
            result = run_retrieval(
                questions,
                store,
                embedding_client,
                output_path,
                top_k=pipeline_cfg.top_k,
                dataset_path=str(pipeline_cfg.questions_path),
                chunks_path=str(pipeline_cfg.chunks_path),
                index_metadata=index_meta,
            )
            shutil.copy(args.config, run_dir / "config.yaml")

            print(f"Retrieved top-{pipeline_cfg.top_k} chunks for {result.question_count} questions.")
            print(f"Results written to {result.output_path}.")
            if result.metadata_path:
                print(f"Run metadata written to {result.metadata_path}.")
            print(f"Config snapshot written to {run_dir / 'config.yaml'}.")
            return 0

        if args.questions is None:
            print("error: --questions is required when --config is not given.", file=sys.stderr)
            return 1

        try:
            store, index_meta = load_index(args.index_dir)
        except IndexBuildError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        if not args.questions.is_file():
            print(f"Questions file not found: {args.questions}", file=sys.stderr)
            return 1

        try:
            embedding_config = load_embedding_config(args.embedding_config)
        except EmbeddingConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        raw = _read_jsonl_file(args.questions)
        questions = [
            Question(
                question_id=row["question_id"],
                query=row.get("query") or row["question"],
            )
            for row in raw
        ]

        if args.output is not None:
            output_path = args.output
        else:
            run_dir = next_run_dir(args.runs_dir)
            output_path = run_dir / "retrieval_results.jsonl"

        embedding_client = _build_embedding_client(embedding_config)
        result = run_retrieval(
            questions,
            store,
            embedding_client,
            output_path,
            top_k=args.top_k,
            run_id=args.run_id,
            dataset_path=str(args.questions),
            chunks_path=str(args.chunks_path) if args.chunks_path else None,
            index_metadata=index_meta,
        )

        print(f"Retrieved top-{args.top_k} chunks for {result.question_count} questions.")
        print(f"Results written to {result.output_path}.")
        if result.metadata_path:
            print(f"Run metadata written to {result.metadata_path}.")
        return 0

    if args.command == "inspect-retrieval":
        try:
            inspection = load_inspection(
                args.question_id,
                args.run,
                examples_path=args.examples,
            )
        except InspectionError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(format_inspection(inspection, preview_chars=args.preview_chars))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _add_dataset_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DATASET_CONFIG_PATH,
        help="Dataset config YAML path.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory containing questions.jsonl and documents/.",
    )


def _resolve_dataset_config(config_path: Path, data_root: Path | None) -> DatasetConfig:
    if data_root is not None:
        return DatasetConfig.from_data_root(data_root)
    return load_dataset_config(config_path)


def _run_configured_evaluation(
    *,
    config_path: Path,
    run_id: str | None,
    limit: int | None,
):
    run_config = load_evaluation_run_config(config_path)
    if limit is not None:
        run_config = replace(
            run_config,
            settings=replace(run_config.settings, limit=limit),
        )
    llm_client = _build_llm_client(run_config)
    judge_client = _build_judge_client(run_config)
    return run_evaluation_from_config(
        run_config,
        llm_client,
        judge_client=judge_client,
        run_id=run_id,
    )


def _record_report_path(run_metadata_path: Path, report_path: Path) -> None:
    metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise BaselineReportError(
            f"Run metadata must be a JSON object: {run_metadata_path}"
        )
    metadata["report_path"] = str(report_path)
    run_metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _build_llm_client(run_config) -> LLMClient:
    if run_config.model.provider == "mock":
        return MockLLMClient(
            run_config.model,
            responses=["mock response"] * run_config.settings.limit,
        )
    if run_config.model.provider == "ollama":
        return OllamaLLMClient(run_config.model)
    raise LLMConfigError(f"Unsupported LLM provider: {run_config.model.provider}")


def _build_judge_client(run_config) -> LLMClient | None:
    if run_config.judge is None:
        return None
    if run_config.judge.model.provider == "mock":
        return MockLLMClient(
            run_config.judge.model,
            responses=[
                '{"verdict": "incorrect", "reason": "Mock judge response."}'
            ]
            * run_config.settings.limit,
        )
    if run_config.judge.model.provider == "ollama":
        return OllamaLLMClient(run_config.judge.model)
    raise LLMConfigError(
        f"Unsupported judge LLM provider: {run_config.judge.model.provider}"
    )


def _build_index_with_progress(chunks, embedding_client, retrieval_config, index_dir, *, show_progress: bool = False):
    """Call build_index, optionally with a rich progress bar."""
    if not show_progress:
        return build_index(chunks, embedding_client, retrieval_config, index_dir)

    try:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Embedding chunks…", total=len(chunks))

            def on_batch(completed: int, total: int) -> None:
                progress.update(task, completed=completed)

            return build_index(chunks, embedding_client, retrieval_config, index_dir, on_batch=on_batch)

    except ImportError:
        print("Progress bar requires 'rich': pip install rich", file=sys.stderr)
        return build_index(chunks, embedding_client, retrieval_config, index_dir)


def _build_embedding_client(config: EmbeddingConfig):
    if config.provider == "mock":
        return MockEmbeddingClient(config)
    if config.provider == "ollama":
        return OllamaEmbeddingClient(config)
    from financebench_eval_harness.embedding import EmbeddingProviderError
    raise EmbeddingProviderError(f"Unsupported embedding provider: {config.provider}")


def _read_jsonl_file(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _positive_int(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed_value


def _format_evidence_page_check(check: EvidencePageCheck) -> str:
    status = "MATCH" if check.is_match else "MISMATCH"
    document_path = str(check.document_path) if check.document_path is not None else "missing"
    return (
        f"{status} {check.question_id} evidence[{check.evidence_index}] "
        f"doc={check.document_filename} "
        f"path={document_path} "
        f"evidence_page={check.evidence_page_num} "
        f"extracted_page={check.extracted_page_num} "
        f"reason={check.reason} "
        f"method={check.match_method} "
        f"evidence={_quote_excerpt(check.evidence_excerpt)} "
        f"page={_quote_excerpt(check.page_excerpt)}"
    )


def _quote_excerpt(text: str) -> str:
    safe_text = text.replace('"', "'")
    return f'"{safe_text}"'


if __name__ == "__main__":
    raise SystemExit(main())
