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
from financebench_eval_harness.eval_retrieval import (
    format_retrieval_failure_report,
    generate_retrieval_report,
    score_retrieval_run,
)
from financebench_eval_harness.rag_run import RAGRunError, run_rag_from_config
from financebench_eval_harness.rag_run_config import (
    DEFAULT_RAG_RUN_CONFIG_PATH,
    RAGRunConfigError,
    load_rag_run_config,
)
from financebench_eval_harness.analysis import (
    join_retrieval_and_answer_scores,
    summarize_joined_metrics,
)
from financebench_eval_harness.rag_report import (
    RagReportError,
    generate_rag_report,
)
from financebench_eval_harness.rag_score import RAGScoreError, score_rag_run
from financebench_eval_harness.rag_score_config import (
    DEFAULT_RAG_SCORE_CONFIG_PATH,
    RAGScoreConfigError,
    RAGScoreSettings,
    load_rag_score_config,
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

    p_eval = subparsers.add_parser(
        "eval-retrieval",
        help="Score a completed retrieval run and write a markdown report.",
    )
    p_eval.add_argument(
        "--config",
        type=Path,
        default=Path("configs/retrieval.yaml"),
        metavar="PATH",
        help="Retrieval pipeline config YAML (default: configs/retrieval.yaml)",
    )
    p_eval.add_argument(
        "--run-id",
        required=True,
        metavar="RUN_ID",
        help="Run ID to evaluate (subdirectory of runs_dir, e.g. run_001)",
    )
    p_eval.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        metavar="DIR",
        help="Directory to write the markdown report (default: reports/)",
    )

    run_rag_parser = subparsers.add_parser(
        "run-rag",
        help="Generate RAG answers from retrieved chunks.",
    )
    run_rag_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RAG_RUN_CONFIG_PATH,
        help="RAG run config YAML path.",
    )
    run_rag_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional deterministic run directory name.",
    )
    run_rag_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Limit number of examples for smoke tests.",
    )

    p_inspect_failure = subparsers.add_parser(
        "inspect-retrieval-failure",
        help="Print detailed retrieval inspection for a single question from a scored run.",
    )
    p_inspect_failure.add_argument(
        "--config",
        type=Path,
        default=Path("configs/retrieval.yaml"),
        metavar="PATH",
        help="Retrieval pipeline config YAML (default: configs/retrieval.yaml)",
    )
    p_inspect_failure.add_argument(
        "--run-id",
        required=True,
        metavar="RUN_ID",
        help="Run ID to inspect (subdirectory of runs_dir).",
    )
    p_inspect_failure.add_argument(
        "--question-id",
        required=True,
        metavar="QUESTION_ID",
        help="Question ID to inspect.",
    )

    report_rag_parser = subparsers.add_parser(
        "report-rag",
        help="Generate end-to-end RAG evaluation Markdown report.",
    )
    report_rag_parser.add_argument(
        "--joined-dir",
        type=Path,
        required=True,
        help="Directory containing joined_metrics.jsonl and joined_summary.json.",
    )
    report_rag_parser.add_argument(
        "--rag-run-dir",
        type=Path,
        default=None,
        help="Optional RAG run directory for model metadata and rich examples.",
    )
    report_rag_parser.add_argument(
        "--retrieval-summary",
        type=Path,
        default=None,
        help="Optional path to retrieval_summary.json for retrieval hit rates.",
    )
    report_rag_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier used in the report title and filename.",
    )
    report_rag_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write the Markdown report (default: reports/).",
    )

    join_metrics_parser = subparsers.add_parser(
        "join-metrics",
        help="Join retrieval scores with answer scores per question.",
    )
    join_metrics_parser.add_argument(
        "--retrieval-scores",
        type=Path,
        required=True,
        help="Path to retrieval_scores.jsonl produced by eval-retrieval.",
    )
    join_metrics_parser.add_argument(
        "--answer-scores",
        type=Path,
        required=True,
        help="Path to scores.jsonl produced by run-rag.",
    )
    join_metrics_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write joined_metrics.jsonl and joined_summary.json.",
    )
    join_metrics_parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="k value for hit@k retrieval signal (default: 5).",
    )

    p_inspect_rag = subparsers.add_parser(
        "inspect-rag",
        help="Inspect one RAG result end-to-end for a single question.",
    )
    p_inspect_rag.add_argument(
        "--run",
        type=Path,
        required=True,
        metavar="RAG_RUN_DIR",
        help="RAG run directory (contains rag_predictions.jsonl and scores.jsonl).",
    )
    p_inspect_rag.add_argument(
        "--question-id",
        required=True,
        metavar="QUESTION_ID",
        help="Question ID to inspect.",
    )
    p_inspect_rag.add_argument(
        "--retrieval-run",
        type=Path,
        default=None,
        metavar="RETRIEVAL_RUN_DIR",
        help="Retrieval run directory for chunk text and gold evidence (optional; auto-resolved from metadata if omitted).",
    )
    p_inspect_rag.add_argument(
        "--joined-dir",
        type=Path,
        default=None,
        help="Analysis directory with joined_metrics.jsonl for failure labels (optional).",
    )

    score_rag_parser = subparsers.add_parser(
        "score-rag",
        help="Score existing RAG predictions: lexical metrics, answer verdict, grounding verdict.",
    )
    score_rag_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RAG_SCORE_CONFIG_PATH,
        help="RAG score config YAML path.",
    )
    score_rag_parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Override run_dir from config (directory with rag_predictions.jsonl).",
    )

    grounding_parser = subparsers.add_parser(
        "analyze-grounding",
        help="Run M7 hallucination and grounding analysis on a completed RAG run.",
    )
    grounding_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/grounding_analysis.yaml"),
        help="Grounding analysis config YAML.",
    )
    grounding_parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for the generated Markdown report.",
    )
    grounding_parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID for the report filename (inferred from run_dir if omitted).",
    )

    inspect_failure_parser = subparsers.add_parser(
        "inspect-failure",
        help="Inspect hallucination and grounding details for one question.",
    )
    inspect_failure_parser.add_argument(
        "--run",
        type=Path,
        required=True,
        metavar="RUN_DIR",
        help="Path to the run directory containing grounding analysis outputs.",
    )
    inspect_failure_parser.add_argument(
        "--question-id",
        required=True,
        metavar="QUESTION_ID",
        help="Question ID to inspect.",
    )

    cmp_parser = subparsers.add_parser(
        "compare-embeddings",
        help="Run retrieval comparison across multiple embedding models (M6).",
    )
    cmp_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/embedding_comparison.yaml"),
        help="Embedding comparison config YAML.",
    )
    cmp_parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for the generated Markdown report.",
    )
    cmp_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate config and list models without running embeddings.",
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

    if args.command == "eval-retrieval":
        try:
            pipeline_cfg = load_pipeline_config(args.config)
        except PipelineConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        run_dir = pipeline_cfg.runs_dir / args.run_id
        results_path = run_dir / "retrieval_results.jsonl"
        if not results_path.is_file():
            print(f"Retrieval results not found: {results_path}", file=sys.stderr)
            return 1

        summary = score_retrieval_run(pipeline_cfg, run_dir)

        report_path = generate_retrieval_report(
            summary, args.run_id, pipeline_cfg, output_dir=args.report_dir, run_dir=run_dir
        )

        print(f"Scored {summary['example_count']} questions.")
        print(f"Retrieval report: {report_path}")
        return 0

    if args.command == "inspect-retrieval-failure":
        try:
            pipeline_cfg = load_pipeline_config(args.config)
        except PipelineConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        run_dir = pipeline_cfg.runs_dir / args.run_id
        for p in (run_dir / "retrieval_results.jsonl", run_dir / "retrieval_scores.jsonl"):
            if not p.is_file():
                print(f"File not found: {p}", file=sys.stderr)
                return 1
        try:
            report = format_retrieval_failure_report(pipeline_cfg, run_dir, args.question_id)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(report)
        return 0

    if args.command == "run-rag":
        try:
            result = _run_configured_rag_evaluation(
                config_path=args.config,
                run_id=args.run_id,
                limit=args.limit,
            )
        except (RAGRunConfigError, RAGRunError, LLMConfigError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"RAG run output: {result.output_dir}")
        print(f"Wrote config snapshot to {result.config_path}")
        print(f"Wrote {result.example_count} predictions to {result.predictions_path}")
        print(f"Wrote {result.example_count} score rows to {result.scores_path}")
        print(f"Wrote run metadata to {result.run_metadata_path}")
        print(f"Attempted: {result.attempted_count}")
        print(f"Succeeded: {result.success_count}")
        print(f"Errors: {result.error_count}")
        return 0

    if args.command == "report-rag":
        try:
            result = generate_rag_report(
                args.joined_dir,
                rag_run_dir=args.rag_run_dir,
                retrieval_summary_path=args.retrieval_summary,
                output_dir=args.output_dir,
                run_id=args.run_id,
            )
        except (RagReportError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"RAG report: {result.report_path}")
        print(f"Questions evaluated: {result.example_count}")
        return 0

    if args.command == "join-metrics":
        if not args.retrieval_scores.is_file():
            print(f"Retrieval scores file not found: {args.retrieval_scores}", file=sys.stderr)
            return 1
        if not args.answer_scores.is_file():
            print(f"Answer scores file not found: {args.answer_scores}", file=sys.stderr)
            return 1

        retrieval_scores = _read_jsonl_file(args.retrieval_scores)
        answer_scores = _read_jsonl_file(args.answer_scores)
        rows = join_retrieval_and_answer_scores(retrieval_scores, answer_scores, k=args.k)
        summary = summarize_joined_metrics(rows)

        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        joined_path = output_dir / "joined_metrics.jsonl"
        summary_path = output_dir / "joined_summary.json"

        with joined_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print(f"Joined {summary['example_count']} questions.")
        print(f"  retrieval_hit_answer_correct:  {summary['retrieval_hit_answer_correct']}")
        print(f"  retrieval_hit_answer_wrong:    {summary['retrieval_hit_answer_wrong']}")
        print(f"  retrieval_miss_answer_correct: {summary['retrieval_miss_answer_correct']}")
        print(f"  retrieval_miss_answer_wrong:   {summary['retrieval_miss_answer_wrong']}")
        print(f"Wrote joined metrics to {joined_path}")
        print(f"Wrote summary to {summary_path}")
        return 0

    if args.command == "inspect-rag":
        from financebench_eval_harness.rag_inspect import (
            RagInspectError,
            format_rag_inspection,
            load_rag_inspection,
        )

        try:
            result = load_rag_inspection(
                args.run,
                args.question_id,
                retrieval_run_dir=args.retrieval_run,
                joined_dir=args.joined_dir,
            )
        except RagInspectError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(format_rag_inspection(result))
        return 0

    if args.command == "score-rag":
        try:
            score_config = load_rag_score_config(args.config)
        except RAGScoreConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        if args.run_dir is not None:
            from dataclasses import replace as _replace
            score_config = _replace(
                score_config,
                settings=_replace(score_config.settings, run_dir=args.run_dir),
            )

        answer_judge_client = _build_score_judge_client(score_config.answer_judge)
        grounding_judge_client = _build_score_judge_client(score_config.grounding_judge)

        try:
            result = score_rag_run(
                score_config,
                answer_judge_client=answer_judge_client,
                grounding_judge_client=grounding_judge_client,
            )
        except RAGScoreError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"score-rag output: {result.output_dir}")
        print(f"Scored {result.scored_count}/{result.example_count} examples ({result.error_count} errors)")
        print(f"  answer scores:   {result.answer_scores_path}")
        print(f"  grounding scores: {result.grounding_scores_path}")
        print(f"  combined scores:  {result.combined_scores_path}")
        return 0

    if args.command == "compare-embeddings":
        from financebench_eval_harness.embedding_comparison_config import (
            EmbeddingComparisonConfigError,
            load_embedding_comparison_config,
        )
        from financebench_eval_harness.embedding_comparison import run_embedding_comparison

        try:
            cmp_config = load_embedding_comparison_config(args.config)
        except EmbeddingComparisonConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        if args.dry_run:
            import yaml as _yaml
            print(f"[dry-run] {len(cmp_config.embedding_models)} models to compare:")
            for spec in cmp_config.embedding_models:
                print(f"  {spec.provider}/{spec.name} ({spec.category})")
            print(f"[dry-run] Fixed chunks: {cmp_config.retrieval.chunks_path}")
            print(f"[dry-run] top_k={cmp_config.retrieval.top_k}")
            # Write stub run dir so the report generator has something to read.
            dry_run_dir = cmp_config.runs_dir / cmp_config.run_id
            dry_run_dir.mkdir(parents=True, exist_ok=True)
            (dry_run_dir / "embedding_leaderboard.json").write_text("[]", encoding="utf-8")
            (dry_run_dir / "embedding_decision.json").write_text("{}", encoding="utf-8")
            (dry_run_dir / "config.yaml").write_text(
                _yaml.dump({
                    "run_id": cmp_config.run_id,
                    "dry_run": True,
                    "chunks_path": str(cmp_config.retrieval.chunks_path),
                    "questions_path": str(cmp_config.retrieval.questions_path),
                    "top_k": cmp_config.retrieval.top_k,
                    "evidence_overlap_threshold": cmp_config.retrieval.evidence_overlap_threshold,
                    "embedding_models": [
                        {"name": s.name, "provider": s.provider, "category": s.category}
                        for s in cmp_config.embedding_models
                    ],
                }),
                encoding="utf-8",
            )
            try:
                from financebench_eval_harness.embedding_comparison_report import (
                    generate_embedding_comparison_report,
                )
                report_path = generate_embedding_comparison_report(
                    dry_run_dir, output_dir=args.report_dir, dry_run=True
                )
                print(f"[dry-run] Report:   {report_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"[dry-run] Warning: report generation failed: {exc}", file=sys.stderr)
            return 0

        result = _run_embedding_comparison_with_progress(cmp_config, run_embedding_comparison)

        _print_comparison_summary(result, top_k=cmp_config.retrieval.top_k)

        try:
            from financebench_eval_harness.embedding_comparison_report import (
                generate_embedding_comparison_report,
            )
            report_path = generate_embedding_comparison_report(
                result.run_dir, output_dir=args.report_dir
            )
            print(f"Comparison report: {report_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: report generation failed: {exc}", file=sys.stderr)

        return 0 if result.failed_count == 0 else 2

    if args.command == "analyze-grounding":
        from financebench_eval_harness.grounding_analysis import analyze_grounding
        from financebench_eval_harness.grounding_analysis_config import (
            GroundingAnalysisConfigError,
            load_grounding_analysis_config,
        )
        from financebench_eval_harness.hallucination_report import generate_hallucination_report

        try:
            config = load_grounding_analysis_config(args.config)
        except GroundingAnalysisConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"Analyzing grounding for run: {config.settings.run_dir}")
        try:
            result = analyze_grounding(config)
        except Exception as exc:  # noqa: BLE001
            print(f"Grounding analysis failed: {exc}", file=sys.stderr)
            return 1

        run_id = args.run_id or config.settings.run_dir.name
        print(f"Grounding scores:  {result.grounding_scores_path}")
        print(f"Citation scores:   {result.citation_scores_path}")
        print(f"Failure analysis:  {result.failure_analysis_path}")
        print(f"Failure summary:   {result.failure_summary_path}")

        try:
            report_result = generate_hallucination_report(
                result.failure_analysis_path,
                result.failure_summary_path,
                run_id=run_id,
                output_dir=args.report_dir,
            )
            print(f"Report:            {report_result.report_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: report generation failed: {exc}", file=sys.stderr)

        return 0

    if args.command == "inspect-failure":
        from financebench_eval_harness.failure_inspector import (
            format_failure_inspection,
            load_failure_inspection,
        )

        result = load_failure_inspection(args.run, args.question_id)
        print(format_failure_inspection(result))
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


def _run_configured_rag_evaluation(
    *,
    config_path: Path,
    run_id: str | None,
    limit: int | None,
):
    rag_config = load_rag_run_config(config_path)
    effective_limit = limit if limit is not None else 1000
    llm_client = _build_rag_llm_client(rag_config, limit=effective_limit)
    judge_client = _build_rag_judge_client(rag_config, limit=effective_limit)
    return run_rag_from_config(
        rag_config,
        llm_client,
        judge_client=judge_client,
        run_id=run_id,
        limit=limit,
    )


def _build_rag_llm_client(config, *, limit: int) -> LLMClient:
    if config.model.provider == "mock":
        return MockLLMClient(config.model, responses=["mock response"] * limit)
    if config.model.provider == "ollama":
        return OllamaLLMClient(config.model)
    raise LLMConfigError(f"Unsupported LLM provider: {config.model.provider}")


def _build_score_judge_client(judge_config) -> LLMClient | None:
    if judge_config is None:
        return None
    if judge_config.model.provider == "mock":
        return MockLLMClient(
            judge_config.model,
            responses=['{"verdict": "correct", "reason": "Mock."}'] * 1000,
        )
    if judge_config.model.provider == "ollama":
        return OllamaLLMClient(judge_config.model)
    raise LLMConfigError(f"Unsupported judge LLM provider: {judge_config.model.provider}")


def _build_rag_judge_client(config, *, limit: int) -> LLMClient | None:
    if config.judge is None:
        return None
    if config.judge.model.provider == "mock":
        return MockLLMClient(
            config.judge.model,
            responses=['{"verdict": "incorrect", "reason": "Mock judge response."}'] * limit,
        )
    if config.judge.model.provider == "ollama":
        return OllamaLLMClient(config.judge.model)
    raise LLMConfigError(f"Unsupported judge LLM provider: {config.judge.model.provider}")


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


def _run_embedding_comparison_with_progress(cmp_config, run_embedding_comparison):
    """Run compare-embeddings with a rich startup panel and live progress bars."""
    try:
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
        from rich.text import Text
    except ImportError:
        return run_embedding_comparison(cmp_config)

    console = Console()

    n = len(cmp_config.embedding_models)
    model_names = ", ".join(s.name for s in cmp_config.embedding_models)
    info = Text()
    info.append(f"  {n} model{'s' if n != 1 else ''}  ·  ")
    info.append(model_names, style="cyan")
    info.append(f"\n  chunks     {cmp_config.retrieval.chunks_path}")
    info.append(f"\n  questions  {cmp_config.retrieval.questions_path}")
    info.append(f"\n  top_k={cmp_config.retrieval.top_k}  ·  fail_fast={cmp_config.fail_fast}")
    console.print(Panel(info, title="Embedding Comparison", border_style="blue"))

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        model_task = progress.add_task("Models", total=n)
        chunk_task = progress.add_task("Chunks", total=1, visible=False)

        def on_event(event: str, info: dict) -> None:
            if event == "model_start":
                progress.update(
                    model_task,
                    description=f"[{info['index']}/{info['total']}] {info['model']}",
                )
                progress.update(chunk_task, visible=False, completed=0, total=1)
            elif event in ("model_done", "model_failed"):
                progress.advance(model_task)
            elif event == "embed_start":
                total = info["total_to_embed"]
                hits = info["cache_hits"]
                progress.update(
                    chunk_task,
                    description=f"  Embedding  ({hits} cached)",
                    total=total,
                    completed=0,
                    visible=total > 0,
                )
            elif event == "embed_progress":
                progress.update(chunk_task, completed=info["completed"])

        return run_embedding_comparison(cmp_config, on_event=on_event)


def _print_comparison_summary(result, *, top_k: int = 10) -> None:
    """Print post-run summary table."""
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print(f"Comparison run: {result.run_dir}")
        print(f"Models succeeded: {result.succeeded_count}")
        print(f"Models failed: {result.failed_count}")
        if result.failed_models:
            for name, err in result.failed_models.items():
                print(f"  FAILED {name}: {err}", file=sys.stderr)
        return

    console = Console()
    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Provider")
    table.add_column("Category")
    table.add_column("Hit@10", justify="right")
    table.add_column("Embed latency", justify="right")
    table.add_column("Cost", justify="right")

    for mr in result.model_results:
        spec = mr.spec
        if mr.succeeded:
            hit = mr.summary.get(f"evidence_text_hit@{top_k}_rate", 0.0) if mr.summary else 0.0
            latency = f"{mr.embedding_latency_s:.1f}s" if mr.embedding_latency_s else "cached"
            cost = f"${mr.estimated_cost_usd:.4f}" if mr.estimated_cost_usd else "free"
            table.add_row(spec.name, spec.provider, spec.category, f"{hit:.3f}", latency, cost)
        else:
            table.add_row(
                f"[red]{spec.name}[/red]",
                spec.provider,
                spec.category,
                "[red]FAILED[/red]",
                "—",
                "—",
            )

    console.print()
    console.print(table)
    console.print(f"Run directory: [dim]{result.run_dir}[/dim]")
    if result.failed_models:
        for name, err in result.failed_models.items():
            console.print(f"  [red]✗[/red] {name}: {err}")
    console.print()


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
    if config.provider == "sentence_transformers":
        from financebench_eval_harness.embedding import SentenceTransformersEmbeddingClient
        return SentenceTransformersEmbeddingClient(config)
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
