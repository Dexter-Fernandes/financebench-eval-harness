import financebench_eval_harness.data as data


def test_data_module_reexports_public_api() -> None:
    public_names = [
        "DocumentExtractionError",
        "DocumentExtractionFailure",
        "DocumentExtractionResult",
        "DocumentPage",
        "DocumentPageLoadError",
        "DocumentRegistryValidationResult",
        "EvidencePageCheck",
        "EvidencePageValidationResult",
        "FinanceBenchDataLayout",
        "FinanceBenchEvidence",
        "FinanceBenchExample",
        "FinanceBenchQuestionLoadError",
        "MissingFinanceBenchDataError",
        "ProcessedExamplesBuildResult",
        "ProcessedFinanceBenchExampleLoadError",
        "build_document_registry",
        "build_processed_financebench_examples",
        "canonical_pdf_name",
        "extract_document_pages",
        "extract_financebench_documents",
        "load_financebench_examples",
        "load_processed_financebench_examples",
        "validate_financebench_data_layout",
        "validate_financebench_dataset",
        "validate_financebench_document_registry",
        "validate_financebench_evidence_pages",
    ]

    for name in public_names:
        assert hasattr(data, name), f"missing public data API: {name}"
