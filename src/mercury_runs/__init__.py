__all__ = [
    "ArtifactLayout",
    "CSCGConfig",
    "MercuryConfig",
    "MercuryCSCGHybridConfig",
    "PIPELINE_SCHEMA_VERSION",
    "POCMLConfig",
    "RunConfigModel",
    "RunIdentity",
    "RunStatusModel",
    "SensoryConfig",
    "StudyDefaults",
    "StudyGridConfig",
    "SubsystemStatus",
    "compute_cscg_paper_precision",
    "compute_cscg_precision",
    "compute_mercury_paper_precision",
    "compute_mercury_precision",
    "compute_pocml_paper_precision",
    "compute_pocml_precision",
    "compute_sensory_paper_precision_metrics",
    "evaluate_pocml_sequence",
    "execute_pipeline_run",
    "generate_mercury_datasets",
    "main",
    "parse_arguments",
    "run_cscg",
    "run_mercury",
    "run_mercury_cscg_hybrid",
    "run_pocml",
    "run_sensory",
    "run_single",
    "run_study",
]


def __getattr__(name: str):
    if name in {
        "CSCGConfig",
        "MercuryConfig",
        "MercuryCSCGHybridConfig",
        "POCMLConfig",
        "SensoryConfig",
        "compute_cscg_paper_precision",
        "compute_cscg_precision",
        "compute_mercury_paper_precision",
        "compute_mercury_precision",
        "compute_pocml_paper_precision",
        "compute_pocml_precision",
        "compute_sensory_paper_precision_metrics",
        "evaluate_pocml_sequence",
        "generate_mercury_datasets",
        "run_cscg",
        "run_mercury",
        "run_mercury_cscg_hybrid",
        "run_pocml",
        "run_sensory",
    }:
        from . import algorithms

        return getattr(algorithms, name)
    if name in {"execute_pipeline_run", "run_single", "run_study"}:
        from . import application

        return getattr(application, name)
    if name in {
        "ArtifactLayout",
        "PIPELINE_SCHEMA_VERSION",
        "RunConfigModel",
        "RunIdentity",
        "RunStatusModel",
        "StudyDefaults",
        "StudyGridConfig",
        "SubsystemStatus",
    }:
        from . import domain

        return getattr(domain, name)
    if name in {"main", "parse_arguments"}:
        from . import interfaces

        return getattr(interfaces, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
