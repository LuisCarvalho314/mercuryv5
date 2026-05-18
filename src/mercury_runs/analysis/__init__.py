__all__ = [
    "StudyDataset",
    "generate_bmu_attribution_plots",
    "generate_method_graph_plots",
    "load_mercury_study_dataset",
]


def __getattr__(name: str):
    if name == "generate_bmu_attribution_plots":
        from .bmu_attribution_plots import generate_bmu_attribution_plots

        return generate_bmu_attribution_plots
    if name == "generate_method_graph_plots":
        from .graph_plots import generate_method_graph_plots

        return generate_method_graph_plots
    if name in {"StudyDataset", "load_mercury_study_dataset"}:
        from .study_dataset import StudyDataset, load_mercury_study_dataset

        return {"StudyDataset": StudyDataset, "load_mercury_study_dataset": load_mercury_study_dataset}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
