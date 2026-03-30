"""
Hyperparameter optimisation analysis (within-level):
1) Permutation importance per level for predicting latent_precision_mean
2) Top-K rows per level by:
   - predicted mean (from a per-level model)
   - observed mean (ground-truth best in the data)
3) Multi-objective selection per level:
   - Pareto front for (max mean, min coeff_var, min node_count_mean)
   - A simple scalar tradeoff score (z-scored within each level)

Outputs:
- permutation_importance_within_level.csv
- best_predicted_per_level.csv
- best_observed_per_level.csv
- pareto_per_level.csv
- best_tradeoff_per_level.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------
# Config
# --------------------------
parquet_path = Path("/Users/lcarv/PycharmProjects/mercuryv5/results/studies"
                    "/study2/study_summary.parquet")

group_column_name = "level"

target_mean_column_name = "latent_precision_mean"
target_coeffvar_column_name = "latent_precision_coeff_var"
target_nodecount_column_name = "latent_node_count_mean"

feature_column_names: List[str] = [
    "ambiguity_threshold",
    "trace_decay",
    "mixture_alpha",
    "mixture_beta",
    "memory_replay",
    "memory_disambiguation",
    "am_lr",
]

minimum_rows_per_level = 10
test_size = 0.25
random_seed = 0

n_estimators = 500
min_samples_leaf = 2
permutation_repeats = 10
top_k_importances = 20

top_k_predicted = 5
top_k_observed = 5
top_k_tradeoff = 5

# Tradeoff weights: bigger => penalise more
coeffvar_weight = 1.0
nodecount_weight = 1.0


# --------------------------
# Helpers
# --------------------------
def build_pipeline_for_frame(feature_frame: pd.DataFrame) -> Pipeline:
    categorical_feature_names = [c for c in feature_frame.columns if feature_frame[c].dtype == "object"]
    numeric_feature_names = [c for c in feature_frame.columns if c not in categorical_feature_names]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_feature_names),
            ("numeric", "passthrough", numeric_feature_names),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_seed,
        n_jobs=-1,
        min_samples_leaf=min_samples_leaf,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def get_post_ohe_feature_names(pipeline: Pipeline, original_feature_names: List[str]) -> List[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    categorical_feature_names = []
    numeric_feature_names = []

    # Recover which were treated as categorical vs numeric
    for name in original_feature_names:
        if name in preprocessor.feature_names_in_:
            # This is stable enough for this use case
            pass

    # We can infer from the fitted transformer
    categorical_transformer = None
    for transformer_name, transformer, column_names in preprocessor.transformers_:
        if transformer_name == "categorical":
            categorical_transformer = transformer
            categorical_feature_names = list(column_names)
        elif transformer_name == "numeric":
            numeric_feature_names = list(column_names)

    if categorical_transformer is not None and len(categorical_feature_names) > 0:
        encoded_categorical_names = list(categorical_transformer.get_feature_names_out(categorical_feature_names))
    else:
        encoded_categorical_names = []

    return encoded_categorical_names + numeric_feature_names


def pareto_front_max_mean_min_cv_min_nodes(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Keep non-dominated points for:
      - maximise target_mean_column_name
      - minimise target_coeffvar_column_name
      - minimise target_nodecount_column_name

    Implementation: O(n^2) but n is small per level here.
    """
    values = frame[[target_mean_column_name, target_coeffvar_column_name, target_nodecount_column_name]].to_numpy()
    keep_mask = np.ones(len(frame), dtype=bool)

    for i in range(len(frame)):
        if not keep_mask[i]:
            continue
        mean_i, cv_i, nodes_i = values[i]

        for j in range(len(frame)):
            if i == j or not keep_mask[i]:
                continue
            mean_j, cv_j, nodes_j = values[j]

            dominates = (
                (mean_j >= mean_i)
                and (cv_j <= cv_i)
                and (nodes_j <= nodes_i)
                and (
                    (mean_j > mean_i)
                    or (cv_j < cv_i)
                    or (nodes_j < nodes_i)
                )
            )
            if dominates:
                keep_mask[i] = False

    return frame.loc[keep_mask].copy()


def z_score(series: pd.Series) -> pd.Series:
    denom = series.std(ddof=0)
    if denom == 0 or np.isnan(denom):
        return series * 0.0
    return (series - series.mean()) / (denom + 1e-12)


# --------------------------
# Load data
# --------------------------
dataframe = pq.read_table(str(parquet_path)).to_pandas()

required_columns = (
    [group_column_name]
    + feature_column_names
    + [target_mean_column_name, target_coeffvar_column_name, target_nodecount_column_name]
)
missing_columns = [c for c in required_columns if c not in dataframe.columns]
if missing_columns:
    raise KeyError(f"Missing columns in parquet: {missing_columns}")

dataframe = dataframe[required_columns].copy()

# Drop rows missing any objective (features can have NaNs; we handle by dropping rows used in fitting)
dataframe = dataframe.dropna(subset=[group_column_name, target_mean_column_name, target_coeffvar_column_name, target_nodecount_column_name])


# --------------------------
# 1) Permutation importance per level
# --------------------------
importance_rows: List[pd.DataFrame] = []

# --------------------------
# 2) Top-K predicted per level (fit per-level model)
#    and Top-K observed per level (no model)
# --------------------------
best_predicted_rows: List[pd.DataFrame] = []
best_observed_rows: List[pd.DataFrame] = []

# --------------------------
# 3) Pareto + tradeoff per level
# --------------------------
pareto_rows: List[pd.DataFrame] = []
tradeoff_rows: List[pd.DataFrame] = []

for level_value, level_frame in dataframe.groupby(group_column_name):
    level_frame = level_frame.copy()

    if len(level_frame) < minimum_rows_per_level:
        importance_rows.append(
            pd.DataFrame(
                {
                    group_column_name: [level_value],
                    "feature": ["<skipped>"],
                    "perm_importance_mean": [np.nan],
                    "perm_importance_std": [np.nan],
                    "n_rows": [len(level_frame)],
                    "note": ["too_few_rows"],
                }
            )
        )
        continue

    # Deduplicate for output tables only; keep full data for modeling
    level_frame_dedup = level_frame.drop_duplicates(
        subset=feature_column_names + [target_mean_column_name, target_coeffvar_column_name, target_nodecount_column_name]
    )

    # --------------------------
    # Fit model for mean
    # --------------------------
    feature_frame = level_frame[feature_column_names]
    target_series = level_frame[target_mean_column_name]

    # Drop rows with missing feature values for modeling
    modeling_mask = ~feature_frame.isna().any(axis=1)
    modeling_frame = level_frame.loc[modeling_mask].copy()

    if len(modeling_frame) < minimum_rows_per_level:
        importance_rows.append(
            pd.DataFrame(
                {
                    group_column_name: [level_value],
                    "feature": ["<skipped>"],
                    "perm_importance_mean": [np.nan],
                    "perm_importance_std": [np.nan],
                    "n_rows": [len(modeling_frame)],
                    "note": ["too_many_feature_nans"],
                }
            )
        )
        continue

    X = modeling_frame[feature_column_names]
    y = modeling_frame[target_mean_column_name]

    pipeline = build_pipeline_for_frame(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    pipeline.fit(X_train, y_train)

    permutation = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=permutation_repeats,
        random_state=random_seed,
        n_jobs=-1,
    )

    post_ohe_feature_names = get_post_ohe_feature_names(pipeline, feature_column_names)

    importance_frame = (
        pd.DataFrame(
            {
                group_column_name: level_value,
                "feature": post_ohe_feature_names,
                "perm_importance_mean": permutation.importances_mean,
                "perm_importance_std": permutation.importances_std,
                "n_rows": len(modeling_frame),
            }
        )
        .sort_values("perm_importance_mean", ascending=False)
        .head(top_k_importances)
    )
    importance_rows.append(importance_frame)

    # --------------------------
    # Predicted best (using per-level model)
    # --------------------------
    modeling_frame["predicted_mean"] = pipeline.predict(modeling_frame[feature_column_names])

    predicted_rank_frame = (
        modeling_frame[
            [group_column_name]
            + feature_column_names
            + [target_mean_column_name, target_coeffvar_column_name, target_nodecount_column_name, "predicted_mean"]
        ]
        .drop_duplicates(
            subset=feature_column_names
            + [target_mean_column_name, target_coeffvar_column_name, target_nodecount_column_name, "predicted_mean"]
        )
        .dropna(subset=["predicted_mean"])
        .sort_values("predicted_mean", ascending=False)
        .head(top_k_predicted)
        .copy()
    )
    best_predicted_rows.append(predicted_rank_frame)

    # --------------------------
    # Observed best (ground truth in this dataset)
    # --------------------------
    observed_rank_frame = (
        level_frame_dedup[
            [group_column_name]
            + feature_column_names
            + [target_mean_column_name, target_coeffvar_column_name, target_nodecount_column_name]
        ]
        .sort_values([target_mean_column_name, target_coeffvar_column_name, target_nodecount_column_name],
                     ascending=[False, True, True])
        .head(top_k_observed)
        .copy()
    )
    best_observed_rows.append(observed_rank_frame)

    # --------------------------
    # Pareto front (3 objectives)
    # --------------------------
    pareto_input_frame = level_frame_dedup.dropna(
        subset=[target_mean_column_name, target_coeffvar_column_name, target_nodecount_column_name]
    ).copy()

    pareto_frame = pareto_front_max_mean_min_cv_min_nodes(pareto_input_frame)
    pareto_frame = pareto_frame[
        [group_column_name]
        + feature_column_names
        + [target_mean_column_name, target_coeffvar_column_name, target_nodecount_column_name]
    ].sort_values(
        [target_mean_column_name, target_coeffvar_column_name, target_nodecount_column_name],
        ascending=[False, True, True],
    )
    pareto_rows.append(pareto_frame)

    # --------------------------
    # Tradeoff score (z-scored within level)
    # --------------------------
    tradeoff_frame = pareto_input_frame.copy()
    tradeoff_frame["mean_z"] = z_score(tradeoff_frame[target_mean_column_name])
    tradeoff_frame["coeffvar_z"] = z_score(tradeoff_frame[target_coeffvar_column_name])
    tradeoff_frame["nodecount_z"] = z_score(tradeoff_frame[target_nodecount_column_name])

    tradeoff_frame["tradeoff_score"] = (
        tradeoff_frame["mean_z"]
        - coeffvar_weight * tradeoff_frame["coeffvar_z"]
        - nodecount_weight * tradeoff_frame["nodecount_z"]
    )

    tradeoff_best_frame = (
        tradeoff_frame[
            [group_column_name]
            + feature_column_names
            + [target_mean_column_name, target_coeffvar_column_name, target_nodecount_column_name, "tradeoff_score"]
        ]
        .sort_values("tradeoff_score", ascending=False)
        .head(top_k_tradeoff)
        .copy()
    )
    tradeoff_rows.append(tradeoff_best_frame)


# --------------------------
# Concatenate + save outputs
# --------------------------
all_importances = pd.concat(importance_rows, ignore_index=True)
all_importances.to_csv("permutation_importance_within_level.csv", index=False)

predicted_columns = (
    [group_column_name]
    + feature_column_names
    + [target_mean_column_name, target_coeffvar_column_name, target_nodecount_column_name, "predicted_mean"]
)
observed_columns = (
    [group_column_name]
    + feature_column_names
    + [target_mean_column_name, target_coeffvar_column_name, target_nodecount_column_name]
)
tradeoff_columns = (
    [group_column_name]
    + feature_column_names
    + [target_mean_column_name, target_coeffvar_column_name, target_nodecount_column_name, "tradeoff_score"]
)

best_predicted_per_level = (
    pd.concat(best_predicted_rows, ignore_index=True)
    if best_predicted_rows
    else pd.DataFrame(columns=predicted_columns)
)
best_predicted_per_level.to_csv("best_predicted_per_level.csv", index=False)

best_observed_per_level = (
    pd.concat(best_observed_rows, ignore_index=True)
    if best_observed_rows
    else pd.DataFrame(columns=observed_columns)
)
best_observed_per_level.to_csv("best_observed_per_level.csv", index=False)

pareto_per_level = (
    pd.concat(pareto_rows, ignore_index=True)
    if pareto_rows
    else pd.DataFrame(columns=observed_columns)
)
pareto_per_level.to_csv("pareto_per_level.csv", index=False)

best_tradeoff_per_level = (
    pd.concat(tradeoff_rows, ignore_index=True)
    if tradeoff_rows
    else pd.DataFrame(columns=tradeoff_columns)
)
best_tradeoff_per_level.to_csv("best_tradeoff_per_level.csv", index=False)


# --------------------------
# Print summaries
# --------------------------
for level_value in sorted(dataframe[group_column_name].unique()):
    level_importances = all_importances[all_importances[group_column_name] == level_value]
    if len(level_importances) > 0:
        print(f"\n=== level={level_value} | permutation importance (predicting {target_mean_column_name}) ===")
        print(
            level_importances[
                ["feature", "perm_importance_mean", "perm_importance_std", "n_rows"]
            ].to_string(index=False)
        )

    level_predicted = best_predicted_per_level[best_predicted_per_level[group_column_name] == level_value]
    if len(level_predicted) > 0:
        print(f"\n=== level={level_value} | top {top_k_predicted} predicted (max predicted_mean) ===")
        print(
            level_predicted.drop(columns=[group_column_name]).to_string(index=False)
        )

    level_observed = best_observed_per_level[best_observed_per_level[group_column_name] == level_value]
    if len(level_observed) > 0:
        print(f"\n=== level={level_value} | top {top_k_observed} observed (max mean, min coeffvar, min nodes) ===")
        print(
            level_observed.drop(columns=[group_column_name]).to_string(index=False)
        )

    level_tradeoff = best_tradeoff_per_level[best_tradeoff_per_level[group_column_name] == level_value]
    if len(level_tradeoff) > 0:
        print(f"\n=== level={level_value} | top {top_k_tradeoff} tradeoff (max score) ===")
        print(
            level_tradeoff.drop(columns=[group_column_name]).to_string(index=False)
        )
