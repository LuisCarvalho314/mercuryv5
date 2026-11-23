# transCSSR evaluation-walk pilot

This is an exploratory analysis of existing cached evaluation walks. It does not
regenerate walks or enter the benchmark suite. Walk 0 trains each model; walks
1 and 2 are filtered independently. The fixed cache uses canonical-v4,
cardinal-distance range 1, random-action probability 0.3, evaluation seed 0,
10,000 steps per walk, and `valid_trajectories_only=false`.

## Reproduce

The run used upstream [`ddarmon/transCSSR`](https://github.com/ddarmon/transCSSR)
commit `23a53ba21101cf26bdfcb83570d60bb6147d07ba` (token-first v1.1.0).
Install it outside the project environment, then expose only the isolated site
packages to the project interpreter for the analysis script:

```sh
UV_CACHE_DIR=/private/tmp/transcssr-uvcache uv venv --python 3.13 /private/tmp/transcssr-pilot-venv
UV_CACHE_DIR=/private/tmp/transcssr-uvcache uv pip install --python /private/tmp/transcssr-pilot-venv/bin/python \
  'git+https://github.com/ddarmon/transCSSR@23a53ba21101cf26bdfcb83570d60bb6147d07ba' scikit-learn
PYTHONPATH=/private/tmp/transcssr-pilot-venv/lib/python3.13/site-packages:. \
  MPLCONFIGDIR=/private/tmp/transcssr-mpl .venv/bin/python scripts/analysis/transcssr_pilot/run.py
```

The project dependency files are untouched. `results/current/transcssr_pilot/`
contains the cache identity and token maps, serialized fitted models, held-out
state paths, and `summary.json` with metrics and pairwise comparisons.

## Semantics

Each action and cardinal-distance vector is mapped to a distinct integer token;
the maps are saved with the results. Integer encoding is necessary because the
upstream filter treats a sequence of tuple tokens as an ensemble. The Cartesian
control uses the cache's exact physical-state ID as its output token. Physical
IDs are never passed to a cardinal-distance fit. The input and output alphabets
are fixed from the selected walks before fitting so the filter can encode every
held-out token; this exposes token *identities*, not held-out transitions or
labels, to training.

The cache stores the position and observation **after** each action. At step
`t`, transCSSR's filtered state predicts that step's output using the action at
`t` and the preceding input/output history. Thus physical and preceding
observation relationships compare filtered state `t` with cache row `t-1`.
Row 0 has no cached predecessor and is omitted from those scores. The current
output relationship separately compares state `t` with observation row `t`.
Labels `-1` are unsynchronized; they contribute to the synchronization rate
but are excluded from fixed-map precision, recall, and adjusted Rand scores.

The fixed-map scores call the project's existing
`compute_fixed_map_precision_details`: precision is the sum of each inferred
state's largest physical overlap divided by synchronized steps; recall is the
sum of each physical state's largest inferred overlap divided by synchronized
steps. A low physical recall can reflect a sensible merge when physical states
have equivalent observable futures. Compare model stability and cooccurrence
structure alongside physical-state scores.
