# Oracle Moore-machine audit

This standalone utility reads privileged physical states from cached evaluation
walks and certifies the deterministic Moore machines of levels 13, 20, 21, and
22. It is not part of training, model evaluation, or the benchmark comparison
pipeline. Its results are in [`RESULTS.json`](RESULTS.json) and the short
[`REPORT.md`](REPORT.md). A separate check of the 2,000-step datasets and all
10,000-step evaluation walks is in [`SAMPLE_CHECKS.md`](SAMPLE_CHECKS.md) and
[`SAMPLE_CHECKS.json`](SAMPLE_CHECKS.json).

## Reproduce

Use Java 17 or newer, Maven, and the existing project Python environment:

```sh
PYTHONPATH=. .venv/bin/python scripts/analysis/automatalib_moore_audit/run.py \
  --maven /path/to/mvn
PYTHONPATH=. .venv/bin/python scripts/analysis/automatalib_moore_audit/check_samples.py \
  --maven /path/to/mvn
```

The isolated `pom.xml` pins AutomataLib 13.0.0. Maven build files are created
in a temporary directory and dependencies are cached under the system temporary
directory. No project dependency file or learning pipeline is changed.

The Python extractor reads walks 0–2 from the existing
`ambiguity_response_ablation` canonical-v4 cache. It checks each action vector,
physical state, collision, observation, and transition against the maze layout;
requires every physical state and all four actions per state; then gives the
complete table to the Java runner. The Java runner uses AutomataLib's
`CompactMoore`, `HopcroftMinimizer.minimizeUniversal` with `DONT_PRUNE`, and
`Automata.findSeparatingWord` between physical states. It checks that the
pairwise equivalence classes match the minimized count. Python replays every
separating word against the full table before saving results.

The cache records states and observations *after* each action. The audit uses
row `t-1` as the source of action `t`, and omits row 0 from transition inference
because its pre-action state is unavailable. Row 0 can still establish its
post-action state's observation. Action IDs are north=0, east=1, south=2,
west=3. Physical IDs are row-major open-position IDs from `LEVEL.py`.

`check_samples.py` scans every existing 2,000-step, cardinal-distance range-1,
all-step dataset on the four levels. Since these parquet files contain no
physical-state path, it replays actions from the privileged initial position
recorded in metadata and checks each result against the certified machine. It
also checks all 100 cached 10,000-step evaluation walks per level. Coverage is
reported per trajectory. For each complete sample, it builds the full table
from that sample's rows, deduplicates identical tables, and runs AutomataLib
on every distinct table. Each complete sample receives its minimized count,
physical-state classes, and separating words. An incomplete sample is not
treated as a complete machine even when every observed row agrees with the
oracle.

The benchmark report also includes a global characterizing set `W` produced
by AutomataLib's `Automata.characterizingSet`, which calls its established
`CharacterizingSets` machinery. The audit replays every word from every pair
of physical states and records which words distinguish each observation-aliased
pair. The empty word compares the Moore machine's current state outputs.
`Automata.stateCharacterizingSet` is a separate per-state API and is not needed
for this global certificate. AutomataLib's directly named
`findShortestSeparatingWord` API compares two automata; the audit keeps its
existing state-pair separating-word call and does not add a shortest-word
analysis.
