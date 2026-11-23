
import json
from collections import defaultdict
from pathlib import Path

from scripts.analysis.plot_selected_transition_residual_maps import CSCG_PRESETS

ROOT = Path(
    "results/current/studies/"
    "mercury_cscg_hybrid_vs_cscg_batch_size_30_seed_common_eval_walks2"
)

LEVELS = (13, 20, 21)
SEEDS = range(30)
RAND_PROBS = (0.0, 0.3)
BATCHES = (1, 10, 40, 100)
PRESETS = tuple(preset for _, preset in CSCG_PRESETS)

actual = defaultdict(list)
unreadable = []

for run_dir in ROOT.iterdir():
    if not run_dir.is_dir():
        continue

    status_path = run_dir / "run_status.json"

    try:
        status = json.loads(status_path.read_text())
        config = status["config"]

        method = str(config["method"])
        preset = (
            None
            if method == "mercury_cscg_hybrid"
            else config.get("cscg_clone_preset")
        )

        key = (
            method,
            int(config["level"]),
            int(config["seed"]),
            float(config["rand_prob"]),
            int(config["cscg_batch_size"]),
            preset,
        )

        actual[key].append(run_dir.name)

    except Exception as exc:
        unreadable.append((run_dir.name, repr(exc)))

expected = set()

for level in LEVELS:
    for seed in SEEDS:
        for rand_prob in RAND_PROBS:
            for batch in BATCHES:
                expected.add(
                    (
                        "mercury_cscg_hybrid",
                        level,
                        seed,
                        rand_prob,
                        batch,
                        None,
                    )
                )

                for preset in PRESETS:
                    expected.add(
                        (
                            "cscg",
                            level,
                            seed,
                            rand_prob,
                            batch,
                            preset,
                        )
                    )

actual_keys = set(actual)
missing = sorted(expected - actual_keys)
unexpected = sorted(actual_keys - expected)
duplicates = {
    key: run_ids
    for key, run_ids in actual.items()
    if len(run_ids) > 1
}

print("Expected configurations:", len(expected))
print("Run directories:", sum(path.is_dir() for path in ROOT.iterdir()))
print("Unique configuration keys:", len(actual_keys))
print("Missing configurations:", len(missing))
print("Unexpected configurations:", len(unexpected))
print("Duplicate configuration keys:", len(duplicates))
print("Unreadable directories:", len(unreadable))

print("\nMISSING")
for key in missing:
    print(key)

print("\nDUPLICATES")
for key, run_ids in duplicates.items():
    print(key, "->", run_ids)

print("\nUNEXPECTED")
for key in unexpected:
    print(key, "->", actual[key])

print("\nUNREADABLE")
for item in unreadable:
    print(item)
