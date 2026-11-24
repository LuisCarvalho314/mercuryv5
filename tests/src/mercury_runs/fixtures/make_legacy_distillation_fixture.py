"""Regenerate the legacy online JEPA-CSCG golden fixture.

The fixture was produced by `distillation.py` at commit 852cf3b, before the
`jepa_mix` objective was introduced. It must NOT be regenerated with later code
unless legacy learning behaviour is intentionally changed:

    PYTHONPATH=src python tests/src/mercury_runs/fixtures/make_legacy_distillation_fixture.py [module.py]
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "legacy_distillation_golden.npz"


def problem():
    rng = np.random.default_rng(11)
    n_clones = np.array([2, 3, 2])
    T = rng.uniform(.1, 1., (2, 7, 7))
    T /= T.sum(-1, keepdims=True)
    labels = rng.integers(0, 3, 61)
    actions = rng.integers(0, 2, 60)
    return T, n_clones, labels, actions


def cases():
    """(name, method, config kwargs) covering every legacy code path."""
    for method, (updates, coupled), inference, forgetting in itertools.product(
            ["short", "long", "predictive", "posterior", "nll_control"],
            [(2, False), (5, True), (5, False)], ["filter", "smooth"], [1., .99]):
        kwargs = dict(window=4, forgetting=forgetting, local_updates=updates,
                      coupled_auxiliary=coupled, student_inference=inference, penalty_weight=1.)
        yield f"{method}|K{updates}|c{int(coupled)}|{inference}|l{forgetting}", method, kwargs
    for method in ["predictive", "nll_control"]:
        yield f"{method}|eta0.3|tw6", method, dict(window=4, teacher_window=6, penalty_weight=.3,
                                                   local_updates=5, coupled_auxiliary=True,
                                                   student_inference="smooth", forgetting=1.)


def main(module_path):
    spec = importlib.util.spec_from_file_location("legacy_distillation", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    T, n_clones, x, a = problem()
    arrays = {}
    for name, method, kwargs in cases():
        result = module.train_experiment(T, n_clones, x, a, method, module.ExperimentConfig(**kwargs),
                                         range(len(x) + 1), episode_starts=[30])
        arrays[name + "|T"] = np.stack([result["snapshots"][s] for s in range(len(x) + 1)])
        arrays[name + "|teacher"] = result["teacher_T"]
        arrays[name + "|counts"] = result["counts"]
    np.savez_compressed(OUTPUT, **arrays)
    print(f"wrote {len(arrays)} arrays to {OUTPUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE.parents[3] / "src/mercury_runs/algorithms/cscg/distillation.py")
