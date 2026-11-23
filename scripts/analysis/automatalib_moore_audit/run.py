"""Certify cached deterministic mazes as privileged Moore machines with AutomataLib.

This is an oracle benchmark audit. It is deliberately outside all model training
and evaluation entry points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from src.maze_environment.LEVEL import levels


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CACHE = (
    ROOT
    / "results/current/studies/ambiguity_response_ablation/evaluation_caches"
    / "canonical-v4-1e15aac97fcf16ae"
)
LEVELS = (13, 20, 21, 22)
ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # north, east, south, west
WALKS = (0, 1, 2)


def _reference_positions(level: int) -> list[tuple[int, int]]:
    return [
        (row, col)
        for row, line in enumerate(levels[level])
        for col, tile in enumerate(line)
        if tile != "X"
    ]


def _expected_successor(
    source: int, action: int, positions: list[tuple[int, int]]
) -> int:
    row, col = positions[source]
    dr, dc = ACTIONS[action]
    return {position: state for state, position in enumerate(positions)}.get(
        (row + dr, col + dc), source
    )


def _validate_complete_table(
    state_count: int,
    outputs: dict[int, tuple[int, ...]],
    transitions: dict[tuple[int, int], int],
) -> None:
    expected_states = set(range(state_count))
    expected_pairs = {(state, action) for state in expected_states for action in range(4)}
    if set(outputs) != expected_states:
        raise ValueError(f"Missing physical states: {sorted(expected_states - set(outputs))}")
    if set(transitions) != expected_pairs:
        raise ValueError(f"Missing state-action pairs: {sorted(expected_pairs - set(transitions))}")
    if any(target not in expected_states for target in transitions.values()):
        raise ValueError("Transition target outside physical-state set")


def extract_machine(level: int, cache_root: Path = CACHE) -> dict:
    filename = f"level={level}_sensor=cardinal distance_range=1_rand=0.3_eval-seed=0.npz"
    path = cache_root / "walks" / filename
    positions = _reference_positions(level)
    outputs: dict[int, tuple[int, ...]] = {}
    transitions: dict[tuple[int, int], int] = {}
    walk_details = []
    with np.load(path, allow_pickle=False) as cached:
        if int(cached["n_walks"]) <= max(WALKS):
            raise ValueError(f"Cache has fewer than {len(WALKS)} walks: {path}")
        for walk in WALKS:
            prefix = f"walk_{walk:05d}"
            if not bool(cached[f"{prefix}_valid"]):
                raise ValueError(f"Invalid selected walk {walk} for level {level}")
            obs = np.asarray(cached[f"{prefix}_obs"])
            vectors = np.asarray(cached[f"{prefix}_act"])
            gt = np.asarray(cached[f"{prefix}_gt_exact"])
            gt_actions = np.asarray(cached[f"{prefix}_gt_action"])
            collisions = np.asarray(cached[f"{prefix}_collisions"], dtype=bool)
            if not (
                obs.ndim == vectors.ndim == 2
                and obs.shape == vectors.shape
                and obs.shape[1] == 4
                and gt.shape == gt_actions.shape == collisions.shape == (len(obs),)
                and len(obs) >= 2
            ):
                raise ValueError(f"Misaligned cached arrays in level {level} walk {walk}")
            if not np.all((vectors == 0) | (vectors == 1)) or not np.all(vectors.sum(axis=1) == 1):
                raise ValueError(f"Non-one-hot action in level {level} walk {walk}")
            action_ids = vectors.argmax(axis=1)
            if not np.array_equal(action_ids, gt_actions):
                raise ValueError(f"Action IDs disagree with vectors in level {level} walk {walk}")
            if not np.all((gt >= 0) & (gt < len(positions))):
                raise ValueError(f"Invalid physical state in level {level} walk {walk}")
            for step, state_value in enumerate(gt):
                state = int(state_value)
                output = tuple(int(value) for value in obs[step])
                if state in outputs and outputs[state] != output:
                    raise ValueError(f"Conflicting observations for level {level} state {state}")
                outputs[state] = output
                if step == 0:
                    continue  # the cache does not record the pre-action state of row 0
                source = int(gt[step - 1])
                action = int(action_ids[step])
                target = state
                expected = _expected_successor(source, action, positions)
                if target != expected:
                    raise ValueError(
                        f"Action/transition mismatch at level {level} walk {walk} step {step}"
                    )
                if bool(collisions[step]) != (source == target):
                    raise ValueError(
                        f"Collision/transition mismatch at level {level} walk {walk} step {step}"
                    )
                key = (source, action)
                if key in transitions and transitions[key] != target:
                    raise ValueError(f"Nondeterministic transition at level {level} {key}")
                transitions[key] = target
            walk_details.append({"walk": walk, "rows": len(gt)})
    _validate_complete_table(len(positions), outputs, transitions)
    return {
        "level": level,
        "cache": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "physical_state_count": len(positions),
        "state_action_pair_count": len(transitions),
        "positions": [list(position) for position in positions],
        "outputs": [list(outputs[state]) for state in range(len(positions))],
        "transitions": [
            [state, action, transitions[(state, action)]]
            for state in range(len(positions))
            for action in range(4)
        ],
        "walks": walk_details,
        "coverage_complete": True,
        "deterministic": True,
    }


def _machine_tsv(machine: dict, path: Path) -> None:
    lines = [f"N\t{machine['physical_state_count']}"]
    lines.extend(
        f"S\t{state}\t{','.join(map(str, output))}"
        for state, output in enumerate(machine["outputs"])
    )
    lines.extend(f"T\t{s}\t{a}\t{t}" for s, a, t in machine["transitions"])
    path.write_text("\n".join(lines) + "\n")


def _output_trace(
    machine: dict, start: int, actions: list[int], transitions: dict[tuple[int, int], int]
) -> list[str]:
    state = start
    trace = [",".join(map(str, machine["outputs"][state]))]
    for action in actions:
        state = transitions[(state, action)]
        trace.append(",".join(map(str, machine["outputs"][state])))
    return trace


def _parse_java(text: str, machine: dict) -> dict:
    count = machine["physical_state_count"]
    classes = []
    witnesses = []
    characterizing_words: dict[int, list[int]] = {}
    minimized = None
    for row in text.rstrip("\r\n").splitlines():
        parts = row.split("\t")
        if parts[0] == "P" and len(parts) == 3:
            if int(parts[1]) != count or minimized is not None:
                raise ValueError("AutomataLib returned inconsistent physical-state count")
            minimized = int(parts[2])
        elif parts[0] == "B" and len(parts) == 2:
            classes.append([int(state) for state in parts[1].split(",")])
        elif parts[0] == "W" and len(parts) == 6:
            witnesses.append({
                "states": [int(parts[1]), int(parts[2])],
                "actions": [int(action) for action in parts[3].split(",")],
                "output_traces": [parts[4].split(";"), parts[5].split(";")],
            })
        elif parts[0] == "C" and len(parts) == 3:
            index = int(parts[1])
            if index < 0 or index in characterizing_words:
                raise ValueError("Duplicate or negative characterizing-word index")
            actions = [] if not parts[2] else [int(action) for action in parts[2].split(",")]
            if any(action < 0 or action >= 4 for action in actions):
                raise ValueError("Characterizing word contains invalid action")
            characterizing_words[index] = actions
        else:
            raise ValueError(f"Unexpected Java output: {row}")
    if minimized is None or minimized != len(classes):
        raise ValueError("Missing or inconsistent minimized state count")
    if sorted(state for block in classes for state in block) != list(range(count)):
        raise ValueError("AutomataLib classes do not partition physical states")
    aliases = {
        (left, right)
        for left in range(count)
        for right in range(left + 1, count)
        if machine["outputs"][left] == machine["outputs"][right]
    }
    equivalent = {
        (left, right)
        for block in classes
        for left in block
        for right in block
        if left < right
    }
    witnessed = {tuple(witness["states"]) for witness in witnesses}
    if witnessed != aliases - equivalent:
        raise ValueError("Missing or unexpected separating sequences for aliased pairs")
    transitions = {(s, a): t for s, a, t in machine["transitions"]}
    for witness in witnesses:
        traces = [
            _output_trace(machine, start, witness["actions"], transitions)
            for start in witness["states"]
        ]
        if traces != witness["output_traces"] or traces[0] == traces[1]:
            raise ValueError("AutomataLib separating sequence failed independent replay")
    if sorted(characterizing_words) != list(range(len(characterizing_words))):
        raise ValueError("Characterizing-word indices are not contiguous")
    words = [characterizing_words[index] for index in range(len(characterizing_words))]
    pair_coverage = []
    for left in range(count):
        for right in range(left + 1, count):
            distinguishing = [
                index for index, word in enumerate(words)
                if _output_trace(machine, left, word, transitions)
                != _output_trace(machine, right, word, transitions)
            ]
            pair = (left, right)
            if (pair in equivalent) == bool(distinguishing):
                raise ValueError(f"Characterizing set disagrees with state equivalence for {pair}")
            pair_coverage.append({
                "states": [left, right],
                "distinguishing_word_indices": distinguishing,
            })
    lengths = [len(word) for word in words]
    aliased_coverage = [
        pair for pair in pair_coverage if tuple(pair["states"]) in aliases
    ]
    return {
        "minimized_moore_state_count": minimized,
        "equivalence_classes": classes,
        "all_classes_singleton": all(len(block) == 1 for block in classes),
        "perceptually_aliased_pair_count": len(aliases),
        "separating_sequences": witnesses,
        "characterizing_set": [
            {"actions": word, "length": len(word)} for word in words
        ],
        "characterizing_set_word_count": len(words),
        "characterizing_set_word_lengths": lengths,
        "characterizing_set_max_word_length": max(lengths, default=0),
        "characterizing_set_mean_word_length": statistics.mean(lengths) if lengths else None,
        "characterizing_set_median_word_length": statistics.median(lengths) if lengths else None,
        "characterizing_set_covers_all_distinguishable_pairs": True,
        "characterizing_set_distinguishes_all_distinct_physical_states": not equivalent,
        "characterizing_set_pair_coverage": pair_coverage,
        "characterizing_set_aliased_pair_coverage": aliased_coverage,
    }


def _compile_java_runner(maven: Path, temp: Path) -> str:
    repo = Path(tempfile.gettempdir()) / "automatalib-moore-m2"
    subprocess.run(
        [str(maven), "-q", "-f", str(HERE / "pom.xml"),
         f"-Dmaven.repo.local={repo}", f"-Daudit.build.dir={temp / 'build'}",
         "compile", "dependency:build-classpath", f"-Dmdep.outputFile={temp / 'classpath'}"],
        check=True,
    )
    return f"{temp / 'build/classes'}:{(temp / 'classpath').read_text().strip()}"


def _report(results: dict) -> str:
    lines = [
        "# AutomataLib oracle Moore-machine certification",
        "",
        "Each machine uses physical position as state, north/east/south/west as input, "
        "and the cached range-1 cardinal-distance observation vector as state output. "
        "The selected walks are 0–2 from the canonical-v4 cache in `ambiguity_response_ablation`.",
        "",
        "| Level | Physical states | Moore classes | State–action rows | Singleton classes | Aliased pairs separated | Words in W | Max word length |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for machine in results["levels"]:
        lines.append(
            f"| {machine['level']} | {machine['physical_state_count']} | "
            f"{machine['minimized_moore_state_count']} | {machine['state_action_pair_count']} | "
            f"{'yes' if machine['all_classes_singleton'] else 'no'} | "
            f"{len(machine['separating_sequences'])}/{machine['perceptually_aliased_pair_count']} | "
            f"{machine['characterizing_set_word_count']} | "
            f"{machine['characterizing_set_max_word_length']} |"
        )
    lines += [
        "",
        "## Classes and witnesses",
        "",
        "Witness action IDs: 0=north, 1=east, 2=south, 3=west.",
        "",
    ]
    for machine in results["levels"]:
        lines.append(
            f"- **Level {machine['level']}:** classes "
            + ", ".join("{" + ", ".join(map(str, block)) + "}" for block in machine["equivalence_classes"])
            + "."
        )
        if machine["separating_sequences"]:
            words = "; ".join(
                f"{tuple(witness['states'])} → {witness['actions']}"
                for witness in machine["separating_sequences"]
            )
            lines.append(f"  Aliased-pair witnesses: {words}.")
    lines += ["", "## Global characterizing sets", "",
              "The empty word ε compares the current state outputs; a Moore trace "
              "includes that output before any action.", ""]
    for machine in results["levels"]:
        labeled = "; ".join(
            f"W{index}={word['actions'] if word['actions'] else 'ε'} "
            f"(length {word['length']})"
            for index, word in enumerate(machine["characterizing_set"])
        )
        lines.append(f"- **Level {machine['level']}:** {labeled}.")
        covered = sum(
            bool(pair["distinguishing_word_indices"])
            for pair in machine["characterizing_set_pair_coverage"]
        )
        pairs = machine["physical_state_count"] * (machine["physical_state_count"] - 1) // 2
        lines.append(
            f"  Independent replay separates {covered}/{pairs} physical-state pairs; "
            f"mean/median word length {machine['characterizing_set_mean_word_length']:g}/"
            f"{machine['characterizing_set_median_word_length']:g}."
        )
        aliases = "; ".join(
            f"{tuple(pair['states'])}: "
            + ", ".join(f"W{index}" for index in pair["distinguishing_word_indices"])
            for pair in machine["characterizing_set_aliased_pair_coverage"]
        )
        lines.append(f"  Observation-aliased pairs distinguished by: {aliases}.")
    lines += [
        "",
        "## What this establishes",
        "",
        "All physical states and all four actions per state occur in the selected walks. "
        "Their outputs and successors are deterministic and agree with the maze geometry. "
        "AutomataLib 13.0.0 minimizes the resulting complete Moore machines; its "
        "state-pair separating words give the listed witnesses. The Java runner checks "
        "that pairwise classes match its minimized count. AutomataLib constructs a "
        "global characterizing set W; Python independently replays every word "
        "and verifies all physical-state pairs. It also replays every pairwise "
        "witness. Cache hashes, full tables, classes, words, and pair coverage are in `RESULTS.json`.",
        "",
        "## What this does not establish",
        "",
        "This is an oracle statement about these fixed benchmark machines and their "
        "privileged physical states. It establishes behavioral irreducibility and "
        "finite distinguishing action sequences. It does not reconstruct an "
        "epsilon-transducer, equate physical states with its causal states, show "
        "that a learner can infer these states from sensorimotor history, or provide "
        "a baseline comparison with MERCURY, CSCG, TMGWR, or transCSSR. Completeness "
        "is conditional on the benchmark layout, cached ground truth, and checked alignment.",
        "",
        "AutomataLib APIs: [CompactMoore](https://learnlib.de/automatalib/maven-site/13.0.0/apidocs/net.automatalib.core/net/automatalib/automaton/transducer/impl/CompactMoore.html), "
        "[HopcroftMinimizer](https://learnlib.de/automatalib/maven-site/13.0.0/apidocs/net.automatalib.util/net/automatalib/util/automaton/minimizer/HopcroftMinimizer.html), "
        "[Automata.characterizingSet](https://learnlib.de/automatalib/maven-site/13.0.0/apidocs/net.automatalib.util/net/automatalib/util/automaton/Automata.html), "
        "[CharacterizingSets](https://learnlib.de/automatalib/maven-site/13.0.0/apidocs/net.automatalib.util/net/automatalib/util/automaton/equivalence/CharacterizingSets.html), "
        "and [Automata.findSeparatingWord](https://learnlib.de/automatalib/maven-site/13.0.0/apidocs/net.automatalib.util/net/automatalib/util/automaton/Automata.html).",
        "",
    ]
    return "\n".join(lines)


def run(maven: Path, output_dir: Path, cache_root: Path = CACHE) -> dict:
    protocol = json.loads((cache_root / "protocol.json").read_text())
    required = {
        "schema_version": "canonical_v4",
        "evaluation_seed": 0,
        "sensor": "cardinal distance",
        "sensor_range": 1,
        "rand_prob": 0.3,
        "valid_trajectories_only": False,
    }
    for key, value in required.items():
        if protocol.get(key) != value:
            raise ValueError(f"Unexpected cache protocol {key}: {protocol.get(key)!r}")
    machines = [extract_machine(level, cache_root) for level in LEVELS]
    with tempfile.TemporaryDirectory(prefix="automatalib-moore-") as temporary:
        temp = Path(temporary)
        classpath = _compile_java_runner(maven, temp)
        # Exercise an actual Moore merge and a nonempty separating word before
        # interpreting benchmark output. This catches API/semantics drift.
        synthetic = {
            "physical_state_count": 4,
            "outputs": [[0], [1], [1], [0]],
            "transitions": [[s, a, (1, 1, 2, 3)[s]] for s in range(4) for a in range(4)],
        }
        synthetic_path = temp / "synthetic.tsv"
        _machine_tsv(synthetic, synthetic_path)
        check = subprocess.run(
            ["java", "-cp", classpath, "MooreAudit", str(synthetic_path)],
            check=True, capture_output=True, text=True,
        )
        checked = _parse_java(check.stdout, synthetic)
        if checked["equivalence_classes"] != [[0], [1, 2], [3]]:
            raise ValueError("AutomataLib synthetic merge check failed")
        if len(checked["separating_sequences"]) != 1 or not checked["separating_sequences"][0]["actions"]:
            raise ValueError("AutomataLib synthetic distinguishing-word check failed")
        if not checked["characterizing_set_covers_all_distinguishable_pairs"]:
            raise ValueError("AutomataLib synthetic characterizing-set check failed")
        if checked["characterizing_set_distinguishes_all_distinct_physical_states"]:
            raise ValueError("Synthetic equivalent states were incorrectly distinguished")
        if next(pair for pair in checked["characterizing_set_pair_coverage"]
                if pair["states"] == [1, 2])["distinguishing_word_indices"]:
            raise ValueError("Synthetic equivalent pair has a characterizing witness")
        for machine in machines:
            input_path = temp / f"level={machine['level']}.tsv"
            _machine_tsv(machine, input_path)
            java = subprocess.run(
                ["java", "-cp", classpath, "MooreAudit", str(input_path)],
                check=True, capture_output=True, text=True,
            )
            machine.update(_parse_java(java.stdout, machine))
    result = {
        "automatalib_version": "13.0.0",
        "protocol": protocol,
        "selected_walks": list(WALKS),
        "levels": machines,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_dir / "REPORT.md").write_text(_report(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maven", type=Path, default=Path("mvn"))
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    run(args.maven, args.output_dir)
