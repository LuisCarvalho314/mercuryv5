"""Longer teacher smoothing context (teacher_window) in online JEPA-CSCG."""
from __future__ import annotations

import numpy as np
import pytest

from src.mercury_runs.algorithms.cscg import distillation as module
from src.mercury_runs.algorithms.cscg.distillation import (
    ExperimentConfig, clone_groups, row_normalize, smooth_window, train_experiment,
)

METHODS = ["short", "long", "predictive", "posterior", "nll_control"]
WORK_KEYS = ["transitions", "base_msteps", "refinements", "em_passes", "penalty_updates",
             "statistics_commits", "teacher_updates"]
SCHEDULES = [(2, False), (5, True)]


def tiny_problem():
    rng = np.random.default_rng(7)
    return row_normalize(rng.uniform(.1, 1., (2, 5, 5))), np.array([2, 3])


def stream():
    return np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]), np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])


def capture(monkeypatch):
    """Record every teacher/student smoothing call and every replay evaluation."""
    smooths, replays = [], []
    original_smooth, original_replay = module.smooth_window, module.replay_terms
    def smooth(T, groups, observations, actions, boundary):
        smooths.append(dict(T=T.copy(), x=np.array(observations), a=np.array(actions), boundary=np.array(boundary)))
        return original_smooth(T, groups, observations, actions, boundary)
    def replay(T, beliefs, actions, ids, targets):
        replays.append(dict(beliefs=[np.array(b) for b in beliefs], actions=list(actions), targets=np.array(targets)))
        return original_replay(T, beliefs, actions, ids, targets)
    monkeypatch.setattr(module, "smooth_window", smooth)
    monkeypatch.setattr(module, "replay_terms", replay)
    return smooths, replays


@pytest.mark.parametrize("value", [2, 1, True, 3.0])
def test_teacher_window_must_be_an_integer_at_least_the_window(value):
    with pytest.raises(ValueError, match="teacher_window"):
        ExperimentConfig(window=3, teacher_window=value)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("inference", ["filter", "smooth"])
def test_teacher_window_equal_to_window_is_the_unchanged_default(method, schedule, inference):
    T, clones = tiny_problem(); x, a = stream()
    kwargs = dict(window=3, local_updates=schedule[0], coupled_auxiliary=schedule[1], student_inference=inference)
    default = train_experiment(T, clones, x, a, method, ExperimentConfig(**kwargs), range(12))
    explicit = train_experiment(T, clones, x, a, method, ExperimentConfig(teacher_window=3, **kwargs), range(12))
    for step in range(12):
        np.testing.assert_array_equal(default["snapshots"][step], explicit["snapshots"][step])
    np.testing.assert_array_equal(default["teacher_T"], explicit["teacher_T"])
    assert default["work"] == explicit["work"] and default["diagnostics"] == explicit["diagnostics"]


@pytest.mark.parametrize("method", ["short", "long", "nll_control"])
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_arms_without_a_teacher_target_are_unchanged(method, schedule):
    T, clones = tiny_problem(); x, a = stream()
    kwargs = dict(window=3, local_updates=schedule[0], coupled_auxiliary=schedule[1])
    base = train_experiment(T, clones, x, a, method, ExperimentConfig(**kwargs), range(12))
    longer = train_experiment(T, clones, x, a, method, ExperimentConfig(teacher_window=7, **kwargs), range(12))
    for step in range(12):
        np.testing.assert_array_equal(base["snapshots"][step], longer["snapshots"][step])
    np.testing.assert_array_equal(base["teacher_T"], longer["teacher_T"])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_update_budgets_match_the_short_teacher_window(method, schedule):
    T, clones = tiny_problem(); x, a = stream()
    kwargs = dict(window=3, local_updates=schedule[0], coupled_auxiliary=schedule[1])
    base = train_experiment(T, clones, x, a, method, ExperimentConfig(**kwargs), [11])
    longer = train_experiment(T, clones, x, a, method, ExperimentConfig(teacher_window=7, **kwargs), [11])
    assert {k: base["work"][k] for k in WORK_KEYS} == {k: longer["work"][k] for k in WORK_KEYS}


@pytest.mark.parametrize("method", ["predictive", "posterior"])
def test_longer_teacher_window_changes_the_teacher_arms(method):
    T, clones = tiny_problem(); x, a = stream()
    base = train_experiment(T, clones, x, a, method, ExperimentConfig(window=3), [11])
    longer = train_experiment(T, clones, x, a, method, ExperimentConfig(window=3, teacher_window=7), [11])
    assert not np.array_equal(base["final_T"], longer["final_T"])


def test_teacher_targets_are_the_long_smoothing_sliced_to_the_replay_window(monkeypatch):
    """At arrival t: teacher smooths x_{ts:t} (ts = t-W_T+1, clipped) from its cached
    boundary at ts; replay uses only the targets for x_{s+1:t}, with s = t-W+1."""
    T, clones = tiny_problem(); x, a = stream(); groups = clone_groups(clones)
    window, teacher_window = 3, 7
    smooths, replays = capture(monkeypatch)
    # No checkpoints: a checkpoint adds a post-update diagnostic replay call.
    train_experiment(T, clones, x, a, "predictive", ExperimentConfig(window=window, teacher_window=teacher_window), [])
    # Predictive, 2 local updates (historical): per arrival one teacher smoothing call, one replay.
    assert len(smooths) == len(replays) == len(x) - 1
    for t, (call, rep) in enumerate(zip(smooths, replays), start=1):
        ts, s = max(0, t - teacher_window + 1), max(0, t - window + 1)
        np.testing.assert_array_equal(call["x"], x[ts:t + 1])          # never beyond x_t
        np.testing.assert_array_equal(call["a"], a[ts:t])
        q, _ = smooth_window(call["T"], groups, x[ts:t + 1], a[ts:t], call["boundary"])
        np.testing.assert_array_equal(rep["targets"], q[s - ts + 1:])
        assert rep["actions"] == list(a[s:t]) and len(rep["beliefs"]) == t - s


def test_teacher_window_is_clipped_at_episode_starts(monkeypatch):
    T, clones = tiny_problem(); x, a = stream()
    smooths, _ = capture(monkeypatch)
    train_experiment(T, clones, x, a, "predictive", ExperimentConfig(window=3, teacher_window=7), [], episode_starts=[5])
    arrivals = [t for t in range(1, len(x)) if t != 5]   # arrival 5 starts an episode: no transition
    assert len(smooths) == len(arrivals)
    for t, call in zip(arrivals, smooths):
        start = max(0, t - 6) if t < 5 else max(5, t - 6)
        np.testing.assert_array_equal(call["x"], x[start:t + 1])


def test_teacher_window_never_reads_future_observations():
    """Perturbing x_{t+1:} leaves every snapshot up to t unchanged."""
    T, clones = tiny_problem(); x, a = stream()
    config = ExperimentConfig(window=3, teacher_window=7, local_updates=5, coupled_auxiliary=True)
    full = train_experiment(T, clones, x, a, "predictive", config, range(12))
    for t in range(2, len(x)):
        y = x.copy(); y[t:] = 1 - y[t:]
        part = train_experiment(T, clones, y, a, "predictive", config, range(12))
        for step in range(t + 1):
            np.testing.assert_array_equal(full["snapshots"][step], part["snapshots"][step])
