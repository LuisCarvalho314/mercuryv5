from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Hashable, Mapping


State = Hashable
Action = Hashable
Observation = Hashable


def shannon_entropy(probabilities: Mapping[Hashable, float], log_base: float = 2.0) -> float:
    """Compute Shannon entropy of a discrete distribution."""
    entropy = 0.0
    for probability in probabilities.values():
        if probability > 0.0:
            entropy -= probability * (math.log(probability) / math.log(log_base))
    return entropy


def compute_conditional_next_state_entropy_given_observation_and_action(
    transition_probabilities: Mapping[State, Mapping[Action, Mapping[State, float]]],
    observation_by_state: Mapping[State, Observation],
    state_prior: Mapping[State, float] | None = None,
    log_base: float = 2.0,
) -> float:
    """
    Compute H(S_{t+1} | O_t, A_t).

    Parameters
    ----------
    transition_probabilities:
        Nested mapping of the form:
            transition_probabilities[current_state][action][next_state] = probability
    observation_by_state:
        Mapping:
            observation_by_state[current_state] = observation
    state_prior:
        Optional prior over current states p(s_t).
        If omitted, a uniform prior over states is used.
    log_base:
        Entropy base. Use 2 for bits, e for nats.

    Returns
    -------
    float
        The conditional entropy H(S_{t+1} | O_t, A_t).
    """
    states = list(transition_probabilities.keys())

    if state_prior is None:
        uniform_probability = 1.0 / len(states)
        state_prior = {state: uniform_probability for state in states}

    # Normalise prior in case the caller passes unnormalised weights.
    prior_total = sum(state_prior.values())
    if prior_total <= 0.0:
        raise ValueError("state_prior must have positive total mass.")
    normalised_state_prior = {
        state: state_prior.get(state, 0.0) / prior_total for state in states
    }

    # Collect available actions.
    all_actions: set[Action] = set()
    for state in states:
        all_actions.update(transition_probabilities[state].keys())

    # p(o, a) and p(s_{t+1} | o, a)
    joint_observation_action_probability: dict[tuple[Observation, Action], float] = defaultdict(float)
    next_state_distribution_given_observation_and_action: dict[
        tuple[Observation, Action], dict[State, float]
    ] = defaultdict(lambda: defaultdict(float))

    for current_state in states:
        current_observation = observation_by_state[current_state]
        current_state_probability = normalised_state_prior[current_state]

        for action in all_actions:
            # Here we assume the conditioning is over available actions uniformly.
            # If you want a policy-weighted version, replace this with pi(a | s_t).
            action_probability_given_state = 1.0 / len(all_actions)

            joint_probability = current_state_probability * action_probability_given_state
            observation_action_key = (current_observation, action)

            joint_observation_action_probability[observation_action_key] += joint_probability

            next_state_distribution = transition_probabilities[current_state].get(action, {})
            for next_state, transition_probability in next_state_distribution.items():
                next_state_distribution_given_observation_and_action[observation_action_key][
                    next_state
                ] += joint_probability * transition_probability

    conditional_entropy = 0.0

    for observation_action_key, joint_probability in joint_observation_action_probability.items():
        if joint_probability <= 0.0:
            continue

        unnormalised_next_state_distribution = next_state_distribution_given_observation_and_action[
            observation_action_key
        ]
        conditional_next_state_distribution = {
            next_state: probability_mass / joint_probability
            for next_state, probability_mass in unnormalised_next_state_distribution.items()
        }

        conditional_entropy += joint_probability * shannon_entropy(
            conditional_next_state_distribution,
            log_base=log_base,
        )

    return conditional_entropy


if __name__ == "__main__":
    transition_probabilities = {
        "s1": {
            "up": {"s1": 1.0},
            "left": {"s1": 1.0},
            "right": {"s2": 1.0},
            "down": {"s1": 1.0},
        },
        "s2": {
            "up": {"s2": 1.0},
            "left": {"s1": 1.0},
            "right": {"s3": 1.0},
            "down": {"s2": 1.0},
        },
        "s3": {
            "up": {"s3": 1.0},
            "left": {"s2": 1.0},
            "right": {"s4": 1.0},
            "down": {"s3": 1.0},
        },
        "s4": {
            "up": {"s4": 1.0},
            "left": {"s4": 1.0},
            "right": {"s7": 1.0},
            "down": {"s4": 1.0},
        },
        # "s5": {
        #     "up": {"s5": 1.0},
        #     "left": {"s4": 1.0},
        #     "right": {"s7": 1.0},
        #     "down": {"s5": 1.0},
        # },
        # "s6": {
        #     "up": {"s6": 1.0},
        #     "left": {"s5": 1.0},
        #     "right": {"s7": 1.0},
        #     "down": {"s6": 1.0},
        # },
        "s7": {
            "up": {"s7": 1.0},
            "left": {"s4": 1.0},
            "right": {"s7": 1.0},
            "down": {"s7": 1.0},
        },
    }

    observation_by_state = {
        "s1": "A",
        "s2": "B",
        "s3": "B",
        "s4": "B",
        # "s5": "B",
        "s7": "C",

    }

    entropy_bits = compute_conditional_next_state_entropy_given_observation_and_action(
        transition_probabilities=transition_probabilities,
        observation_by_state=observation_by_state,
    )
    # entropy_bits = 0

    print(f"H(S_t+1 | O_t, A_t) = {entropy_bits:.4f} bits")
    print(f"H(S_t+1 | O_t, A_t)/|states| = "
          f"{entropy_bits/math.log(len(observation_by_state)):.4f} bits")

    print(f"Num states {2**entropy_bits:.4f} states")