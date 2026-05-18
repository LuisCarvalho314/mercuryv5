import math
from collections import defaultdict


def compute_H_S_given_O(observation_by_state, state_prior=None, log_base=2):
    """
    Compute H(S | O)

    Parameters
    ----------
    observation_by_state : dict
        Mapping: state -> observation
    state_prior : dict or None
        Mapping: state -> p(s). If None, uses uniform.
    log_base : int
        2 for bits, e for nats

    Returns
    -------
    float
        H(S | O)
    """

    states = list(observation_by_state.keys())

    # Uniform prior if not provided
    if state_prior is None:
        p_s = {s: 1.0 / len(states) for s in states}
    else:
        total = sum(state_prior.values())
        p_s = {s: state_prior[s] / total for s in states}

    # Group states by observation
    states_by_observation = defaultdict(list)
    for s, o in observation_by_state.items():
        states_by_observation[o].append(s)

    # Compute p(o)
    p_o = {}
    for o, states_o in states_by_observation.items():
        p_o[o] = sum(p_s[s] for s in states_o)

    # Compute H(S | O)
    H = 0.0

    for o, states_o in states_by_observation.items():
        if p_o[o] == 0:
            continue

        # p(s | o)
        H_o = 0.0
        for s in states_o:
            p_s_given_o = p_s[s] / p_o[o]
            if p_s_given_o > 0:
                H_o -= p_s_given_o * (math.log(p_s_given_o) / math.log(log_base))

        H += p_o[o] * H_o

    return H


if __name__ == "__main__":
    # observation_by_state = {
    #     "s1": "A",
    #     "s2": "B",
    #     "s3": "B",
    #     "s4": "B",
    #     "s5": "C",
    # }
    observation_by_state = {
        "s1": "A",
        "s2": "B",
        "s3": "C",
        "s4": "D",
        "s5": "E",
        "s6": "E",
        "s7": "F",
        "s8": "F",
    }

    H = compute_H_S_given_O(observation_by_state)
    H_norm = H / math.log(len(observation_by_state))
    print(f"H(S | O) = {H:.4f} bits")
    print(f"H(S | O) / log(n) = {H_norm:.4f} bits/state")
    print(f"I_norm(S | O) = {1 - H_norm:.4f} bits/state")
    print(f"Num states {2**H:.4f} states")
