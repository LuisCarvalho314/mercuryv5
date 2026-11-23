"""
noisy_trajectories.py

Synthetic experiment for trajectory-history noise under perceptual aliasing.

The nominal environment is a 1-D corridor embedded in a 2-D action space:

                 wall
                  ↑
    0 -- 1 -- 2 -- 3 -- ... -- N-1
                  ↓
                 wall

The agent has four actions:

    LEFT
    RIGHT
    UP
    DOWN

LEFT / RIGHT move along the corridor when traversable.

UP / DOWN hit a wall everywhere and therefore:

    state_{t+1} = state_t

The point of the "stall" noise is therefore NOT that the same action
mysteriously fails.

Instead:

    clean:
        state 4 --RIGHT--> state 5

    noisy:
        state 4 --UP--> state 4
        state 4 --RIGHT--> state 5

The latent state and observation are unchanged after the blocked action,
but the ACTION HISTORY is different.

This mimics trajectory noise where the agent briefly takes an irrelevant
or blocked action before continuing.

Observations
------------
Use four binary channels in globally fixed cardinal directions:

    [N, E, S, W]

For the corridor:

    N = 0 everywhere
    S = 0 everywhere

    E = 1 unless at right boundary
    W = 1 unless at left boundary

Therefore:

    left boundary:
        [0, 1, 0, 0]

    interior:
        [0, 1, 0, 1]

    right boundary:
        [0, 0, 0, 1]

All interior states remain perceptually aliased.

Tokens are interleaved:

    O_t, A_t, O_{t+1}, A_{t+1}, ...

Encodings
---------
1. Exact positional strip
2. Weighted-sum / decay-only context
3. Random-position FHRR
4. Structured rotary FHRR

Main controlled question
------------------------
For a clean trajectory ending in state 7, compare:

    clean_7
    blocked-UP trajectory ending in state 7
    blocked-DOWN trajectory ending in state 7
    two blocked actions ending in state 7
    reversal ending in state 7

against genuinely different clean trajectories:

    clean_6
    clean_8

Desired behaviour:

    S(noisy_7, clean_7)
        >
    max(
        S(clean_6, clean_7),
        S(clean_8, clean_7)
    )

The important quantity is therefore the state-separation margin, not
raw similarity alone.
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    seed: int = 7

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    n_states: int = 5

    target_states: tuple = tuple(
        range(1, 4)
    )

    # --------------------------------------------------------
    # Context representation
    # --------------------------------------------------------

    # Tokens are interleaved:
    #
    #     O, A, O, A, ...
    #
    # 40 tokens therefore corresponds to roughly 20 actions.
    window: int = 40

    # Number of complex FHRR dimensions.
    #
    # Equivalent real dimensionality = 2 * dim.
    dim: int = 1024

    # Explicit recency weighting.
    #
    # Keep at 1 initially to isolate positional geometry.
    lam: float = 1.0

    # --------------------------------------------------------
    # Structured positional encoding
    # --------------------------------------------------------

    rotary_mode: str = "rope"

    # Used for the optional Gaussian kernel.
    position_scale: float = 1.0

    # RoPE frequency base.
    rope_base: float = 100.0

    # --------------------------------------------------------
    # Monte Carlo noise
    # --------------------------------------------------------

    n_samples: int = 300

    noise_levels: tuple = tuple(
        np.linspace(
            0.0,
            0.5,
            11,
        )
    )

    # At maximum noise severity, probability that before
    # progressing the agent selects a blocked UP/DOWN action.
    max_blocked_prob: float = 0.55

    # Probability of a genuine LEFT->RIGHT backtrack.
    max_backtrack_prob: float = 0.30

    # Multiple blocked actions may occur before progress.
    max_consecutive_blocked: int = 3

    # --------------------------------------------------------
    # RoPE base sweep
    # --------------------------------------------------------

    rope_bases: tuple = (
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
        200.0,
        500.0,
        1000.0,
        10000.0,
    )

    # --------------------------------------------------------
    # Experiment toggles
    # --------------------------------------------------------

    run_monte_carlo: bool = True

    run_base_sweep: bool = True

    show_plots: bool = True


CFG = Config()


# ============================================================
# ACTIONS
# ============================================================

LEFT = "LEFT"
RIGHT = "RIGHT"
UP = "UP"
DOWN = "DOWN"


# ============================================================
# ENVIRONMENT
# ============================================================

def observation(
    state,
    cfg,
):
    """
    Four globally fixed binary traversability channels:

        [N, E, S, W]

    The corridor is enclosed by walls north and south.
    """

    north_free = 0

    east_free = int(
        state
        < cfg.n_states - 1
    )

    south_free = 0

    west_free = int(
        state > 0
    )

    return (
        north_free,
        east_free,
        south_free,
        west_free,
    )


def observation_token(
    state,
    cfg,
):
    n, e, s, w = observation(
        state,
        cfg,
    )

    return (
        f"O_{n}{e}{s}{w}"
    )


def action_token(action):

    if action == LEFT:
        return "A_L"

    if action == RIGHT:
        return "A_R"

    if action == UP:
        return "A_U"

    if action == DOWN:
        return "A_D"

    raise ValueError(
        f"Unknown action: {action}"
    )


def transition(
    state,
    action,
    cfg,
):
    """
    Apply one action.

    UP and DOWN always hit corridor walls.

    LEFT and RIGHT move when the corresponding neighbouring
    corridor state exists.
    """

    # --------------------------------------------------------
    # Blocked orthogonal actions
    # --------------------------------------------------------

    if action in (
        UP,
        DOWN,
    ):
        return state

    # --------------------------------------------------------
    # Horizontal movement
    # --------------------------------------------------------

    if action == LEFT:

        if state == 0:
            return state

        return state - 1

    if action == RIGHT:

        if state == cfg.n_states - 1:
            return state

        return state + 1

    raise ValueError(
        f"Unknown action: {action}"
    )


# ============================================================
# TRAJECTORY GENERATION
# ============================================================

def clean_trajectory(
    start,
    target,
    cfg,
):
    """
    Clean shortest trajectory along the corridor.

    Returns
    -------
    tokens:
        O_0, A_0, O_1, A_1, ...

    states:
        visited latent states

    actions:
        issued actions
    """

    if not (0 <= target < cfg.n_states):
        raise ValueError(
            f"Target state {target} is outside corridor "
            f"[0, {cfg.n_states - 1}]"
        )

    state = start

    tokens = [
        observation_token(
            state,
            cfg,
        )
    ]

    states = [state]

    actions = []

    while state != target:

        action = (
            RIGHT
            if target > state
            else LEFT
        )

        actions.append(
            action
        )

        tokens.append(
            action_token(
                action
            )
        )

        state = transition(
            state,
            action,
            cfg,
        )

        states.append(
            state
        )

        tokens.append(
            observation_token(
                state,
                cfg,
            )
        )

    return (
        tokens,
        states,
        actions,
    )


def noisy_trajectory(
    start,
    target,
    rng,
    blocked_prob,
    backtrack_prob,
    cfg,
):
    """
    Follow the same overall route to target while adding trajectory noise.

    Noise type 1: blocked orthogonal action
    ---------------------------------------

        state s
          |
         UP
          |
        wall

    The action is recorded but state does not change.

    Example:

        ... R O_s
            U O_s
            R O_{s+1}

    Noise type 2: genuine reversal
    ------------------------------

        s -> s-1 -> s -> s+1

    The final target state remains unchanged.
    """

    if not (0 <= target < cfg.n_states):
        raise ValueError(
            f"Target state {target} is outside corridor "
            f"[0, {cfg.n_states - 1}]"
        )

    state = start

    tokens = [
        observation_token(
            state,
            cfg,
        )
    ]

    states = [state]

    actions = []

    while state != target:

        intended = (
            RIGHT
            if target > state
            else LEFT
        )

        opposite = (
            LEFT
            if intended == RIGHT
            else RIGHT
        )

        # ----------------------------------------------------
        # Genuine short reversal / backtrack
        # ----------------------------------------------------

        if (
            rng.random()
            < backtrack_prob
        ):

            detour_state = transition(
                state,
                opposite,
                cfg,
            )

            # Only create a reversal if the opposite movement
            # is actually possible.
            if detour_state != state:

                actions.append(
                    opposite
                )

                tokens.append(
                    action_token(
                        opposite
                    )
                )

                state = detour_state

                states.append(
                    state
                )

                tokens.append(
                    observation_token(
                        state,
                        cfg,
                    )
                )

                # Return to where we were.
                actions.append(
                    intended
                )

                tokens.append(
                    action_token(
                        intended
                    )
                )

                state = transition(
                    state,
                    intended,
                    cfg,
                )

                states.append(
                    state
                )

                tokens.append(
                    observation_token(
                        state,
                        cfg,
                    )
                )

        # ----------------------------------------------------
        # One or more blocked UP/DOWN actions
        # ----------------------------------------------------

        n_blocked = 0

        while (
            n_blocked
            < cfg.max_consecutive_blocked
            and rng.random()
            < blocked_prob
        ):

            blocked_action = (
                UP
                if rng.random() < 0.5
                else DOWN
            )

            actions.append(
                blocked_action
            )

            tokens.append(
                action_token(
                    blocked_action
                )
            )

            # This action hits the corridor wall.
            new_state = transition(
                state,
                blocked_action,
                cfg,
            )

            assert (
                new_state == state
            )

            state = new_state

            states.append(
                state
            )

            # Observation is therefore unchanged.
            tokens.append(
                observation_token(
                    state,
                    cfg,
                )
            )

            n_blocked += 1

        # ----------------------------------------------------
        # Eventually perform intended progress action
        # ----------------------------------------------------

        actions.append(
            intended
        )

        tokens.append(
            action_token(
                intended
            )
        )

        state = transition(
            state,
            intended,
            cfg,
        )

        states.append(
            state
        )

        tokens.append(
            observation_token(
                state,
                cfg,
            )
        )

    return (
        tokens,
        states,
        actions,
    )


# ============================================================
# TRAJECTORY FROM EXPLICIT ACTIONS
# ============================================================

def trajectory_from_actions(
    start,
    actions,
    cfg,
):
    """
    Construct an exact controlled trajectory from an explicit
    sequence of actions.

    This is preferable to inferring actions from state sequences,
    because a repeated state could have resulted from:

        UP
        DOWN
        blocked LEFT
        blocked RIGHT

    and those are sensorimotorically different.
    """

    state = start

    tokens = [
        observation_token(
            state,
            cfg,
        )
    ]

    states = [
        state
    ]

    for action in actions:

        tokens.append(
            action_token(
                action
            )
        )

        state = transition(
            state,
            action,
            cfg,
        )

        states.append(
            state
        )

        tokens.append(
            observation_token(
                state,
                cfg,
            )
        )

    return (
        tokens,
        states,
    )


# ============================================================
# VOCABULARY
# ============================================================

VOCAB = [
    # observations
    "O_0100",   # left boundary
    "O_0101",   # interior
    "O_0001",   # right boundary

    # actions
    "A_L",
    "A_R",
    "A_U",
    "A_D",
]

TOKEN_TO_INDEX = {
    token: i
    for i, token
    in enumerate(VOCAB)
}

VOCAB_SIZE = len(
    VOCAB
)


def token_one_hot(
    token,
):
    x = np.zeros(
        VOCAB_SIZE,
        dtype=float,
    )

    x[
        TOKEN_TO_INDEX[
            token
        ]
    ] = 1.0

    return x


# ============================================================
# FHRR CONTENT CODES
# ============================================================

def random_fhrr(
    dim,
    rng,
):
    phase = rng.uniform(
        -np.pi,
        np.pi,
        size=dim,
    )

    return np.exp(
        1j * phase
    )


def make_content_codes(
    dim,
    rng,
):
    return {
        token: random_fhrr(
            dim,
            rng,
        )
        for token in VOCAB
    }


# ============================================================
# POSITION OPERATORS
# ============================================================

def make_rotary_theta(
    dim,
    mode,
    rng,
    rope_base,
    position_scale,
):
    """
    Construct angular frequencies.

    Position k:

        p_k = exp(j * k * theta)

    Therefore:

        conj(p_m) * p_n
            =
        exp(j * (n-m) * theta)

    so the positional relationship depends only on relative
    displacement.
    """

    if mode == "gaussian":

        sigma = (
            1.0
            / position_scale
        )

        return rng.normal(
            loc=0.0,
            scale=sigma,
            size=dim,
        )

    if mode == "rope":

        i = np.arange(
            dim
        )

        return (
            rope_base
            ** (-i / dim)
        )

    raise ValueError(
        f"Unknown rotary mode: "
        f"{mode}"
    )


def make_random_position_theta(
    dim,
    rng,
):
    """
    Conventional random FHRR positional generator.

    In high dimensions, distinct integer offsets are generally
    close to orthogonal.
    """

    return rng.uniform(
        -np.pi,
        np.pi,
        size=dim,
    )


def position(
    theta,
    lag,
):
    return np.exp(
        1j
        * lag
        * theta
    )


def position_kernel(
    theta,
    lag,
):
    """
    Similarity between positional code 0 and positional code lag.

        K(lag)
            =
        mean_i cos(lag * theta_i)
    """

    return np.mean(
        np.cos(
            lag * theta
        )
    )


# ============================================================
# SIMILARITY
# ============================================================

def cosine(
    a,
    b,
):
    denom = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denom == 0:
        return 0.0

    if np.iscomplexobj(a):

        return (
            np.real(
                np.vdot(
                    a,
                    b,
                )
            )
            / denom
        )

    return (
        np.dot(
            a,
            b,
        )
        / denom
    )


# ============================================================
# ENCODER 1:
# EXACT POSITIONAL STRIP
# ============================================================

def encode_strip(
    tokens,
    cfg,
):
    """
    One explicit one-hot slot per lag.

    Any inserted action-observation pair shifts all earlier
    content by two token positions.
    """

    representation = (
        np.zeros(
            cfg.window
            * VOCAB_SIZE,
            dtype=float,
        )
    )

    recent = tokens[
        -cfg.window:
    ]

    recent = list(
        reversed(
            recent
        )
    )

    for lag, token in enumerate(
        recent
    ):

        offset = (
            lag
            * VOCAB_SIZE
        )

        representation[
            offset
            + TOKEN_TO_INDEX[
                token
            ]
        ] = (
            cfg.lam
            ** lag
        )

    return representation


# ============================================================
# ENCODER 2:
# WEIGHTED-SUM / DECAY-ONLY CONTEXT
# ============================================================

def encode_linear(
    tokens,
    cfg,
):
    """
    Content accumulation without explicit positional binding.

        h =
        sum_k lambda^k x_{t-k}

    Included only as a simple collapsed-history baseline.
    """

    h = np.zeros(
        VOCAB_SIZE,
        dtype=float,
    )

    recent = tokens[
        -cfg.window:
    ]

    for lag, token in enumerate(
        reversed(
            recent
        )
    ):

        h += (
            cfg.lam
            ** lag
        ) * token_one_hot(
            token
        )

    return h


# ============================================================
# ENCODER 3 / 4:
# FHRR POSITIONAL MEMORY
# ============================================================

def encode_fhrr(
    tokens,
    content_codes,
    theta,
    cfg,
):
    """
    Finite-window FHRR history:

        h =
        sum_k
            lambda^k
            x_{t-k}
            p^k
    """

    h = np.zeros(
        cfg.dim,
        dtype=np.complex128,
    )

    recent = tokens[
        -cfg.window:
    ]

    for lag, token in enumerate(
        reversed(
            recent
        )
    ):

        h += (
            cfg.lam
            ** lag
        ) * (
            content_codes[
                token
            ]
            * position(
                theta,
                lag,
            )
        )

    return h


# ============================================================
# ENCODER COLLECTION
# ============================================================

class Encoders:

    def __init__(
        self,
        cfg,
        rng,
    ):

        self.cfg = cfg

        self.content_codes = (
            make_content_codes(
                cfg.dim,
                rng,
            )
        )

        self.rotary_theta = (
            make_rotary_theta(
                dim=cfg.dim,
                mode=cfg.rotary_mode,
                rng=rng,
                rope_base=
                    cfg.rope_base,
                position_scale=
                    cfg.position_scale,
            )
        )

        self.random_theta = (
            make_random_position_theta(
                cfg.dim,
                rng,
            )
        )

    def encode_all(
        self,
        tokens,
    ):

        return {

            "strip":
                encode_strip(
                    tokens,
                    self.cfg,
                ),

            "linear":
                encode_linear(
                    tokens,
                    self.cfg,
                ),

            "random_vsa":
                encode_fhrr(
                    tokens,
                    self.content_codes,
                    self.random_theta,
                    self.cfg,
                ),

            "rotary_vsa":
                encode_fhrr(
                    tokens,
                    self.content_codes,
                    self.rotary_theta,
                    self.cfg,
                ),
        }


# ============================================================
# CONTROLLED TRAJECTORY TEST
# ============================================================

def controlled_trajectory_test(
    encoders,
    cfg,
    verbose=True,
):
    """
    Controlled trajectory-noise test for a 5-state corridor.

    Reference latent state:

        state 2

    All noisy trajectories end at state 2.

    Genuine neighbouring clean trajectories terminate at:

        state 1
        state 3

    Desired condition:

        S(noisy_2, clean_2)
            >
        max(
            S(clean_1, clean_2),
            S(clean_3, clean_2)
        )

    Positive margin therefore means that trajectory noise has not
    made the representation more similar to a different latent state
    than to the correct one.
    """

    trajectories = {

        # ----------------------------------------------------
        # Reference
        # ----------------------------------------------------

        "clean_2":
            [RIGHT, RIGHT],

        # ----------------------------------------------------
        # One blocked action
        # ----------------------------------------------------

        # Move to state 1, hit upper wall, then continue.
        "blocked_late":
            [
                RIGHT,
                UP,
                RIGHT,
            ],

        # Hit lower wall at the beginning, then continue.
        "blocked_early":
            [
                DOWN,
                RIGHT,
                RIGHT,
            ],

        # ----------------------------------------------------
        # Two blocked actions
        # ----------------------------------------------------

        "two_blocked":
            [
                RIGHT,
                UP,
                DOWN,
                RIGHT,
            ],

        # ----------------------------------------------------
        # Genuine reversal
        # ----------------------------------------------------

        # Reach state 2, move back to state 1, then return.
        "reversal":
            [
                RIGHT,
                RIGHT,
                LEFT,
                RIGHT,
            ],

        # ----------------------------------------------------
        # Genuine neighbouring clean states
        # ----------------------------------------------------

        "clean_1":
            [RIGHT],

        "clean_3":
            [
                RIGHT,
                RIGHT,
                RIGHT,
            ],
    }

    encoded = {}
    latent_end_states = {}

    for name, actions in trajectories.items():

        tokens, states = trajectory_from_actions(
            start=0,
            actions=actions,
            cfg=cfg,
        )

        encoded[name] = encoders.encode_all(
            tokens
        )

        latent_end_states[name] = states[-1]

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assert (
        latent_end_states["clean_2"] == 2
    )

    for name in [
        "blocked_late",
        "blocked_early",
        "two_blocked",
        "reversal",
    ]:
        assert (
            latent_end_states[name] == 2
        ), (
            name,
            latent_end_states[name],
        )

    assert (
        latent_end_states["clean_1"] == 1
    )

    assert (
        latent_end_states["clean_3"] == 3
    )

    methods = [
        "strip",
        "linear",
        "random_vsa",
        "rotary_vsa",
    ]

    results = {}

    if verbose:

        print()
        print("=" * 95)
        print("CONTROLLED TRAJECTORY TEST")
        print("=" * 95)

        print(
            f"rotary_mode = "
            f"{cfg.rotary_mode}"
        )

        if cfg.rotary_mode == "rope":
            print(
                f"rope_base = "
                f"{cfg.rope_base}"
            )

        if cfg.rotary_mode == "gaussian":
            print(
                f"position_scale = "
                f"{cfg.position_scale}"
            )

        print(
            f"lambda = "
            f"{cfg.lam}"
        )

        print()
        print("Action sequences:")

        for name, actions in trajectories.items():
            print(
                f"  {name:16s} "
                f"{format_actions(actions)} "
                f"-> state "
                f"{latent_end_states[name]}"
            )

    # --------------------------------------------------------
    # Similarity test
    # --------------------------------------------------------

    for method in methods:

        ref = encoded[
            "clean_2"
        ][
            method
        ]

        sim_1 = cosine(
            ref,
            encoded[
                "clean_1"
            ][
                method
            ],
        )

        sim_3 = cosine(
            ref,
            encoded[
                "clean_3"
            ][
                method
            ],
        )

        nearest_neighbour = max(
            sim_1,
            sim_3,
        )

        method_results = {
            "clean_1":
                sim_1,

            "clean_3":
                sim_3,

            "nearest_neighbour":
                nearest_neighbour,
        }

        if verbose:

            print()
            print(method)

            print(
                f"  clean_1:            "
                f"{sim_1:.4f}"
            )

            print(
                f"  clean_3:            "
                f"{sim_3:.4f}"
            )

            print(
                f"  nearest neighbour:  "
                f"{nearest_neighbour:.4f}"
            )

        for name in [
            "blocked_late",
            "blocked_early",
            "two_blocked",
            "reversal",
        ]:

            same_similarity = cosine(
                ref,
                encoded[
                    name
                ][
                    method
                ],
            )

            margin = (
                same_similarity
                - nearest_neighbour
            )

            method_results[
                name
            ] = {
                "similarity":
                    same_similarity,

                "margin":
                    margin,
            }

            if verbose:

                print(
                    f"  {name:16s} "
                    f"sim="
                    f"{same_similarity:.4f} "
                    f"margin="
                    f"{margin:+.4f}"
                )

        results[
            method
        ] = method_results

    return results

# ============================================================
# CLEAN PROTOTYPES
# ============================================================

def build_prototypes(
    encoders,
    cfg,
):
    """
    One clean shortest-path prototype for each target state.
    """

    methods = [
        "strip",
        "linear",
        "random_vsa",
        "rotary_vsa",
    ]

    prototypes = {
        method: {}
        for method in methods
    }

    for target in (
        cfg.target_states
    ):

        (
            tokens,
            _,
            _,
        ) = clean_trajectory(
            start=0,
            target=target,
            cfg=cfg,
        )

        encoded = (
            encoders.encode_all(
                tokens
            )
        )

        for (
            method,
            vector,
        ) in encoded.items():

            prototypes[
                method
            ][
                target
            ] = vector

    return prototypes


# ============================================================
# NEAREST CLEAN PROTOTYPE
# ============================================================

def classify(
    representation,
    prototypes,
):
    scores = {
        state: cosine(
            representation,
            prototype,
        )
        for (
            state,
            prototype,
        ) in prototypes.items()
    }

    prediction = max(
        scores,
        key=scores.get,
    )

    return (
        prediction,
        scores,
    )


# ============================================================
# MONTE CARLO EXPERIMENT
# ============================================================

def run_experiment(
    rng,
    encoders,
    prototypes,
    cfg,
):
    methods = list(
        prototypes.keys()
    )

    results = {
        method: {
            "same_similarity": [],
            "other_similarity": [],
            "margin": [],
            "accuracy": [],
        }
        for method in methods
    }

    noise_levels = np.asarray(
        cfg.noise_levels
    )

    max_noise = (
        noise_levels[-1]
    )

    for noise in (
        noise_levels
    ):

        blocked_prob = (
            noise
            * cfg.max_blocked_prob
            / max_noise
        )

        backtrack_prob = (
            noise
            * cfg.max_backtrack_prob
            / max_noise
        )

        accum = {
            method: {
                "same": [],
                "other": [],
                "margin": [],
                "correct": [],
            }
            for method in methods
        }

        for target in (
            cfg.target_states
        ):

            for _ in range(
                cfg.n_samples
            ):

                (
                    tokens,
                    states,
                    actions,
                ) = noisy_trajectory(
                    start=0,
                    target=target,
                    rng=rng,
                    blocked_prob=
                        blocked_prob,
                    backtrack_prob=
                        backtrack_prob,
                    cfg=cfg,
                )

                # Sanity check:
                # noise must not change final target.
                assert (
                    states[-1]
                    == target
                )

                encoded = (
                    encoders.encode_all(
                        tokens
                    )
                )

                for method in methods:

                    representation = (
                        encoded[
                            method
                        ]
                    )

                    own_prototype = (
                        prototypes[
                            method
                        ][
                            target
                        ]
                    )

                    same = cosine(
                        representation,
                        own_prototype,
                    )

                    (
                        prediction,
                        scores,
                    ) = classify(
                        representation,
                        prototypes[
                            method
                        ],
                    )

                    other_scores = [
                        score
                        for (
                            state,
                            score,
                        ) in scores.items()
                        if state != target
                    ]

                    best_other = max(
                        other_scores
                    )

                    accum[
                        method
                    ][
                        "same"
                    ].append(
                        same
                    )

                    accum[
                        method
                    ][
                        "other"
                    ].append(
                        best_other
                    )

                    accum[
                        method
                    ][
                        "margin"
                    ].append(
                        same
                        - best_other
                    )

                    accum[
                        method
                    ][
                        "correct"
                    ].append(
                        prediction
                        == target
                    )

        for method in methods:

            results[
                method
            ][
                "same_similarity"
            ].append(
                np.mean(
                    accum[
                        method
                    ][
                        "same"
                    ]
                )
            )

            results[
                method
            ][
                "other_similarity"
            ].append(
                np.mean(
                    accum[
                        method
                    ][
                        "other"
                    ]
                )
            )

            results[
                method
            ][
                "margin"
            ].append(
                np.mean(
                    accum[
                        method
                    ][
                        "margin"
                    ]
                )
            )

            results[
                method
            ][
                "accuracy"
            ].append(
                np.mean(
                    accum[
                        method
                    ][
                        "correct"
                    ]
                )
            )

    return results


# ============================================================
# EXAMPLE TRAJECTORIES
# ============================================================

def format_actions(
    actions,
):
    mapping = {
        LEFT: "L",
        RIGHT: "R",
        UP: "U",
        DOWN: "D",
    }

    return [
        mapping[action]
        for action in actions
    ]


def show_examples(
    rng,
    cfg,
):
    target = 3

    print()

    print(
        "=" * 70
    )

    print(
        "EXAMPLE TRAJECTORIES"
    )

    print(
        "=" * 70
    )

    (
        _,
        clean_states,
        clean_actions,
    ) = clean_trajectory(
        0,
        target,
        cfg,
    )

    print()

    print(
        "Clean:"
    )

    print(
        "states: ",
        clean_states,
    )

    print(
        "actions:",
        format_actions(
            clean_actions
        ),
    )

    examples = [
        (0.10, 0.00),
        (0.25, 0.10),
        (0.50, 0.25),
    ]

    for (
        blocked,
        backtrack,
    ) in examples:

        (
            _,
            states,
            actions,
        ) = noisy_trajectory(
            start=0,
            target=target,
            rng=rng,
            blocked_prob=blocked,
            backtrack_prob=
                backtrack,
            cfg=cfg,
        )

        print()

        print(
            f"blocked="
            f"{blocked:.2f}, "
            f"backtrack="
            f"{backtrack:.2f}:"
        )

        print(
            "states: ",
            states,
        )

        print(
            "actions:",
            format_actions(
                actions
            ),
        )


# ============================================================
# ALTERNATIVE-ROUTE CONTROL
# ============================================================

def alternative_route_control(
    encoders,
    cfg,
):
    """
    Reach the same latent target from opposite sides.

    This is intentionally NOT classified as nuisance noise.
    """

    target = (
        cfg.n_states
        // 2
    )

    (
        left_tokens,
        _,
        _,
    ) = clean_trajectory(
        start=0,
        target=target,
        cfg=cfg,
    )

    (
        right_tokens,
        _,
        _,
    ) = clean_trajectory(
        start=cfg.n_states - 1,
        target=target,
        cfg=cfg,
    )

    left = (
        encoders.encode_all(
            left_tokens
        )
    )

    right = (
        encoders.encode_all(
            right_tokens
        )
    )

    print()

    print(
        "=" * 70
    )

    print(
        "ALTERNATIVE-ROUTE CONTROL"
    )

    print(
        "=" * 70
    )

    print(
        f"Same latent target "
        f"state: {target}"
    )

    print(
        "One trajectory approaches "
        "from left, the other from right."
    )

    print()

    for method in left:

        score = cosine(
            left[
                method
            ],
            right[
                method
            ],
        )

        print(
            f"{method:12s}: "
            f"{score: .4f}"
        )


# ============================================================
# PRINT POSITION KERNEL
# ============================================================

def print_kernel_values(
    encoders,
):
    print()

    print(
        "=" * 70
    )

    print(
        "ROTARY POSITION KERNEL"
    )

    print(
        "=" * 70
    )

    print(
        f"{'lag':>5s} "
        f"{'K(lag)':>12s}"
    )

    for lag in range(
        0,
        17,
    ):

        k = position_kernel(
            encoders.rotary_theta,
            lag,
        )

        marker = ""

        # A single blocked action inserts:
        #
        #     A_U, O_same
        #
        # or
        #
        #     A_D, O_same
        #
        # i.e. 2 extra tokens.
        if (
            lag > 0
            and lag % 2 == 0
        ):
            marker = (
                "  <- blocked-action "
                "offset multiple"
            )

        print(
            f"{lag:5d} "
            f"{k:12.4f}"
            f"{marker}"
        )


# ============================================================
# MONTE CARLO SUMMARY
# ============================================================

def print_results(
    results,
    cfg,
):
    methods = list(
        results.keys()
    )

    print()

    print(
        "=" * 90
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 90
    )

    for i, noise in enumerate(
        cfg.noise_levels
    ):

        print()

        print(
            f"Noise severity = "
            f"{noise:.2f}"
        )

        print(
            f"{'method':12s} "
            f"{'same':>9s} "
            f"{'other':>9s} "
            f"{'margin':>9s} "
            f"{'accuracy':>9s}"
        )

        for method in methods:

            values = (
                results[
                    method
                ]
            )

            print(
                f"{method:12s} "
                f"{values['same_similarity'][i]:9.3f} "
                f"{values['other_similarity'][i]:9.3f} "
                f"{values['margin'][i]:9.3f} "
                f"{values['accuracy'][i]:9.3f}"
            )


# ============================================================
# POSITIONAL KERNEL PLOT
# ============================================================

def plot_position_kernel(
    encoders,
    cfg,
):
    lags = np.arange(
        0,
        21,
    )

    rotary = np.array([
        position_kernel(
            encoders.rotary_theta,
            lag,
        )
        for lag in lags
    ])

    random = np.array([
        position_kernel(
            encoders.random_theta,
            lag,
        )
        for lag in lags
    ])

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        lags,
        rotary,
        marker="o",
        label=
            "structured rotary",
    )

    plt.plot(
        lags,
        random,
        marker="o",
        label=
            "random FHRR",
    )

    # One blocked action adds one action-observation pair,
    # hence two token positions.
    for lag in [
        2,
        4,
        6,
        8,
    ]:

        plt.axvline(
            lag,
            linewidth=0.7,
            linestyle=":",
        )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.xlabel(
        "Relative token displacement"
    )

    plt.ylabel(
        "Position similarity"
    )

    plt.title(
        f"Positional kernel "
        f"({cfg.rotary_mode}, "
        f"base={cfg.rope_base:g})"
    )

    plt.legend()

    plt.tight_layout()


# ============================================================
# MONTE CARLO PLOTS
# ============================================================

def plot_results(
    results,
    cfg,
):
    noise_levels = np.asarray(
        cfg.noise_levels
    )

    # --------------------------------------------------------
    # Same-state similarity
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    for (
        method,
        values,
    ) in results.items():

        plt.plot(
            noise_levels,
            values[
                "same_similarity"
            ],
            marker="o",
            label=method,
        )

    plt.xlabel(
        "Trajectory-noise severity"
    )

    plt.ylabel(
        "Similarity to clean "
        "same-state trajectory"
    )

    plt.title(
        "Robustness to "
        "trajectory noise"
    )

    plt.legend()

    plt.tight_layout()

    # --------------------------------------------------------
    # Margin
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    for (
        method,
        values,
    ) in results.items():

        plt.plot(
            noise_levels,
            values[
                "margin"
            ],
            marker="o",
            label=method,
        )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.xlabel(
        "Trajectory-noise severity"
    )

    plt.ylabel(
        "Same-state similarity "
        "- best other-state similarity"
    )

    plt.title(
        "State-separation margin"
    )

    plt.legend()

    plt.tight_layout()

    # --------------------------------------------------------
    # Prototype classification
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    for (
        method,
        values,
    ) in results.items():

        plt.plot(
            noise_levels,
            values[
                "accuracy"
            ],
            marker="o",
            label=method,
        )

    plt.xlabel(
        "Trajectory-noise severity"
    )

    plt.ylabel(
        "Nearest clean-state "
        "prototype accuracy"
    )

    plt.ylim(
        -0.02,
        1.02,
    )

    plt.title(
        "Latent-state retrieval "
        "under trajectory noise"
    )

    plt.legend()

    plt.tight_layout()


# ============================================================
# ROPE BASE SWEEP
# ============================================================

def run_rope_base_sweep(
    cfg,
):
    """
    Sweep RoPE base.

    RNG is reset for every base so content hypervectors remain
    identical across the sweep.
    """

    print()

    print(
        "=" * 95
    )

    print(
        "ROPE BASE SWEEP"
    )

    print(
        "=" * 95
    )

    cases = [
        "blocked_late",
        "blocked_early",
        "two_blocked",
        "reversal",
    ]

    sweep_results = {
        case: []
        for case in cases
    }

    neighbour_similarities = []

    bases = np.asarray(
        cfg.rope_bases,
        dtype=float,
    )

    for base in bases:

        sweep_cfg = Config(
            **{
                **cfg.__dict__,
                "rope_base":
                    float(base),
                "rotary_mode":
                    "rope",
            }
        )

        # Reset seed so only RoPE frequencies change.
        rng = (
            np.random.default_rng(
                cfg.seed
            )
        )

        encoders = Encoders(
            sweep_cfg,
            rng,
        )

        result = (
            controlled_trajectory_test(
                encoders,
                sweep_cfg,
                verbose=False,
            )
        )

        rotary = result[
            "rotary_vsa"
        ]

        neighbour = (
            rotary[
                "nearest_neighbour"
            ]
        )

        neighbour_similarities.append(
            neighbour
        )

        print()

        print(
            f"ROPE_BASE = "
            f"{base:g}"
        )

        print(
            f"  neighbour = "
            f"{neighbour:.4f}"
        )

        for case in cases:

            similarity = (
                rotary[
                    case
                ][
                    "similarity"
                ]
            )

            margin = (
                rotary[
                    case
                ][
                    "margin"
                ]
            )

            sweep_results[
                case
            ].append(
                margin
            )

            print(
                f"  {case:16s} "
                f"sim={similarity:.4f} "
                f"margin={margin:+.4f}"
            )

    # --------------------------------------------------------
    # Margin plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 5)
    )

    for case in cases:

        plt.plot(
            bases,
            sweep_results[
                case
            ],
            marker="o",
            label=case,
        )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.xscale(
        "log"
    )

    plt.xlabel(
        "RoPE base"
    )

    plt.ylabel(
        "Noisy same-state similarity "
        "- nearest clean neighbour"
    )

    plt.title(
        "Trajectory-noise margin "
        "vs RoPE base"
    )

    plt.legend()

    plt.tight_layout()

    # --------------------------------------------------------
    # Clean neighbour collapse
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        bases,
        neighbour_similarities,
        marker="o",
    )

    plt.xscale(
        "log"
    )

    plt.xlabel(
        "RoPE base"
    )

    plt.ylabel(
        "Nearest clean-state "
        "neighbour similarity"
    )

    plt.title(
        "Clean-state similarity "
        "vs RoPE base"
    )

    plt.tight_layout()

    return sweep_results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "=" * 70
    )

    print(
        "TRAJECTORY NOISE EXPERIMENT"
    )

    print(
        "=" * 70
    )

    print(
        f"rotary mode: "
        f"{CFG.rotary_mode}"
    )

    print(
        f"RoPE base: "
        f"{CFG.rope_base}"
    )

    print(
        f"lambda: "
        f"{CFG.lam}"
    )

    print(
        f"complex dimensions: "
        f"{CFG.dim}"
    )

    print(
        f"window: "
        f"{CFG.window} tokens"
    )

    # --------------------------------------------------------
    # Fixed configuration
    # --------------------------------------------------------

    rng = np.random.default_rng(
        CFG.seed
    )

    encoders = Encoders(
        CFG,
        rng,
    )

    show_examples(
        rng,
        CFG,
    )

    print_kernel_values(
        encoders
    )

    controlled_trajectory_test(
        encoders,
        CFG,
        verbose=True,
    )

    alternative_route_control(
        encoders,
        CFG,
    )

    # --------------------------------------------------------
    # Monte Carlo experiment
    # --------------------------------------------------------

    if CFG.run_monte_carlo:

        # Fresh RNG so the stochastic Monte Carlo experiment
        # does not depend on how many examples were generated
        # above.
        monte_rng = (
            np.random.default_rng(
                CFG.seed + 1
            )
        )

        prototypes = (
            build_prototypes(
                encoders,
                CFG,
            )
        )

        results = (
            run_experiment(
                rng=monte_rng,
                encoders=encoders,
                prototypes=prototypes,
                cfg=CFG,
            )
        )

        print_results(
            results,
            CFG,
        )

        if CFG.show_plots:

            plot_results(
                results,
                CFG,
            )

    # --------------------------------------------------------
    # RoPE base sweep
    # --------------------------------------------------------

    if (
        CFG.run_base_sweep
        and CFG.rotary_mode
        == "rope"
    ):

        run_rope_base_sweep(
            CFG
        )

    # --------------------------------------------------------
    # Kernel plot
    # --------------------------------------------------------

    if CFG.show_plots:

        plot_position_kernel(
            encoders,
            CFG,
        )

        plt.show()