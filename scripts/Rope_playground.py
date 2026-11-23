"""
rotary_vsa_playground.py

Experiments with:

    VSA content encoding
        +
    structured rotary / phasor positional binding
        +
    explicit temporal decay

The core representation is

    h_t = sum_k lambda^k * x_{t-k} * p^k

where:

    x_t     = FHRR content hypervector
    p       = one-step positional phasor
    p^k     = position/lag k
    lambda  = explicit forgetting factor

For an infinite exponentially decaying memory:

    h_t = x_t + lambda * p * h_{t-1}

For an exact finite window of length L:

    h_t = x_t
          + lambda * p * h_{t-1}
          - lambda^L * x_{t-L} * p^L

Complex multiplication is FHRR binding.

Each complex dimension is equivalent to one 2D real rotation plane.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 7

# Number of COMPLEX dimensions.
# Equivalent real dimensionality is 2 * D.
D = 1024

# Memory horizon
L = 40

# Explicit forgetting.
# Set to 1.0 to remove amplitude decay entirely.
LAMBDA = 0.90

# Positional-frequency scheme:
#
#   "rope"
#       Original RoPE-style logarithmic frequencies.
#
#   "gaussian"
#       Frequencies sampled from a zero-centred Gaussian.
#       Expected position kernel is approximately:
#
#           K(delta) = exp(-0.5 * (delta / scale)^2)
#
#   "laplacian"
#       Frequencies sampled from a Cauchy distribution.
#       Expected kernel is approximately:
#
#           K(delta) = exp(-|delta| / scale)
#
#   "uniform"
#       Random frequencies over [-pi, pi].
#       Different integer positions become nearly orthogonal.
#
#   "period_logspace"
#       Explicitly choose a logarithmic range of periods.
#
POSITION_MODE = "rope"

# Parameters for Gaussian / Laplacian modes
POSITION_SCALE = 4.0

# Parameters for period_logspace
MIN_PERIOD = 2.0
MAX_PERIOD = 80.0

# Standard RoPE base
ROPE_BASE = 1000.0

# How far to display positional similarity
MAX_LAG_PLOT = 60


# ============================================================
# BASIC FHRR OPERATIONS
# ============================================================

def random_fhrr(d, rng):
    """
    Generate an FHRR hypervector.

    Every component has unit magnitude and random phase.
    """
    phase = rng.uniform(-np.pi, np.pi, size=d)
    return np.exp(1j * phase)


def bind(a, b):
    """
    FHRR binding.

    For complex unit vectors this simply adds phases.
    """
    return a * b


def inverse(a):
    """
    FHRR inverse / unbinding vector.
    """
    return np.conj(a)


def unbind(a, b):
    """
    Remove b from a.
    """
    return a * np.conj(b)


def complex_cosine(a, b):
    """
    Cosine similarity in the equivalent real-valued space.

    A complex D-vector corresponds to a real 2D-vector:

        [Re(x1), Im(x1), Re(x2), Im(x2), ...]

    Therefore the appropriate real inner product is Re(a^H b).
    """
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return np.real(np.vdot(a, b)) / denominator


# ============================================================
# POSITIONAL FREQUENCIES
# ============================================================

def make_theta(
    d,
    mode,
    rng,
    rope_base=10000.0,
    scale=4.0,
    min_period=2.0,
    max_period=80.0,
):
    """
    Construct angular frequencies theta_i.

    Position k is represented as:

        p_k[i] = exp(j * k * theta_i)

    Consequently:

        conj(p_m) * p_n
            = exp(j * (n-m) * theta)

    so relative displacement appears naturally.
    """

    if mode == "rope":
        # One complex dimension corresponds to one 2D RoPE plane.
        #
        # Original real-valued RoPE:
        #
        #   theta_i = base^(-2i / d_real)
        #
        # with d_real = 2*d_complex, giving:
        #
        #   theta_i = base^(-i / d_complex)
        #
        i = np.arange(d)
        theta = rope_base ** (-i / d)

    elif mode == "gaussian":
        # If:
        #
        #   theta ~ Normal(0, 1/scale^2)
        #
        # then approximately:
        #
        #   E[cos(delta * theta)]
        #       = exp(-delta^2 / (2 scale^2))
        #
        sigma = 1.0 / scale
        theta = rng.normal(
            loc=0.0,
            scale=sigma,
            size=d,
        )

    elif mode == "laplacian":
        # A Cauchy frequency distribution gives approximately:
        #
        #   E[cos(delta * theta)]
        #       = exp(-|delta| / scale)
        #
        gamma = 1.0 / scale
        theta = rng.standard_cauchy(size=d) * gamma

        # Wrapping is optional mathematically, but keeps numerical
        # phases in a more convenient range.
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

    elif mode == "uniform":
        # Essentially an ordinary random FHRR positional generator.
        #
        # For integer non-zero displacement, similarities tend
        # towards zero in high dimensions.
        theta = rng.uniform(
            -np.pi,
            np.pi,
            size=d,
        )

    elif mode == "period_logspace":
        # Explicit periods:
        #
        #   theta = 2*pi / period
        #
        periods = np.geomspace(
            min_period,
            max_period,
            d,
        )

        theta = 2 * np.pi / periods

    else:
        raise ValueError(
            f"Unknown POSITION_MODE: {mode}"
        )

    return theta


def one_step_operator(theta):
    """
    The positional operation corresponding to one time step.
    """
    return np.exp(1j * theta)


def position(theta, k):
    """
    Position / temporal lag k.
    """
    return np.exp(1j * k * theta)


# ============================================================
# POSITIONAL KERNEL
# ============================================================

def positional_similarity(theta, delta):
    """
    Similarity between position 0 and position delta.

    Equivalent to:

        mean_i cos(delta * theta_i)

    for unit-magnitude phasors.
    """
    return np.mean(np.cos(delta * theta))


def positional_kernel(theta, lags):
    return np.array([
        positional_similarity(theta, lag)
        for lag in lags
    ])


# ============================================================
# EXPLICIT HISTORY REPRESENTATION
# ============================================================

def explicit_history(
    stream,
    t,
    theta,
    lam=1.0,
    window=None,
):
    """
    Construct history explicitly from raw input.

    Useful as a reference implementation.

    h_t =
        sum_k lambda^k
              x_{t-k}
              p^k
    """

    if window is None:
        max_k = t + 1
    else:
        max_k = min(window, t + 1)

    h = np.zeros_like(stream[0])

    for k in range(max_k):
        x = stream[t - k]

        h += (
            (lam ** k)
            * bind(
                x,
                position(theta, k)
            )
        )

    return h


# ============================================================
# ONLINE ROTARY MEMORY
# ============================================================

class RotaryMemory:
    """
    Exact finite-window online memory.

    State:

        h_t =
            sum_{k=0}^{L-1}
                lambda^k
                x_{t-k}
                p^k

    Update:

        h <- x_t + lambda * p * h

    and remove the expired term:

        h <- h - lambda^L * x_{t-L} * p^L
    """

    def __init__(
        self,
        theta,
        window,
        lam=1.0,
    ):
        self.theta = np.asarray(theta)

        self.p = one_step_operator(theta)

        self.window = window
        self.lam = lam

        self.state = np.zeros(
            len(theta),
            dtype=np.complex128,
        )

        self.buffer = deque()

        self.expired_position = position(
            theta,
            window,
        )

    def update(self, x):
        # Shift all previous content one temporal position
        # and apply forgetting.
        self.state = (
            x
            + self.lam
            * bind(
                self.state,
                self.p,
            )
        )

        self.buffer.append(x.copy())

        # Remove contribution that has just moved to lag L.
        if len(self.buffer) > self.window:
            expired = self.buffer.popleft()

            self.state -= (
                (self.lam ** self.window)
                * bind(
                    expired,
                    self.expired_position,
                )
            )

        return self.state.copy()


class InfiniteRotaryMemory:
    """
    Infinite exponentially decaying version.

        h_t = x_t + lambda * p * h_{t-1}

    No FIFO is needed.
    """

    def __init__(
        self,
        theta,
        lam=0.9,
    ):
        self.theta = np.asarray(theta)

        self.p = one_step_operator(theta)

        self.lam = lam

        self.state = np.zeros(
            len(theta),
            dtype=np.complex128,
        )

    def update(self, x):
        self.state = (
            x
            + self.lam
            * bind(
                self.state,
                self.p,
            )
        )

        return self.state.copy()


# ============================================================
# SEQUENCE ENCODING
# ============================================================

def encode_sequence(
    sequence,
    theta,
    lam=1.0,
):
    """
    Sequence order convention:

        sequence[-1] = lag 0 / newest item
        sequence[-2] = lag 1
        ...

    Example:

        [A, B, C]

    becomes:

        A*p^2 + B*p^1 + C*p^0
    """

    h = np.zeros_like(sequence[0])

    for lag, x in enumerate(reversed(sequence)):
        h += (
            (lam ** lag)
            * bind(
                x,
                position(theta, lag)
            )
        )

    return h


def shift_representation(
    representation,
    theta,
    steps=1,
):
    """
    Move an entire encoded sequence backwards by `steps`.
    """
    return bind(
        representation,
        position(theta, steps),
    )


# ============================================================
# EXPERIMENTS
# ============================================================

def experiment_relative_position(theta):
    print()
    print("=" * 70)
    print("EXPERIMENT 1: RELATIVE POSITION ALGEBRA")
    print("=" * 70)

    x = random_fhrr(D, rng)

    pairs = [
        (2, 5),
        (7, 10),
        (1, 4),
    ]

    for m, n in pairs:

        xm = bind(
            x,
            position(theta, m),
        )

        xn = bind(
            x,
            position(theta, n),
        )

        similarity = complex_cosine(
            xm,
            xn,
        )

        expected = positional_similarity(
            theta,
            n - m,
        )

        print(
            f"m={m:2d}, "
            f"n={n:2d}, "
            f"delta={n-m:2d} | "
            f"similarity={similarity: .6f} | "
            f"K(delta)={expected: .6f}"
        )


def experiment_content_vs_position(theta):
    print()
    print("=" * 70)
    print("EXPERIMENT 2: CONTENT VS POSITION")
    print("=" * 70)

    A = random_fhrr(D, rng)
    B = random_fhrr(D, rng)

    print()
    print("Same content at different temporal offsets:")

    for lag in [0, 1, 2, 3, 4, 8, 16, 32]:

        A0 = bind(
            A,
            position(theta, 0),
        )

        Ak = bind(
            A,
            position(theta, lag),
        )

        print(
            f"A@0 vs A@{lag:2d}: "
            f"{complex_cosine(A0, Ak): .4f}"
        )

    print()
    print("Different content:")

    for lag in [0, 1, 2, 4, 8]:

        A0 = bind(
            A,
            position(theta, 0),
        )

        Bk = bind(
            B,
            position(theta, lag),
        )

        print(
            f"A@0 vs B@{lag:2d}: "
            f"{complex_cosine(A0, Bk): .4f}"
        )


def experiment_sequence_shift(theta):
    print()
    print("=" * 70)
    print("EXPERIMENT 3: WHOLE-SEQUENCE TRANSLATION")
    print("=" * 70)

    A = random_fhrr(D, rng)
    B = random_fhrr(D, rng)
    C = random_fhrr(D, rng)

    sequence = [A, B, C]

    representation = encode_sequence(
        sequence,
        theta,
        lam=1.0,
    )

    print()
    print("Same sequence shifted in temporal coordinates:")

    for shift in range(0, 9):

        shifted = shift_representation(
            representation,
            theta,
            steps=shift,
        )

        similarity = complex_cosine(
            representation,
            shifted,
        )

        print(
            f"shift {shift:2d}: "
            f"{similarity: .4f}"
        )


def experiment_order_sensitivity(theta):
    print()
    print("=" * 70)
    print("EXPERIMENT 4: ORDER SENSITIVITY")
    print("=" * 70)

    A = random_fhrr(D, rng)
    B = random_fhrr(D, rng)
    C = random_fhrr(D, rng)
    Dv = random_fhrr(D, rng)

    sequences = {
        "A B C D": [A, B, C, Dv],
        "D C B A": [Dv, C, B, A],
        "A B D C": [A, B, Dv, C],
        "B C D A": [B, C, Dv, A],
    }

    encoded = {
        name: encode_sequence(
            seq,
            theta,
            lam=LAMBDA,
        )
        for name, seq in sequences.items()
    }

    names = list(encoded.keys())

    print()

    print(" " * 12, end="")

    for name in names:
        print(f"{name:>12}", end="")

    print()

    for name1 in names:

        print(
            f"{name1:>12}",
            end="",
        )

        for name2 in names:

            sim = complex_cosine(
                encoded[name1],
                encoded[name2],
            )

            print(
                f"{sim:12.3f}",
                end="",
            )

        print()


def experiment_online_update(theta):
    print()
    print("=" * 70)
    print("EXPERIMENT 5: ONLINE UPDATE VS FULL RECONSTRUCTION")
    print("=" * 70)

    stream = [
        random_fhrr(D, rng)
        for _ in range(100)
    ]

    memory = RotaryMemory(
        theta=theta,
        window=L,
        lam=LAMBDA,
    )

    errors = []

    for t, x_t in enumerate(stream):

        online = memory.update(x_t)

        explicit = explicit_history(
            stream,
            t,
            theta,
            lam=LAMBDA,
            window=L,
        )

        error = np.linalg.norm(
            online - explicit
        )

        errors.append(error)

    print()
    print(
        "Maximum ||online - explicit||:",
        f"{max(errors):.3e}",
    )

    print(
        "Mean ||online - explicit||:   ",
        f"{np.mean(errors):.3e}",
    )


def experiment_random_vs_structured():
    print()
    print("=" * 70)
    print("EXPERIMENT 6: STRUCTURED VS RANDOM POSITION GENERATORS")
    print("=" * 70)

    structured_theta = make_theta(
        d=D,
        mode=POSITION_MODE,
        rng=rng,
        rope_base=ROPE_BASE,
        scale=POSITION_SCALE,
        min_period=MIN_PERIOD,
        max_period=MAX_PERIOD,
    )

    random_theta = make_theta(
        d=D,
        mode="uniform",
        rng=rng,
    )

    print()
    print(
        f"{'lag':>4} "
        f"{'structured':>14} "
        f"{'random':>14}"
    )

    for lag in range(0, 16):

        s_struct = positional_similarity(
            structured_theta,
            lag,
        )

        s_random = positional_similarity(
            random_theta,
            lag,
        )

        print(
            f"{lag:4d} "
            f"{s_struct:14.4f} "
            f"{s_random:14.4f}"
        )


# ============================================================
# PLOTS
# ============================================================

def plot_position_kernels():
    lags = np.arange(
        0,
        MAX_LAG_PLOT + 1,
    )

    schemes = [
        "rope",
        "gaussian",
        "laplacian",
        "uniform",
        "period_logspace",
    ]

    plt.figure(figsize=(10, 6))

    for mode in schemes:

        local_rng = np.random.default_rng(
            SEED
        )

        theta = make_theta(
            d=D,
            mode=mode,
            rng=local_rng,
            rope_base=ROPE_BASE,
            scale=POSITION_SCALE,
            min_period=MIN_PERIOD,
            max_period=MAX_PERIOD,
        )

        kernel = positional_kernel(
            theta,
            lags,
        )

        plt.plot(
            lags,
            kernel,
            label=mode,
        )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.xlabel(
        "Relative temporal displacement"
    )

    plt.ylabel(
        "Positional similarity"
    )

    plt.title(
        "Relative-position kernels"
    )

    plt.legend()

    plt.tight_layout()


def plot_decay_vs_position(theta):
    lags = np.arange(
        0,
        MAX_LAG_PLOT + 1,
    )

    position_only = positional_kernel(
        theta,
        lags,
    )

    amplitude_decay = (
        LAMBDA ** lags
    )

    combined = (
        position_only
        * amplitude_decay
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        lags,
        position_only,
        label="positional similarity only",
    )

    plt.plot(
        lags,
        amplitude_decay,
        label="lambda^lag only",
    )

    plt.plot(
        lags,
        combined,
        label="combined",
    )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.xlabel(
        "Lag"
    )

    plt.ylabel(
        "Relative contribution / similarity"
    )

    plt.title(
        "Position geometry vs explicit forgetting"
    )

    plt.legend()

    plt.tight_layout()


def plot_target_kernel_comparison():
    """
    For Gaussian and Laplacian frequency schemes,
    compare finite-dimensional approximation against
    their theoretical target kernels.
    """

    lags = np.arange(
        0,
        MAX_LAG_PLOT + 1,
    )

    # ----------------------------
    # Gaussian
    # ----------------------------

    local_rng = np.random.default_rng(
        SEED
    )

    gaussian_theta = make_theta(
        D,
        "gaussian",
        local_rng,
        scale=POSITION_SCALE,
    )

    measured_gaussian = positional_kernel(
        gaussian_theta,
        lags,
    )

    target_gaussian = np.exp(
        -0.5
        * (lags / POSITION_SCALE) ** 2
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        lags,
        measured_gaussian,
        label="sampled VSA kernel",
    )

    plt.plot(
        lags,
        target_gaussian,
        linestyle="--",
        label="theoretical Gaussian kernel",
    )

    plt.xlabel("Lag")
    plt.ylabel("Similarity")

    plt.title(
        f"Gaussian positional kernel "
        f"(scale={POSITION_SCALE})"
    )

    plt.legend()
    plt.tight_layout()

    # ----------------------------
    # Laplacian
    # ----------------------------

    local_rng = np.random.default_rng(
        SEED
    )

    laplacian_theta = make_theta(
        D,
        "laplacian",
        local_rng,
        scale=POSITION_SCALE,
    )

    measured_laplacian = positional_kernel(
        laplacian_theta,
        lags,
    )

    target_laplacian = np.exp(
        -lags / POSITION_SCALE
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        lags,
        measured_laplacian,
        label="sampled VSA kernel",
    )

    plt.plot(
        lags,
        target_laplacian,
        linestyle="--",
        label="theoretical exponential kernel",
    )

    plt.xlabel("Lag")
    plt.ylabel("Similarity")

    plt.title(
        f"Laplacian positional kernel "
        f"(scale={POSITION_SCALE})"
    )

    plt.legend()
    plt.tight_layout()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    rng = np.random.default_rng(
        SEED
    )

    theta = make_theta(
        d=D,
        mode=POSITION_MODE,
        rng=rng,
        rope_base=ROPE_BASE,
        scale=POSITION_SCALE,
        min_period=MIN_PERIOD,
        max_period=MAX_PERIOD,
    )

    print()
    print("=" * 70)
    print("ROTARY VSA PLAYGROUND")
    print("=" * 70)

    print(
        f"complex dimensions: {D}"
    )

    print(
        f"equivalent real dimensions: {2 * D}"
    )

    print(
        f"window length: {L}"
    )

    print(
        f"lambda: {LAMBDA}"
    )

    print(
        f"position mode: {POSITION_MODE}"
    )

    print(
        f"position scale: {POSITION_SCALE}"
    )

    experiment_relative_position(
        theta
    )

    experiment_content_vs_position(
        theta
    )

    experiment_sequence_shift(
        theta
    )

    experiment_order_sensitivity(
        theta
    )

    experiment_online_update(
        theta
    )

    experiment_random_vs_structured()

    plot_position_kernels()

    plot_decay_vs_position(
        theta
    )

    plot_target_kernel_comparison()

    plt.show()