import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from typing import Any
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from maze_environment.LEVEL import levels
from maze_environment.maze_environment_v3 import MazeEnvironment



plt.style.use(["science", "no-latex"])

# sns.set_theme(
#     context="paper",
#     style="whitegrid",
#     font_scale=1.1,
#     rc={
#         "figure.dpi": 150,
#         "savefig.dpi": 300,
#         "axes.spines.top": False,
#         "axes.spines.right": False,
#         "axes.linewidth": 0.8,
#         "grid.linewidth": 0.4,
#         "lines.linewidth": 1.8,
#         "legend.frameon": False,
#     },
# )

sns.set_theme(style="white")

DEFAULT_LEVEL_INDEX = 21

DEFAULT_AGENT_SENSORS = {"sensor": "cardinal distance", "range": 1}
MAZE_CMAP = ListedColormap(["peru", "olivedrab"])
MAZE_LEGEND_HANDLES = [
    Patch(facecolor="peru", edgecolor="dimgray", label="Free space"),
    Patch(facecolor="olivedrab", edgecolor="dimgray", label="Occupied space"),
]


def load_environment(
    level_index: int = DEFAULT_LEVEL_INDEX,
    agent_sensors: dict | None = None,
    seed: int = 0,
) -> MazeEnvironment:
    """Build an import-safe maze environment for downstream plotting."""
    if agent_sensors is None:
        agent_sensors = DEFAULT_AGENT_SENSORS

    return MazeEnvironment(
        level=levels[level_index],
        plotting=False,
        agent_sensors=agent_sensors,
        seed=seed,
    )


def environment_to_plot_data(env: MazeEnvironment) -> dict:
    """Extract the environment state needed by plotting code."""
    return {
        "maze": np.asarray(env.maze),
        "agent_position": tuple(env.agent.position),
        "initial_agent_position": tuple(env.initial_agent_position),
        "action_keys": tuple(env.action_keys),
    }


def free_grid_positions(env: MazeEnvironment) -> list[tuple[int, int]]:
    """Return every non-wall grid coordinate as (row, col)."""
    free_rows, free_cols = np.where(env.maze == 0)
    return list(zip(free_rows.tolist(), free_cols.tolist(), strict=True))


def normalise_observation(observation: Any) -> Any:
    """Convert numpy observations to plain Python values for printing/export."""
    if isinstance(observation, np.ndarray):
        return observation.tolist()
    if isinstance(observation, tuple):
        return tuple(normalise_observation(value) for value in observation)
    if isinstance(observation, list):
        return [normalise_observation(value) for value in observation]
    if isinstance(observation, np.generic):
        return observation.item()
    return observation


def observation_key(observation: Any) -> Any:
    """Convert an observation to a hashable key for categorical colouring."""
    observation = normalise_observation(observation)
    if isinstance(observation, list):
        return tuple(observation_key(value) for value in observation)
    if isinstance(observation, tuple):
        return tuple(observation_key(value) for value in observation)
    return observation


def free_grid_observations(env: MazeEnvironment) -> list[dict[str, Any]]:
    """Get the configured sensor observation from each free grid space."""
    original_agent_position = tuple(env.agent.position)
    original_env_position = tuple(env.agent_position)
    original_observation = env.agent.observation

    observations = []
    try:
        for position in free_grid_positions(env):
            env.agent.position = position
            env.agent_position = position
            observation = env.agent.make_observation(env.maze)
            observations.append(
                {
                    "position": position,
                    "observation": normalise_observation(observation),
                }
            )
    finally:
        env.agent.position = original_agent_position
        env.agent_position = original_env_position
        env.agent.observation = original_observation

    return observations


def observation_heatmap_data(env: MazeEnvironment) -> tuple[dict[tuple[int, int], int], list[Any]]:
    """Create position-keyed category data for unique observations."""
    observations = free_grid_observations(env)
    unique_observations = sorted(
        {observation_key(item["observation"]) for item in observations},
        key=repr,
    )
    observation_to_category = {
        observation: category_id
        for category_id, observation in enumerate(unique_observations)
    }

    heatmap_data = {}
    for item in observations:
        position = item["position"]
        heatmap_data[position] = observation_to_category[observation_key(item["observation"])]

    return heatmap_data, unique_observations


def plot_observation_heatmap(
    env: MazeEnvironment,
    output_path: str | None = None,
) -> Axes:
    """Plot ground-truth tile positions coloured by each tile's observation."""
    heatmap_data, unique_observations = observation_heatmap_data(env)
    observation_colours = sns.color_palette("husl", len(unique_observations)).as_hex()

    _, ax = plt.subplots()
    for (row, col), category_id in heatmap_data.items():
        colour = observation_colours[category_id]
        ax.add_patch(
            Rectangle(
                (col, row),
                width=1,
                height=1,
                facecolor=colour,
                edgecolor="dimgray",
                linewidth=0.5,
            )
        )

    legend_handles = [
        *[
            Patch(
                facecolor=colour,
                edgecolor="dimgray",
                label=f"O{index}",
            )
            for index, colour in enumerate(
                observation_colours,
            )
        ],
    ]
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, env.maze.shape[1])
    ax.set_ylim(env.maze.shape[0], 0)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.figure.subplots_adjust(right=0.68)

    if output_path is not None:
        ax.figure.savefig(output_path, dpi=300, bbox_inches="tight")

    return ax


def main() -> None:
    env = load_environment()
    plot_data = environment_to_plot_data(env)

    observation_data = free_grid_observations(env)
    for item in observation_data:
        print(item)

    ax = sns.heatmap(
        plot_data["maze"],
        cmap=MAZE_CMAP,
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="dimgray",
        cbar=False,
    )
    ax.legend(
        handles=MAZE_LEGEND_HANDLES,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(MAZE_LEGEND_HANDLES),
        frameon=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.subplots_adjust(bottom=0.18)
    plt.savefig(f"{DEFAULT_LEVEL_INDEX}_maze.png", dpi=300)
    plt.show()

    plot_observation_heatmap(
        env,
        output_path=f"{DEFAULT_LEVEL_INDEX}_observation_heatmap.png",
    )
    plt.show()

    # Keep imports and extracted data available for interactive plotting work.
    _ = plt, plot_data, observation_data


if __name__ == "__main__":
    main()
