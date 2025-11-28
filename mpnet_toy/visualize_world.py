from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

Point2D = Tuple[float, float]


def show_occupancy_grid(grid: np.ndarray, ax=None) -> None:
    """
    Visualize occupancy grid (H, W) with 1=occupied in black.
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(grid, cmap="gray_r", origin="lower")
    ax.set_title("Occupancy Grid")
    ax.set_xlabel("i")
    ax.set_ylabel("j")


def show_path_on_grid(
    grid: np.ndarray,
    path: List[Point2D],
    ax=None,
    color="red",
    label="Path",
) -> None:
    """
    Overlay a continuous-space path on the grid visualization.

    The path points are in continuous coords [0,1], so we map to grid pixels:
        px = x * W
        py = y * H
    """
    H, W = grid.shape

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(grid, cmap="gray_r", origin="lower")
    xs = [p[0] * W for p in path]
    ys = [p[1] * H for p in path]
    ax.plot(xs, ys, color=color, linewidth=2, label=label)
    ax.scatter(xs[0], ys[0], c="green", s=30, label="Start")
    ax.scatter(xs[-1], ys[-1], c="red", s=30, label="Goal")
    ax.set_title(label)
    ax.legend()


def compare_paths(
    grid: np.ndarray,
    path_astar: List[Point2D],
    path_mpnet: List[Point2D],
    title: str = "A* vs MPNet",
) -> None:
    """
    Plot A* path and MPNet-predicted path side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    show_path_on_grid(grid, path_astar, ax=axes[0], color="blue", label="A* Path")
    axes[0].set_title("A* Path")

    show_path_on_grid(grid, path_mpnet, ax=axes[1], color="red", label="MPNet Path")
    axes[1].set_title("MPNet Path")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
