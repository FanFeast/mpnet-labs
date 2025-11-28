import heapq
import math
from typing import List, Optional, Tuple

import numpy as np
from env2d import World2D

GridIndex = Tuple[int, int]  # (i, j) where i is x-index, j is y-index
Point2D = Tuple[float, float]


def point_to_grid(world: World2D, x: float, y: float) -> GridIndex:
    """
    Map a continuous point (x, y) in [0, 1] x [0, 1] to grid indices (i, j).

    Convention:
    - i in [0, grid_size-1] is x index
    - j in [0, grid_size-1] is y index
    - cell (i, j) center is:
            x_c = (i + 0.5) / grid_size
            y_c = (j + 0.5) / grid_size

    For mapping continuous -> index, we can use:
        i = int(x * grid_size)
        j = int(y * grid_size)
    and clamp i, j to [0, grid_size-1].
    """
    return (
        min(max(int(x * world.grid_size), 0), world.grid_size - 1),
        min(max(int(y * world.grid_size), 0), world.grid_size - 1),
    )


def grid_to_point(world: World2D, idx: GridIndex) -> Point2D:
    """
    Map grid indices (i, j) to the continuous center (x, y) of that cell.
    """
    return ((idx[0] + 0.5) / world.grid_size, (idx[1] + 0.5) / world.grid_size)


def astar_on_grid(
    occ: np.ndarray,
    start: GridIndex,
    goal: GridIndex,
) -> Optional[List[GridIndex]]:
    """
    Run A* on a 2D occupancy grid.

    Parameters
    ----------
    occ : np.ndarray of shape (H, W)
        Binary grid. 1 = occupied, 0 = free.
    start, goal : (i, j) tuples
        Start and goal grid indices.

    Returns
    -------
    path : list of (i, j) from start to goal (inclusive),
        or None if no path exists.
    """
    height, width = occ.shape

    def in_bounds(i: int, j: int) -> bool:
        return 0 <= i < width and 0 <= j < height

    def is_free(i: int, j: int) -> bool:
        return occ[j, i] == 0

    def neighbors(i: int, j: int) -> List[GridIndex]:
        """
        4-connected neighbors: up, down, left, right.
        """
        result: List[GridIndex] = []
        candidates = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
        for ni, nj in candidates:
            if in_bounds(ni, nj) and is_free(ni, nj):
                result.append((ni, nj))
        return result

    def heuristic(a: GridIndex, b: GridIndex) -> float:
        """
        Heuristic distance between two grid cells.
        We can use Euclidean distance.
        """
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    # A* boilerplate
    open_heap: List[Tuple[float, GridIndex]] = []
    # g_cost[(i, j)] = cost from start to this node.
    g_cost = {}
    # parent[(i, j)] = previous node in path.
    parent = {}

    # Initialize start
    g_cost[start] = 0.0
    f_start = heuristic(start, goal)
    heapq.heappush(open_heap, (f_start, start))

    while open_heap:
        current_f, current = heapq.heappop(open_heap)
        if current == goal:
            # Reconstruct path from goal back to start.
            path: List[GridIndex] = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            path.reverse()
            return path

        current_g = g_cost[current]

        for nb in neighbors(*current):
            # Cost between adjacent cells (4-connected) is 1.
            tentative_g = current_g + 1.0

            if nb not in g_cost or tentative_g < g_cost[nb]:
                g_cost[nb] = tentative_g
                parent[nb] = current
                f_nb = tentative_g + heuristic(nb, goal)
                heapq.heappush(open_heap, (f_nb, nb))

    # No path found
    return None


def plan_path_astar(
    world: World2D,
    start_xy: Point2D,
    goal_xy: Point2D,
) -> Optional[List[Point2D]]:
    """
    High-level helper:
    1. Convert continuous start/goal to grid indices.
    2. Run A* on the world occupancy grid.
    3. Convert the resulting grid path back to continuous waypoints.

    Returns
    -------
    waypoints : list of (x, y) from start to goal (continuous),
                or None if no path is found.
    """
    # 1. Convert start and goal to grid indices.
    occ = world.get_occupancy_grid()

    start_idx = point_to_grid(world, start_xy[0], start_xy[1])
    goal_idx = point_to_grid(world, goal_xy[0], goal_xy[1])

    sx, sy = start_idx
    gx, gy = goal_idx
    if occ[sy, sx] != 0 or occ[gy, gx] != 0:
        return None

    # 2. Run A* on the occupancy grid.
    occ = world.get_occupancy_grid()
    grid_path = astar_on_grid(occ, start_idx, goal_idx)
    if grid_path is None:
        return None

    # 3. Convert each grid cell in the path back to a continuous waypoint.
    waypoints: List[Point2D] = [grid_to_point(world, idx) for idx in grid_path]
    return waypoints
