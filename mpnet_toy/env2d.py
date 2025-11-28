# env2d.py
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class RectObstacle:
    """
    Axis-aligned rectangular obstacle in [0, 1] x [0, 1].

    Attributes
    ----------
    xmin, ymin, xmax, ymax : float
        Coordinates of the rectangle bounds.
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains_point(self, x: float, y: float) -> bool:
        """
        Return True if (x, y) lies inside or on the boundary of this rectangle.
        """
        return (self.xmin <= x <= self.xmax) and (self.ymin <= y <= self.ymax)


class World2D:
    """
    Simple 2D world with axis-aligned rectangular obstacles.

    The continuous workspace is [0, 1] x [0, 1].

    An internal occupancy grid with shape (grid_size, grid_size) is built,
    where:
        1 = occupied (inside an obstacle)
        0 = free
    """

    def __init__(self, obstacles: List[RectObstacle], grid_size: int = 64):
        """
        Parameters
        ----------
        obstacles : list of RectObstacle
            List of obstacles in the world.
        grid_size : int
            Resolution of the occupancy grid along each dimension.
        """
        self.obstacles = obstacles
        self.grid_size = grid_size

        # Build occupancy once at construction time
        self.occupancy = self._rasterize_obstacles()

    @classmethod
    def random_world(
        cls,
        num_obstacles: int = 3,
        grid_size: int = 64,
        min_size: float = 0.1,
        max_size: float = 0.3,
        seed: int = None,
    ) -> "World2D":
        """
        Create a random world with a few random axis-aligned rectangles.

        Each rectangle:
          - width and height are sampled in [min_size, max_size]
          - position is chosen so the rectangle lies inside [0, 1]^2
        """
        # TODO:
        # 1. Set random seeds if seed is not None.
        # 2. Loop num_obstacles times:
        #    - sample width w and height h in [min_size, max_size]
        #    - sample xmin, ymin so that xmax = xmin + w, ymax = ymin + h
        #      and the rectangle stays inside [0, 1].
        #    - create RectObstacle and append to a list.
        # 3. Return cls(obstacles=that_list, grid_size=grid_size).
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        obstacles: List[RectObstacle] = []
        for _ in range(num_obstacles):
            w = random.uniform(min_size, max_size)
            h = random.uniform(min_size, max_size)
            xmin = random.uniform(0, 1 - w)
            ymin = random.uniform(0, 1 - h)
            xmax = xmin + w
            ymax = ymin + h
            obstacles.append(RectObstacle(xmin, ymin, xmax, ymax))

        return World2D(obstacles=obstacles, grid_size=grid_size)

    def _point_in_any_obstacle(self, x: float, y: float) -> bool:
        """
        Return True if (x, y) is inside any obstacle.
        """
        for obstacle in self.obstacles:
            if obstacle.contains_point(x, y):
                return True
        return False

    def _rasterize_obstacles(self) -> np.ndarray:
        """
        Build and return a binary occupancy grid of shape (grid_size, grid_size).

        Convention:
          - occupancy[j, i] corresponds to the cell whose center is at:
                x_c = (i + 0.5) / grid_size
                y_c = (j + 0.5) / grid_size
          - occupancy[j, i] = 1 if that center lies inside any obstacle
            else 0.
        """
        occ = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        for j in range(self.grid_size):
            y_c = (j + 0.5) / self.grid_size
            for i in range(self.grid_size):
                x_c = (i + 0.5) / self.grid_size
                if self._point_in_any_obstacle(x_c, y_c):
                    occ[j, i] = 1
        return occ

    def get_occupancy_grid(self) -> np.ndarray:
        """
        Return a copy of the occupancy grid (values 0 or 1, shape (H, W)).
        """
        return self.occupancy.copy()

    def is_point_in_collision(self, x: float, y: float) -> bool:
        """
        Check if a single point in continuous space is in collision.

        Rules:
          - If the point is outside [0, 1]^2, treat it as collision.
          - Otherwise, collision iff it lies inside any obstacle.
        """
        # TODO:
        # 1. If x or y is outside [0, 1], return True.
        # 2. Else, return self._point_in_any_obstacle(x, y).
        if 0 < x < 1 and 0 < y < 1:
            return self._point_in_any_obstacle(x, y)
        else:
            return True

    def is_segment_in_collision(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        num_samples: int = 50,
    ) -> bool:
        """
        Check if the line segment p0 -> p1 collides with any obstacle.

        Implementation idea:
          - Sample num_samples+1 points along the segment using linear
            interpolation:
                t in [0, 1], x = (1 - t)*x0 + t*x1, y similar
          - If any sampled point is in collision, return True.
          - If none are in collision, return False.
        """
        x0, y0 = p0
        x1, y1 = p1

        for i in range(num_samples + 1):
            t = i / num_samples
            x = (1 - t) * x0 + t * x1
            y = (1 - t) * y0 + t * y1
            if self.is_point_in_collision(x, y):
                return True
        return False

    def sample_free_point(self, max_tries: int = 1000) -> Tuple[float, float]:
        """
        Randomly sample a collision-free point in the world.

        Algorithm:
          - Try up to max_tries times:
              * sample x, y uniformly in [0, 1]
              * if not in collision, return (x, y)
          - If we fail max_tries times in a row, raise RuntimeError.
        """
        for _ in range(max_tries):
            x = random.random()
            y = random.random()
            if not self.is_point_in_collision(x, y):
                return (x, y)
        raise RuntimeError("Failed to sample a free point within max_tries.")
