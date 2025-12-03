# demo_astar_vis.py
import matplotlib.pyplot as plt
from env2d import World2D
from grid_planner import plan_path_astar
from visualize_world import show_path_on_grid


def main() -> None:
    world = World2D.random_world(num_obstacles=9, grid_size=64, seed=1)
    grid = world.get_occupancy_grid()

    start = world.sample_free_point()
    goal = world.sample_free_point()

    path = plan_path_astar(world, start, goal)
    if path is None:
        print("No path found.")
        return

    show_path_on_grid(grid, path, label="A* Path")
    plt.show()


if __name__ == "__main__":
    main()
