# test_planner.py
from env2d import World2D
from grid_planner import plan_path_astar


def main() -> None:
    world = World2D.random_world(num_obstacles=3, grid_size=64, seed=0)

    start = world.sample_free_point()
    goal = world.sample_free_point()
    print("Start:", start)
    print("Goal:", goal)

    path = plan_path_astar(world, start, goal)
    if path is None:
        print("No path found.")
        return

    print("Path length (num waypoints):", len(path))
    print("First few waypoints:")
    for p in path[:5]:
        print("  ", p)
    print("Last waypoint:", path[-1])


if __name__ == "__main__":
    main()
