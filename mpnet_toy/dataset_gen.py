import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
from env2d import World2D
from grid_planner import plan_path_astar

Point2D = Tuple[float, float]


def collect_samples_from_path(
    grid: np.ndarray,
    path: List[Point2D],
) -> List[Dict[str, np.ndarray]]:
    """
    Given a single expert path (sequence of waypoints) and its occupancy grid,
    create one-step training samples:
        (grid, x_cur, x_goal, x_next)

    We use:
      - x_cur  = path[t]
      - x_next = path[t+1]
      - x_goal = path[-1]  (same for all t)

    Returns a list of dicts, each dict containing:
      "grid"   : (H, W) float32 or uint8
      "x_cur"  : (2,) float32
      "x_goal" : (2,) float32
      "x_next" : (2,) float32
    """
    samples: List[Dict[str, np.ndarray]] = []

    if len(path) < 2:
        return samples

    grid_f = grid.astype(np.float32)  # keep a single copy for all steps
    x_goal = np.asarray(path[-1], dtype=np.float32)

    for t in range(len(path) - 1):
        x_cur = np.asarray(path[t], dtype=np.float32)
        x_next = np.asarray(path[t + 1], dtype=np.float32)

        sample = {
            "grid": grid_f,  # we can share the same array reference
            "x_cur": x_cur,
            "x_goal": x_goal,
            "x_next": x_next,
        }
        samples.append(sample)

    return samples


def generate_dataset_for_world(
    world: World2D,
    num_paths: int = 20,
    max_tries_per_path: int = 100,
) -> List[Dict[str, np.ndarray]]:
    """
    Generate one-step samples from a single random world.

    Algorithm:
      - For each of num_paths:
          * Try up to max_tries_per_path times:
              - sample a random free start and goal
              - run A* to get a path
              - if path is found and has at least 2 waypoints:
                    * create samples from that path
                    * break out of the retry loop for this path
      - Return the list of samples for this world.
    """
    all_samples: List[Dict[str, np.ndarray]] = []
    grid = world.get_occupancy_grid()

    for _ in range(num_paths):
        found = False
        for _ in range(max_tries_per_path):
            start = world.sample_free_point()
            goal = world.sample_free_point()
            path = plan_path_astar(world, start, goal)
            if path is None or len(path) < 2:
                continue
            samples = collect_samples_from_path(grid, path)
            all_samples.extend(samples)
            found = True
            break
        # Optional: you could log or print something if not found,
        # but for now we just skip if we fail.
        if not found:
            print("Warning: No valid path found after max tries; skipping this path.")
            pass

    return all_samples


def generate_dataset_many_worlds(
    num_worlds: int = 10,
    num_paths_per_world: int = 20,
    grid_size: int = 64,
    seed: int = 0,
    num_threads: int = 4,
) -> Dict[str, np.ndarray]:
    """
    Generate a full dataset over many random worlds using multithreading.

    Saves intermediate results to disk and combines them at the end.

    Returns a dict of numpy arrays suitable for saving to .npz or feeding
    into a PyTorch Dataset later:

      {
        "grids":  (N, 1, H, W) float32
        "x_cur":  (N, 2) float32
        "x_goal": (N, 2) float32
        "x_next": (N, 2) float32
      }

    N is the total number of one-step samples across all worlds.
    """

    rng = np.random.RandomState(seed)

    # Pre-generate seeds for each world to avoid thread safety issues with rng
    world_seeds = rng.randint(0, 1_000_000, size=num_worlds)

    # Create a temporary directory to store intermediate files
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory for intermediate datasets: {temp_dir}")

    def process_world_and_save(args: Tuple[int, int]) -> str:
        """Process a single world, save samples to disk, and return filename."""
        world_idx, world_seed = args

        world = World2D.random_world(
            num_obstacles=3,
            grid_size=grid_size,
            min_size=0.1,
            max_size=0.3,
            seed=int(world_seed),
        )

        samples = generate_dataset_for_world(
            world,
            num_paths=num_paths_per_world,
            max_tries_per_path=100,
        )

        if not samples:
            return ""

        # Convert samples to arrays for this world
        w_grids = [s["grid"] for s in samples]
        w_x_cur = [s["x_cur"] for s in samples]
        w_x_goal = [s["x_goal"] for s in samples]
        w_x_next = [s["x_next"] for s in samples]

        grids_arr = np.stack(w_grids, axis=0)
        x_cur_arr = np.stack(w_x_cur, axis=0)
        x_goal_arr = np.stack(w_x_goal, axis=0)
        x_next_arr = np.stack(w_x_next, axis=0)

        # Add channel dim for grid: (N, 1, H, W)
        grids_arr = grids_arr[:, None, :, :].astype(np.float32)

        dataset_chunk = {
            "grids": grids_arr,
            "x_cur": x_cur_arr.astype(np.float32),
            "x_goal": x_goal_arr.astype(np.float32),
            "x_next": x_next_arr.astype(np.float32),
        }

        filename = os.path.join(temp_dir, f"world_{world_idx}.npz")
        np.savez_compressed(filename, **dataset_chunk)
        return filename

    saved_files: List[str] = []

    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Pass both index and seed
            results = executor.map(
                process_world_and_save, zip(range(num_worlds), world_seeds)
            )

            for fname in results:
                if fname:
                    saved_files.append(fname)

        if not saved_files:
            raise RuntimeError("No samples generated. Check world parameters.")

        # Combine all datasets
        all_grids = []
        all_x_cur = []
        all_x_goal = []
        all_x_next = []

        print(f"Combining {len(saved_files)} intermediate datasets...")
        for fname in saved_files:
            with np.load(fname) as data:
                all_grids.append(data["grids"])
                all_x_cur.append(data["x_cur"])
                all_x_goal.append(data["x_goal"])
                all_x_next.append(data["x_next"])

        # Concatenate
        final_grids = np.concatenate(all_grids, axis=0)
        final_x_cur = np.concatenate(all_x_cur, axis=0)
        final_x_goal = np.concatenate(all_x_goal, axis=0)
        final_x_next = np.concatenate(all_x_next, axis=0)

        dataset = {
            "grids": final_grids,
            "x_cur": final_x_cur,
            "x_goal": final_x_goal,
            "x_next": final_x_next,
        }

    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")

    return dataset


def save_dataset_npz(path: str, dataset: Dict[str, np.ndarray]) -> None:
    """
    Save the dataset dict to a .npz file.
    """
    np.savez_compressed(path, **dataset)
