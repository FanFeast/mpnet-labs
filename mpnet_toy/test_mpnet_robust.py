"""
Lightweight robustness test for MPNet2D.

We sample random worlds and attempt to reach random goals by rolling out the
network in closed loop. Useful for checking whether recent model changes still
produce collision-free, goal-reaching behavior.
"""

from __future__ import annotations

import argparse
import random
from typing import Tuple

import numpy as np
import torch
from env2d import World2D
from mpnet_model import MPNet2D
from torch.amp import autocast


def clamp_step(
    x_cur: torch.Tensor,
    x_next: torch.Tensor,
    max_step: float | None,
) -> torch.Tensor:
    """
    Optionally clamp the step size to avoid wild jumps.
    """
    if max_step is None:
        return x_next

    delta = x_next - x_cur
    step_norm = torch.linalg.norm(delta, dim=-1, keepdim=True) + 1e-9
    scale = torch.clamp(max_step / step_norm, max=1.0)
    return x_cur + delta * scale


def rollout_episode(
    model: MPNet2D,
    world: World2D,
    device: torch.device,
    max_steps: int,
    goal_tolerance: float,
    max_step: float | None,
) -> Tuple[bool, int]:
    """
    Run one start->goal attempt. Returns (success flag, steps taken).
    """
    grid = world.get_occupancy_grid().astype(np.float32)[None, None, :, :]
    grid_t = torch.from_numpy(grid).to(device)

    start = world.sample_free_point()
    goal = world.sample_free_point()

    x_cur = torch.tensor(start, device=device, dtype=torch.float32).unsqueeze(0)
    x_goal = torch.tensor(goal, device=device, dtype=torch.float32).unsqueeze(0)

    with autocast(device_type=device.type, enabled=device.type == "cuda"):
        z = model.encode_env(grid_t)

    for step_idx in range(1, max_steps + 1):
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            x_next = model.step(z, x_cur, x_goal)

        x_next = clamp_step(x_cur, x_next, max_step=max_step)
        x_next = torch.clamp(x_next, 0.0, 1.0)
        next_pt = tuple(x_next.squeeze(0).tolist())

        if world.is_segment_in_collision(
            start if step_idx == 1 else tuple(x_cur.squeeze(0).tolist()), next_pt
        ):
            return False, step_idx

        # Check goal
        if np.linalg.norm(np.array(next_pt) - np.array(goal)) < goal_tolerance:
            return True, step_idx

        x_cur = x_next

    return False, max_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Closed-loop rollout test for MPNet2D."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/mukul/fanfeast-stuff/mpnet-labs/mpnet_toy_model.pt",
        help="Path to the saved model weights.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of random start/goal attempts to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=64,
        help="Maximum rollout length per episode.",
    )
    parser.add_argument(
        "--goal-tolerance",
        type=float,
        default=0.05,
        help="Distance to goal considered success.",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=0.2,
        help="Optional clamp on per-step movement (set to 0 to disable).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=64,
        help="Latent dimension used when creating MPNet2D.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for the planning MLP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print("Using device:", device)

    model = MPNet2D(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    raw_state = torch.load(args.checkpoint, map_location=device)
    clean_state = {
        k.replace("._orig_mod.", "."): v
        for k, v in raw_state.items()
    }
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing or unexpected:
        print("Warning: load_state_dict mismatches")
        print("  missing:", missing)
        print("  unexpected:", unexpected)
    model.eval()

    successes = 0
    steps_for_success = []

    max_step = args.max_step if args.max_step > 0 else None

    for ep in range(1, args.episodes + 1):
        world = World2D.random_world(num_obstacles=3, grid_size=64, seed=args.seed + ep)
        success, steps = rollout_episode(
            model,
            world,
            device,
            max_steps=args.max_steps,
            goal_tolerance=args.goal_tolerance,
            max_step=max_step,
        )
        successes += int(success)
        if success:
            steps_for_success.append(steps)
        print(f"Episode {ep:02d}: {'SUCCESS' if success else 'FAIL'} in {steps} steps")

    success_rate = successes / max(args.episodes, 1)
    mean_steps = (
        (sum(steps_for_success) / len(steps_for_success))
        if steps_for_success
        else float("nan")
    )

    print("\nSummary:")
    print(f"  Success rate: {success_rate * 100:.2f}% ({successes}/{args.episodes})")
    print(f"  Mean steps (successful episodes): {mean_steps:.2f}")


if __name__ == "__main__":
    main()
