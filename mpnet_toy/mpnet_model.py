from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnvEncoder(nn.Module):
    """
    Simple CNN-based environment encoder for 2D occupancy grids.

    Input:  (B, 1, 64, 64) in {0, 1} (float)
    Output: (B, latent_dim)
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)

        # For 64x64 input with stride 2 three times:
        # 64 -> 32 -> 16 -> 8
        # so feature map is (B, 64, 8, 8)
        self.fc = nn.Linear(64 * 8 * 8, latent_dim)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class PlanningNet(nn.Module):
    """
    MLP that predicts next config given:
      - env latent z    : (B, latent_dim)
      - current config  : (B, 2)
      - goal config     : (B, 2)

    Output:
      - next config     : (B, 2)
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        in_dim = latent_dim + 2 + 2
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        z: torch.Tensor,
        x_cur: torch.Tensor,
        x_goal: torch.Tensor,
    ) -> torch.Tensor:
        inp = torch.cat([z, x_cur, x_goal], dim=-1)
        h = F.relu(self.fc1(inp))
        h = F.relu(self.fc2(h))
        x_next = self.fc3(h)
        return x_next


class MPNet2D(nn.Module):
    """
    Minimal 2D MPNet-like model:
      - EnvEncoder
      - PlanningNet
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.encoder = EnvEncoder(latent_dim=latent_dim)
        self.planner = PlanningNet(latent_dim=latent_dim, hidden_dim=hidden_dim)

    def encode_env(self, grid: torch.Tensor) -> torch.Tensor:
        return self.encoder(grid)

    def step(
        self,
        latent: torch.Tensor,
        x_cur: torch.Tensor,
        x_goal: torch.Tensor,
    ) -> torch.Tensor:
        return self.planner(latent, x_cur, x_goal)
