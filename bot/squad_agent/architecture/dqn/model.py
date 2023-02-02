"""
Bring the Encoder, the DQN and DQN target network together for ease
This makes saving and loading the model simpler
"""
import numpy as np
import torch
from torch import nn, Tensor

# relative import required for training with docker
try:
    from bot.squad_agent.architecture.dqn.dqn_model import Network
    from bot.squad_agent.architecture.encoder import Encoder
except ImportError:
    from ...architecture.encoder import Encoder
    from ...architecture.dqn.dqn_model import Network


class Model(nn.Module):
    def __init__(
        self,
        grid: np.ndarray,
        y: int,
        x: int,
        obs_dim: int,
        action_dim: int,
        support: Tensor,
        device,
    ) -> None:
        super(Model, self).__init__()
        self.atom_size: int = 128
        self.support: Tensor = support
        self.encoder = Encoder(device, grid, y, x)

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim, self.atom_size, support).to(device)
        self.dqn_target = Network(obs_dim, action_dim, self.atom_size, support).to(
            device
        )
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        # the target network should not be updated by the optimizer
        for param in self.dqn_target.parameters():
            param.requires_grad = False
        self.dqn_target.eval()

    def forward(self, state: Tensor) -> Tensor:
        """Forward method implementation.
        Note, state here should be the output from the encoder
        """
        dist = self.dqn.dist(state)
        q = torch.sum(dist * self.support, dim=2)

        return q
