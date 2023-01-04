from typing import Optional

from torch import cat, flatten, nn
from torch.distributions import Categorical
import numpy as np

from bot.squad_agent.architecture.encoder import Encoder


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(
        self,
        action_space_size: int,
        device,
        grid: Optional[np.ndarray],
        height: int,
        width: int,
    ):
        super().__init__()
        self.shared_layers = Encoder(device, grid, height, width)

        self.lstm = nn.LSTM(292, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.policy_layers = layer_init(nn.Linear(128, action_space_size), std=0.01)
        self.value_layers = layer_init(nn.Linear(128, 1), std=1)

    def get_states(
        self, spatial, entity, scalar, locations, lstm_state, done, process_spatial=True
    ):
        hidden, processed_spatial = self.shared_layers(
            spatial, entity, scalar, locations, process_spatial
        )

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = flatten(cat(new_hidden), 0, 1)
        return new_hidden, lstm_state, processed_spatial

    def get_value(
        self, spatial, entity, scalar, locations, lstm_state, done, process_spatial
    ):
        hidden, _, _ = self.get_states(
            spatial, entity, scalar, locations, lstm_state, done, process_spatial
        )
        return self.value_layers(hidden)

    def get_action_and_value(
        self,
        spatial,
        entity,
        scalar,
        locations,
        lstm_state,
        done,
        action=None,
        process_spatial=True,
    ):
        hidden, lstm_state, processed_spatial = self.get_states(
            spatial, entity, scalar, locations, lstm_state, done, process_spatial
        )
        logits = self.policy_layers(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.value_layers(hidden),
            lstm_state,
            processed_spatial,
        )

    def forward(self, spatial, entity, scalar):
        z, processed_spatial = self.shared_layers(spatial, entity, scalar)
        policy_logits = self.policy_layers(z)
        value = self.value_layers(z)
        return policy_logits, value
