"""
The offline agent goal is to collect state, actions and rewards and store them to disk
(Note: Fine to use this agent to test a trained model by setting InferenceMode: True)
RL Training (back propagation) should then be carried out via a separate process / script
    after the game is complete
"""

from os import path
from typing import Dict, List

import torch
from sc2.data import Result
from torch import optim, nn

from bot.botai_ext import BotAIExt
from bot.consts import ConfigSettings, SQUAD_ACTIONS
from bot.pathing import Pathing
from bot.squad_agent.architecture.actor_critic import ActorCritic
from bot.squad_agent.base_agent import BaseAgent
from bot.squad_agent.features import Features
from sc2.position import Point2
from sc2.units import Units

from loguru import logger

NUM_ENVS: int = 1
NUM_ROLLOUT_STEPS: int = 64
SPATIAL_SHAPE: tuple[int, int, int, int] = (1, 37, 120, 120)
ENTITY_SHAPE: tuple[int, int, int] = (1, 256, 405)
SCALAR_SHAPE: tuple[int, int] = (1, 10)


class OfflineAgent(BaseAgent):
    __slots__ = (
        "features",
        "pathing",
        "model",
        "optimizer",
        "initial_lstm_state",
        "current_lstm_state",
        "scalars",
        "actions",
        "locations",
        "logprobs",
        "rewards",
        "dones",
        "values",
        "current_rollout_step",
    )

    def __init__(self, ai: BotAIExt, config: Dict, pathing: Pathing):
        # we will use the aiarena docker to play multiple simultaneous games to collect state, action, rewards etc.
        # so use "cpu" here and the separate training script should use "cuda" if available
        super().__init__(ai, config, "cpu")

        self.features: Features = Features(ai, config, 256, self.device)
        self.pathing: Pathing = pathing

        grid = self.pathing.map_data.get_pyastar_grid()
        self.model = ActorCritic(len(SQUAD_ACTIONS), self.device, self.ai, grid).to(
            self.device
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2.5e-4, eps=1e-5)
        if path.isfile(self.CHECKPOINT_PATH):
            self.model, self.optimizer, self.epoch = self.load_checkpoint(
                self.model, self.optimizer, self.device
            )
        # nothing stored on disk yet, there should be something there for the training script later
        else:
            self.save_checkpoint(self.model, self.optimizer)

        self.model.train() if self.training_active else self.model.eval()

        self.initial_lstm_state = (
            torch.zeros(
                self.model.lstm.num_layers, NUM_ENVS, self.model.lstm.hidden_size
            ).to(self.device),
            torch.zeros(
                self.model.lstm.num_layers, NUM_ENVS, self.model.lstm.hidden_size
            ).to(self.device),
        )

        self.current_lstm_state = self.initial_lstm_state

        self.scalars = torch.zeros((NUM_ROLLOUT_STEPS,) + SCALAR_SHAPE).to(self.device)
        self.actions = torch.zeros((NUM_ROLLOUT_STEPS,) + (1,)).to(self.device)
        self.locations = torch.zeros((NUM_ROLLOUT_STEPS,) + (1, 256, 2)).to(self.device)
        self.logprobs = torch.zeros((NUM_ROLLOUT_STEPS, NUM_ENVS)).to(self.device)
        self.rewards = torch.zeros((NUM_ROLLOUT_STEPS, NUM_ENVS)).to(self.device)
        self.dones = torch.zeros((NUM_ROLLOUT_STEPS, NUM_ENVS)).to(self.device)
        self.values = torch.zeros((NUM_ROLLOUT_STEPS, NUM_ENVS)).to(self.device)
        self.current_rollout_step: int = 0

    def choose_action(
        self,
        squads: List,
        pos_of_squad: Point2,
        all_close_enemy: Units,
        squad_units: Units,
        attack_target: Point2,
        rally_point: Point2,
    ) -> int:
        super(OfflineAgent, self).choose_action(
            squads,
            pos_of_squad,
            all_close_enemy,
            squad_units,
            attack_target,
            rally_point,
        )
        reward: float = self.reward
        obs = self.features.transform_obs(
            self.pathing.ground_grid, pos_of_squad, attack_target, rally_point
        )
        spatial, entity, scalar, locations = obs
        locations = locations.to(self.device)
        spatial = spatial.to(self.device)
        entity = entity.to(self.device)
        entity = nn.functional.normalize(entity)

        self.cumulative_reward += reward
        self.squad_reward = 0.0
        with torch.no_grad():
            (
                action,
                logprob,
                _,
                value,
                self.current_lstm_state,
                processed_spatial,
            ) = self.model.get_action_and_value(
                spatial,
                entity,
                scalar,
                locations,
                self.current_lstm_state,
                self.dones,
            )
            self.action_distribution[action] += 1
            if self.current_rollout_step < NUM_ROLLOUT_STEPS:
                self.current_rollout_step += 1
            else:
                # TODO: Store things to disk
                pass
            return action.item()

    def on_episode_end(self, result):
        if self.training_active:
            logger.info("On episode end called")
            _reward = 5.0 if result == Result.Victory else -5.0
            self.store_episode_data(
                result,
                self.epoch,
                self.cumulative_reward + _reward,
                self.action_distribution,
            )

            current_step: int = self.current_rollout_step
            if current_step == NUM_ROLLOUT_STEPS:
                current_step = NUM_ROLLOUT_STEPS - 1
            self.rewards[current_step] = _reward
            self.dones[current_step] = 1
            # TODO: Store things to disk
