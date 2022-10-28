"""
All agents should inherit from this base class
"""

import json
from abc import ABCMeta, abstractmethod
from datetime import datetime
from os import path

import torch
from loguru import logger
from typing import List, Dict, Optional

from sc2.data import Result
from torch.utils.tensorboard import SummaryWriter

from sc2.bot_ai import BotAI

from bot.consts import DATA_DIR, SQUAD_ACTIONS
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units


class BaseAgent(metaclass=ABCMeta):
    __slots__ = (
        "ai",
        "device",
        "epoch",
        "current_action",
        "cumulative_reward",
        "squad_reward",
        "all_episode_data",
        "previous_close_enemy",
        "previous_main_squad",
        "writer",
        "action_distribution",
        "ml_training_file_path",
        "CHECKPOINT_PATH",
        "num_actions",
    )

    def __init__(self, ai: BotAI):
        super().__init__()
        self.ai: BotAI = ai

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using {self.device}")

        self.epoch: int = 0
        self.current_action = 0

        self.cumulative_reward: float = 0.0
        self.squad_reward: float = 0.0
        self.ml_training_file_path: path = path.join(
            DATA_DIR, "agent_training_history.json"
        )
        self.all_episode_data: List[Dict] = []
        self.previous_close_enemy: Optional[Units] = None
        self.previous_main_squad: Optional[Units] = None

        self.writer = SummaryWriter("data/runs")
        self.action_distribution: List[int] = [0 for _ in range(len(SQUAD_ACTIONS))]

        self.num_actions: int = len(SQUAD_ACTIONS)
        self.CHECKPOINT_PATH: path = path.join(DATA_DIR, "checkpoint.pt")

    @property
    def reward(self) -> float:
        reward = self.squad_reward
        # clip the reward between -1 and 1 to help training
        reward = min(reward, 1.0) if reward >= 0.0 else max(reward, -1.0)
        return reward

    @abstractmethod
    def choose_action(
        self,
        squads: List,
        pos_of_squad: Point2,
        all_close_enemy: Units,
        squad_units: Units,
    ) -> int:
        self.previous_main_squad = squad_units
        self.previous_close_enemy = all_close_enemy

    @abstractmethod
    def on_episode_end(self, result):
        pass

    def on_unit_destroyed(self, tag: int) -> None:
        if self.previous_main_squad and tag in self.previous_main_squad.tags:
            unit: Unit = self.previous_main_squad.find_by_tag(tag)
            value = self.ai.calculate_unit_value(unit.type_id)
            value = (value.minerals + value.vespene * 1.5) / 700.0
            self.squad_reward -= value
        elif self.previous_close_enemy and tag in self.previous_close_enemy.tags:
            unit: Unit = self.previous_close_enemy.find_by_tag(tag)
            value = self.ai.calculate_unit_value(unit.type_id)
            value = (value.minerals + value.vespene * 1.5) / 700.0
            self.squad_reward += value

    def get_episode_data(self, get_default: bool = True) -> List[Dict]:
        if path.isfile(self.ml_training_file_path):
            with open(self.ml_training_file_path, "r") as f:
                episode_data = json.load(f)
        elif get_default:
            # no data, create a dummy version
            episode_data = [
                {
                    "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "GlobalStep": 0,
                    "Race": str(self.ai.enemy_race),
                    "Reward": 0.0,
                    "Result": 0,
                    "OppID": self.ai.opponent_id,
                    "ActionDistribution": [],
                }
            ]
        else:
            episode_data = []
        self.all_episode_data = episode_data
        return episode_data

    def store_episode_data(self, result, steps, reward, action_distribution) -> None:

        episode_data = self.get_episode_data(get_default=False)
        step = 0 if len(episode_data) == 0 else episode_data[-1]["GlobalStep"]

        result_id = self._get_result_id(result)
        episode_info = {
            "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "GlobalStep": steps + step,
            "Race": str(self.ai.enemy_race),
            "Reward": reward,
            "Result": result_id,
            "OppID": self.ai.opponent_id,
            "ActionDistribution": action_distribution,
        }
        if len(self.all_episode_data) >= 1:
            self.all_episode_data.append(episode_info)
        else:
            self.all_episode_data = [episode_info]
        with open(self.ml_training_file_path, "w") as f:
            json.dump(self.all_episode_data, f)

    def save_checkpoint(self, model, optimizer):
        state = {
            "epoch": self.epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        f_path = self.CHECKPOINT_PATH
        torch.save(state, f_path)

    def load_checkpoint(self, model, optimizer, device):
        logger.info("loaded existing model")
        checkpoint = torch.load(self.CHECKPOINT_PATH, map_location=torch.device(device))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return model, optimizer, checkpoint["epoch"]

    @staticmethod
    def _get_result_id(result) -> int:
        result_id: int = 1
        if result == Result.Victory:
            result_id = 2
        elif result == Result.Defeat:
            result_id = 0
        return result_id
