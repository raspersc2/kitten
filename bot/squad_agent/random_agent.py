from random import randint
from typing import List

from loguru import logger

from bot.squad_agent.base_agent import BaseAgent
from sc2.bot_ai import BotAI
from sc2.data import Result
from sc2.position import Point2
from sc2.units import Units


class RandomAgent(BaseAgent):
    def __init__(self, ai: BotAI):
        super().__init__(ai)

    def choose_action(
        self,
        squads: List,
        pos_of_squad: Point2,
        all_close_enemy: Units,
        squad_units: Units,
    ) -> int:
        self.cumulative_reward += self.reward
        self.squad_reward = 0.0
        action: int = randint(0, self.num_actions - 1)
        self.action_distribution[action] += 1
        return action

    def on_episode_end(self, result):
        logger.info("On episode end called")
        _reward = 50.0 if result == Result.Victory else -50.0
        self.store_episode_data(
            result=result,
            steps=self.epoch,
            reward=self.cumulative_reward + _reward,
            action_distribution=self.action_distribution,
        )