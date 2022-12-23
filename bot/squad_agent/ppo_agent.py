from typing import Dict, List

from bot.pathing import Pathing
from sc2.units import Units

from sc2.position import Point2

from bot.squad_agent.base_agent import BaseAgent
from bot.botai_ext import BotAIExt

from bot.squad_agent.features import Features


class PPOAgent(BaseAgent):
    def __init__(self, ai: BotAIExt, config: Dict, pathing: Pathing):
        super().__init__(ai, config)
        self.features: Features = Features(ai, 256, self.device)
        self.pathing: Pathing = pathing

    def choose_action(
        self,
        squads: List,
        pos_of_squad: Point2,
        all_close_enemy: Units,
        squad_units: Units,
    ) -> int:
        obs = self.features.transform_obs(self.pathing.ground_grid)
        spatial, entity, locations = obs
        return 1

    def on_episode_end(self, result):
        pass

    #
