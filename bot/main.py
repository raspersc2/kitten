from typing import Dict

import yaml  # type: ignore
from sc2.data import Result
from sc2.ids.ability_id import AbilityId
from sc2.position import Point3
from sc2.unit import Unit

from bot.botai_ext import BotAIExt
from bot.consts import AgentClass, ConfigSettings, UnitRoleTypes
from bot.modules.macro import Macro
from bot.modules.map_scouter import MapScouter
from bot.modules.pathing import Pathing
from bot.modules.terrain import Terrain
from bot.modules.unit_roles import UnitRoles
from bot.modules.workers import WorkersManager
from bot.squad_agent.agents.base_agent import BaseAgent
from bot.squad_agent.agents.dqn_agent import DQNAgent
from bot.squad_agent.agents.dqn_rainbow_agent import DQNRainbowAgent
from bot.squad_agent.agents.offline_agent import OfflineAgent
from bot.squad_agent.agents.ppo_agent import PPOAgent
from bot.squad_agent.agents.random_agent import RandomAgent
from bot.state import State
from bot.unit_squads import UnitSquads
from MapAnalyzer.MapData import MapData


class Kitten(BotAIExt):
    __slots__ = (
        "map_data",
        "unit_roles",
        "unit_squads",
        "workers_manager",
        "macro",
        "pathing",
        "CONFIG_FILE",
        "config",
        "debug",
        "sent_chat",
    )

    agent: BaseAgent
    map_data: MapData
    pathing: Pathing
    macro: Macro
    unit_squads: UnitSquads

    def __init__(self) -> None:
        super().__init__()

        self.config: Dict = dict()
        self.CONFIG_FILE = "config.yaml"
        with open(f"{self.CONFIG_FILE}", "r") as config_file:
            self.config = yaml.safe_load(config_file)
        self.debug: bool = self.config[ConfigSettings.DEBUG]

        self.unit_roles: UnitRoles = UnitRoles(self)
        self.terrain: Terrain = Terrain(self)
        self.map_scouter: MapScouter = MapScouter(self, self.unit_roles, self.terrain)

        self.workers_manager: WorkersManager = WorkersManager(
            self, self.unit_roles, self.terrain
        )
        self.sent_chat: bool = False

    async def on_start(self) -> None:
        self.map_data = MapData(self)
        self.pathing = Pathing(self, self.map_data)

        # TODO: Improve this, handle invalid options in config and don't use if/else
        agent_class: str = self.config[ConfigSettings.SQUAD_AGENT][
            ConfigSettings.AGENT_CLASS
        ]
        try:
            if agent_class == AgentClass.OFFLINE_AGENT:
                self.agent = OfflineAgent(self, self.config, self.pathing)
            elif agent_class == AgentClass.PPO_AGENT:
                self.agent = PPOAgent(self, self.config, self.pathing)
            elif agent_class == AgentClass.DQN_AGENT:
                self.agent = DQNAgent(self, self.config, self.pathing)
            elif agent_class == AgentClass.DQN_RAINBOW_AGENT:
                self.agent = DQNRainbowAgent(self, self.config, self.pathing)
            elif agent_class == AgentClass.RANDOM_AGENT:
                self.agent = RandomAgent(self, self.config, self.pathing)
        except ValueError:
            raise ValueError("Invalid AgentClass name in config.yaml")

        self.macro = Macro(
            self, self.unit_roles, self.workers_manager, self.map_data, self.debug
        )
        self.unit_squads = UnitSquads(self, self.unit_roles, self.agent, self.terrain)
        self.client.game_step = self.config[ConfigSettings.GAME_STEP]
        self.client.raw_affects_selection = True
        self.agent.get_episode_data()

        await self.terrain.initialize()
        await self.map_scouter.initialize()
        for worker in self.workers:
            self.unit_roles.assign_role(worker.tag, UnitRoleTypes.GATHERING)

    async def on_step(self, iteration: int) -> None:

        if self.time > 1200.0:
            await self.client.leave()
        state: State = State(self)
        await self.unit_squads.update(iteration, self.pathing)
        await self.macro.update(state, iteration)
        self.workers_manager.update(state, iteration)
        self.map_scouter.update()
        # reasonable assumption the pathing module does not need updating early on
        if self.time > 60.0:
            self.pathing.update(iteration)

        if self.time > 5.0 and not self.sent_chat:
            num_episodes: int = len(self.agent.all_episode_data)
            await self.chat_send(
                f"Meow! This kitty has trained for {num_episodes} episodes (happy)"
            )
            self.sent_chat = True

        if self.time > 179.0 and self.state.game_loop % 672 == 0:
            reward: float = self.agent.cumulative_reward
            emotion: str = (
                "meow" if reward == 0.0 else ("growl" if reward < 0.0 else "purr")
            )
            await self.chat_send(
                f"Cumulative episode reward: {round(reward, 4)} ...{emotion}"
            )

        if self.debug:
            height: float = self.get_terrain_z_height(self.terrain.own_nat)
            self.client.debug_text_world(
                "Own nat", Point3((*self.terrain.own_nat, height)), size=11
            )
            self.client.debug_text_world(
                "Enemy nat", Point3((*self.terrain.enemy_nat, height)), size=11
            )
            for unit in self.all_units:
                self.client.debug_text_world(f"{unit.tag}", unit, size=9)

    async def on_unit_created(self, unit: Unit) -> None:
        self.unit_roles.catch_unit(unit)

    async def on_unit_destroyed(self, unit_tag: int) -> None:
        self.agent.on_unit_destroyed(unit_tag)
        self.unit_squads.remove_tag(unit_tag)
        self.pathing.remove_unit_tag(unit_tag)
        self.workers_manager.remove_worker_from_mineral(unit_tag)
        self.workers_manager.remove_worker_from_vespene(unit_tag)

    async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
        if not unit.is_structure:
            return

        compare_health: float = max(50.0, unit.health_max * 0.09)
        if unit.health < compare_health:
            unit(AbilityId.CANCEL_BUILDINPROGRESS)

    async def on_end(self, game_result: Result) -> None:
        self.agent.on_episode_end(game_result)
