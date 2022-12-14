from typing import Optional, Dict

import yaml

from MapAnalyzer.MapData import MapData
from bot.botai_ext import BotAIExt
from bot.consts import AgentClass, ConfigSettings
from bot.modules.macro import Macro
from bot.modules.map_scouter import MapScouter
from bot.modules.pathing import Pathing
from bot.modules.terrain import Terrain
from bot.modules.unit_roles import UnitRoles
from bot.modules.workers import WorkersManager
from bot.squad_agent.base_agent import BaseAgent
from bot.squad_agent.offline_agent import OfflineAgent
from bot.squad_agent.random_agent import RandomAgent
from bot.state import State
from bot.unit_squads import UnitSquads
from sc2.data import Result
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point3
from sc2.unit import Unit


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

    def __init__(self):
        super().__init__()

        # initiate in `on_start`
        self.agent: Optional[BaseAgent] = None
        self.map_data: Optional[MapData] = None
        self.pathing: Optional[Pathing] = None
        self.macro: Optional[Macro] = None
        self.unit_squads: Optional[UnitSquads] = None

        self.config: Dict = dict()
        self.CONFIG_FILE = "config.yaml"
        with open(f"{self.CONFIG_FILE}", "r") as config_file:
            self.config = yaml.safe_load(config_file)
        self.debug: bool = self.config[ConfigSettings.DEBUG]

        self.unit_roles: UnitRoles = UnitRoles(self)
        self.terrain: Terrain = Terrain(self)
        self.map_scouter: MapScouter = MapScouter(self, self.unit_roles, self.terrain)

        self.workers_manager: WorkersManager = WorkersManager(self, self.unit_roles)
        self.sent_chat: bool = False

    async def on_start(self) -> None:
        self.map_data = MapData(self)
        self.pathing = Pathing(self, self.map_data)

        # TODO: Improve this, handle invalid options in config and don't use if/else
        if (
            self.config[ConfigSettings.SQUAD_AGENT][ConfigSettings.AGENT_CLASS]
            == AgentClass.OFFLINE_AGENT
        ):
            self.agent = OfflineAgent(self, self.config, self.pathing)
        else:
            self.agent = RandomAgent(self, self.config, self.pathing)

        self.macro: Macro = Macro(
            self, self.unit_roles, self.workers_manager, self.map_data, self.debug
        )
        self.unit_squads: UnitSquads = UnitSquads(
            self, self.unit_roles, self.agent, self.terrain
        )
        self.client.game_step = self.config[ConfigSettings.GAME_STEP]
        self.client.raw_affects_selection = True
        self.agent.get_episode_data()

        await self.terrain.initialize()
        await self.map_scouter.initialize()

        for worker in self.units(UnitTypeId.SCV):
            worker.gather(self.mineral_field.closest_to(worker))
            self.unit_roles.catch_unit(worker)

    async def on_step(self, iteration: int) -> None:
        # unit_position_list: List[List[float]] = [
        #     [unit.position.x, unit.position.y] for unit in self.enemy_units
        # ]
        # if unit_position_list:
        #     self.enemy_tree = KDTree(unit_position_list)

        state: State = State(self)
        await self.unit_squads.update(iteration, self.pathing)
        await self.macro.update(state, iteration)
        self.workers_manager.update(state, iteration)
        self.map_scouter.update()
        # reasonable assumption the pathing module does not need updating early on
        if self.time > 60.0:
            self.pathing.update(iteration)

        if (
            self.time > 5.0
            and not self.sent_chat
            and isinstance(self.agent, OfflineAgent)
        ):
            await self.chat_send(
                f"Meow! This kitty has trained for {len(self.agent.all_episode_data)} episodes (happy)"
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
                f"Own nat", Point3((*self.terrain.own_nat, height)), size=11
            )
            self.client.debug_text_world(
                f"Enemy nat", Point3((*self.terrain.enemy_nat, height)), size=11
            )
            for unit in self.all_units:
                self.client.debug_text_world(f"{unit.tag}", unit, size=9)

    async def on_unit_created(self, unit: Unit) -> None:
        self.unit_roles.catch_unit(unit)

    async def on_unit_destroyed(self, unit_tag: int) -> None:
        self.agent.on_unit_destroyed(unit_tag)
        self.unit_squads.remove_tag(unit_tag)
        self.pathing.remove_unit_tag(unit_tag)

    async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
        if not unit.is_structure:
            return

        compare_health: float = max(50.0, unit.health_max * 0.09)
        if unit.health < compare_health:
            unit(AbilityId.CANCEL_BUILDINPROGRESS)

    async def on_end(self, game_result: Result) -> None:
        self.agent.on_episode_end(game_result)
