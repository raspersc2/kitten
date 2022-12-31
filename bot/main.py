from typing import Optional, List, Dict

from sc2.data import Result
from scipy.spatial import KDTree

from bot.botai_ext import BotAIExt
from bot.pathing import Pathing
from bot.squad_agent.base_agent import BaseAgent
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit
import yaml

from bot.consts import AgentClass, ConfigSettings
from bot.macro import Macro
from bot.state import State
from bot.unit_roles import UnitRoles
from bot.unit_squads import UnitSquads
from bot.workers_manager import WorkersManager

from bot.squad_agent.random_agent import RandomAgent
from bot.squad_agent.offline_agent import OfflineAgent
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
        self.unit_squads: UnitSquads = UnitSquads(self, self.unit_roles, self.agent)
        self.client.game_step = self.config[ConfigSettings.GAME_STEP]
        self.client.raw_affects_selection = True
        self.agent.get_episode_data()
        for worker in self.units(UnitTypeId.SCV):
            worker.gather(self.mineral_field.closest_to(worker))
            self.unit_roles.catch_unit(worker)

    async def on_step(self, iteration: int) -> None:
        unit_position_list: List[List[float]] = [
            [unit.position.x, unit.position.y] for unit in self.enemy_units
        ]
        if unit_position_list:
            self.enemy_tree = KDTree(unit_position_list)

        state: State = State(self)
        await self.unit_squads.update(iteration, self.pathing)
        await self.macro.update(state, iteration)
        self.workers_manager.update(state, iteration)
        # reasonable assumption the pathing module does not need updating early on
        # if self.time > 60.0:
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

        if self.debug:
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
