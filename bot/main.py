from typing import Optional, List

from scipy.spatial import KDTree

from bot.botai_ext import BotAIExt
from bot.pathing import Pathing
from bot.squad_agent.base_agent import BaseAgent
from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit

from bot.macro import Macro
from bot.state import State
from bot.unit_roles import UnitRoles
from bot.unit_squads import UnitSquads
from bot.workers_manager import WorkersManager

from bot.squad_agent.random_agent import RandomAgent
from MapAnalyzer.MapData import MapData


class Kitten(BotAIExt):
    __slots__ = (
        "map_data",
        "unit_roles",
        "unit_squads",
        "workers_manager",
        "macro",
        "pathing",
    )

    def __init__(self):
        super().__init__()

        # initiate in `on_start`
        self.map_data: Optional[MapData] = None
        self.pathing: Optional[Pathing] = None

        self.agent: BaseAgent = RandomAgent(self)
        self.unit_roles: UnitRoles = UnitRoles(self)
        self.unit_squads: UnitSquads = UnitSquads(self, self.unit_roles, self.agent)
        self.workers_manager: WorkersManager = WorkersManager(self, self.unit_roles)
        self.macro: Macro = Macro(self, self.unit_roles, self.workers_manager)

    async def on_start(self) -> None:
        self.map_data = MapData(self)
        self.pathing = Pathing(self, self.map_data)
        self.client.game_step = 2
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

    async def on_unit_created(self, unit: Unit) -> None:
        self.unit_roles.catch_unit(unit)

    async def on_unit_destroyed(self, unit_tag: int) -> None:
        # self.agent.on_unit_destroyed(unit_tag)
        self.unit_squads.remove_tag(unit_tag)
        self.pathing.remove_unit_tag(unit_tag)

    async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
        if not unit.is_structure:
            return

        compare_health: float = max(50.0, unit.health_max * 0.09)
        if unit.health < compare_health:
            unit(AbilityId.CANCEL_BUILDINPROGRESS)
