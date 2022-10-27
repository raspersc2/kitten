from typing import Optional, Set

from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.macro import Macro
from bot.state import State
from bot.unit_roles import UnitRoles
from bot.workers_manager import WorkersManager


class Kitten(BotAI):
    __slots__ = "unit_roles", "workers_manager", "macro"

    def __init__(self):
        super().__init__()
        self.unit_roles: UnitRoles = UnitRoles(self)
        self.workers_manager: WorkersManager = WorkersManager(self, self.unit_roles)
        self.macro: Macro = Macro(self, self.unit_roles, self.workers_manager)

    async def on_start(self) -> None:
        self.client.game_step = 4
        for worker in self.units(UnitTypeId.SCV):
            worker.gather(self.mineral_field.closest_to(worker))
            self.unit_roles.catch_unit(worker)

    async def on_step(self, iteration: int) -> None:
        state: State = State(self)
        await self.macro.update(state, iteration)
        self.workers_manager.update(state)

    async def on_unit_created(self, unit: Unit) -> None:
        self.unit_roles.catch_unit(unit)

    async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
        if not unit.is_structure:
            return

        compare_health: float = max(50.0, unit.health_max * 0.09)
        if unit.health < compare_health:
            unit(AbilityId.CANCEL_BUILDINPROGRESS)
