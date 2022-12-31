import itertools
from typing import Optional

from bot.modules.terrain import Terrain
from sc2.position import Point2

from sc2.ids.unit_typeid import UnitTypeId

from sc2.unit import Unit

from sc2.units import Units

from bot.modules.unit_roles import UnitRoles
from sc2.bot_ai import BotAI
from bot.consts import UnitRoleTypes


class MapScouter:
    """
    Assign a marine to scout around the map so agent can pick up more data
    """

    def __init__(self, ai: BotAI, unit_roles: UnitRoles, terrain: Terrain) -> None:
        self.ai: BotAI = ai
        self.unit_roles: UnitRoles = unit_roles
        self.terrain: Terrain = terrain

        self.expansions_generator = None
        self.next_base_location: Optional[Point2] = None

        self.STEAL_FROM: set[UnitRoleTypes] = {UnitRoleTypes.ATTACKING}

    async def initialize(self) -> None:
        # set up the expansion generator, so we can keep cycling through expansion locations
        base_locations: list[Point2] = [
            el[0] for el in self.terrain.expansion_distances[1:]
        ]
        self.expansions_generator = itertools.cycle(base_locations)
        self.next_base_location = next(self.expansions_generator)

    def update(self) -> None:
        if self.ai.time < 260.0:
            return

        existing_map_scouters: Units = self.unit_roles.get_units_from_role(
            UnitRoleTypes.MAP_SCOUTER
        )

        if not existing_map_scouters:
            self._assign_map_scouter()
        else:
            for scout in existing_map_scouters:
                self._scout_map(scout)

    def _assign_map_scouter(self):
        steal_from: Units = self.unit_roles.get_units_from_role(
            UnitRoleTypes.ATTACKING, UnitTypeId.MARINE
        )
        if steal_from:
            marine: Unit = steal_from.closest_to(self.ai.start_location)
            self.unit_roles.assign_role(marine.tag, UnitRoleTypes.MAP_SCOUTER)

    def _scout_map(self, scout: Unit):
        if self.next_base_location and self.ai.is_visible(self.next_base_location):
            self.next_base_location = next(self.expansions_generator)

        if scout.order_target != self.next_base_location:
            scout.move(self.next_base_location)
