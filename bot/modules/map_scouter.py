import itertools
from typing import Any, Iterator

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.consts import UnitRoleTypes
from bot.modules.terrain import Terrain
from bot.modules.unit_roles import UnitRoles


class MapScouter:
    """
    Assign a marine to scout around the map so agent can pick up more data
    """

    expansions_generator: Iterator[Any]

    def __init__(self, ai: BotAI, unit_roles: UnitRoles, terrain: Terrain) -> None:
        self.ai: BotAI = ai
        self.unit_roles: UnitRoles = unit_roles
        self.terrain: Terrain = terrain

        self.next_base_location: Point2 = Point2((1, 1))

        self.STEAL_FROM: set[UnitRoleTypes] = {UnitRoleTypes.ATTACKING}

    async def initialize(self) -> None:
        # set up the expansion generator,
        # so we can keep cycling through expansion locations
        base_locations: list[Point2] = [
            el[0] for el in self.terrain.expansion_distances[1:]
        ]
        self.expansions_generator = itertools.cycle(base_locations)
        self.next_base_location = next(self.expansions_generator)

    def update(self) -> None:
        if self.ai.time < 185.0:
            return

        existing_map_scouters: Units = self.unit_roles.get_units_from_role(
            UnitRoleTypes.MAP_SCOUTER
        )

        if not existing_map_scouters:
            self._assign_map_scouter()
        else:
            for scout in existing_map_scouters:
                self._scout_map(scout)

    def _assign_map_scouter(self) -> None:
        if steal_from := self.unit_roles.get_units_from_role(
            UnitRoleTypes.ATTACKING, UnitTypeId.MARINE
        ):
            if len(steal_from) > 20:
                marine: Unit = steal_from.closest_to(self.ai.start_location)
                self.unit_roles.assign_role(marine.tag, UnitRoleTypes.MAP_SCOUTER)

    def _scout_map(self, scout: Unit) -> None:
        if self.next_base_location and self.ai.is_visible(self.next_base_location):
            self.next_base_location = next(self.expansions_generator)

        if scout.order_target != self.next_base_location:
            scout.move(self.next_base_location)
