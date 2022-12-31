import itertools
from typing import Optional

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

    def __init__(self, ai: BotAI, unit_roles: UnitRoles) -> None:
        self.ai: BotAI = ai
        self.unit_roles: UnitRoles = unit_roles
        self.expansion_distances: list[tuple[Point2, float]] = []
        self.expansions_generator = None
        self.next_base_location: Optional[Point2] = None

        self.STEAL_FROM: set[UnitRoleTypes] = {UnitRoleTypes.ATTACKING}

    async def initialize(self) -> None:
        # store all expansion locations, sorted by distance to spawn
        for el in self.ai.expansion_locations_list:
            if self.ai.start_location.distance_to(el) < self.ai.EXPANSION_GAP_THRESHOLD:
                continue

            distance = await self.ai.client.query_pathing(self.ai.start_location, el)
            if distance:
                self.expansion_distances.append((el, distance))

        # sort by path length to each expansion
        self.expansion_distances = sorted(
            self.expansion_distances, key=lambda x: x[1], reverse=True
        )

        # set up the expansion generator, so we can keep cycling through expansion locations
        base_locations: list[Point2] = [el[0] for el in self.expansion_distances[1:]]
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
