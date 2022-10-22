"""
Basic macro that focuses on bio production and upgrades
"""
from typing import Optional, List

from bot.consts import UnitRoleTypes
from bot.state import State
from bot.unit_roles import UnitRoles
from bot.workers_manager import WorkersManager
from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units


class Macro:
    __slots__ = "ai", "unit_roles", "workers_manager", "state", "max_workers"

    def __init__(
        self, ai: BotAI, unit_roles: UnitRoles, workers_manager: WorkersManager
    ):
        self.ai: BotAI = ai
        self.unit_roles: UnitRoles = unit_roles
        self.workers_manager: WorkersManager = workers_manager
        self.state: Optional[State] = None
        self.max_workers: int = 19

    async def update(self, state: State, iteration: int) -> None:
        self.state = state
        if len(self.ai.townhalls) > 1:
            self.max_workers = 41

        building_scvs: Units = self.unit_roles.get_units_from_role(
            UnitRoleTypes.BUILDING
        )
        for scv in building_scvs:
            if len(scv.orders) == 0:
                self.unit_roles.assign_role(scv.tag, UnitRoleTypes.GATHERING)

        available_scvs: Units = self.unit_roles.get_units_from_role(
            UnitRoleTypes.GATHERING
        ).filter(lambda u: u.is_gathering and not u.is_carrying_resource)

        self._manage_upgrades()
        self._build_refineries(available_scvs)
        await self._build_supply(iteration, available_scvs)
        self._produce_workers()
        await self._build_barracks_and_addons(available_scvs)
        # don't make army till orbitals have started
        if not self.state.ccs:
            self._produce_army()

        # 2 townhalls at all times
        if (
            len(self.ai.townhalls) < 2
            and self.ai.can_afford(UnitTypeId.COMMANDCENTER)
            and available_scvs
        ):
            location: Optional[Point2] = await self.ai.get_next_expansion()
            if location:
                await self._build_structure(
                    UnitTypeId.COMMANDCENTER,
                    self.state.build_area,
                    available_scvs,
                    specific_location=location,
                )

        # not the greatest solution here, and should be improved
        # but better then nothing
        for structure in self.ai.structures_without_construction_SCVs:
            structure(AbilityId.CANCEL_BUILDINPROGRESS)

    def _produce_army(self) -> None:
        barracks: Units = self.state.barracks.idle
        if not barracks or not self.ai.can_afford(UnitTypeId.MARINE):
            return

        for rax in barracks:
            if rax.is_idle:
                if rax.has_techlab:
                    if self.ai.can_afford(UnitTypeId.MARAUDER):
                        rax(AbilityId.BARRACKSTRAIN_MARAUDER)
                    continue
                rax(AbilityId.BARRACKSTRAIN_MARINE)

    def _produce_workers(self) -> None:
        if (
            self.ai.supply_workers >= self.max_workers
            or not self.ai.can_afford(UnitTypeId.SCV)
            or self.ai.supply_left <= 0
            or not self.ai.townhalls
        ):
            return

        # no rax yet, all ths can build scvs
        if self.state.barracks.ready.amount < 1:
            for th in self.ai.townhalls.idle:
                th.train(UnitTypeId.SCV)
        # rax present, only orbitals / pfs can build scvs
        # TODO: Adjust this if we build PFs
        else:
            for th in self.state.orbitals.idle:
                th.train(UnitTypeId.SCV)

    async def _build_supply(self, iteration: int, available_scvs: Units):
        # lower existing depots
        if iteration % 12 == 0:
            for depot in self.state.depots.ready:
                depot(AbilityId.MORPH_SUPPLYDEPOT_LOWER)

        max_building: int = 1 if len(self.state.barracks) < 3 else 2
        if (
            self.ai.supply_used < 14
            or self.ai.supply_left >= 5
            or self.ai.already_pending(UnitTypeId.SUPPLYDEPOT) >= max_building
            or not self.ai.can_afford(UnitTypeId.SUPPLYDEPOT)
            or not available_scvs
            or not self.ai.townhalls
        ):
            return

        await self._build_structure(
            UnitTypeId.SUPPLYDEPOT, self.state.build_area, available_scvs
        )

    async def _build_barracks_and_addons(self, available_scvs: Units) -> None:
        def barracks_points_to_build_addon(sp_position: Point2) -> List[Point2]:
            """Return all points that need to be checked when trying to build an addon. Returns 4 points."""
            addon_offset: Point2 = Point2((2.5, -0.5))
            addon_position: Point2 = sp_position + addon_offset
            addon_points = [
                (addon_position + Point2((x - 0.5, y - 0.5))).rounded
                for x in range(0, 2)
                for y in range(0, 2)
            ]
            return addon_points

        rax: Units = self.state.barracks

        add_ons: Units = self.ai.structures(UnitTypeId.BARRACKSTECHLAB)
        if len(add_ons) < 3 and self.ai.can_afford(UnitTypeId.TECHLAB) and rax.idle:
            for b in rax:
                if not b.has_add_on:
                    addon_points: List[Point2] = barracks_points_to_build_addon(
                        b.position
                    )
                    if all(
                        self.ai.in_map_bounds(addon_point)
                        and self.ai.in_placement_grid(addon_point)
                        and self.ai.in_pathing_grid(addon_point)
                        for addon_point in addon_points
                    ):
                        b.build(UnitTypeId.BARRACKSTECHLAB)

        max_barracks: int = 2 if len(self.ai.townhalls) <= 1 else 8
        rax: Units = self.state.barracks
        if (
            self.ai.tech_requirement_progress(UnitTypeId.BARRACKS) != 1
            or len(rax) >= max_barracks
            or not available_scvs
            or not self.ai.can_afford(UnitTypeId.BARRACKS)
        ):
            return

        await self._build_structure(
            UnitTypeId.BARRACKS, self.state.build_area, available_scvs
        )

    def _manage_upgrades(self):
        ccs: Units = self.state.ccs
        if ccs and self.ai.can_afford(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND):
            ccs.first(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND)

    async def _build_structure(
        self,
        structure_type: UnitTypeId,
        placement_area: Point2,
        available_workers: Units,
        specific_location: Optional[Point2] = None,
        placement_step: int = 3,
    ) -> None:
        if specific_location:
            location: Point2 = specific_location
        else:
            location: Point2 = await self.ai.find_placement(
                structure_type, placement_area, placement_step=placement_step
            )
        if location:
            worker: Unit = available_workers.closest_to(location)
            self.unit_roles.assign_role(worker.tag, UnitRoleTypes.BUILDING)
            self.workers_manager.remove_worker_from_mineral(worker.tag)
            worker.build(structure_type, location)

    def _build_refineries(self, available_scvs: Units):
        # 2 gas buildings
        max_gas: int = 0 if len(self.ai.townhalls) < 2 else 2
        current_gas_num = (
            self.ai.already_pending(UnitTypeId.REFINERY) + self.ai.gas_buildings.amount
        )
        # Build refineries (on nearby vespene) when at least one barracks is in construction
        if (
            current_gas_num >= max_gas
            or len(self.state.barracks) < 2
            or not self.ai.can_afford(UnitTypeId.REFINERY)
            or not available_scvs
        ):
            return

        # Loop over all townhalls nearly complete
        for th in self.ai.townhalls.filter(lambda _th: _th.build_progress > 0.6):
            # Find all vespene geysers that are closer than range 10 to this townhall
            vgs: Units = self.ai.vespene_geyser.closer_than(10, th)
            for vg in vgs:
                if self.ai.gas_buildings.filter(lambda gb: gb.distance_to(vg) < 3.0):
                    continue

                worker: Unit = available_scvs.closest_to(vg)
                self.unit_roles.assign_role(worker.tag, UnitRoleTypes.BUILDING)
                self.workers_manager.remove_worker_from_mineral(worker.tag)
                worker.build_gas(vg)
                break
