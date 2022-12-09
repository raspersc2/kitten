"""
Basic macro that focuses on bio production and upgrades
This should probably be rewritten / refactored into separate files for anything more complicated
"""
from typing import Optional, List, Set, Tuple

import numpy as np
from sc2.ids.upgrade_id import UpgradeId

from MapAnalyzer import MapData, Region
from bot.botai_ext import BotAIExt
from bot.consts import UnitRoleTypes
from bot.state import State
from bot.unit_roles import UnitRoles
from bot.workers_manager import WorkersManager
from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2, Point3, Pointlike
from sc2.unit import Unit
from sc2.units import Units


class Macro:
    __slots__ = (
        "ai",
        "unit_roles",
        "workers_manager",
        "map_data",
        "debug",
        "state",
        "depot_positions",
        "max_workers",
    )

    def __init__(
        self,
        ai: BotAIExt,
        unit_roles: UnitRoles,
        workers_manager: WorkersManager,
        map_data: MapData,
        debug: bool,
    ):
        self.ai: BotAIExt = ai
        self.unit_roles: UnitRoles = unit_roles
        self.workers_manager: WorkersManager = workers_manager
        self.map_data: MapData = map_data
        self.debug: bool = debug
        self.state: Optional[State] = None
        ramp = self.ai.main_base_ramp
        corner_depots = list(ramp.corner_depots)
        self.depot_positions: List[Point2] = [
            ramp.depot_in_middle,
            corner_depots[0],
            corner_depots[1],
        ]
        self.max_workers: int = 19
        self._calculate_supply_placements()

    async def update(self, state: State, iteration: int) -> None:
        self.state = state
        if len(self.ai.townhalls) > 1:
            self.max_workers = 41

        if building_scvs := self.unit_roles.get_units_from_role(
            UnitRoleTypes.BUILDING
        ).filter(lambda u: u.is_idle):
            for scv in building_scvs:
                self.unit_roles.assign_role(scv.tag, UnitRoleTypes.GATHERING)

        available_scvs: Units = self.unit_roles.get_units_from_role(
            UnitRoleTypes.GATHERING
        ).filter(lambda u: u.is_gathering and not u.is_carrying_resource)

        self._manage_upgrades()
        self._build_refineries(available_scvs)
        await self._build_supply(iteration, available_scvs)
        self._produce_workers()
        await self._build_addons()
        self._produce_army()
        await self._build_barracks(available_scvs)
        await self._build_factory(available_scvs)
        await self._build_starport(available_scvs)

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
                    self.state.natural_build_area,
                    available_scvs,
                    specific_location=location,
                )

        # not the greatest solution here, and should be improved
        # but better then nothing
        for structure in self.ai.structures_without_construction_SCVs:
            structure(AbilityId.CANCEL_BUILDINPROGRESS)

        if self.debug:
            for position in self.depot_positions:
                pos: Point3 = Point3(
                    (position.x, position.y, self.ai.get_terrain_z_height(position))
                )
                self.ai.client.debug_box2_out(pos)

    def _produce_army(self) -> None:
        if not self.ai.can_afford(UnitTypeId.MARINE) or self.ai.supply_left <= 0:
            return
        barracks: Units = self.state.barracks.filter(lambda u: u.is_ready and u.is_idle)
        ports: Units = self.state.starports.filter(lambda u: u.is_ready and u.is_idle)

        for rax in barracks:
            if self.ai.supply_left <= 0:
                break
            if rax.has_techlab:
                if self.ai.can_afford(UnitTypeId.MARAUDER):
                    self.ai.train(UnitTypeId.MARAUDER)
                continue
            self.ai.train(UnitTypeId.MARINE)

        for _ in ports:
            if self.ai.supply_left <= 0:
                break
            if self.ai.can_afford(UnitTypeId.MEDIVAC):
                self.ai.train(UnitTypeId.MEDIVAC)

    def _produce_workers(self) -> None:
        if (
            self.ai.supply_workers >= self.max_workers
            or not self.ai.can_afford(UnitTypeId.SCV)
            or self.ai.supply_left <= 0
            or not self.ai.townhalls.idle
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

    async def _build_supply(self, iteration: int, available_scvs: Units) -> None:
        # lower existing depots
        if iteration % 12 == 0:
            for depot in self.state.depots.ready:
                depot(AbilityId.MORPH_SUPPLYDEPOT_LOWER)

        max_building: int = 1 if len(self.state.barracks) < 3 else 2
        if (
            self.ai.supply_used < 14
            or self.ai.supply_left >= 5
            or self.ai.supply_cap >= 200
            or self.ai.already_pending(UnitTypeId.SUPPLYDEPOT) >= max_building
            or not self.ai.can_afford(UnitTypeId.SUPPLYDEPOT)
            or not available_scvs
            or not self.ai.townhalls
        ):
            return

        # try to build depot at one of our precalculated spots
        if len(self.depot_positions) > 0:
            pos: Point2 = self.depot_positions[0]
            # this might rarely happen, remove the point, return and try with new point next time
            if not self.ai.in_placement_grid(pos):
                self.depot_positions = self.depot_positions[1:]
                return

            self.depot_positions = self.depot_positions[1:]
            await self._build_structure(
                UnitTypeId.SUPPLYDEPOT,
                self.state.main_build_area,
                available_scvs,
                specific_location=pos,
            )
        # else generic depot placement
        else:
            await self._build_structure(
                UnitTypeId.SUPPLYDEPOT, self.state.natural_build_area, available_scvs
            )

    async def _build_addons(self) -> None:
        add_ons: Units = self.ai.structures(UnitTypeId.BARRACKSTECHLAB)
        if len(add_ons) < 2 and self.ai.can_afford(UnitTypeId.TECHLAB):
            rax: Units = self.state.barracks.filter(lambda u: u.is_ready and u.is_idle)
            for b in rax:
                if not b.has_add_on:
                    add_on_location: Pointlike = b.position.offset(Point2((2.5, -0.5)))
                    if await self.ai.can_place(UnitTypeId.SUPPLYDEPOT, add_on_location):
                        b.build(UnitTypeId.BARRACKSTECHLAB)

    async def _build_barracks(self, available_scvs: Units) -> None:
        max_barracks: int = 2 if len(self.ai.townhalls) <= 1 else 8
        rax: Units = self.state.barracks
        if (
            self.ai.tech_requirement_progress(UnitTypeId.BARRACKS) != 1
            or len(rax) >= max_barracks
            or not available_scvs
            or not self.ai.can_afford(UnitTypeId.BARRACKS)
        ):
            return

        build_area: Point2 = (
            self.state.main_build_area
            if len(rax) < 5
            else self.state.natural_build_area
        )

        await self._build_structure(UnitTypeId.BARRACKS, build_area, available_scvs)

    def _manage_upgrades(self):
        ccs: Units = self.state.ccs
        if ccs and self.ai.can_afford(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND):
            ccs.first(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND)

        if techlabs := self.ai.structures.filter(
            lambda u: u.type_id == UnitTypeId.BARRACKSTECHLAB and u.is_idle
        ):
            if self.ai.already_pending_upgrade(
                UpgradeId.SHIELDWALL
            ) == 0 and self.ai.can_afford(UpgradeId.SHIELDWALL):
                self.ai.research(UpgradeId.SHIELDWALL)
                return
            if self.ai.already_pending_upgrade(
                UpgradeId.STIMPACK
            ) == 0 and self.ai.can_afford(UpgradeId.STIMPACK):
                self.ai.research(UpgradeId.STIMPACK)
                return
            if self.ai.already_pending_upgrade(
                UpgradeId.PUNISHERGRENADES
            ) == 0 and self.ai.can_afford(UpgradeId.PUNISHERGRENADES):
                self.ai.research(UpgradeId.PUNISHERGRENADES)
                return

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
        max_gas: int = (
            2
            if len(self.state.barracks) >= 5
            else (1 if len(self.state.barracks) >= 3 else 0)
        )
        current_gas_num = (
            self.ai.already_pending(UnitTypeId.REFINERY) + self.ai.gas_buildings.amount
        )
        # Build refineries (on nearby vespene) when at least one barracks is in construction
        if (
            current_gas_num >= max_gas
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

    def _calculate_supply_placements(self) -> None:
        """
        Depots placements around the edge of the main base to make room for everything else
        TODO: Current status -> Good enough
            But try to get depot placements all round the edge
        """
        region: Region = self.map_data.in_region_p(self.ai.start_location)
        # reposition these points slightly inwards
        potential_depot_positions: Set[Point2] = {
            p.towards(region.center, 2.0).rounded
            for p in region.perimeter_points
            if self.ai.in_placement_grid(p.towards(region.center, 2.0).rounded)
            and p.distance_to(self.ai.main_base_ramp.top_center) > 5.5
        }

        placement_grid = self.ai.game_info.placement_grid.data_numpy.copy()
        for pos in potential_depot_positions:
            valid, placement_grid = self.ai.valid_two_by_two_position(
                pos, placement_grid
            )
            if valid:
                self.depot_positions.append(pos)

        self.depot_positions = sorted(
            self.depot_positions, key=lambda x: x.distance_to(self.ai.start_location)
        )

    async def _build_factory(self, available_scvs: Units) -> None:
        factories: Units = self.state.factories
        if f := factories.not_flying:
            f[0](AbilityId.LIFT_FACTORY)
            self.unit_roles.assign_role(f[0].tag, UnitRoleTypes.ATTACKING)

        if (
            self.ai.tech_requirement_progress(UnitTypeId.FACTORY) != 1
            or len(factories) >= 1
            or not available_scvs
            or not self.ai.can_afford(UnitTypeId.FACTORY)
            or self.ai.already_pending(UnitTypeId.FACTORY)
        ):
            return

        await self._build_structure(
            UnitTypeId.FACTORY, self.state.main_build_area, available_scvs
        )

    async def _build_starport(self, available_scvs: Units) -> None:
        ports: Units = self.state.starports
        if (
            self.ai.tech_requirement_progress(UnitTypeId.STARPORT) != 1
            or len(ports) >= 1
            or not available_scvs
            or not self.ai.can_afford(UnitTypeId.STARPORT)
            or self.ai.already_pending(UnitTypeId.STARPORT)
        ):
            return

        await self._build_structure(
            UnitTypeId.STARPORT, self.state.main_build_area, available_scvs
        )
