"""
Basic macro that focuses on bio production and upgrades
This should probably be rewritten / refactored into separate files for anything more complicated
"""
from typing import Optional, List, Set

from MapAnalyzer import MapData, Region
from bot.botai_ext import BotAIExt
from bot.consts import UnitRoleTypes
from bot.modules.unit_roles import UnitRoles
from bot.modules.workers import WorkersManager
from bot.state import State
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2, Point3, Pointlike
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

        self._manage_upgrades()
        self._build_refineries()
        await self._build_supply(iteration)
        self._produce_workers()
        await self._build_addons()
        self._produce_army()
        await self._build_factory()
        await self._build_starport()
        await self._build_barracks()
        await self._build_bays()

        # catch any scvs not doing anything and send back to mining
        if building_scvs := self.unit_roles.get_units_from_role(
            UnitRoleTypes.BUILDING
        ).filter(lambda u: len(u.orders) == 0 or u.is_carrying_vespene or u.is_idle):
            for scv in building_scvs:
                self.unit_roles.assign_role(scv.tag, UnitRoleTypes.GATHERING)

        # 2 townhalls at all times
        if len(self.ai.townhalls) < 2 and self.ai.can_afford(UnitTypeId.COMMANDCENTER):
            location: Optional[Point2] = await self.ai.get_next_expansion()
            if location:
                await self._build_structure(
                    UnitTypeId.COMMANDCENTER,
                    self.state.natural_build_area,
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
            if rax.has_techlab and self.ai.minerals >= 100:
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

    async def _build_supply(self, iteration: int) -> None:
        # lower existing depots
        if iteration % 12 == 0:
            for depot in self.state.depots.ready:
                depot(AbilityId.MORPH_SUPPLYDEPOT_LOWER)

        num_rax: int = len(self.state.barracks)
        max_building: int = 1 if num_rax < 3 else (2 if num_rax < 8 else 3)
        build_when_supply_left: int = 10 if num_rax >= 8 else 5
        if (
            self.ai.supply_used < 14
            or self.ai.supply_left >= build_when_supply_left
            or self.ai.supply_cap >= 200
            or self.ai.already_pending(UnitTypeId.SUPPLYDEPOT) >= max_building
            or not self.ai.can_afford(UnitTypeId.SUPPLYDEPOT)
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
                specific_location=pos,
            )
        # else generic depot placement
        else:
            await self._build_structure(
                UnitTypeId.SUPPLYDEPOT, self.state.natural_build_area
            )

    async def _build_addons(self) -> None:
        ready_rax: Units = self.state.barracks.filter(lambda u: u.is_ready)
        if len(ready_rax) < 3 or self.ai.vespene < 25:
            return

        add_ons: Units = self.ai.structures(UnitTypeId.BARRACKSTECHLAB)
        max_add_ons: int = 2 if len(self.state.barracks) > 5 else 1
        if len(add_ons) < max_add_ons and self.ai.can_afford(UnitTypeId.TECHLAB):
            rax: Units = ready_rax.filter(lambda u: u.is_idle)
            for b in rax:
                if not b.has_add_on:
                    add_on_location: Pointlike = b.position.offset(Point2((2.5, -0.5)))
                    if await self.ai.can_place(UnitTypeId.SUPPLYDEPOT, add_on_location):
                        b.build(UnitTypeId.STARPORTTECHLAB)

    async def _build_barracks(self) -> None:
        max_barracks: int = (
            2 if len(self.ai.townhalls) <= 1 else (4 if not self.state.factories else 8)
        )
        if self.ai.minerals > 500:
            max_barracks = 9

        rax: Units = self.state.barracks
        if self._dont_build(
            rax,
            UnitTypeId.BARRACKS,
            num_existing=max_barracks,
            max_pending=5,
        ):
            return

        build_area: Point2 = (
            self.state.main_build_area
            if len(rax) < 5
            else self.state.natural_build_area
        )

        await self._build_structure(UnitTypeId.BARRACKS, build_area)

    def _manage_upgrades(self):
        ccs: Units = self.state.ccs
        if ccs and self.ai.can_afford(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND):
            ccs.first(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND)

        if self.ai.structures.filter(
            lambda u: u.type_id == UnitTypeId.BARRACKSTECHLAB and u.is_idle
        ):
            if self.ai.already_pending_upgrade(
                UpgradeId.SHIELDWALL
            ) == 0 and self.ai.can_afford(UpgradeId.SHIELDWALL):
                self.ai.research(UpgradeId.SHIELDWALL)
                return
            # if self.ai.already_pending_upgrade(
            #     UpgradeId.STIMPACK
            # ) == 0 and self.ai.can_afford(UpgradeId.STIMPACK):
            #     self.ai.research(UpgradeId.STIMPACK)
            #     return
            if (
                self.ai.already_pending_upgrade(UpgradeId.PUNISHERGRENADES) == 0
                and self.ai.can_afford(UpgradeId.PUNISHERGRENADES)
                # and UpgradeId.STIMPACK in self.ai.state.upgrades
                and UpgradeId.SHIELDWALL in self.ai.state.upgrades
            ):
                self.ai.research(UpgradeId.PUNISHERGRENADES)
                return

        if self.ai.already_pending_upgrade(
            UpgradeId.TERRANINFANTRYWEAPONSLEVEL1
        ) == 0 and self.ai.can_afford(UpgradeId.PUNISHERGRENADES):
            self.ai.research(UpgradeId.TERRANINFANTRYWEAPONSLEVEL1)
            return

        if self.ai.already_pending_upgrade(
            UpgradeId.TERRANINFANTRYARMORSLEVEL1
        ) == 0 and self.ai.can_afford(UpgradeId.PUNISHERGRENADES):
            self.ai.research(UpgradeId.TERRANINFANTRYARMORSLEVEL1)
            return

    async def _build_structure(
        self,
        structure_type: UnitTypeId,
        placement_area: Point2,
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
            if worker := self.workers_manager.select_worker(location, force=True):
                self.unit_roles.assign_role(worker.tag, UnitRoleTypes.BUILDING)
                self.workers_manager.remove_worker_from_mineral(worker.tag)
                worker.build(structure_type, location)

    def _build_refineries(self):
        # 2 gas buildings
        num_rax: int = len(self.state.barracks)
        max_gas: int = 2 if num_rax >= 4 else 0
        current_gas_num = (
            self.ai.already_pending(UnitTypeId.REFINERY) + self.ai.gas_buildings.amount
        )
        # Build refineries (on nearby vespene) when at least one barracks is in construction
        if current_gas_num >= max_gas or not self.ai.can_afford(UnitTypeId.REFINERY):
            return

        # Loop over all townhalls nearly complete
        for th in self.ai.townhalls.filter(lambda _th: _th.build_progress > 0.6):
            # Find all vespene geysers that are closer than range 10 to this townhall
            vgs: Units = self.ai.vespene_geyser.closer_than(10, th)
            for vg in vgs:
                if self.ai.gas_buildings.filter(lambda gb: gb.distance_to(vg) < 3.0):
                    continue

                if worker := self.workers_manager.select_worker(
                    vg.position, force=True
                ):
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

    async def _build_factory(self) -> None:
        factories: Units = self.state.factories

        # we only care about factories for the starport
        if self.state.starports:
            return

        if self._dont_build(factories, UnitTypeId.FACTORY):
            return

        await self._build_structure(UnitTypeId.FACTORY, self.state.main_build_area)

    async def _build_starport(self) -> None:
        ports: Units = self.state.starports
        if self._dont_build(ports, UnitTypeId.STARPORT):
            return

        await self._build_structure(UnitTypeId.STARPORT, self.state.main_build_area)

    def _dont_build(
        self,
        structures: Units,
        structure_type: UnitTypeId,
        num_existing: int = 1,
        max_pending: int = 1,
    ) -> bool:
        return (
            self.ai.tech_requirement_progress(structure_type) != 1
            or len(structures) >= num_existing
            or not self.ai.can_afford(structure_type)
            or self.ai.already_pending(structure_type) >= max_pending
        )

    async def _build_bays(self):
        bay_type: UnitTypeId = UnitTypeId.ENGINEERINGBAY
        bays = self.ai.structures(bay_type)

        if self._dont_build(bays, bay_type) or len(self.state.factories) < 1:
            return

        await self._build_structure(bay_type, self.state.main_build_area)
