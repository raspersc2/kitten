from collections import defaultdict
from typing import Dict, Set, List, DefaultDict, Optional

from bot.consts import UnitRoleTypes, WORKERS_DEFEND_AGAINST
from bot.modules.unit_roles import UnitRoles
from bot.state import State
from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units


class WorkersManager:
    __slots__ = (
        "ai",
        "unit_roles",
        "workers_per_gas",
        "worker_to_mineral_patch_dict",
        "mineral_patch_to_list_of_workers",
        "mineral_tag_to_mineral",
        "mineral_object_to_worker_units_object",
        "worker_tag_to_townhall_tag",
        "worker_to_geyser_dict",
        "geyser_to_list_of_workers",
        "enemy_committed_worker_rush",
        "worker_defence_tags",
        "long_distance_mfs",
        "locked_action_tags",
    )

    def __init__(self, ai: BotAI, unit_roles: UnitRoles) -> None:
        self.ai: BotAI = ai
        self.unit_roles: UnitRoles = unit_roles
        self.workers_per_gas: int = 3
        self.worker_to_mineral_patch_dict: Dict[int, int] = {}
        self.mineral_patch_to_list_of_workers: Dict[int, Set[int]] = {}
        self.mineral_tag_to_mineral: Dict[int, Unit] = {}
        self.mineral_object_to_worker_units_object: DefaultDict[
            Unit, List[Unit]
        ] = defaultdict(list)
        # store which townhall the worker is closest to
        self.worker_tag_to_townhall_tag: Dict[int, int] = {}

        self.worker_to_geyser_dict: Dict[int, int] = {}
        self.geyser_to_list_of_workers: Dict[int, Set[int]] = {}
        self.enemy_committed_worker_rush: bool = False
        self.worker_defence_tags: Set = set()
        self.long_distance_mfs: Units = Units([], self.ai)
        self.locked_action_tags: Dict[int, float] = dict()

    @property
    def available_minerals(self) -> Units:
        """
        Find all mineral fields available near a townhall that don't have 2 workers assigned to it yet
        """
        available_minerals: Units = Units([], self.ai)
        townhalls: Units = self.ai.townhalls.filter(lambda th: th.build_progress > 0.85)
        if not townhalls or not self.ai.mineral_field:
            return available_minerals

        for townhall in townhalls:
            # we want workers on closest mineral patch first
            minerals_sorted: Units = self.ai.mineral_field.filter(
                lambda mf: mf.is_visible
                and not mf.is_snapshot
                and mf.distance_to(townhall) < 10
                and len(self.mineral_patch_to_list_of_workers.get(mf.tag, [])) < 2
            ).sorted_by_distance_to(townhall)

            if minerals_sorted:
                available_minerals.extend(minerals_sorted)

        return available_minerals

    def update(self, state: State, iteration: int) -> None:
        gatherers: Units = self.unit_roles.get_units_from_role(UnitRoleTypes.GATHERING)
        self._assign_workers(gatherers)
        if iteration % 4 == 0:
            self._collect_resources(gatherers)

        for oc in state.orbitals.filter(lambda x: x.energy >= 50):
            mfs: Units = self.ai.mineral_field.closer_than(10, oc)
            if mfs:
                mf: Unit = max(mfs, key=lambda x: x.mineral_contents)
                oc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mf)

        self._handle_worker_rush()

    def select_worker(self, target_position: Point2) -> Optional[Unit]:
        """
        Note: Make sure to change the worker role once selected. Otherwise, it is selected to mine again
        This doesn't select workers from geysers, so make sure to remove workers from gas if low on workers
        """
        workers: Units = self.ai.workers.tags_in(self.worker_to_mineral_patch_dict)
        # there is a chance we have no workers
        if not workers:
            return

        # if there are workers not assigned to mine (probably long distance mining), choose one of those and return
        unassigned_workers: Units = workers.tags_not_in(
            list(self.worker_to_mineral_patch_dict) + list(self.worker_to_geyser_dict)
        )
        if unassigned_workers:
            return unassigned_workers.closest_to(target_position)

        if available_workers := workers.filter(
            lambda w: w.tag in self.worker_to_mineral_patch_dict
            and not w.is_carrying_resource
        ):
            # find townhalls with plenty of mineral patches
            townhalls: Units = self.ai.townhalls.filter(
                lambda th: th.is_ready
                and self.ai.mineral_field.closer_than(10, th).amount >= 8
            ).sorted_by_distance_to(target_position)
            # seems there are no townhalls with plenty of resources, don't be fussy at this point
            if not townhalls:
                return available_workers.closest_to(target_position)

            # go through townhalls, we loop through the min fields by distance to townhall
            # that way there is a good chance we pick a worker at a far mineral patch
            for townhall in townhalls:
                minerals_sorted_by_distance: Units = self.ai.mineral_field.closer_than(
                    10, townhall
                ).sorted_by_distance_to(townhall)
                for mineral in reversed(minerals_sorted_by_distance):
                    # we have record of the patch, with some worker tags saved
                    if mineral.tag in self.mineral_patch_to_list_of_workers:
                        # try to get a worker at this patch that is not carrying resources
                        if _workers := available_workers.filter(
                            lambda w: w.tag
                            in self.mineral_patch_to_list_of_workers[mineral.tag]
                            and not w.is_carrying_resource
                            and not w.is_collecting
                        ):
                            worker: Unit = _workers.first
                            # make sure to remove worker, so a new one can be assigned to mine
                            self.remove_worker_from_mineral(worker.tag)
                            return worker

            # somehow got here without finding a worker, anyone will do
            worker: Unit = available_workers.closest_to(target_position)
            self.remove_worker_from_mineral(worker.tag)
            return worker

    def _assign_workers(self, workers: Units) -> None:
        """
        Assign workers to mineral patches and gas buildings
        @param workers:
        @return:
        """
        if not workers or not self.ai.townhalls:
            return

        # This takes priority, ok to remove from minerals
        if self.ai.gas_buildings:
            self._assign_worker_to_gas_buildings(self.ai.gas_buildings)

        unassigned_workers: Units = workers.filter(
            lambda u: u.tag not in self.worker_to_geyser_dict
            and u.tag not in self.worker_to_mineral_patch_dict
        )

        if self.available_minerals:
            self._assign_workers_to_mineral_patches(
                self.available_minerals, unassigned_workers
            )

    def _assign_worker_to_gas_buildings(self, gas_buildings: Units) -> None:
        """
        We only assign one worker per step, with the hope of grabbing workers on far mineral patches
        @param gas_buildings:
        @return:
        """
        if not self.ai.townhalls:
            return

        for gas in gas_buildings.ready:
            # don't assign if there is no townhall nearby
            if not self.ai.townhalls.closer_than(12, gas):
                continue
            # too many workers assigned, this can happen if we want to pull workers off gas
            if (
                len(self.geyser_to_list_of_workers.get(gas.tag, []))
                > self.workers_per_gas
            ):
                workers_on_gas: Units = self.ai.workers.tags_in(
                    self.geyser_to_list_of_workers[gas.tag]
                )
                if workers_on_gas:
                    self._remove_worker_from_vespene(workers_on_gas.first.tag)
                continue
            # already perfect amount of workers assigned
            if (
                len(self.geyser_to_list_of_workers.get(gas.tag, []))
                == self.workers_per_gas
            ):
                continue

            # Assign worker closest to the gas building
            worker: Optional[Unit] = self.select_worker(gas.position)

            if not worker or worker.tag in self.geyser_to_list_of_workers:
                continue
            if (
                len(self.geyser_to_list_of_workers.get(gas.tag, []))
                < self.workers_per_gas
            ):
                if len(self.geyser_to_list_of_workers.get(gas.tag, [])) == 0:
                    self.geyser_to_list_of_workers[gas.tag] = {worker.tag}
                else:
                    if worker.tag not in self.geyser_to_list_of_workers[gas.tag]:
                        self.geyser_to_list_of_workers[gas.tag].add(worker.tag)
                self.worker_to_geyser_dict[worker.tag] = gas.tag
                self.worker_tag_to_townhall_tag[
                    worker.tag
                ] = self.ai.townhalls.closest_to(gas).tag
                # if this worker was collecting minerals, we need to remove it
                self.remove_worker_from_mineral(worker.tag)
                break

    def _assign_workers_to_mineral_patches(
        self, available_minerals: Units, workers: Units
    ) -> None:
        """
        Given some minerals and workers, assign two to each mineral patch
        Thanks to burny's example worker stacking code:
        https://github.com/BurnySc2/python-sc2/blob/develop/examples/worker_stack_bot.py
        @param available_minerals:
        @param workers:
        @return:
        """
        if len(workers) == 0 or not self.ai.townhalls:
            return

        _minerals: Units = available_minerals

        for worker in workers:
            tag: int = worker.tag
            # run out of minerals to assign
            if not _minerals:
                return
            if (
                tag in self.worker_to_mineral_patch_dict
                or tag in self.worker_to_geyser_dict
            ):
                continue

            # find the closest mineral, then find the nearby minerals that are closest to the townhall
            closest_mineral: Unit = _minerals.closest_to(worker)
            nearby_minerals: Units = _minerals.closer_than(10, closest_mineral)
            th: Unit = self.ai.townhalls.closest_to(closest_mineral)
            mineral: Unit = nearby_minerals.closest_to(th)

            if len(self.mineral_patch_to_list_of_workers.get(mineral.tag, [])) < 2:
                self._assign_worker_to_patch(mineral, worker)

            # enough have been assigned to this patch, don't consider it on next iteration over loop
            if len(self.mineral_patch_to_list_of_workers.get(mineral.tag, [])) >= 2:
                _minerals.remove(mineral)

    def _assign_worker_to_patch(self, mineral_field: Unit, worker: Unit) -> None:
        mineral_tag: int = mineral_field.tag
        worker_tag: int = worker.tag
        if len(self.mineral_patch_to_list_of_workers.get(mineral_tag, [])) == 0:
            self.mineral_patch_to_list_of_workers[mineral_tag] = {worker_tag}
        else:
            if worker_tag not in self.mineral_patch_to_list_of_workers[mineral_tag]:
                self.mineral_patch_to_list_of_workers[mineral_tag].add(worker_tag)
        self.worker_to_mineral_patch_dict[worker_tag] = mineral_tag
        self.worker_tag_to_townhall_tag[worker_tag] = self.ai.townhalls.closest_to(
            mineral_field
        ).tag

    def _collect_resources(self, workers: Units) -> None:
        if not workers or not self.ai.townhalls:
            return

        calculated_long_distance_mfs: bool = False
        gas_buildings: Dict[int, Unit] = {gas.tag: gas for gas in self.ai.gas_buildings}
        minerals: Dict[int, Unit] = {
            mineral.tag: mineral for mineral in self.ai.mineral_field
        }
        for worker in workers:
            worker_tag: int = worker.tag
            if worker_tag in self.locked_action_tags:
                if self.ai.time > self.locked_action_tags[worker_tag] + 0.4:
                    self.locked_action_tags.pop(worker_tag)
                continue

            if worker_tag in self.worker_to_mineral_patch_dict:
                mineral_tag: int = self.worker_to_mineral_patch_dict[worker_tag]
                mineral: Optional[Unit] = minerals.get(mineral_tag, None)
                if mineral is None:
                    # Mined out or no vision? Remove it
                    self._remove_mineral_field(mineral_tag)
                    continue

                if (
                    not worker.is_carrying_minerals
                    and worker.order_target != mineral.tag
                ):
                    worker.gather(mineral)
                    self.locked_action_tags[worker_tag] = self.ai.time
                    # to reduce apm only click one scv on to their patch per iteration
                    # let in-game engine sort things out inbetween
                    break
            elif worker_tag in self.worker_to_geyser_dict:
                gas_building_tag: int = self.worker_to_geyser_dict[worker.tag]
                gas_building: Optional[Unit] = gas_buildings.get(gas_building_tag, None)
                townhall: Unit = self.ai.townhalls.closest_to(worker)

                if not gas_building or not gas_building.vespene_contents:
                    self._remove_gas_building(gas_building_tag)
                elif (
                    worker.order_target != gas_building.tag
                    and worker.order_target != townhall.tag
                ):
                    worker.gather(gas_building)

            else:
                if not worker.is_carrying_resource and not worker.is_gathering:
                    if not calculated_long_distance_mfs:
                        self.long_distance_mfs = self.ai.mineral_field.filter(
                            lambda mf: not self.ai.townhalls.closer_than(
                                15.0, mf.position
                            )
                        )
                        calculated_long_distance_mfs = True
                    if len(self.ai.townhalls) <= 1:
                        worker.gather(self.ai.mineral_field.closest_to(worker))
                    else:
                        worker.gather(self.long_distance_mfs.closest_to(worker))

    def remove_worker_from_mineral(self, worker_tag: int) -> None:
        """
        Remove worker from internal data structures.
        This happens if worker gets assigned to do something else
        @param worker_tag:
        @return:
        """
        if worker_tag in self.worker_to_mineral_patch_dict:
            # found the worker, get the min tag before deleting
            min_patch_tag: int = self.worker_to_mineral_patch_dict[worker_tag]
            del self.worker_to_mineral_patch_dict[worker_tag]
            if worker_tag in self.worker_tag_to_townhall_tag:
                del self.worker_tag_to_townhall_tag[worker_tag]

            # using the min patch tag, we can remove from other collection
            self.mineral_patch_to_list_of_workers[min_patch_tag].remove(worker_tag)

    def _remove_worker_from_vespene(self, worker_tag: int) -> None:
        """
        Remove worker from internal data structures.
        This happens if worker gets assigned to do something else, or removing workers from gas
        @param worker_tag:
        @return:
        """
        if worker_tag in self.worker_to_geyser_dict:
            # found the worker, get the gas building tag before deleting
            gas_building_tag: int = self.worker_to_geyser_dict[worker_tag]
            del self.worker_to_geyser_dict[worker_tag]
            if worker_tag in self.worker_tag_to_townhall_tag:
                del self.worker_tag_to_townhall_tag[worker_tag]

            # using the gas building tag, we can remove from other collection
            self.geyser_to_list_of_workers[gas_building_tag].remove(worker_tag)

    def _remove_gas_building(self, gas_building_tag):
        """Remove gas building and assigned workers from bookkeeping"""
        if gas_building_tag in self.geyser_to_list_of_workers:
            del self.geyser_to_list_of_workers[gas_building_tag]
            self.worker_to_geyser_dict = {
                key: val
                for key, val in self.worker_to_geyser_dict.items()
                if val != gas_building_tag
            }

    def _remove_mineral_field(self, mineral_field_tag: int) -> None:
        """Remove mineral field and assigned workers from bookkeeping"""
        if mineral_field_tag in self.mineral_patch_to_list_of_workers:
            del self.mineral_patch_to_list_of_workers[mineral_field_tag]
            self.worker_to_mineral_patch_dict = {
                key: val
                for key, val in self.worker_to_mineral_patch_dict.items()
                if val != mineral_field_tag
            }

    def _handle_worker_rush(self) -> None:
        """zerglings too !"""
        # got to a point in time we don't care about this anymore, hopefully there are reapers around
        # scvs should go idle, at which point the gathering resources logic should kick in
        if (
            self.ai.time > 200.0 and not self.enemy_committed_worker_rush
        ) or not self.ai.workers:
            self.worker_defence_tags = set()
            return

        enemy_workers: Units = self.ai.enemy_units.filter(
            lambda u: u.type_id in WORKERS_DEFEND_AGAINST
            and (u.distance_to(self.ai.start_location) < 25.0)
        )
        enemy_lings: Units = enemy_workers(UnitTypeId.ZERGLING)

        if enemy_workers.amount > 8 and self.ai.time < 180:
            self.enemy_committed_worker_rush = True

        # calculate how many workers we should use to defend
        num_enemy_workers: int = enemy_workers.amount
        if num_enemy_workers > 0:
            workers_needed: int = (
                num_enemy_workers
                if num_enemy_workers <= 6 and enemy_lings.amount <= 3
                else len(self.ai.workers)
            )
            if len(self.worker_defence_tags) < workers_needed:
                workers_to_take: int = workers_needed - len(self.worker_defence_tags)
                unassigned_workers: Units = self.ai.workers.tags_not_in(
                    self.worker_defence_tags
                )
                if workers_to_take > 0:
                    workers: Units = unassigned_workers.take(workers_to_take)
                    for worker in workers:
                        self.worker_defence_tags.add(worker.tag)
                        self.unit_roles.assign_role(
                            worker.tag, UnitRoleTypes.WORKER_DEFENDER
                        )

        # actually defend if there is a worker threat
        if len(self.worker_defence_tags) > 0 and self.ai.mineral_field:
            defence_workers: Units = self.ai.workers.tags_in(self.worker_defence_tags)
            close_mineral_patch: Unit = self.ai.mineral_field.closest_to(
                self.ai.start_location
            )
            if defence_workers and enemy_workers:
                for worker in defence_workers:
                    # in attack range of enemy, prioritise attacking
                    if (
                        worker.weapon_cooldown == 0
                        and enemy_workers.in_attack_range_of(worker)
                    ):
                        worker.attack(enemy_workers.closest_to(worker))
                    # attack the workers
                    elif worker.weapon_cooldown == 0 and enemy_workers:
                        worker.attack(enemy_workers.closest_to(worker))
                    else:
                        worker.gather(close_mineral_patch)
            elif defence_workers:
                for worker in defence_workers:
                    worker.gather(close_mineral_patch)
                    self.unit_roles.assign_role(worker.tag, UnitRoleTypes.GATHERING)
                self.worker_defence_tags = set()
