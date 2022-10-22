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

ENEMY_WORKER_TYPES: Set[UnitTypeId] = {
    UnitTypeId.DRONE,
    UnitTypeId.PROBE,
    UnitTypeId.SCV,
    UnitTypeId.ZERGLING,
}


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

        """
        A RL agent to controls the main army squad
        """

        """
        Logic below this point is all the scripted parts:
         - Macro
         - Mining
         - Basic worker defence
         
         All rudimentary and could be heavily improved
        """
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

    def _handle_worker_rush(self) -> None:
        """zerglings too !"""
        # got to a point in time we don't care about this anymore, hopefully there are reapers around
        # scvs should go idle, at which point the gathering resources logic should kick in
        if self.time > 200.0 and not self.enemy_committed_worker_rush:
            self.worker_defence_tags = set()
            return

        enemy_workers: Units = self.enemy_units.filter(
            lambda u: u.type_id in ENEMY_WORKER_TYPES
            and (u.distance_to(self.start_location) < 25.0)
        )
        enemy_lings: Units = enemy_workers(UnitTypeId.ZERGLING)

        if enemy_workers.amount > 8 and self.time < 180:
            self.enemy_committed_worker_rush = True

        # calculate how many workers we should use to defend
        num_enemy_workers: int = enemy_workers.amount
        if num_enemy_workers > 0 and self.workers_manager:
            workers_needed: int = (
                num_enemy_workers
                if num_enemy_workers <= 6 and enemy_lings.amount <= 3
                else self.workers_manager.amount
            )
            if len(self.worker_defence_tags) < workers_needed:
                workers_to_take: int = workers_needed - len(self.worker_defence_tags)
                unassigned_workers: Units = self.workers_manager.tags_not_in(
                    self.worker_defence_tags
                )
                if workers_to_take > 0:
                    workers: Units = unassigned_workers.take(workers_to_take)
                    for worker in workers:
                        self.worker_defence_tags.add(worker.tag)

        # actually defend if there is a worker threat
        if len(self.worker_defence_tags) > 0 and self.mineral_field:
            defence_workers: Units = self.workers_manager.tags_in(
                self.worker_defence_tags
            )
            close_mineral_patch: Unit = self.mineral_field.closest_to(
                self.start_location
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
                self.worker_defence_tags = set()
