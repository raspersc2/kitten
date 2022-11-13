from typing import Set, Union, Optional, Dict

from bot.pathing import Pathing
from sc2.unit import Unit

from bot.botai_ext import BotAIExt
from sc2.position import Point2
from sc2.units import Units
from sc2.ids.ability_id import AbilityId


class UnitSquad:
    __slots__ = (
        "ai",
        "squad_id",
        "squad_position",
        "squad_units",
        "current_action",
        "current_action_position",
        "stutter_forward",
        "stuttering",
        "action_updated_this_step",
    )

    def __init__(self, ai: BotAIExt, squad_id: str, squad_units: Units):
        self.ai: BotAIExt = ai
        self.squad_id: str = squad_id
        self.squad_units: Units = squad_units
        self.squad_position: Point2 = squad_units.center

        self.current_action: AbilityId = AbilityId.ATTACK
        self.current_action_position: Point2 = self.ai.game_info.map_center
        # if set to False, units will kite back by default
        self.stutter_forward: bool = False
        # if we store this, then the stutter action is only issued once
        self.stuttering: bool = False

        self.action_updated_this_step: bool = False

    def set_squad_units(self, units: Units) -> None:
        self.squad_units = units

    def update_action(
        self, action: AbilityId, position: Point2, stutter_forward: bool = False
    ) -> None:
        if action != self.current_action or position != self.current_action_position:
            self.current_action = action
            self.current_action_position = position
            self.stutter_forward = stutter_forward
            self.action_updated_this_step = True

    async def do_action(
        self, squad_tags: Set[int], pathing: Pathing, main_squad: bool = False
    ) -> None:
        # currently only main squad uses RL agent, all other squads have some scripted logic
        if not main_squad:
            await self._do_scripted_squad_action(squad_tags)
        else:

            if self.current_action == AbilityId.HOLDPOSITION:
                perform_action: bool = False
                for unit in self.squad_units:
                    if not unit.is_using_ability(AbilityId.HOLDPOSITION):
                        perform_action = True
                        break
                if perform_action:
                    await self.ai.give_units_same_order(
                        AbilityId.HOLDPOSITION, squad_tags
                    )
            elif self.current_action == AbilityId.ATTACK:
                await self._do_squad_attack_action(squad_tags, pathing)
            else:
                await self._do_squad_move_action(squad_tags, pathing)

    async def _do_scripted_squad_action(self, squad_tags: Set[int]) -> None:
        """
        Used for smaller squads
        Main goal is to join up with the main squad
        Without spamming too many actions
        """
        unit: Unit = self.squad_units[0]
        target: Union[Point2, int, None] = unit.order_target
        if (
            target
            and isinstance(target, Point2)
            and target.distance_to(self.current_action_position) < 10.0
        ):
            return

        await self.ai.give_units_same_order(
            self.current_action, squad_tags, self.current_action_position
        )

    def should_stutter(self, close_enemy: Dict[int, Units]) -> bool:
        avg_weapon_cooldown: float = sum(
            [u.weapon_cooldown for u in self.squad_units]
        ) / len(self.squad_units)
        # all weapons are ready, should stay on attack command
        if avg_weapon_cooldown == 0.0:
            return False

        if self.stutter_forward:
            # if all units are in range of something, don't worry about moving
            all_in_range: bool = True
            for unit in self.squad_units:
                enemy: Units = close_enemy[unit.tag]
                if not enemy.in_attack_range_of(unit):
                    all_in_range = False
                    break
            if all_in_range:
                return False
            return avg_weapon_cooldown > 3.5
        else:
            return avg_weapon_cooldown > 5.5

    async def _do_squad_attack_action(
        self, squad_tags: Set[int], pathing: Pathing
    ) -> None:
        close_enemy: Dict[int, Units] = self.ai.enemies_in_range(self.squad_units, 15.0)
        should_stutter: bool = self.should_stutter(close_enemy)
        sample_unit: Unit = self.squad_units[0]

        # all units weapons are ready, fire!
        if not should_stutter and not sample_unit.is_attacking:
            self.stuttering = False
            await self.ai.give_units_same_order(
                self.current_action,
                squad_tags,
                self.current_action_position,
            )
        # else move depending on the agent's action type
        elif should_stutter and not self.stuttering:
            # only call this once till weapons are ready again, so action isn't spammed
            self.stuttering = True
            if self.stutter_forward:
                pos: Point2 = self.current_action_position
            else:
                # get a path back home, and kite back using that
                pos: Point2 = pathing.find_path_next_point(
                    start=self.squad_position,
                    target=self.ai.start_location,
                    grid=pathing.ground_grid,
                    sensitivity=12,
                )
            await self.ai.give_units_same_order(AbilityId.MOVE, squad_tags, pos)

    async def _do_squad_move_action(
        self, squad_tags: Set[int], pathing: Pathing
    ) -> None:
        if self.squad_position.distance_to(self.current_action_position) < 2.0:
            return

        sample_unit: Unit = self.squad_units[0]

        pos: Point2 = pathing.find_path_next_point(
            start=self.squad_position,
            target=self.current_action_position,
            grid=pathing.ground_grid,
            sensitivity=5,
        )
        order_target: Optional[int, Point2] = sample_unit.order_target
        if (
            order_target
            and isinstance(order_target, Point2)
            and order_target.distance_to(pos) > 1.0
        ):
            await self.ai.give_units_same_order(self.current_action, squad_tags, pos)
