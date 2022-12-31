"""
Create and manage unit squad bookkeeping.
Note: squad actions are carried out in `unit_squad.py`
"""
import uuid
from typing import Set, Dict, Any, List, Optional

from loguru import logger

from bot.botai_ext import BotAIExt
from bot.consts import UnitRoleTypes, SQUAD_ACTIONS, SquadActionType
from bot.modules.pathing import Pathing
from bot.modules.unit_roles import UnitRoles
from bot.squad_agent.base_agent import BaseAgent
from bot.unit_squad import UnitSquad
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.units import Units


class UnitSquads:
    __slots__ = (
        "ai",
        "unit_roles",
        "agent",
        "assigned_unit_tags",
        "squads",
        "squads_dict",
        "attack_target",
        "rally_point",
        "AGENT_FRAME_SKIP",
        "SQUAD_OBJECT",
        "SQUAD_RADIUS",
        "TAGS",
    )

    def __init__(self, ai: BotAIExt, unit_roles: UnitRoles, agent: BaseAgent):

        self.ai: BotAIExt = ai
        self.unit_roles: UnitRoles = unit_roles
        self.agent: BaseAgent = agent

        self.squads: List[UnitSquad] = []

        self.assigned_unit_tags: Set[int] = set()
        # key -> squad_id, value -> tags, squad object
        self.squads_dict: Dict[str, Dict[str, Any]] = dict()

        self.attack_target: Optional[Point2] = None
        self.rally_point: Optional[Point2] = None

        # How often we get a new squad action (22.4 FPS)
        self.AGENT_FRAME_SKIP: int = 20
        self.SQUAD_OBJECT: str = "squad_object"
        self.SQUAD_RADIUS: float = 15.0
        self.TAGS: str = "tags"

    async def update(self, iteration: int, pathing: Pathing) -> None:
        if iteration % 8 == 0:
            self._set_rally_point()
            self._set_attack_target()

        army: Units = self.unit_roles.get_units_from_role(UnitRoleTypes.ATTACKING)

        # handle unit squad assignment not currently in our records
        self._squad_assignment(army.tags_not_in(self.assigned_unit_tags))
        # handle existing squads merging / splitting
        self._handle_existing_squads_assignment(army)
        # update the unit collections associated with each squad
        self._regenerate_squad_units(army)
        # control the unit squads
        if len(self.squads) > 0:
            await self._handle_squads(iteration, pathing)

    def remove_tag(self, tag: int) -> None:
        """'on_unit_destroyed' calls this"""
        if tag in self.assigned_unit_tags:
            found_squad: bool = False
            squad_id_to_remove_from = ""
            for squad_id, squad_info in self.squads_dict.items():
                if tag in squad_info[self.TAGS]:
                    squad_id_to_remove_from = squad_id
                    found_squad = True
                    break
            if found_squad:
                self._remove_unit_tag(tag, squad_id_to_remove_from)

    async def _handle_squads(self, iteration: int, pathing: Pathing) -> None:
        (
            id_of_largest_squad,
            pos_of_largest_squad,
            largest_squad,
        ) = self._get_largest_squad(self.squads)

        for squad in self.squads:
            # for the main squad, we use the agent to decide on an action
            # aim for a new agent action once every 20 frames
            # the individual unit squad scripted control will be based on this action
            if squad.squad_id == id_of_largest_squad:
                # update the action once every 20 frames (just under once per second (in-game time))
                if iteration % (self.AGENT_FRAME_SKIP // self.ai.client.game_step) == 0:
                    action: int = self.agent.choose_action(
                        self.squads,
                        pos_of_largest_squad,
                        self.ai.all_enemy_units.filter(
                            lambda u: u.distance_to(pos_of_largest_squad) < 15.0
                        ),
                        squad.squad_units,
                        self.attack_target,
                        self.rally_point,
                    )
                    logger.info(f"Chosen action: {SQUAD_ACTIONS[action]}")
                    action_type: SquadActionType = SQUAD_ACTIONS[action]
                    if action_type == SquadActionType.ATTACK_STUTTER_BACK:
                        squad.update_action(AbilityId.ATTACK, self.attack_target)
                    elif action_type == SquadActionType.ATTACK_STUTTER_FORWARD:
                        squad.update_action(AbilityId.ATTACK, self.attack_target, True)
                    elif action_type == SquadActionType.MOVE_TO_MAIN_OFFENSIVE_THREAT:
                        squad.update_action(AbilityId.MOVE, self.attack_target)
                    elif action_type == SquadActionType.HOLD_POSITION:
                        squad.update_action(AbilityId.HOLDPOSITION, self.attack_target)
                    elif action_type == SquadActionType.RETREAT_TO_RALLY_POINT:
                        squad.update_action(AbilityId.MOVE, self.rally_point)
                    elif action_type == SquadActionType.HOLD_POSITION:
                        squad.update_action(AbilityId.HOLDPOSITION, self.rally_point)
                    elif action_type == SquadActionType.STIM:
                        squad.set_stim_status(True)
            else:
                squad.update_action(AbilityId.ATTACK, pos_of_largest_squad)

            await squad.do_action(
                squad_tags=self.squads_dict[squad.squad_id][self.TAGS],
                pathing=pathing,
                main_squad=squad.squad_id == id_of_largest_squad,
            )

    def _squad_assignment(self, unassigned_units: Units) -> None:
        if not unassigned_units:
            return

        for unit in unassigned_units:
            tag: int = unit.tag
            # check if unit may join an existing squad
            squad_to_join: str = self._closest_squad_id(
                unit.position, self.SQUAD_RADIUS
            )
            # found an existing squad to join
            if squad_to_join != "":
                self.squads_dict[squad_to_join][self.TAGS].add(tag)
                self.assigned_unit_tags.add(tag)
            # otherwise create a new squad containing just this unit
            else:
                self._create_squad({tag})

    def _handle_existing_squads_assignment(self, army: Units):
        """
        Handle units straying from squads, or multiple squads overlapping etc.
        """
        # Stray units get too far from squad -> Remove from current squad
        for squad in self.squads:
            squad_id = squad.squad_id
            in_range_tags: Set[int] = army.closer_than(
                self.SQUAD_RADIUS, squad.squad_position
            ).tags
            for unit in squad.squad_units:
                if unit.tag not in in_range_tags:
                    self._remove_unit_tag(unit.tag, squad_id)

        # Multiple squads overlapping -> Merge
        for squad in self.squads:
            squad_id = squad.squad_id
            if self._merge_with_closest_squad(squad_id):
                # only merge one squad per frame, makes managing this somewhat easier
                break

    def _closest_squad_id(
        self, position: Point2, distance_to_check: float, avoid_squad_id: str = ""
    ) -> str:
        if not self.squads:
            return ""

        closest_squad: Optional[UnitSquad] = None
        min_distance: float = 9998.9
        for squad in self.squads:
            if squad.squad_id == avoid_squad_id:
                continue

            current_distance: float = position.distance_to(squad.squad_position)
            if current_distance < min_distance:
                closest_squad = squad
                min_distance = current_distance

        return closest_squad.squad_id if min_distance < distance_to_check else ""

    def _create_squad(self, tags: Set[int]) -> None:
        squad_id: str = uuid.uuid4().hex
        squad_units = self.ai.units.tags_in(tags)
        squad: UnitSquad = UnitSquad(self.ai, squad_id, squad_units)
        self.squads_dict[squad_id] = {}
        self.squads_dict[squad_id][self.TAGS] = tags
        self.squads_dict[squad_id][self.SQUAD_OBJECT] = squad
        self.squads.append(squad)
        for tag in tags:
            self.assigned_unit_tags.add(tag)

    def _remove_unit_tag(self, tag: int, squad_id: str):
        """
        Remove a unit tag from any data structures
        """
        if tag in self.assigned_unit_tags:
            self.assigned_unit_tags.remove(tag)

        if tag in self.squads_dict[squad_id][self.TAGS]:
            self.squads_dict[squad_id][self.TAGS].remove(tag)

        # if this was the only unit in the squad, then remove the squad too
        if len(self.squads_dict[squad_id][self.TAGS]) == 0:
            self._remove_squad(squad_id)

    def _remove_squad(self, squad_id: str, squad_id_to_join: str = "") -> None:
        """
        Remove squad from bookkeeping
        Optionally pass a new squad id for remaining units to join
        """
        # get any leftover units in this squad before deleting anything
        units: Units = self.squads_dict[squad_id][self.SQUAD_OBJECT].squad_units

        del self.squads_dict[squad_id]
        self.squads = [squad for squad in self.squads if squad_id != squad.squad_id]

        # if providing another squad to join then add the units tags to the squad dict
        # (these units will then be added to the squad on the next frame)
        if squad_id_to_join != "" and squad_id_to_join in self.squads_dict:
            for unit in units:
                self.squads_dict[squad_id_to_join][self.TAGS].add(unit.tag)
        # no squad to join, remove from assigned_unit_tags so the units can be repurposed
        else:
            for unit in units:
                tag = unit.tag
                if tag in self.assigned_unit_tags:
                    self.assigned_unit_tags.remove(tag)

    def _merge_with_closest_squad(self, squad_id: str, distance: float = 15.0) -> bool:
        squad: UnitSquad = self.squads_dict[squad_id][self.SQUAD_OBJECT]
        closest_squad_id: str = self._closest_squad_id(
            squad.squad_position, distance, squad_id
        )
        if closest_squad_id != "":
            # remove this squad
            tags = squad.squad_units.tags
            self._remove_squad(squad_id)
            # add tags to new squad id
            self.squads_dict[closest_squad_id][self.TAGS].update(tags)
            for tag in tags:
                self.assigned_unit_tags.add(tag)
            return True

        return False

    def _get_largest_squad(self, squads):
        """
        TODO: Largest based on supply instead
            Easier to calculate on number of units initially
        """
        main_group_id = ""
        # default value, last known position of main squad
        position_of_squad: Point2 = self.ai.start_location
        largest_squad: Optional[UnitSquad] = None
        num_units_in_main_group: int = 0

        for squad in squads:
            amount: int = len(squad.squad_units)
            if amount >= num_units_in_main_group:
                main_group_id = squad.squad_id
                position_of_squad = squad.squad_units.center
                num_units_in_main_group = amount
                largest_squad = squad

        return main_group_id, position_of_squad, largest_squad

    def _regenerate_squad_units(self, army: Units) -> None:
        """
        Using the recorded tags of each squad, regenerate a fresh Units object for this frame
        """
        squads_to_remove: List[Dict] = []
        for squad_id in self.squads_dict:
            squad_units: Units = Units(
                army.tags_in(self.squads_dict[squad_id][self.TAGS]), self.ai
            )
            # squads may contain no more units (we don't clear up the tags of dead units)
            if not squad_units:
                squads_to_remove.append(
                    {
                        "id": squad_id,
                        "role": self.squads_dict[squad_id][self.SQUAD_OBJECT].role,
                    }
                )
                continue

            self.squads_dict[squad_id][self.SQUAD_OBJECT].set_squad_units(squad_units)

        # remove any squads with empty units
        for squad_to_remove in squads_to_remove:
            self._remove_squad(squad_to_remove["id"])

    def _set_rally_point(self):
        if len(self.ai.townhalls) > 1:
            self.rally_point = self.ai.townhalls.furthest_to(
                self.ai.start_location
            ).position.towards(self.ai.game_info.map_center, 5)
        else:
            self.rally_point = self.ai.main_base_ramp.top_center

    def _set_attack_target(self) -> None:
        if enemy_units := self.ai.enemy_units:
            center_mass, num = self.ai.center_mass(enemy_units)
            if num >= 5:
                self.attack_target = center_mass
                return

        if enemy_structures := self.ai.enemy_structures:
            self.attack_target = enemy_structures.closest_to(
                self.ai.game_info.map_center
            ).position
            return

        self.attack_target = self.ai.enemy_start_locations[0]
