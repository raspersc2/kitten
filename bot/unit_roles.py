from typing import Dict, Set, Union, Optional, List

from sc2.bot_ai import BotAI
from bot.consts import UnitRoleTypes
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit
from sc2.units import Units


class UnitRoles:
    __slots__ = "ai", "unit_role_dict", "tag_to_role_dict", "all_assigned_tags", "WORKER_TYPES"

    def __init__(self, ai: BotAI):
        self.ai: BotAI = ai
        self.unit_role_dict: Dict[UnitRoleTypes, Set[int]] = {
            role: set() for role in UnitRoleTypes
        }
        self.tag_to_role_dict: Dict[int, UnitRoleTypes] = {}
        self.all_assigned_tags: Set[int] = set()
        self.WORKER_TYPES: Set[UnitTypeId] = {UnitTypeId.SCV, UnitTypeId.MULE}

    def assign_role(self, tag: int, role: UnitRoleTypes) -> None:
        """
        Assign a unit a role.
        """
        self.clear_role(tag)
        self.unit_role_dict[role].add(tag)
        self.tag_to_role_dict[tag] = role

    def clear_role(self, tag: int) -> None:
        """
        Clear a unit's role.
        """
        for role in self.unit_role_dict:
            if tag in self.unit_role_dict[role]:
                self.unit_role_dict[role].remove(tag)

    def catch_unit(self, unit: Unit) -> None:
        if unit.type_id in self.WORKER_TYPES:
            self.assign_role(unit.tag, UnitRoleTypes.GATHERING)
        else:
            self.assign_role(unit.tag, UnitRoleTypes.ATTACKING)

    def get_single_type_from_single_role(
        self,
        unit_type: UnitTypeId,
        role: UnitRoleTypes,
        restrict_to: Optional[Units] = None,
    ) -> List[Unit]:
        """
        Get all units of a given type that have a specified role. Moved to a function to avoid duplicated code.
        If restrict_to is Units, this will only get the units of the specified type and role that are also in
        restrict_to.
        @param unit_type:
        @param role:
        @param restrict_to:
        @return:
        """
        # get set of tags of units with the role
        unit_with_role_tags: Set[int] = self.unit_role_dict[role]
        # get the tags of units of the type
        units_of_type_tags: Set[int] = self.ai.units.filter(
            lambda u: u.type_id == unit_type
        ).tags

        # take the intersection of the sets to get the shared tags
        # this will be the units of the specified type with the specified role
        if not restrict_to:
            shared_tags: Set[int] = unit_with_role_tags & units_of_type_tags
        else:
            shared_tags: Set[int] = (
                unit_with_role_tags & units_of_type_tags & restrict_to.tags
            )

        return self.ai.units.tags_in(shared_tags)

    def get_units_from_role(
        self,
        role: UnitRoleTypes,
        unit_type: Optional[Union[UnitTypeId, Set[UnitTypeId]]] = None,
        restrict_to: Optional[Units] = None,
    ) -> Units:
        """
        Get a Units object containing units with a given role. If a UnitID or set of UnitIDs are given, it will only
        return units of those types, otherwise it will return all units with the role. If restrict_to is specified, it
        will only retrieve units from that object.
        """
        if unit_type:
            if isinstance(unit_type, UnitTypeId):
                # single unit type, use the single type and role function
                return Units(
                    self.get_single_type_from_single_role(unit_type, role, restrict_to),
                    self.ai,
                )
            else:
                # will crash if not an iterable, but we should be careful with typing anyway
                retrieved_units: List[Unit] = []
                for type_id in unit_type:
                    retrieved_units.extend(
                        self.get_single_type_from_single_role(
                            type_id, role, restrict_to
                        )
                    )
                return Units(retrieved_units, self.ai)
        else:
            # get every unit with the role
            if restrict_to:
                tags_to_get: Set[int] = self.unit_role_dict[role] & restrict_to.tags
            else:
                tags_to_get: Set[int] = self.unit_role_dict[role]
            # get the List[Unit] from UnitCacheManager and return as Units
            return self.ai.units.tags_in(tags_to_get)
