from typing import Union, List, Set, Tuple, Optional, Dict

import numpy as np
from scipy.spatial import KDTree

from bot.consts import ALL_STRUCTURES
from sc2.units import Units

from sc2.position import Point2

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId

from sc2.bot_ai import BotAI
from s2clientprotocol import raw_pb2 as raw_pb
from s2clientprotocol import sc2api_pb2 as sc_pb


class BotAIExt(BotAI):
    def __init__(self):
        self.enemy_tree: Optional[KDTree] = None

    async def on_step(self, iteration: int):
        pass

    def enemies_in_range(self, units: Units, distance: float) -> Dict[int, Units]:
        """
        Get all enemies in range of multiple units in one call
        :param units:
        :param distance:
        :return: Dictionary: Key -> Unit tag, Value -> Units in range of that unit
        """
        if not self.enemy_tree or not self.enemy_units:
            return {units[index].tag: Units([], self) for index in range(len(units))}

        unit_positions: List[Point2] = [u.position for u in units]
        in_range_list: List[Units] = []
        if unit_positions:
            query_result = self.enemy_tree.query_ball_point(unit_positions, distance)
            for result in query_result:
                in_range_units = Units(
                    [self.enemy_units[index] for index in result], self
                )
                in_range_list.append(in_range_units)
        return {units[index].tag: in_range_list[index] for index in range(len(units))}

    @staticmethod
    def center_mass(units: Units, distance: float = 5.0) -> Tuple[Point2, int]:
        """
        :param units:
        :param distance:
        :return: Position where most units reside, num units at that position
        """
        center_mass: Point2 = units[0].position
        max_num_units: int = 0
        for unit in units:
            pos: Point2 = unit.position
            close: Units = units.closer_than(distance, pos)
            if len(close) > max_num_units:
                center_mass = pos
                max_num_units = len(close)

        return center_mass, max_num_units

    async def give_units_same_order(
        self,
        order: AbilityId,
        unit_tags: Union[List[int], Set[int]],
        target: Optional[Union[Point2, int]] = None,
    ):
        """
        Give units corresponding to the given tags the same order.
        @param order: the order to give to all units
        @param unit_tags: the tags of the units to give the order to
        @param target: either a Point2 of the location to target or the tag of the unit to target
        """
        if not target:
            # noinspection PyProtectedMember
            await self.client._execute(
                action=sc_pb.RequestAction(
                    actions=[
                        sc_pb.Action(
                            action_raw=raw_pb.ActionRaw(
                                unit_command=raw_pb.ActionRawUnitCommand(
                                    ability_id=order.value,
                                    unit_tags=unit_tags,
                                )
                            )
                        ),
                    ]
                )
            )
        elif isinstance(target, Point2):
            # noinspection PyProtectedMember
            await self.client._execute(
                action=sc_pb.RequestAction(
                    actions=[
                        sc_pb.Action(
                            action_raw=raw_pb.ActionRaw(
                                unit_command=raw_pb.ActionRawUnitCommand(
                                    ability_id=order.value,
                                    target_world_space_pos=target.as_Point2D,
                                    unit_tags=unit_tags,
                                )
                            )
                        ),
                    ]
                )
            )
        else:
            # noinspection PyProtectedMember
            await self.client._execute(
                action=sc_pb.RequestAction(
                    actions=[
                        sc_pb.Action(
                            action_raw=raw_pb.ActionRaw(
                                unit_command=raw_pb.ActionRawUnitCommand(
                                    ability_id=order.value,
                                    target_unit_tag=target,
                                    unit_tags=unit_tags,
                                )
                            )
                        ),
                    ]
                )
            )

    @staticmethod
    def valid_two_by_two_position(
        position: Point2, placement_grid: np.ndarray
    ) -> Tuple[bool, np.ndarray]:
        """
        Have a possible building pos, see if it can go there
        :param position:
        :param placement_grid:
        :return:is valid bool, updated placement grid
        """
        points_to_check: List[List[int]] = [[0, 0] for _ in range(4)]
        # (x, y)
        points_to_check[0][0] = position[0]
        points_to_check[0][1] = position[1]
        # (x - 1, y)
        points_to_check[1][0] = position[0] - 1
        points_to_check[1][1] = position[1]
        # (x, y - 1)
        points_to_check[2][0] = position[0]
        points_to_check[2][1] = position[1] - 1
        # (x - 1, y - 1)
        points_to_check[3][0] = position[0] - 1
        points_to_check[3][1] = position[1] - 1

        valid: bool = True
        for point in points_to_check:
            x = point[0]
            y = point[1]
            if (
                x >= placement_grid.shape[0]
                or x < 0
                or y >= placement_grid.shape[1]
                or y < 0
            ):
                return False, placement_grid

            if placement_grid[point[0]][point[1]] == 0:
                valid = False
                break

        if valid:
            # update our copy of placement grid
            for point in points_to_check:
                placement_grid[point[0]][point[1]] = 0

        return valid, placement_grid

    def get_total_supply(self, units: Units) -> int:
        """
        Get total supply of units.
        @param units:
        @return:
        """
        return sum(
            [
                UNIT_DATA[unit.type_id]["supply"]
                for unit in units
                # yes we did have a crash getting supply of a nuke!
                if unit.type_id not in ALL_STRUCTURES and unit.type_id != UnitTypeId.NUKE
            ]
        )
