from typing import Union, List, Set, Tuple, Optional

from scipy.spatial.distance import cdist

from sc2.units import Units

from sc2.position import Point2

from sc2.ids.ability_id import AbilityId

from sc2.bot_ai import BotAI
from s2clientprotocol import raw_pb2 as raw_pb
from s2clientprotocol import sc2api_pb2 as sc_pb


class BotAIExt(BotAI):
    async def on_step(self, iteration: int):
        pass

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
