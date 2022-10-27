from enum import Enum, auto
from typing import Set

from sc2.ids.unit_typeid import UnitTypeId


class UnitRoleTypes(Enum):
    ATTACKING = auto()
    BUILDING = auto()
    GATHERING = auto()
    WORKER_DEFENDER = auto()


WORKERS_DEFEND_AGAINST: Set[UnitTypeId] = {
    UnitTypeId.DRONE,
    UnitTypeId.PROBE,
    UnitTypeId.SCV,
    UnitTypeId.ZERGLING
}
