from enum import Enum, auto


class UnitRoleTypes(Enum):
    ATTACKING = auto()
    BUILDING = auto()
    GATHERING = auto()
    WORKER_DEFENDER = auto()
