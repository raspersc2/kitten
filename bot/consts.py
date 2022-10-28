from enum import Enum, auto
from typing import Set, Dict

from sc2.ids.unit_typeid import UnitTypeId

DATA_DIR: str = "./data"


class SquadActionType(Enum):
    ATTACK_MOVE = auto()
    ATTACK_STUTTER_BACK = auto()
    ATTACK_STUTTER_FORWARD = auto()
    DESTROY_CLOSEST_ROCKS = auto()
    MOVE_TO_MAIN_OFFENSIVE_THREAT = auto()
    MOVE_TO_SAFE_SPOT = auto()
    HOLD_POSITION = auto()
    RETREAT_TO_RALLY_POINT = auto()


SQUAD_ACTIONS: Dict[int, SquadActionType] = {
    0: SquadActionType.ATTACK_MOVE,
    1: SquadActionType.ATTACK_STUTTER_BACK,
    2: SquadActionType.ATTACK_STUTTER_FORWARD,
    3: SquadActionType.MOVE_TO_SAFE_SPOT,
    4: SquadActionType.MOVE_TO_MAIN_OFFENSIVE_THREAT,
    5: SquadActionType.HOLD_POSITION,
    6: SquadActionType.DESTROY_CLOSEST_ROCKS,
    7: SquadActionType.RETREAT_TO_RALLY_POINT,
}


class UnitRoleTypes(Enum):
    ATTACKING = auto()
    BUILDING = auto()
    GATHERING = auto()
    WORKER_DEFENDER = auto()


WORKERS_DEFEND_AGAINST: Set[UnitTypeId] = {
    UnitTypeId.DRONE,
    UnitTypeId.PROBE,
    UnitTypeId.SCV,
    UnitTypeId.ZERGLING,
}
