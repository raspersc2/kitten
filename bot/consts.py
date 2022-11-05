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
    0: SquadActionType.ATTACK_STUTTER_BACK,
    1: SquadActionType.ATTACK_STUTTER_FORWARD,
    2: SquadActionType.MOVE_TO_SAFE_SPOT,
    3: SquadActionType.MOVE_TO_MAIN_OFFENSIVE_THREAT,
    4: SquadActionType.RETREAT_TO_RALLY_POINT,
    5: SquadActionType.HOLD_POSITION,
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

ALL_STRUCTURES: Set[UnitTypeId] = {
    UnitTypeId.ARMORY,
    UnitTypeId.ASSIMILATOR,
    UnitTypeId.ASSIMILATORRICH,
    UnitTypeId.AUTOTURRET,
    UnitTypeId.BANELINGNEST,
    UnitTypeId.BARRACKS,
    UnitTypeId.BARRACKSFLYING,
    UnitTypeId.BARRACKSREACTOR,
    UnitTypeId.BARRACKSTECHLAB,
    UnitTypeId.BUNKER,
    UnitTypeId.BYPASSARMORDRONE,
    UnitTypeId.COMMANDCENTER,
    UnitTypeId.COMMANDCENTERFLYING,
    UnitTypeId.CREEPTUMOR,
    UnitTypeId.CREEPTUMORBURROWED,
    UnitTypeId.CREEPTUMORQUEEN,
    UnitTypeId.CYBERNETICSCORE,
    UnitTypeId.DARKSHRINE,
    UnitTypeId.ELSECARO_COLONIST_HUT,
    UnitTypeId.ENGINEERINGBAY,
    UnitTypeId.EVOLUTIONCHAMBER,
    UnitTypeId.EXTRACTOR,
    UnitTypeId.EXTRACTORRICH,
    UnitTypeId.FACTORY,
    UnitTypeId.FACTORYFLYING,
    UnitTypeId.FACTORYREACTOR,
    UnitTypeId.FACTORYTECHLAB,
    UnitTypeId.FLEETBEACON,
    UnitTypeId.FORGE,
    UnitTypeId.FUSIONCORE,
    UnitTypeId.GATEWAY,
    UnitTypeId.GHOSTACADEMY,
    UnitTypeId.GREATERSPIRE,
    UnitTypeId.HATCHERY,
    UnitTypeId.HIVE,
    UnitTypeId.HYDRALISKDEN,
    UnitTypeId.INFESTATIONPIT,
    UnitTypeId.LAIR,
    UnitTypeId.LURKERDENMP,
    UnitTypeId.MISSILETURRET,
    UnitTypeId.NEXUS,
    UnitTypeId.NYDUSCANAL,
    UnitTypeId.NYDUSCANALATTACKER,
    UnitTypeId.NYDUSCANALCREEPER,
    UnitTypeId.NYDUSNETWORK,
    UnitTypeId.ORACLESTASISTRAP,
    UnitTypeId.ORBITALCOMMAND,
    UnitTypeId.ORBITALCOMMANDFLYING,
    UnitTypeId.PHOTONCANNON,
    UnitTypeId.PLANETARYFORTRESS,
    UnitTypeId.POINTDEFENSEDRONE,
    UnitTypeId.PYLON,
    UnitTypeId.PYLONOVERCHARGED,
    UnitTypeId.RAVENREPAIRDRONE,
    UnitTypeId.REACTOR,
    UnitTypeId.REFINERY,
    UnitTypeId.REFINERYRICH,
    UnitTypeId.RESOURCEBLOCKER,
    UnitTypeId.ROACHWARREN,
    UnitTypeId.ROBOTICSBAY,
    UnitTypeId.ROBOTICSFACILITY,
    UnitTypeId.SENSORTOWER,
    UnitTypeId.SHIELDBATTERY,
    UnitTypeId.SPAWNINGPOOL,
    UnitTypeId.SPINECRAWLER,
    UnitTypeId.SPINECRAWLERUPROOTED,
    UnitTypeId.SPIRE,
    UnitTypeId.SPORECRAWLER,
    UnitTypeId.SPORECRAWLERUPROOTED,
    UnitTypeId.STARGATE,
    UnitTypeId.STARPORT,
    UnitTypeId.STARPORTFLYING,
    UnitTypeId.STARPORTREACTOR,
    UnitTypeId.STARPORTTECHLAB,
    UnitTypeId.SUPPLYDEPOT,
    UnitTypeId.SUPPLYDEPOTLOWERED,
    UnitTypeId.TECHLAB,
    UnitTypeId.TEMPLARARCHIVE,
    UnitTypeId.TWILIGHTCOUNCIL,
    UnitTypeId.ULTRALISKCAVERN,
    UnitTypeId.WARPGATE,
}

INFLUENCE_COSTS: Dict[UnitTypeId, Dict] = {
    UnitTypeId.ADEPT: {"AirCost": 0, "GroundCost": 9, "AirRange": 0, "GroundRange": 5},
    UnitTypeId.ADEPTPHASESHIFT: {
        "AirCost": 0,
        "GroundCost": 9,
        "AirRange": 0,
        "GroundRange": 5,
    },
    UnitTypeId.AUTOTURRET: {
        "AirCost": 31,
        "GroundCost": 31,
        "AirRange": 7,
        "GroundRange": 7,
    },
    UnitTypeId.ARCHON: {
        "AirCost": 40,
        "GroundCost": 40,
        "AirRange": 3,
        "GroundRange": 3,
    },
    UnitTypeId.BANELING: {
        "AirCost": 0,
        "GroundCost": 20,
        "AirRange": 0,
        "GroundRange": 3,
    },
    UnitTypeId.BANSHEE: {
        "AirCost": 0,
        "GroundCost": 12,
        "AirRange": 0,
        "GroundRange": 6,
    },
    UnitTypeId.BATTLECRUISER: {
        "AirCost": 31,
        "GroundCost": 50,
        "AirRange": 6,
        "GroundRange": 6,
    },
    UnitTypeId.BUNKER: {
        "AirCost": 22,
        "GroundCost": 22,
        "AirRange": 6,
        "GroundRange": 6,
    },
    UnitTypeId.CARRIER: {
        "AirCost": 20,
        "GroundCost": 20,
        "AirRange": 11,
        "GroundRange": 11,
    },
    UnitTypeId.CORRUPTOR: {
        "AirCost": 10,
        "GroundCost": 0,
        "AirRange": 6,
        "GroundRange": 0,
    },
    UnitTypeId.CYCLONE: {
        "AirCost": 27,
        "GroundCost": 27,
        "AirRange": 7,
        "GroundRange": 7,
    },
    UnitTypeId.GHOST: {
        "AirCost": 10,
        "GroundCost": 10,
        "AirRange": 6,
        "GroundRange": 6,
    },
    UnitTypeId.HELLION: {
        "AirCost": 0,
        "GroundCost": 8,
        "AirRange": 0,
        "GroundRange": 8,
    },
    UnitTypeId.HYDRALISK: {
        "AirCost": 20,
        "GroundCost": 20,
        "AirRange": 6,
        "GroundRange": 6,
    },
    UnitTypeId.INFESTOR: {
        "AirCost": 30,
        "GroundCost": 30,
        "AirRange": 10,
        "GroundRange": 10,
    },
    UnitTypeId.LIBERATOR: {
        "AirCost": 10,
        "GroundCost": 0,
        "AirRange": 5,
        "GroundRange": 0,
    },
    UnitTypeId.MARINE: {
        "AirCost": 10,
        "GroundCost": 10,
        "AirRange": 5,
        "GroundRange": 5,
    },
    UnitTypeId.MOTHERSHIP: {
        "AirCost": 23,
        "GroundCost": 23,
        "AirRange": 7,
        "GroundRange": 7,
    },
    UnitTypeId.MUTALISK: {
        "AirCost": 8,
        "GroundCost": 8,
        "AirRange": 3,
        "GroundRange": 3,
    },
    UnitTypeId.ORACLE: {
        "AirCost": 0,
        "GroundCost": 24,
        "AirRange": 0,
        "GroundRange": 4,
    },
    UnitTypeId.PHOENIX: {
        "AirCost": 15,
        "GroundCost": 0,
        "AirRange": 7,
        "GroundRange": 0,
    },
    UnitTypeId.PHOTONCANNON: {
        "AirCost": 22,
        "GroundCost": 22,
        "AirRange": 7,
        "GroundRange": 7,
    },
    UnitTypeId.QUEEN: {
        "AirCost": 12.6,
        "GroundCost": 11.2,
        "AirRange": 7,
        "GroundRange": 5,
    },
    UnitTypeId.SENTRY: {
        "AirCost": 8.4,
        "GroundCost": 8.4,
        "AirRange": 5,
        "GroundRange": 5,
    },
    UnitTypeId.SPINECRAWLER: {
        "AirCost": 0,
        "GroundCost": 15,
        "AirRange": 0,
        "GroundRange": 7,
    },
    UnitTypeId.STALKER: {
        "AirCost": 10,
        "GroundCost": 10,
        "AirRange": 6,
        "GroundRange": 6,
    },
    UnitTypeId.TEMPEST: {
        "AirCost": 17,
        "GroundCost": 17,
        "AirRange": 14,
        "GroundRange": 10,
    },
    UnitTypeId.THOR: {
        "AirCost": 28,
        "GroundCost": 28,
        "AirRange": 11,
        "GroundRange": 7,
    },
    UnitTypeId.VIKINGASSAULT: {
        "AirCost": 0,
        "GroundCost": 17,
        "AirRange": 0,
        "GroundRange": 6,
    },
    UnitTypeId.VIKINGFIGHTER: {
        "AirCost": 14,
        "GroundCost": 0,
        "AirRange": 9,
        "GroundRange": 0,
    },
    UnitTypeId.VOIDRAY: {
        "AirCost": 20,
        "GroundCost": 20,
        "AirRange": 6,
        "GroundRange": 6,
    },
    UnitTypeId.WIDOWMINEBURROWED: {
        "AirCost": 150,
        "GroundCost": 150,
        "AirRange": 5.5,
        "GroundRange": 5.5,
    },
}
