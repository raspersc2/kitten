from dataclasses import dataclass
from typing import Optional

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.units import Units


@dataclass
class State:
    ai: BotAI
    barracks: Optional[Units] = None
    build_area: Optional[Point2] = None
    ccs: Optional[Units] = None
    depots: Optional[Units] = None
    orbitals: Optional[Units] = None

    def __post_init__(self):
        self.barracks = self.ai.structures(UnitTypeId.BARRACKS)

        self.ccs = self.ai.townhalls(UnitTypeId.COMMANDCENTER)
        self.depots = self.ai.structures(UnitTypeId.SUPPLYDEPOT)
        self.orbitals = self.ai.townhalls(UnitTypeId.ORBITALCOMMAND)
        if len(self.ai.townhalls.ready) > 1:
            self.build_area: Point2 = self.ai.townhalls.furthest_to(
                self.ai.start_location
            ).position.towards(self.ai.game_info.map_center, 5.5)
        else:
            self.build_area: Point2 = self.ai.start_location.towards(
                self.ai.game_info.map_center, 7.0
            )
