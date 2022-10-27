from sc2.bot_ai import BotAI
from sc2.position import Point2
from sc2.units import Units


class UnitSquad:
    __slots__ = "ai", "squad_id", "squad_position", "squad_units"

    def __init__(self, ai: BotAI, squad_id: str, squad_units: Units):
        self.ai: BotAI = ai
        self.squad_id: str = squad_id
        self.squad_units: Units = squad_units
        self.squad_position: Point2 = squad_units.center
