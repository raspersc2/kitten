from sc2.position import Point2

from sc2.bot_ai import BotAI


class Terrain:
    def __init__(self, ai: BotAI) -> None:
        self.ai: BotAI = ai
        self.expansion_distances: list[tuple[Point2, float]] = []

    @property
    def enemy_nat(self) -> Point2:
        return self.expansion_distances[0][0]

    @property
    def own_nat(self) -> Point2:
        return self.expansion_distances[-1][0]

    async def initialize(self) -> None:
        # store all expansion locations, sorted by distance to enemy
        for el in self.ai.expansion_locations_list:
            if self.ai.start_location.distance_to(el) < self.ai.EXPANSION_GAP_THRESHOLD:
                continue

            distance = await self.ai.client.query_pathing(self.ai.start_location, el)
            if distance:
                self.expansion_distances.append((el, distance))

        # sort by path length to each expansion
        self.expansion_distances = sorted(
            self.expansion_distances, key=lambda x: x[1], reverse=True
        )
