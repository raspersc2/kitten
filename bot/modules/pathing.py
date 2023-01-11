from typing import Optional, Dict, List

import numpy as np
from sc2.ids.effect_id import EffectId
from scipy import spatial

from MapAnalyzer import MapData
from bot.consts import ALL_STRUCTURES, INFLUENCE_COSTS, EFFECT_COSTS
from sc2.bot_ai import BotAI
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units


class Pathing:
    __slots__ = (
        "ai",
        "map_data",
        "memory_unit_tags",
        "memory_units",
        "effects_grid",
        "ground_grid",
        "TIME_IN_MEMORY",
        "RANGE_BUFFER",
    )

    def __init__(self, ai: BotAI, map_data: Optional[MapData]):
        self.ai: BotAI = ai
        self.map_data: Optional[MapData] = map_data
        self.memory_unit_tags: Dict[int, Dict] = dict()
        self.memory_units: Units = Units([], self.ai)
        self.effects_grid: Optional[np.ndarray] = None
        self.ground_grid: Optional[np.ndarray] = None
        self.TIME_IN_MEMORY: float = 15.0
        # this buffer is fairly large, since we are pathing as a squad in this project
        # rather than precise individual unit control
        self.RANGE_BUFFER: float = 5.5

    def update(self, iteration: int) -> None:
        # get clean grid
        if self.ground_grid is None or iteration % 64 == 0:
            grid = self.map_data.get_pyastar_grid()
            self.ground_grid = grid.copy()
            self.effects_grid = grid.copy()

        self._add_effects()

        self.memory_units = Units([], self.ai)
        # Add enemy influence to ground grid and refresh memory of enemy units
        for unit in self.ai.all_enemy_units:
            tag: int = unit.tag
            if unit.type_id in ALL_STRUCTURES:
                self._add_structure_influence(unit)
            else:
                self._add_unit_influence(unit)
                # seen an enemy unit, remember it incase we lose vision
                if tag in self.memory_unit_tags:
                    self.memory_unit_tags.update(
                        {
                            tag: {
                                "expiry": self.ai.time + self.TIME_IN_MEMORY,
                                "unit_obj": unit,
                            }
                        }
                    )
                else:
                    self.memory_unit_tags[tag] = {
                        "expiry": self.ai.time + self.TIME_IN_MEMORY,
                        "unit_obj": unit,
                    }

        # remove stale units in memory
        tags_to_remove: List[int] = []
        for enemy_unit_tag in self.memory_unit_tags:
            if self.ai.time > self.memory_unit_tags[enemy_unit_tag]["expiry"]:
                tags_to_remove.append(enemy_unit_tag)
            else:
                self.memory_units.append(
                    self.memory_unit_tags[enemy_unit_tag]["unit_obj"]
                )

        for unit in self.memory_units:
            self._add_unit_influence(unit)

        for tag in tags_to_remove:
            self.remove_unit_tag(tag)

    def _add_effects(self) -> None:
        """Add effects influence to map"""

        for effect in self.ai.state.effects:
            if effect == EffectId.LURKERMP:
                effect_costs = EFFECT_COSTS[effect.id]
                for pos in effect.positions:
                    self.effects_grid = self._add_cost(
                        pos,
                        effect_costs["GroundCost"],
                        effect_costs["GroundRange"],
                        self.effects_grid,
                    )
            elif effect.id in EFFECT_COSTS:
                effect_costs = EFFECT_COSTS[effect.id]
                self.effects_grid = self._add_cost(
                    Point2.center(effect.positions),
                    effect_costs["GroundCost"],
                    effect_costs["GroundRange"],
                    self.effects_grid,
                )

    def find_closest_safe_spot(
        self, from_pos: Point2, grid: np.ndarray, radius: int = 15
    ) -> Point2:
        """
        @param from_pos:
        @param grid:
        @param radius:
        @return:
        """
        all_safe: np.ndarray = self.map_data.lowest_cost_points_array(
            from_pos, radius, grid
        )
        # type hint wants a numpy array but doesn't actually need one - this is faster
        all_dists = spatial.distance.cdist(all_safe, [from_pos], "sqeuclidean")
        min_index = np.argmin(all_dists)

        # safe because the shape of all_dists (N x 1) means argmin will return an int
        return Point2(all_safe[min_index])

    def find_path_next_point(
        self,
        start: Point2,
        target: Point2,
        grid: np.ndarray,
        sensitivity: int = 2,
        smoothing: bool = False,
    ) -> Point2:
        """
        Most commonly used, we need to calculate the right path for a unit
        But only the first element of the path is required
        @param start:
        @param target:
        @param grid:
        @param sensitivity:
        @param smoothing:
        @return: The next point on the path we should move to
        """
        # Note: On rare occasions a path is not found and returns `None`
        path: Optional[List[Point2]] = self.map_data.pathfind(
            start, target, grid, sensitivity=sensitivity, smoothing=smoothing
        )
        if not path or len(path) == 0:
            return target
        else:
            return path[0]

    @staticmethod
    def is_position_safe(
        grid: np.ndarray,
        position: Point2,
        weight_safety_limit: float = 1.0,
    ) -> bool:
        """
        Checks if the current position is dangerous by comparing against default_grid_weights
        @param grid: Grid we want to check
        @param position: Position of the unit etc
        @param weight_safety_limit: The threshold at which we declare the position safe
        @return:
        """
        position = position.rounded
        weight: float = grid[position.x, position.y]
        # np.inf check if drone is pathing near a spore crawler
        return weight == np.inf or weight <= weight_safety_limit

    def remove_unit_tag(self, unit_tag: int) -> None:
        if unit_tag in self.memory_unit_tags:
            self.memory_unit_tags.pop(unit_tag)

    def _add_unit_influence(self, enemy: Unit) -> None:
        """
        Add influence to the relevant grid.
        TODO:
            Add spell castors
            Add units that have no weapon in the API such as BCs, sentries and voids
            Extend this to add influence to an air grid
        @return:
        """
        # this unit is in our dictionary where we define custom weights and ranges
        # it could be this unit doesn't have a weapon in the API or we just want to use custom values
        if enemy.type_id in INFLUENCE_COSTS:
            values: Dict = INFLUENCE_COSTS[enemy.type_id]
            self.ground_grid = self._add_cost(
                enemy.position,
                values["GroundCost"],
                values["GroundRange"] + self.RANGE_BUFFER,
                self.ground_grid,
            )
        # this unit has values in the API and is not in our custom dictionary, take them from there
        elif enemy.can_attack_ground:
            self.ground_grid = self._add_cost(
                enemy.position,
                enemy.ground_dps,
                enemy.ground_range + self.RANGE_BUFFER,
                self.ground_grid,
            )

    def _add_structure_influence(self, enemy: Unit) -> None:
        """
        Add structure influence to the relevant grid.
        TODO:
            Extend this to add influence to an air grid
        @param enemy:
        @return:
        """
        if not enemy.is_ready:
            return

        if enemy.type_id in INFLUENCE_COSTS:
            values: Dict = INFLUENCE_COSTS[enemy.type_id]
            self.ground_grid = self._add_cost(
                enemy.position,
                values["GroundCost"],
                values["GroundRange"] + self.RANGE_BUFFER,
                self.ground_grid,
            )

    def _add_cost(
        self,
        pos: Point2,
        weight: float,
        unit_range: float,
        grid: np.ndarray,
        initial_default_weights: int = 0,
    ) -> np.ndarray:
        """Or add "influence", mostly used to add enemies to a grid"""

        grid = self.map_data.add_cost(
            position=pos.rounded,
            radius=unit_range,
            grid=grid,
            weight=int(weight),
            initial_default_weights=initial_default_weights,
        )
        return grid
