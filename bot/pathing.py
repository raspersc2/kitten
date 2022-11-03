from typing import Optional, Dict, List

import numpy as np

from sc2.position import Point2

from sc2.unit import Unit

from sc2.units import Units

from MapAnalyzer import MapData
from sc2.bot_ai import BotAI
import cv2

from bot.consts import ALL_STRUCTURES, INFLUENCE_COSTS


class Pathing:
    __slots__ = (
        "ai",
        "map_data",
        "memory_unit_tags",
        "memory_units",
        "ground_grid",
        "TIME_IN_MEMORY",
        "RANGE_BUFFER",
    )

    def __init__(self, ai: BotAI, map_data: Optional[MapData]):
        self.ai: BotAI = ai
        self.map_data: Optional[MapData] = map_data
        self.memory_unit_tags: Dict[int, Dict] = dict()
        self.memory_units: Units = Units([], self.ai)
        self.ground_grid: Optional[np.ndarray] = None
        self.TIME_IN_MEMORY: float = 15.0
        self.RANGE_BUFFER: float = 3.0

    def update(self, iteration: int) -> None:
        # TODO: Add effects

        # get clean grid
        self.ground_grid = self.map_data.get_pyastar_grid()

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

        self.map_data.draw_influence_in_game(self.ground_grid)

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
            position=(int(pos.x), int(pos.y)),
            radius=unit_range,
            grid=grid,
            weight=int(weight),
            initial_default_weights=initial_default_weights,
        )
        return grid
