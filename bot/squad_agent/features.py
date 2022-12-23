import enum
from typing import List, Tuple

import numpy as np
import torch
from torch import uint8, int16, float16, float32
from torch.nn.functional import one_hot

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

from bot.botai_ext import BotAIExt

UNIT_TYPES = [
    0,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    149,
    150,
    151,
    268,
    289,
    311,
    321,
    322,
    324,
    330,
    335,
    336,
    341,
    342,
    343,
    344,
    350,
    364,
    365,
    371,
    372,
    373,
    376,
    377,
    472,
    473,
    474,
    475,
    483,
    484,
    485,
    486,
    487,
    488,
    489,
    490,
    493,
    494,
    495,
    496,
    498,
    499,
    500,
    501,
    502,
    503,
    504,
    517,
    518,
    559,
    560,
    561,
    562,
    563,
    564,
    588,
    589,
    590,
    591,
    608,
    609,
    610,
    612,
    628,
    629,
    630,
    638,
    639,
    640,
    641,
    642,
    643,
    648,
    649,
    651,
    661,
    662,
    663,
    664,
    665,
    666,
    687,
    688,
    689,
    690,
    691,
    692,
    693,
    694,
    732,
    733,
    734,
    796,
    797,
    801,
    824,
    830,
    877,
    880,
    881,
    884,
    885,
    886,
    887,
    892,
    893,
    894,
    1904,
    1908,
    1910,
    1911,
    1912,
    1913,
    1955,
    1956,
    1957,
    1958,
    1960,
    1961,
    1995,
]

BUFF_TYPES = [
    0,
    5,
    6,
    7,
    8,
    11,
    12,
    13,
    16,
    17,
    18,
    22,
    24,
    25,
    27,
    28,
    29,
    30,
    33,
    36,
    38,
    49,
    59,
    83,
    89,
    99,
    102,
    116,
    121,
    122,
    129,
    132,
    133,
    134,
    136,
    137,
    145,
    271,
    272,
    273,
    274,
    275,
    277,
    279,
    280,
    281,
    288,
    289,
    20,
    97,
    303,
    304,
    306,
    300,
    293,
]

# since we one hot encode unit type ids we don't want to hot encode 1961 different values
# so convert all unit type ids into our own index
BUFF_TYPE_DICT = dict(zip(BUFF_TYPES, range(0, len(BUFF_TYPES))))
UNIT_TYPE_DICT = dict(zip(UNIT_TYPES, range(0, len(UNIT_TYPES))))

SPATIAL_SIZE = [152, 152]  # y, x
BUFF_LENGTH = 3
UPGRADE_LENGTH = 20
MAX_DELAY = 127
BEGINNING_ORDER_LENGTH = 20
MAX_SELECTED_UNITS_NUM = 64
EFFECT_LEN = 100

NUM_UNIT_TYPES: int = len(UnitTypeId)
NUM_BUFF_TYPES: int = len(BUFF_TYPES)
NUM_UPGRADES: int = len(UpgradeId)


class FeatureType(enum.Enum):
    SCALAR = 1
    CATEGORICAL = 2


class ScoreCategories(enum.IntEnum):
    """Indices for the `score_by_category` observation's second dimension."""

    none = 0
    army = 1
    economy = 2
    technology = 3
    upgrade = 4


class FeatureUnit(enum.IntEnum):
    """Indices for the `feature_unit` observations."""

    unit_type = 0
    alliance = 1
    health_max = 2
    shield_max = 3
    energy_max = 4
    x = 5
    y = 6
    cloak = 7
    is_blip = 8
    is_powered = 9
    weapon_cooldown = 10
    is_hallucination = 11
    buff_id_0 = 12
    buff_id_1 = 13
    is_active = 14
    attack_upgrade_level = 15
    armor_upgrade_level = 16
    shield_upgrade_level = 17


ENTITY_INFO = [
    ("unit_type", int16),
    ("alliance", uint8),
    ("health_ratio", float16),
    ("shield_ratio", float16),
    ("energy_ratio", float16),
    ("x", uint8),
    ("y", uint8),
    ("cloak", uint8),
    ("is_blip", uint8),
    ("is_powered", uint8),
    ("weapon_cooldown", uint8),
    ("is_hallucination", uint8),
    ("buff_id_0", uint8),
    ("buff_id_1", uint8),
    ("is_active", uint8),
    ("attack_upgrade_level", uint8),
    ("armor_upgrade_level", uint8),
    ("shield_upgrade_level", uint8),
]

# (name, dtype, size)
SCALAR_INFO = [
    ("home_race", uint8, ()),
    ("away_race", uint8, ()),
    ("upgrades", int16, (NUM_UPGRADES,)),
    ("time", float32, ()),
    ("unit_counts_bow", uint8, (NUM_UNIT_TYPES,)),
    ("agent_statistics", float32, (10,)),
    ("beginning_order", int16, (BEGINNING_ORDER_LENGTH,)),
    ("last_queued", int16, ()),
    ("last_delay", int16, ()),
    ("last_action_type", int16, ()),
    ("bo_location", int16, (BEGINNING_ORDER_LENGTH,)),
    ("unit_type_bool", uint8, (NUM_UNIT_TYPES,)),
    ("enemy_unit_type_bool", uint8, (NUM_UNIT_TYPES,)),
]

SPATIAL_INFO = [
    ("height_map", uint8),
    ("visibility_map", uint8),
    ("creep", uint8),
    ("player_relative", uint8),
    ("pathable", uint8),
]


def compute_battle_score(obs, enemy=False):
    if obs is None:
        return 0.0
    score_details = obs.observation.score.score_details
    killed_mineral, killed_vespene = 0.0, 0.0
    for s in ScoreCategories:
        if not enemy:
            killed_mineral += getattr(score_details.killed_minerals, s.name)
            killed_vespene += getattr(score_details.killed_vespene, s.name)
        else:
            killed_mineral += getattr(score_details.lost_minerals, s.name)
            killed_vespene += getattr(score_details.lost_vespene, s.name)
    battle_score = killed_mineral + 1.5 * killed_vespene
    return battle_score


class Features:
    def __init__(self, ai: BotAIExt, max_entities: int, device) -> None:
        self.ai: BotAIExt = ai
        self.units: List = []
        self.tags: List[int] = []
        self.max_entities: int = max_entities
        self.device = device
        self.map_size_y = self.ai.game_info.map_size.y

    def reset(self) -> None:
        self.units = []
        self.tags = []

    def append_unit(self, u, alliance: int, unit_type: int) -> None:
        if len(self.units) >= self.max_entities:
            return

        self.units.append(
            [
                UNIT_TYPE_DICT[unit_type],
                0 if alliance == 1 else 1,
                u.health,
                u.health_max,
                u.shield,
                u.shield_max,
                u.energy,
                u.energy_max,
                u.pos.x,
                self.map_size_y - u.pos.y,
                u.cloak,
                u.is_powered,
                u.weapon_cooldown,
                u.is_hallucination,
                BUFF_TYPE_DICT[u.buff_ids[0]] if len(u.buff_ids) >= 1 else 0,
                BUFF_TYPE_DICT[u.buff_ids[1]] if len(u.buff_ids) >= 2 else 0,
                u.is_active,
                u.attack_upgrade_level,
                u.armor_upgrade_level,
                u.shield_upgrade_level,
            ]
        )

    def transform_obs(
        self, ground_grid: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        for unit in self.ai.state.observation_raw.units:
            alliance: int = unit.alliance
            unit_type: int = unit.unit_type
            self.append_unit(unit, alliance, unit_type)

        entity, entities_type, locations = self._process_entity_info()
        spatial = self._process_spatial_info(ground_grid)
        # TODO
        # scalar: torch.Tensor = self._process_scalar_info(obs, pos_of_squad)

        return spatial, entity, locations

    @staticmethod
    def _np_one_hot(targets: np.ndarray, nb_classes: int) -> np.ndarray:
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])

    def _process_entity_info(self):

        units: List = self.units[: self.max_entities]

        entities_array = np.array(units)

        entities_type = entities_array[:, 0].astype(np.int32)

        encoding_list = []

        unit_type_encoding = self._np_one_hot(entities_type, len(UNIT_TYPES) + 1)
        encoding_list.append(unit_type_encoding)

        alliance_encoding = self._np_one_hot(entities_array[:, 1].astype(np.int32), 2)
        encoding_list.append(alliance_encoding)

        health_ratio = np.expand_dims(
            entities_array[:, 2] / (entities_array[:, 3] + 1e-6), axis=1
        )
        shield_ratio = np.expand_dims(
            entities_array[:, 4] / (entities_array[:, 5] + 1e-6), axis=1
        )
        energy_ratio = np.expand_dims(
            entities_array[:, 6] / (entities_array[:, 7] + 1e-6), axis=1
        )
        x = np.expand_dims(entities_array[:, 8].astype(np.int32), axis=1)
        y = np.expand_dims(entities_array[:, 9].astype(np.int32), axis=1)
        encoding_list.extend(
            [
                health_ratio,
                shield_ratio,
                energy_ratio,
                x,  # x
                y,  # y
            ]
        )

        cloak_encoding = self._np_one_hot(entities_array[:, 10].astype(np.int32), 5)
        encoding_list.append(cloak_encoding)

        powered_encoding = self._np_one_hot(entities_array[:, 11].astype(np.int32), 2)
        encoding_list.append(powered_encoding)

        encoding_list.append(
            np.expand_dims(entities_array[:, 12], axis=1)
        )  # weapon cooldown

        halluc_encoding = self._np_one_hot(entities_array[:, 13].astype(np.int32), 2)
        encoding_list.append(halluc_encoding)

        buff_encoding1 = self._np_one_hot(
            entities_array[:, 14].astype(np.int32),
            NUM_BUFF_TYPES + 1,
        )
        buff_encoding2 = self._np_one_hot(
            entities_array[:, 15].astype(np.int32),
            NUM_BUFF_TYPES + 1,
        )
        encoding_list.extend([buff_encoding1, buff_encoding2])

        active_encoding = self._np_one_hot(entities_array[:, 16].astype(np.int32), 2)
        encoding_list.append(active_encoding)

        attack_upgrade_encoding = self._np_one_hot(
            entities_array[:, 17].astype(np.int32), 4
        )
        encoding_list.append(attack_upgrade_encoding)

        armor_upgrade_encoding = self._np_one_hot(
            entities_array[:, 18].astype(np.int32), 4
        )
        encoding_list.append(armor_upgrade_encoding)

        shield_upgrade_encoding = self._np_one_hot(
            entities_array[:, 19].astype(np.int32), 4
        )
        encoding_list.append(shield_upgrade_encoding)

        all_entities_array = np.concatenate(encoding_list, axis=1)

        all_entities_array = np.pad(
            all_entities_array,
            ((0, 256 - all_entities_array.shape[0]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        all_entities_array = torch.from_numpy(all_entities_array)
        all_entities_array = all_entities_array.to(torch.float32)
        all_entities_array = torch.unsqueeze(all_entities_array, 0)

        locations = np.concatenate([x, y], axis=1)
        locations = np.pad(
            locations,
            ((0, 256 - locations.shape[0]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        locations = torch.from_numpy(locations).to(torch.float32)
        locations = torch.unsqueeze(locations, 0)

        return all_entities_array, entities_type, locations

    def _process_spatial_info(self, ground_grid: np.ndarray) -> torch.Tensor:

        spatial_arr = []
        # location_grid = squad_grid.copy()

        ground_grid = ground_grid[None, :]
        ground_grid = torch.from_numpy(ground_grid)
        spatial_arr.append(ground_grid)

        height = self.ai.game_info.terrain_height.data_numpy.T
        height = height[None, :]
        height = torch.from_numpy(height)
        spatial_arr.append(height)

        creep = self.ai.state.creep.data_numpy.T
        creep = torch.from_numpy(creep)
        creep = one_hot(creep.to(torch.int64), 2)
        creep = torch.movedim(creep, 2, 0)
        spatial_arr.append(creep)

        visibility = self.ai.state.visibility.data_numpy.T
        visibility = visibility[None, :]
        visibility = torch.from_numpy(visibility)
        spatial_arr.append(visibility)

        # location_grid[self.mediator.get_rally_point.rounded] = 0.25
        # location_grid[self.mediator.get_harass_target.rounded] = 0.5
        # if (
        #     ground_threats_near_townhall := self.mediator.get_main_ground_threats_near_townhall
        # ):
        #     location_grid[ground_threats_near_townhall.center.rounded] = 0.75
        # location_grid[self.mediator.get_offensive_attack_target.rounded] = 1.0
        # location_grid = location_grid[None, :]
        # location_grid = torch.from_numpy(location_grid)
        # spatial_arr.append(location_grid)

        spatial_arr = torch.cat(spatial_arr)

        return spatial_arr
