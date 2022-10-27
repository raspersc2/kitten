import random
import sys

sys.path.insert(1, "python-sc2")

from sc2 import maps
from sc2.data import Race, Difficulty
from sc2.main import run_game
from sc2.player import Bot, Computer, AIBuild
from ladder import run_ladder_game

from bot.main import Kitten

bot1 = Bot(Race.Terran, Kitten(), "kitten")


def main():
    # Ladder game started by LadderManager
    print("Starting ladder game...")
    result, opponentid = run_ladder_game(bot1)
    print(result, " against opponent ", opponentid)


# Start game
if __name__ == "__main__":
    if "--LadderServer" in sys.argv:
        # Ladder game started by LadderManager
        print("Starting ladder game...")
        result, opponentid = run_ladder_game(bot1)
        print(result, " against opponent ", opponentid)
    else:
        # Local game
        random_map = random.choice(
            [
                # "BerlingradAIE",
                # "InsideAndOutAIE",
                # "MoondanceAIE",
                # "StargazersAIE",
                "WaterfallAIE",
                # "HardwireAIE",
            ]
        )
        random_race = random.choice([Race.Zerg, Race.Terran, Race.Protoss])
        print("Starting local game...")
        run_game(
            maps.get(random_map),
            [
                # bot2,
                bot1,
                Computer(random_race, Difficulty.CheatVision, ai_build=AIBuild.Macro),
                # bot2,
            ],
            realtime=False,
            # 2 lower spawn / 2564 upper spawn
            # random_seed=2,
        )
