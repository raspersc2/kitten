"""
This script will play kitten vs other opponents to collect actions, rewards and states
When games have completed, the ppo_training script is invoked
This process is repeated

Steps:
1. Set up AI-Arena docker image https://github.com/aiarena/local-play-bootstrap
2. Move this python script to the root of `local-play-bootstrap` from the above repo
3. Copy the entire `kitten` folder into the `bots` folder in `local-play-bootstrap`
4. Find some other bots from AI-Arena to train against: https://aiarena.net/bots/downloadable/
5. Configure this script to set up the matches
6. ???
7. Profit

Optional: Run Tensorboard pointing to `kitten/data/runs`
"""

from os import system
from pathlib import Path
from random import choice

MAP_FILE_EXT: str = "SC2Map"
MAPS_PATH: str = "C:\\Program Files (x86)\\StarCraft II\\Maps"

BOTS_PLAYER_ONE = [
    "kitten,T,python,",
]
BOTS_PLAYER_TWO = [
    "VeTerran,T,cpplinux,",
]

NUM_GAMES_TO_PLAY: int = 1000

if __name__ == "__main__":
    maps: list[str] = [
        p.name.replace(f".{MAP_FILE_EXT}", "")
        for p in Path(MAPS_PATH).glob(f"*.{MAP_FILE_EXT}")
        if p.is_file()
    ]
    print("matches started")
    for x in range(NUM_GAMES_TO_PLAY):
        matchString: str = (
            f"{choice(BOTS_PLAYER_ONE)}{choice(BOTS_PLAYER_TWO)}{choice(maps)}"
        )
        with open("matches", "w") as f:
            f.write(f"{matchString}")
        print(f"match {x}: {matchString}")
        system('cmd /c "docker-compose up"')
        system('cmd /c "conda activate Eris"')
        system(
            'cmd /c "python D:\\kitten\\bot\\squad_agent\\training_scripts\\ppo_trainer.py"'
        )
    print("matches ended")
