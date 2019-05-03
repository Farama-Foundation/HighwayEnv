import sys
import baselines.run as run

import highway_env

DEFAULT_ARGUMENTS = [
    "--env=parking-v0",
    "--alg=her",
    "--num_timesteps=1e4",
    "--network=default",
    "--num_env=0",
    "--save_path=~/models/latest",
    "--load_path=~/models/latest",
    "--save_video_interval=0",
    "--play"
]

if __name__ == "__main__":
    args = sys.argv
    if len(args) <= 1:
        args = DEFAULT_ARGUMENTS
    run.main(args)
