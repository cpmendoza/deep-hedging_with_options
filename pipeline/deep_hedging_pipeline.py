"""
Usage:
    1. cd pipeline
    2. python3 deep_hedging_pipeline.py
"""

import os, sys
from pathlib import Path
import warnings

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)
warnings.filterwarnings("ignore")

from src.utils import *
from src.features.nig_simulation import *
from src.features.jivr_simulation import *
from src.models.deep_rl_training import rl_agent

def deel_hedging_pipeline(config_file,config_file_agent):

    # 1) NIG variables simulation
    nig_simulation(config_file)

    # 2) JIVR model simulation
    jivr_model = implied_volatily_surface_vec(config_file)
    jivr_model.jivr_simulation()

    # 3) Deep hedging approach
    config_file_simulation = config_file["simulation"]
    _ = rl_agent(config_file_simulation,config_file_agent)

    return

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    main_folder = str(Path.cwd().parent)
    sys.path.append(main_folder)
    config_file = load_config(os.path.join(main_folder,'cfgs','config_agent.yml'))
    config_file_agent = config_file["agent"]
    config_file = load_config(os.path.join(main_folder,'cfgs','config_simulation.yml'))
    _ = deel_hedging_pipeline(config_file,config_file_agent)