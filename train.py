"""Script for training."""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import torch
import yaml

import utils
from utils.param import FashionTrainParam
from utils.logger import get_logger
import utils.config as cfg
from dataset.fashionset import FashionLoader
from model.fashionnet import FashionNet, get_net
from solver import FashionNetSolver

from icecream import ic


def main(config, logger, debug=False):
    """
    Training task
    """

    # Get data for training
    train_param = config.train_data_param or config.data_param
    logger.info(f"Data set for training: \n{train_param}")
    train_loader = FashionLoader(train_param, logger)

    # Get data for validation
    ##TODO: Modify val_data_param in param config
    val_param = config.test_data_param or config.data_param
    logger.info(f"Data set for training: \n{val_param}")
    val_loader = FashionLoader(val_param, logger)

    # Get net
    net = get_net(config, logger, debug)

    # Get solver
    solver_param = config.solver_param
    logger.info("Initialize a solver for training.")
    logger.info(f"Solver configuration: \n{solver_param}")
    solver = FashionNetSolver(
        solver_param, net, train_loader, val_loader, logger
    )

    # Load solver state
    if config.resume:
        solver.resume(config.resume)
    # run
    solver.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Hash for All Fashion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Hashing for All Fashion scripts",
    )
    parser.add_argument(
        "--cfg",
        default="configs/train/FHN_VSE_T3_visual_both.yaml",
        help="configuration file.",
    )
    parser.add_argument(
        "--env",
        default="local",
        choices=["local", "colab"],
        help="Using for logging option. Using logger if local, using normal print otherwise.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug the network",
    )
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        kwargs = yaml.load(f, Loader=yaml.FullLoader)
    config = FashionTrainParam(**kwargs)
    config.add_timestamp()

    logger = get_logger(args.env, config)
    logger.info(f"Fashion param : {config}")

    main(config, logger, args.debug)
