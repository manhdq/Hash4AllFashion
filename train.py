"""Script for training."""
import argparse
import os
import logging
import warnings

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

import torch
import yaml

import utils
from utils.param import FashionTrainParam
from utils.logger import Logger, config_log
import utils.config as cfg
from dataset.fashionset import FashionLoader
from model import FashionNet
from solver import FashionNetSolver

from icecream import ic


def get_logger(env, config):
    if env == "local":
        ##TODO: Modify this logger name
        logfile = config_log(
            stream_level=config.log_level, log_file=config.log_file
        )
        logger = logging.getLogger("fashion32")
        logger.info("Logging to file %s", logfile)
    elif env == "colab":
        logger = Logger(config)  # Normal logger
        logger.info(f"Logging to file {logger.logfile}")
    return logger


def load_pretrained(state_dict, pretrained_state_dict):
    for name, param in pretrained_state_dict.items():
        if name in state_dict.keys() and "classifier" not in name:
            # print(name)
            param = param.data
            state_dict[name].copy_(param)


def get_net(config, logger):
    """
    Get network.
    """
    # Get net param
    net_param = config.net_param
    logger.info(f"Initializing {utils.colour(config.net_param.name)}")
    logger.info(net_param)
    # Dimension of latent codes
    net = FashionNet(net_param, logger, cfg.SelectCate)
    state_dict = net.state_dict()
    load_trained = net_param.load_trained

    # Load model from pre-trained file
    if load_trained:
        # Load weights from pre-trained model
        num_devices = torch.cuda.device_count()
        map_location = {"cuda:{}".format(i): "cpu" for i in range(num_devices)}
        logger.info(f"Loading pre-trained model from {load_trained}")
        pretrained_state_dict = torch.load(load_trained, map_location=map_location)
        
        # print("Before load pretrained...")
        # for name, param in pretrained_state_dict.items():
        #     if name in state_dict.keys():
        #         param = param.data
        #         print((state_dict[name] == param).all())
                
        # when new user problem from pre-trained model
        if config.cold_start:
            # TODO: fit with new arch
            # reset the user's embedding
            logger.info("Reset the user embedding")
            # TODO: use more decent way to load pre-trained model for new user
            weight = "user_embedding.encoder.weight"
            pretrained_state_dict[weight] = torch.zeros(
                net_param.dim, net_param.num_users
            )
            net.load_state_dict(pretrained_state_dict)
            ##TODO:
            net.user_embedding.init_weights()
        else:
            # load pre-trained model
            # net.load_state_dict(pretrained_state_dict)
            load_pretrained(state_dict, pretrained_state_dict)

            # print("After load pretrained...")
            # state_dict = net.state_dict()
            # for name, param in pretrained_state_dict.items():
            #     if name in state_dict.keys():
            #         param = param.data
            #         print((state_dict[name] == param).all())

    elif config.resume:  # resume training
        logger.info(f"Training resume from {config.resume}")
    else:
        logger.info(f"Loading weights from backbone {net_param.backbone}.")
        net.init_weights()
    logger.info(f"Copying net to GPU-{config.gpus[0]}")
    net.cuda(device=config.gpus[0])
    return net


def main(config, logger):
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
    net = get_net(config, logger)

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
    args = parser.parse_args()
    with open(args.cfg, "r") as f:
        kwargs = yaml.load(f, Loader=yaml.FullLoader)
    config = FashionTrainParam(**kwargs)
    config.add_timestamp()

    ##TODO: Make this dynamic
    os.makedirs("checkpoints", exist_ok=True)

    logger = get_logger(args.env, config)
    logger.info(f"Fashion param : {config}")

    main(config, logger)
