# from .fashionnet import FashionNet
import utils
from .fashionnet import FashionNet
from .basemodel import *
from .backbones import *


def load_pretrained(state_dict, pretrained_state_dict):
    for name, param in pretrained_state_dict.items():
        if name in state_dict.keys() and "classifier" not in name:
            # print(name)
            param = param.data
            state_dict[name].copy_(param)
            

def get_net(config, logger, debug=False):
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

        if debug:
            print("Before load pretrained...")
            for name, param in pretrained_state_dict.items():
                if name in state_dict.keys():
                    param = param.data
                    print((state_dict[name] == param).all())
                
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

            if debug:
                print("After load pretrained...")
                state_dict = net.state_dict()
                for name, param in pretrained_state_dict.items():
                    if name in state_dict.keys():
                        param = param.data
                        print((state_dict[name] == param).all())

    elif config.resume:  # resume training
        logger.info(f"Training resume from {config.resume}")
    else:
        logger.info(f"Loading weights from backbone {net_param.backbone}.")
        net.init_weights()
    logger.info(f"Copying net to GPU-{config.gpus[0]}")
    net.cuda(device=config.gpus[0])
    return net
