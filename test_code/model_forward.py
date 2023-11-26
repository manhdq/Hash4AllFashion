# %%
import yaml
import logging
import importlib

import torch

from utils import to_device
from utils.param import FashionTrainParam
from utils.logger import Logger, config_log
from dataset.fashionset_v2 import FashionDataset, get_dataloader
from dataset.transforms import get_img_trans
import train

importlib.reload(train)

import matplotlib.pyplot as plt

# %%
cfg_file = "../configs/train/FHN_VSE_T3_visual_new.yaml"
with open(cfg_file, "r") as f:
    kwargs = yaml.load(f, Loader=yaml.FullLoader)

# %%
config = FashionTrainParam(**kwargs)
config.add_timestamp()

logger = logging.getLogger("fashion32")
logger.info(f"Fashion param : {config}")

# %% [markdown]
# ### Load dataloader
param = config.train_data_param
cate_selection = param.cate_selection

# # %%
# transforms = get_img_trans(param.phase, param.image_size)
# dataset = FashionDataset(
#     param, transforms, cate_selection, logger
# )

# # %%
# image = dataset.datum.load_image("10269_9708_31264127289.jpg")
# image = dataset.datum.load_image("11436_9728_30492806793.jpg")
# plt.imshow(image)

# %%
dataloader = get_dataloader(param, logger)

# %%
inputs = next(iter(dataloader))
len(inputs)

# %%
inputs[0].shape

# %% [markdown]
# ### Load model
config.load_trained = None
net = train.get_net(config, logger)

# %%
inputs = to_device(inputs, "cuda")
out = net(*inputs)

# # %%
# path = "encoder_o.pth"
# torch.save(net.encoder_o.state_dict(), path)
# encoder_o = net.encoder_o.copy()
# encoder_o.load_state_dict(torch.load(path))

# # %%
# flag = True
# for p1, p2 in zip(net.encoder_o.parameters(), encoder_o.parameters()):
#     if p1.data.ne(p2.data).sum() > 0:
#         flag = False
#         break
# flag

# %%
net.num_groups

# %%
net.classifier_v

# %%
# net.load_state_dict(
