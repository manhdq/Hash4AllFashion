# %%
import yaml
import logging
import importlib

import torch

from utils import to_device
from utils.param import FashionTrainParam
from utils.logger import Logger, config_log
from dataset import fashionset
from dataset.transforms import get_img_trans
from model import fashionnet

importlib.reload(fashionset)
importlib.reload(fashionnet)

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
# ### Load dataset
param = config.train_data_param
cate_selection = param.cate_selection

# %%
print(config.net_param.load_trained)

# %%
transforms = get_img_trans(param.phase, param.image_size)
dataset = fashionset.FashionDataset(
    param, transforms, cate_selection, logger
)

# %%
dataset.posi_df.head()
dataset.nega_df.head()

# # %%
# image = dataset.datum.load_image("10269_9708_31264127289.jpg")
# image = dataset.datum.load_image("11436_9728_30492806793.jpg")
# plt.imshow(image)

# %% [markdown]
# ### Load dataloader
loader = fashionset.get_dataloader(param, logger)
loader.set_data_mode("PosiOnly")
loader.set_data_mode("NegaOnly")

# %%
inputs = next(iter(loader))
inputs = to_device(inputs, "cuda")
inputs

# %% [markdown]
# ### Load model
net = fashionnet.get_net(config, logger)

# %% [markdown]
# ### try forward through the model
scores = net.visual_output(**inputs)
scores

# %%
out = net(**inputs)

# %% [markdown]
# ### Try copying param from pretrained
state_dict = net.state_dict()
for name, param in state_dict.items():
    print(name, state_dict[name].shape)

# %%
pretrained_state_dict = torch.load(load_trained)

# %%
for name, param in pretrained_state_dict.items():
    # print(name, pretrained_state_dict[name].shape)
    if name in state_dict.keys():
        print(name)
        param = param.data
        print((state_dict[name] == param).all())

# %%
for name, param in pretrained_state_dict.items():
    # print(name, pretrained_state_dict[name].shape)
    if name in state_dict.keys():
        print(name)
        param = param.data
        state_dict[name].copy_(param)

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
tensorl = [torch.randn(1, 512) for _ in range(4)]
torch.stack(tensorl, dim=0).shape

# %%
