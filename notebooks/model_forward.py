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
import torchshow as ts

importlib.reload(fashionset)
importlib.reload(fashionnet)

import matplotlib.pyplot as plt

# %%
cfg_file = "../configs/train/FHN_VOE_T3_fashion32.yaml"
with open(cfg_file, "r") as f:
    kwargs = yaml.load(f, Loader=yaml.FullLoader)

# %%
kwargs

# %%
config = FashionTrainParam(**kwargs)
config.add_timestamp()
param = config.train_data_param

logger = logging.getLogger("fashion32")
logger.info(f"Fashion param : {config}")

# %% [markdown]
# Test 2 dataparams
data_param = config.data_param
train_data_param = config.train_data_param
val_data_param = config.test_data_param

# %%
train_param = train_data_param or data_param
print(train_param.phase)
print(train_param)
print(train_param.default)

# %%
val_param = val_data_param or data_param
print(val_param.phase)
print(val_param)
print(val_param.default)

# # %%
# a = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
# b = {'a': 3, 'b': 4, 'c': 5}
# a or b
# b or a

# # %%
# # a.update(b)
# # a
# b.update(a)
# b

# %% [markdown]
# ### Load dataset

# %%
transforms = get_img_trans(param.phase, param.image_size)
print(transforms)
dataset = fashionset.FashionDataset(param, transforms, logger)

# # %%
# dataset.posi_df.head()
# dataset.nega_df.head()

# # %%
# image = dataset.datum.load_image("10269_9708_31264127289.jpg")
# image = dataset.datum.load_image("11436_9728_30492806793.jpg")
# plt.imshow(image)

# %%
param.transforms

# %% [markdown]
# ### Load dataloader
loader = fashionset.get_dataloader(train_param, logger)
# loader.set_data_mode("PosiOnly")
# loader.set_data_mode("NegaOnly")

# %%
inputs = next(iter(loader))
inputs = to_device(inputs, "cuda")
inputs

# %% [markdown]
# display some input images
imgs = inputs["imgs"][0]
print(imgs.shape)
ts.show(imgs)

# %% [markdown]
# ### Load model
print(config.net_param.load_trained)
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
