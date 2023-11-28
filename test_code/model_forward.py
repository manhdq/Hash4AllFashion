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
import train

importlib.reload(fashionset)

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

# %%
transforms = get_img_trans(param.phase, param.image_size)
dataset = fashionset.FashionDataset(
    param, transforms, cate_selection, logger
)

# %%
dataset.posi_df.head()

# # %%
# image = dataset.datum.load_image("10269_9708_31264127289.jpg")
# image = dataset.datum.load_image("11436_9728_30492806793.jpg")
# plt.imshow(image)

# %%
dataloader = fashionset.get_dataloader(param, logger)

# %%
inputs = next(iter(dataloader))
len(inputs)

# %%
inputs[0].shape

# %% [markdown]
# ### Load model
load_trained = "/home/dungmaster/Projects/Machine Learning/HangerAI_outfits_recommendation_system/Hash4AllFashion/checkpoints/H4A_t3_allFashion_noSemantic_classify_best.net"
config.load_trained = None
net = train.get_net(config, logger)

# %% [markdown]
# ### try copying param from pretrained
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

# %% [markdown]
# ### try forward through the model
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
tensorl = [torch.randn(1, 512) for _ in range(4)]
torch.stack(tensorl, dim=0).shape

# %%
