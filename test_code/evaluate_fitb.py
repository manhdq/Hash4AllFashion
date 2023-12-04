# %%
import os.path as osp
import yaml
import logging
import importlib
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import torch

import utils
from utils import to_device
from utils.param import FashionParam
from utils.logger import Logger, config_log
from dataset import fashionset
from dataset.transforms import get_img_trans
from model import fashionnet

importlib.reload(fashionset)
importlib.reload(fashionnet)
importlib.reload(utils.param)

import matplotlib.pyplot as plt
from reproducible_code.tools import io, image_io, plot

importlib.reload(io)

from icecream import ic

# %%
LOGGER = logging.getLogger("fashion32")

# %%
cfg_file = "../configs/eval/FHN_VSE_OSE_T3.yaml"
with open(cfg_file, "r") as f:
    kwargs = yaml.load(f, Loader=yaml.FullLoader)

# %%
config = FashionParam(**kwargs)
config.add_timestamp()
parallel, device = utils.get_device(config.gpus)

# %%
logger = logging.getLogger("fashion32")
logger.info(f"Fashion param : {config}")
param = config.fitb_data_param

# %% [markdown]
# ### Load dataset
print(config.net_param.load_trained)
cate_selection = param.cate_selection
transforms = get_img_trans(param.phase, param.image_size)
dataset = fashionset.FITBDataset(
    param, transforms, logger
)

# # %%
# df = pd.read_csv("/home/dungmaster/Datasets/Fashion-Outfits-Theme-Aware/train/test.csv")

# # %%
# df.loc[0, "footwear"] = "123"
# df.loc[0, "footwear"]

# %% [markdown]
# ### Load dataloader
param.shuffle = False
loader = fashionset.get_dataloader(param, logger)

# %%
inputs = next(iter(loader))
inputs = to_device(inputs, "cuda")
inputs

# %%
print(len(inputs["outf_s"]))
inputs["imgs"][0].shape

# %% [markdown]
# ### Load model
net = fashionnet.get_net(config, logger)
net.eval()

# %% [markdown]
# ### try forward through the model
scores, _, _ = net.visual_output(**inputs)
len(scores)

# %%
torch.argmax(scores[0][0])

# %%
scores[0][1].tolist()

# %% [markdown]
# ### Calculate FITB score
pbar = tqdm(loader, desc="Computing scores")
correct = 0
cnt = 0
bscores = []

for inputs in pbar:
    inputs = utils.to_device(inputs, device)
    with torch.no_grad():
        scores, _, _ = net.visual_output(**inputs)
    scores = scores[0][1]
    bscores.append(scores.tolist())
    # the first item is the groud-truth item
    if torch.argmax(scores).item() == 0:
        correct += 1
    cnt += 1
    pbar.set_description("Accuracy: {:.3f}".format(correct / cnt))
fitb_acc = correct / cnt
LOGGER.info("FITB Accuracy %.4f", fitb_acc)

# %%
bscores

# %% [markdown]
# ### Load outfits' metadata
data_dir = "/home/dungmaster/Datasets/Fashion-Outfits-Theme-Aware"
# outfits_dir = osp.join(data_dir, "outfits")
image_dir = osp.join(data_dir, "images")

df_outfit_meta = io.load_csv(
    osp.join(data_dir, "important", "outfit_meta_v2.csv")
)

# %%
num_sample = 8
new_sizes = (224, 224)

idxs = random.sample(range(len(loader)), num_sample)
idxs

# %%
outf_descs = []
all_outfs = []
border_size = 5

for idx in idxs:
    scores = bscores[idx]
    idx = np.argmax(scores)

    # Get posi outfit and nega outfit from loader
    outf_id, outfs = loader.get_outfits(idx)
    outf_items, outf_cates = outfs
    num_items_each = int(len(outf_items) // 4)

    # Get posi outfit description
    oid, mode = outf_id.split("_")
    outf_meta = df_outfit_meta[df_outfit_meta["id"] == int(oid)]
    outf_desc = outf_meta.en_Outfit_Description
    if mode == "2":
        outf_desc = outf_meta.additional_info
    outf_desc = outf_desc.iloc[0]

    # Stack images of posi outfit and nega outfit vertically
    outfs = []

    items_1 = outf_items[num_items_each*oi: num_items_each*(oi+1)]
    items_2 = outf_items[num_items_each: num_items_each*2]    
    modified_cate_idx = [
        idx for idx in range(num_items_each)
        if items_1[idx] != items_2[idx]
    ][0]

    # through outfits
    for oi in range(4):
        outf_imgs = []
        items = outf_items[num_items_each*oi: num_items_each*(oi+1)]
        cates = outf_cates[num_items_each*oi: num_items_each*(oi+1)]

        # through items of outfit
        for i, item_id in enumerate(items):
            img_path = osp.join(image_dir, item_id)
            try:
                img = image_io.load_image(img_path, toRGB=False)
            except Exception as e:
                print(e)
                continue

            cate = cates[i].split('.')[0]
            img = cv2.resize(img, new_sizes)
            # img = cv2.putText(
            #     img,
            #     cate,
            #     (50, 100),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1,
            #     (0, 255, 255),
            #     2,
            # )
            if i == modified_cate_idx:
                img = cv2.copyMakeBorder(
                    img[
                        border_size:-border_size,
                        border_size:-border_size,
                        :
                    ],
                    top=border_size,
                    bottom=border_size,
                    left=border_size,
                    right=border_size,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[255, 255, 0]
                )
            outf_imgs.append(img)

        # stack items horizontally
        outf_imgs = np.hstack(outf_imgs)

        color = (255, 0, 0)  # default red
        if oi == idx:
            color = (0, 255, 0)  # green

        outf_imgs = cv2.putText(
            outf_imgs,
            f"{scores[oi][0]:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
        outfs.append(outf_imgs)

    # stack outfits vertically
    outfs = np.vstack(outfs)
    all_outfs.append(outfs)

    # add outfits with description
    outf_descs.append(outf_desc)

# %%
plot.display_multiple_images(
    all_outfs, grid_nrows=2, titles=outf_descs, axes_pad=1.3, line_length=8
)

# %%
