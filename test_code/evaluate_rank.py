# %%
import os.path as osp
import yaml
import logging
import importlib
import random

from tqdm import tqdm
import numpy as np
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

logger = logging.getLogger("fashion32")
logger.info(f"Fashion param : {config}")
param = config.data_param

# %% [markdown]
# ### Load dataset
print(config.net_param.load_trained)
cate_selection = param.cate_selection

# %%
transforms = get_img_trans(param.phase, param.image_size)
dataset = fashionset.FashionDataset(
    param, transforms, cate_selection, logger
)

# %%
dataset.nega_df.head()

# %% [markdown]
# ### Load dataloader
param.shuffle = False
loader = fashionset.get_dataloader(param, logger)

# %%
inputs = next(iter(loader))
inputs = to_device(inputs, "cuda")
inputs

# %% [markdown]
# ### Load model
net = fashionnet.get_net(config, logger)
net.eval()

# %% [markdown]
# ### try forward through the model
scores = net.visual_output(**inputs)
scores

# %%
parallel, device = utils.get_device(config.gpus)

# %%
def outfit_scores():
    """Compute rank scores for data set."""
    num_users = net.param.num_users
    scores = [
        [[] for _ in range(num_users)]
        for _ in range(4)
    ]
    u = 0  # 1 user
    for inputs in tqdm(loader, desc="Computing scores"):
        inputs = utils.to_device(inputs, device)
        with torch.no_grad():
            outputs, _, _ = net.visual_output(**inputs)
            outputs = [s.tolist() for s in outputs]

        for u in range(num_users):
            for n, score in enumerate(outputs):
                for s in score:
                    scores[n][u].append(s)  # [N, U, S, 1]
    return scores

# %%
scores = outfit_scores()

# %%
# compute ndcg
mean_ndcg, avg_ndcg = utils.metrics.NDCG(scores[0], scores[2])
mean_ndcg_binary, avg_ndcg_binary = utils.metrics.NDCG(scores[1], scores[3])
aucs, mean_auc = utils.metrics.ROC(scores[0], scores[2])
aucs_binary, mean_auc_binary = utils.metrics.ROC(scores[1], scores[3])
LOGGER.info(
    "Metric:\n"
    "- average ndcg:%.4f\n"
    "- average ndcg(binary):%.4f\n"
    "- mean auc:%.4f\n"
    "- mean auc(binary):%.4f",
    mean_ndcg.mean(),
    mean_ndcg_binary.mean(),
    mean_auc,
    mean_auc_binary,
)

# %%
scores[1][0]

# %% [markdown]
# ### Load outfits' metadata
data_dir = "/home/dungmaster/Datasets/Fashion-Outfits-Theme-Aware"
# outfits_dir = osp.join(data_dir, "outfits")
image_dir = osp.join(data_dir, "images")

df_outfit_meta = io.load_csv(
    osp.join(data_dir, "important", "outfit_meta_v2.csv")
)
num_sample = 9
new_sizes = (240, 240)

# %%
idxs = random.sample(range(len(loader)-1), num_sample)
idxs

# %%
outf_descs = []
outf_pairs = []

for idx in idxs:
    bpscore, bnscore = scores[1][0][idx], scores[3][0][idx]
    h, w = new_sizes
    p_sizes = n_sizes = (h, w)

    # Get posi outfit and nega outfit from loader
    outf_id, outfs = loader.get_outfits(idx)
    len_p, len_n = len(outfs[0][0]), len(outfs[1][0])

    # Upscale images in outfit with less items so that
    # they can be stacked vertically with the other outfit
    ratio = len_p / len_n
    
    if ratio > 1:
        n_sizes = (int(w * ratio), int(h * ratio))
    else:
        p_sizes = (int(w * 1/ratio), int(h * 1/ratio))        

    sizes = [p_sizes, n_sizes]

    # Get posi outfit description
    oid, mode = outf_id.split('_')
    outf_meta = df_outfit_meta[df_outfit_meta["id"] == int(oid)]
    outf_desc = outf_meta.en_Outfit_Description
    if mode == "2":
        outf_desc = outf_meta.additional_info
    outf_desc = outf_desc.iloc[0]

    # Stack images of posi outfit and nega outfit vertically
    outf_imgs = [[], []]

    for idx, outf in enumerate(outfs):
        items, cates = outf

        for i, item_id in enumerate(items):
            img_path = osp.join(image_dir, item_id)
            try:
                img = image_io.load_image(
                    img_path,
                    toRGB=False
                )
            except Exception as e:
                print(e)
                continue

            cate = cates[i]
            img = cv2.resize(img, sizes[idx])
            img = cv2.putText(img, cate, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)            
            outf_imgs[idx].append(img)

    # Stack outfit images horizontally
    posi_outf = np.hstack(outf_imgs[0])
    # pih, piw, _ = posi_outf.shape

    nega_outf = np.hstack(outf_imgs[1])
    # nih, niw, _ = nega_outf.shape

    # Write score to outfit images
    posi_outf = cv2.putText(posi_outf, f"{bpscore[0]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    nega_outf = cv2.putText(nega_outf, f"{bnscore[0]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Stack images of 2 outfits vertically
    outfs = np.vstack([posi_outf, nega_outf])

    # Add outfits with description
    outf_pairs.append(outfs)
    outf_descs.append(outf_desc)

# %%
plot.display_multiple_images(
    outf_pairs,
    grid_nrows=3,
    titles=outf_descs,
    axes_pad=1.,
    line_length=10
)

# %%
