# %%
import os
import os.path as osp
from glob import glob
import sys

import random
import importlib

from tqdm import tqdm
import numpy as np
import seaborn as sns
import cv2

sys.path += ["../"]
from fashion_clip.fashion_clip import FashionCLIP
from reproducible_code.tools import io, plot

importlib.reload(plot)
importlib.reload(io)

sns.set_theme()
sns.set_style("whitegrid", {"axes.grid": False})
tqdm.pandas()

# %% [markdown]
# ### Load data
error_names = []
# data_dir = "/home/dungmaster/Datasets/Fashion-Outfits-Theme-Aware"
data_dir = "/home/dungmaster/Datasets/polyvore_outfits/"
# img_dir = osp.join(data_dir, "images")
img_dir = osp.join(data_dir, "sample_images")

# %%
for name in tqdm(os.listdir(img_dir)):
    path = osp.join(img_dir, name)
    try:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        error_names.append(name)

error_names

# %% [markdown]
# ### Load model and embedding
model = FashionCLIP("fashion-clip")

# %%
image_embeddings = {}
batch_size = 64
img_names = os.listdir(img_dir)
num_imgs = 0

for idx in tqdm(range(0, len(img_names), batch_size)):
    imgs = img_names[idx : idx + batch_size]
    num_imgs += len(imgs)
    paths = [osp.join(img_dir, img) for img in imgs]
    embeddings = model.encode_images(paths, batch_size)
    image_embeddings.update(
        {n: e[np.newaxis, ...] for n, e in zip(imgs, embeddings)}
    )
    if idx % 4000 == 0:
        print("Row ", idx)

# %%
len(img_names), len(image_embeddings)

# %%
visual_encoding_pkl = osp.join(data_dir, "visual_encoding.pkl")

# %%
io.save_pickle(image_embeddings, visual_encoding_pkl)

# %%
image_embeddings_load = io.load_pickle(visual_encoding_pkl)

# %%
vid = random.sample(list(image_embeddings.keys()), 1)[0]
emb = image_embeddings[vid]

# %%
emb_ = image_embeddings_load[vid]
(emb == emb_).all()

# %%
emb.shape

# %%
