# %%
import os
import os.path as osp
from glob import glob
import random
from shutil import copy
import importlib

from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
sns.set_style("whitegrid", {'axes.grid' : False})

from reproducible_code.tools import image_io, io, plot
importlib.reload(image_io)

# %% 
data_dir = "/home/dungmaster/Datasets/Fashion-Outfits-Theme-Aware"
outfits_dir = osp.join(data_dir, "outfits")
img_paths = glob(osp.join(outfits_dir, "*/*.jpg"))
len(img_paths)

# %%
item_paths = [p for p in img_paths if "outfit_" not in p]
len(item_paths)

# %%
new_img_dir = osp.join(data_dir, "images")

# %%
io.create_dir(new_img_dir)

# %% [markdown]
# ### Save some images using opencv and load again to watch their colours
sample_paths = random.sample(item_paths, 16)

# %% [markdown]
# #### reload using opencv then convert to rgb
sample_imgs = []

for img_path in sample_paths:
    # load image with opencv and convert to rgb
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # save image in new folder
    img_fname = osp.basename(img_path)
    new_img_path = osp.join(new_img_dir, img_fname)
    cv2.imwrite(new_img_path, img)

    # load image again
    img = cv2.imread(new_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sample_imgs.append(img)

# %% [markdown]
# display the reloaded sample images
plot.display_multiple_images(sample_imgs)

# %% [markdown]
# #### Reload using PIL.Image
sample_imgs = []

for img_path in sample_paths:
    # load image with opencv and convert to rgb
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # save image in new folder
    img_fname = osp.basename(img_path)
    new_img_path = osp.join(new_img_dir, img_fname)
    cv2.imwrite(new_img_path, img)

    # load image again
    img = Image.open(new_img_path)  # PIL follows RGB
    sample_imgs.append(img)

# %% [markdown]
# display the reloaded sample images
plot.display_multiple_images(sample_imgs)

# %% [markdown]
# So if we first load image using opencv and convert to rgb
# and save it then load again using PIL.Image the colours are
# flipped
# 
# So first load we should not convert to rgb

# %% [markdown]
# #### Reload using opencv then convert to rgb
sample_imgs = []

for img_path in sample_paths:
    # load image with opencv
    img = cv2.imread(img_path)

    # save image in new folder
    img_fname = osp.basename(img_path)
    new_img_path = osp.join(new_img_dir, img_fname)
    cv2.imwrite(new_img_path, img)

    # load image again
    img = cv2.imread(new_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = Image.open(new_img_path)  # PIL follows RGB
    sample_imgs.append(img)

# %% [markdown]
# display the reloaded sample images
plot.display_multiple_images(sample_imgs)

# %% [markdown]
# #### Reload using PIL.Image
sample_imgs = []

for img_path in sample_paths:
    # load image with opencv
    img = cv2.imread(img_path)
    
    # save image in new folder
    img_fname = osp.basename(img_path)
    new_img_path = osp.join(new_img_dir, img_fname)
    cv2.imwrite(new_img_path, img)

    # load image again
    img = Image.open(new_img_path)  # PIL follows RGB
    sample_imgs.append(img)

# %% [markdown]
# display the reloaded sample images
plot.display_multiple_images(sample_imgs)

# %% [markdown]
# ### Save some images using PIL.Image and load again to watch their colours

# %% [markdown]
# #### Reload using opencv then convert to rgb
sample_imgs = []

for img_path in sample_paths:
    # load image with opencv
    img = np.array(Image.open(img_path))

    # save image in new folder
    img_fname = osp.basename(img_path)
    new_img_path = osp.join(new_img_dir, img_fname)
    cv2.imwrite(new_img_path, img)

    # load image again
    img = cv2.imread(new_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = Image.open(new_img_path)  # PIL follows RGB
    sample_imgs.append(img)

# %% [markdown]
# display the reloaded sample images
plot.display_multiple_images(sample_imgs)

# %% [markdown]
# #### Reload using PIL.Image
sample_imgs = []

for img_path in sample_paths:
    # load image with opencv
    img = np.array(Image.open(img_path))
    
    # save image in new folder
    img_fname = osp.basename(img_path)
    new_img_path = osp.join(new_img_dir, img_fname)
    cv2.imwrite(new_img_path, img)

    # load image again
    img = Image.open(new_img_path)  # PIL follows RGB
    sample_imgs.append(img)

# %% [markdown]
# display the reloaded sample images
plot.display_multiple_images(sample_imgs)

# %% [markdown]
# So we should first load image using opencv
#
# The final process is as follow:
#
# - Fist load image using opencv (not converting to RGB) and save image in new folder using opencv
# - Then reload again using opencv (converting to RGB) or PIL.Image

# %%
error_imgs_load = {}
error_imgs_save = {}
io.create_dir(new_img_dir)

for img_path in tqdm(item_paths):
    # load image with opencv
    try:
        img = cv2.imread(img_path)
    except Exception as e:
        error_imgs_load[img_path] = e
        continue

    # save image in new folder
    img_fname = osp.basename(img_path)
    new_img_path = osp.join(new_img_dir, img_fname)

    try:
        cv2.imwrite(new_img_path, img)
    except Exception as e:
        error_imgs_save[img_path] = e

# %%
num_images_new_dir = len(os.listdir(new_img_dir))
num_images_old_dir = len(item_paths)
print("Number of images in new directory:", num_images_new_dir)
print("Number of missing images:", num_images_old_dir - num_images_new_dir)

# %%
len(error_imgs_load), len(error_imgs_save)

# %%
error_saves = list(error_imgs_save.values())
error_saves = [' '.join(str(er).split(':')[2:]) for er in error_saves]
error_saves = list(set(error_saves))
error_saves

# %% [markdown]
# So the missing images are error when loading
#
# Let's save these image file names into a list
error_img_paths = list(error_imgs_save.keys())
len(error_img_paths)

# %%
error_img_fnames = [osp.basename(path) for path in error_img_paths]
error_img_fnames

# %%
io.save_txt(
    error_img_fnames,
    osp.join(data_dir, "others", "final_error_image_fnames.txt")
)

# %% [markdown]
# ### Display sample images in new folder
new_img_paths = glob(osp.join(new_img_dir, "*.jpg"))

# %%
sample_new_paths = random.sample(new_img_paths, 16)
sample_new_imgs = []

for img_path in sample_new_paths:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sample_new_imgs.append(img)

plot.display_multiple_images(sample_new_imgs)

# %%
error_load_news = {}

for img_path in tqdm(new_img_paths):
    try:
        img = image_io.load_image(img_path)
    except Exception as e:
        error_load_news[img_path] = str(e)

# %%
len(error_load_news)

# %%
