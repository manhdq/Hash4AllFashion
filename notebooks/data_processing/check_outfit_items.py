# %%
import os
import os.path as osp
from glob import glob
import sys

import random
import importlib

from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import cv2

sys.path += ["../"]
from reproducible_code.tools import io, plot, image_io
importlib.reload(plot)

sns.set_theme()
sns.set_style("whitegrid", {'axes.grid' : False})
tqdm.pandas()

from icecream import ic

# %% [markdown]
# ### Global params
data_dir = "/home/dungmaster/Datasets/Fashion-Outfits-Theme-Aware"
outfits_dir = osp.join(data_dir, "outfits")
image_dir = osp.join(data_dir, "images")

# %% [markdown]
# Load latest outfits' metadata
df_outfit_descriptions = io.load_csv(
    osp.join(data_dir, "important", "outfit_meta_v2.csv")
)
df_outfit_descriptions.head(5)

# %% [markdown]
# Load latest outfits' items
df_outfit_items = io.load_csv(
    osp.join(data_dir, "important", "clean_theme_outfit_items_v2.csv")
)
df_outfit_items.head()

# %% [markdown]
# Load latest outfits' categories
df_item_categories = io.load_csv(
    osp.join(data_dir, "important", "processed_theme_aware_item_categories.csv")
)
df_item_categories.head(5)

# # %% [markdown]
# # Display sample images of each cate
# for cate in df_item_categories["category"].unique():
#     print("Category:", cate)
#     sample_df = df_item_categories[df_item_categories["category"] == cate].sample(16)
#     # images = [osp.join(outfits_dir, osp.basename(img).split('_')[0], img) for img in sample_df["images"]]
#     images = [osp.join(image_dir, img) for img in sample_df["images"]]    
#     plot.display_multiple_images(images)

# %% [markdown]
# ### Unknown image category: 0 images
df_item_categories["category"].isna().sum()

# %% [markdown]
# ### Create dataframe of outfit items

# %% [markdown]
# Number of items in existing outfit dataframe
(df_outfit_items != "-1").sum().sum() - len(df_outfit_items)

# %% [markdown]
# Load non-exist items
non_exist_items = io.load_txt(
    osp.join(data_dir, "others", "final_error_image_fnames.txt")
)
len(non_exist_items)

# %% [markdown]
# remove non-existing images from df_item_categories
print("Before removing non-exist item:", len(df_item_categories))
removed_non_exist_df_item_categories = df_item_categories.copy()[
    ~df_item_categories["images"].isin(non_exist_items)
]
print("After removing non-exist item:", len(removed_non_exist_df_item_categories))

# %% [markdown]
# create outfit_id column
removed_non_exist_df_item_categories["outfit_id"] = removed_non_exist_df_item_categories["images"] \
.apply(
    lambda x: x.split('_')[0]
)
removed_non_exist_df_item_categories.head()

# %%
unq_outfit_ids = list(set(removed_non_exist_df_item_categories.outfit_id.tolist()))
unq_categories = list(set(removed_non_exist_df_item_categories.category.tolist()))
print(len(unq_outfit_ids))
print(len(unq_categories))

# %% 
df_outfit_items_latest = pd.DataFrame(columns=["outfit_id"]+unq_categories)
df_outfit_items_latest["outfit_id"] = unq_outfit_ids
df_outfit_items_latest = df_outfit_items_latest.set_index(["outfit_id"])
df_outfit_items_latest.head()

# %%
pbar = tqdm(removed_non_exist_df_item_categories.iterrows())

for idx, row in pbar:
    outfit_id = row.outfit_id
    img = row.images
    cate = row.category
    df_outfit_items_latest.loc[outfit_id, cate] = img

# %%
df_outfit_items_latest = df_outfit_items_latest.reset_index()
df_outfit_items_latest.head()

# %%
df_outfit_items_latest["outfit_id"] = df_outfit_items_latest["outfit_id"].astype("int")
len(df_outfit_items_latest), len(df_outfit_items)

# %% [markdown]
# check for missing outfits of latest outfit items dataframe
missing_outf_ids = set(df_outfit_items.outfit_id.tolist()) - set(df_outfit_items_latest.outfit_id.tolist())
missing_outf_ids = list(missing_outf_ids)
len(missing_outf_ids)

# %%
# df_outfit_items_latest.fillna("-1", inplace=True)
# io.save_csv(
#     df_outfit_items_latest,
#     osp.join(data_dir, "important", "clean_theme_outfit_items_v3.csv")
# )

# %% [markdown]
# Check if all missing outfits in process outfit dataframe has all error items
df_missing_outf = df_outfit_items[df_outfit_items.outfit_id.isin(missing_outf_ids)]
(df_missing_outf == "-1").sum().sum() == len(df_missing_outf) * len(unq_categories)

# %% [markdown]
# ### Display sample outfits
n_sample = 8
new_sizes = (224, 224)
show_original = False

sample_outfits = df_outfit_items_latest.sample(n_sample)
sample_images = []
sample_outfit_titles = []
grey_images = []

for i, row in sample_outfits.iterrows():
    outfit_id = row["outfit_id"]
    
    item_images = []
    outfit_images = []

    outfit_info = df_outfit_descriptions[df_outfit_descriptions["id"] == outfit_id]
    outfit_dir = osp.join(outfits_dir, str(outfit_id))

    if show_original:
        outfit_title = outfit_info["en_Outfit_Name"].iloc[0]
        outfit_desc = outfit_info["en_Outfit_Description"].iloc[0]
        outfit_style = outfit_info["en_Outfit_Style"].iloc[0]
        outfit_occasion = outfit_info["en_Outfit_Occasion"].iloc[0]
        outfit_fit = outfit_info["outfit_fit"].iloc[0]

        outfit_text = f"Description: {outfit_desc}\nName: {outfit_title}\nStyle: {outfit_style}\nOccasion: {outfit_occasion}\nFit: {outfit_fit}"

    else:
        outfit_text = outfit_info["en_Outfit_Description"].iloc[0]        
        choice = random.randint(0, 1)
        if choice == 1:
            outfit_text = outfit_info["additional_info"].iloc[0]        
        
    row = row[row != "-1"]
    for cate, item_id in row[1:].items():
        image_path = osp.join(outfit_dir, str(item_id))

        try:
            # image = np.array(Image.open(image_path))
            image = image_io.load_image(
                image_path,
                # toRGB=False
            )
        except Exception as e:
            print(e)
            continue

        if image is None:
            continue
        
        sizes = image.shape
    
        if len(sizes) == 3:
            if sizes[-1] == 4:
                print("4-channel image")
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            print(f"grey image")
            image = image[..., np.newaxis].repeat(3, -1)
            grey_images.append(image)

        image = cv2.resize(image, new_sizes)
        image = cv2.putText(image, cate, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        item_images.append(image)

    if len(item_images) == 0:
        continue
    
    item_images = np.vstack(item_images)
    ih, iw, _ = item_images.shape

    for outfit_image in os.listdir(outfit_dir):
        if "outfit" not in outfit_image:
            continue
        outfit_image_path = osp.join(outfit_dir, outfit_image)
        outfit_image = image_io.load_image(
            outfit_image_path,
            # toRGB=False
        )
        # outfit_image = np.array(Image.open(outfit_image_path))
        outfit_images.append(outfit_image)

    outfit_images = np.vstack(outfit_images)
    oh, ow, _ = outfit_images.shape

    outfit_images = cv2.resize(outfit_images, (int(ih * ow/oh), ih))
    combined_image = np.hstack((item_images, outfit_images))

    sample_images.append(combined_image)
    sample_outfit_titles.append(outfit_text)

plot.display_multiple_images(
    sample_images,
    grid_nrows=2,
    fig_size=24,
    titles=sample_outfit_titles,
    fontsize=10,
    axes_pad=2.4,
    line_length=8
)

# %% [markdown]
# suspicious outfits
susp_ids = [13605, 12861]

# %%
df_outf_susp = df_outfit_items_latest[df_outfit_items_latest.outfit_id.isin(susp_ids)]
print(len(df_outf_susp))
df_outf_susp.head()

# %%
sample_images = []
sample_outfit_titles = []

for _, row in df_outf_susp.iterrows():
    outfit_id = row.outfit_id
    
    item_images = []
    outfit_images = []

    outfit_info = df_outfit_descriptions[df_outfit_descriptions["id"] == outfit_id]
    outfit_dir = osp.join(outfits_dir, str(outfit_id))

    if show_original:
        outfit_title = outfit_info["en_Outfit_Name"].iloc[0]
        outfit_desc = outfit_info["en_Outfit_Description"].iloc[0]
        outfit_style = outfit_info["en_Outfit_Style"].iloc[0]
        outfit_occasion = outfit_info["en_Outfit_Occasion"].iloc[0]
        outfit_fit = outfit_info["outfit_fit"].iloc[0]

        outfit_text = f"Description: {outfit_desc}\nName: {outfit_title}\nStyle: {outfit_style}\nOccasion: {outfit_occasion}\nFit: {outfit_fit}"

    else:
        outfit_text = outfit_info["en_Outfit_Description"].iloc[0]        
        choice = random.randint(0, 1)
        if choice == 1:
            outfit_text = outfit_info["additional_info"].iloc[0]
        outfit_text += f". ID: {outfit_id}"
        
    row = row[row != "-1"]
    for cate, item_id in row[1:].items():
        image_path = osp.join(outfit_dir, str(item_id))

        try:
            # image = np.array(Image.open(image_path))
            image = image_io.load_image(
                image_path,
                # toRGB=False
            )
        except Exception as e:
            print(e)
            continue

        if image is None:
            continue
        
        sizes = image.shape
    
        if len(sizes) == 3:
            if sizes[-1] == 4:
                print("4-channel image")
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            print(f"grey image")
            image = image[..., np.newaxis].repeat(3, -1)
            grey_images.append(image)

        image = cv2.resize(image, new_sizes)
        image = cv2.putText(image, cate, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        item_images.append(image)

    if len(item_images) == 0:
        continue
    
    item_images = np.vstack(item_images)
    ih, iw, _ = item_images.shape

    for outfit_image in os.listdir(outfit_dir):
        if "outfit" not in outfit_image:
            continue
        outfit_image_path = osp.join(outfit_dir, outfit_image)
        outfit_image = image_io.load_image(
            outfit_image_path,
            # toRGB=False
        )
        # outfit_image = np.array(Image.open(outfit_image_path))
        outfit_images.append(outfit_image)

    outfit_images = np.vstack(outfit_images)
    oh, ow, _ = outfit_images.shape

    outfit_images = cv2.resize(outfit_images, (int(ih * ow/oh), ih))
    combined_image = np.hstack((item_images, outfit_images))

    sample_images.append(combined_image)
    sample_outfit_titles.append(outfit_text)

plot.display_multiple_images(
    sample_images,
    grid_nrows=1,
    fig_size=24,
    titles=sample_outfit_titles,
    fontsize=10,
    axes_pad=2.4,
    line_length=8
)

# %% [markdown]
# remove these invalid outfits
latest_path = osp.join(data_dir, "important", "clean_theme_outfit_items_v3.csv")
df_outfit_items_latest = io.load_csv(latest_path)
df_outfit_items_latest = df_outfit_items_latest[~df_outfit_items_latest.outfit_id.isin(susp_ids)]
len(df_outfit_items_latest)

# %%
# io.save_csv(df_outfit_items_latest, latest_path)

# %%
