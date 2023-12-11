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
from PIL import Image
import cv2

sys.path += ["../"]
from reproducible_code.tools import io, plot, image_io

importlib.reload(plot)

sns.set_theme()
sns.set_style("whitegrid", {"axes.grid": False})
tqdm.pandas()

# %% [markdown]
# ### Loading data
data_dir = "/home/dungmaster/Datasets/Fashion-Outfits-Theme-Aware"
outfits_dir = osp.join(data_dir, "outfits")

# %% [markdown]
# ### Load outfits' metadata
df_outfit_meta = io.load_csv(
    osp.join(data_dir, "important", "outfit_meta.csv")
)
df_outfit_meta.head(5)

# %%
df_outfit_meta.isna().sum()

# %%
outfit_title = df_outfit_meta["en_Outfit_Name"].dropna().unique().tolist()
outfit_desc = (
    df_outfit_meta["en_Outfit_Description"].dropna().unique().tolist()
)
outfit_style = df_outfit_meta["en_Outfit_Style"].dropna().unique().tolist()
outfit_occasion = (
    df_outfit_meta["en_Outfit_Occasion"].dropna().unique().tolist()
)
outfit_fit = df_outfit_meta["en_Outfit_Fit"].dropna().unique().tolist()

# %%
outfit_style

# %%
outfit_occasion

# %%
outfit_fit

# %%
outfit_fit_map = {
    "Slim was thin": "Thin",
    "Appear taller": "Tall",
    "Look younger": "Young",
    "Show whiteness": "White skin",
    "Show breasts": "Breasts",
    "Appear long neck": "Long neck",
    "Look small": "Small face",
    "Strong man": "Strong",
    "Show your buttocks": "Bottom",
}
fit_json = osp.join(data_dir, "json", "fit_map.json")
# io.save_json(outfit_fit_map, fit_json)

# %%
mask = df_outfit_meta["en_Outfit_Fit"].notna()
df_outfit_meta.loc[mask, "outfit_fit"] = df_outfit_meta.loc[
    mask, "en_Outfit_Fit"
].progress_apply(lambda x: outfit_fit_map[x])

# %%
outfit_fit = df_outfit_meta["outfit_fit"].dropna().unique().tolist()
outfit_fit

# %%
df_outfit_meta.fillna("", inplace=True)
df_outfit_meta.iloc[1]["cn_Outfit_Style"]

# %%
df_outfit_meta.columns.tolist()

# %%
df_outfit_meta["detailed_description"] = df_outfit_meta.progress_apply(
    lambda row: (
        (
            row["en_Outfit_Description"]
            if len(row["en_Outfit_Description"]) != 0
            else ""
        )
        + (
            row["en_Outfit_Name"] + "."
            if len(row["en_Outfit_Name"]) != 0
            else ""
        )
        + (
            "Gender " + row["en_Outfit_Gender"] + "."
            if len(row["en_Outfit_Gender"]) != 0
            else ""
        )
        + (
            "For " + row["en_Outfit_Occasion"] + " occasion."
            if len(row["en_Outfit_Occasion"]) != 0
            else ""
        )
        + (
            row["en_Outfit_Occasion"] + " style."
            if len(row["en_Outfit_Occasion"]) != 0
            else ""
        )
        + (row["outfit_fit"] + " fit." if len(row["outfit_fit"]) != 0 else "")
    ).lower(),
    axis=1,
)
print(df_outfit_meta.iloc[0]["detailed_description"])

# %%
meta_csv = osp.join(data_dir, "important", "outfit_meta_v2.csv")
# io.to_csv(meta_csv, df_outfit_meta)
df_outfit_meta = io.load_csv(meta_csv)
df_outfit_meta.head()

# %% [markdown]
# ### Load outfit files

# %%
df_outfit_items = io.load_csv(
    osp.join(data_dir, "important", "clean_theme_outfit_items_v3.csv")
)
df_outfit_items.head()

# %%
print(len(df_outfit_meta))
outfit_name = "ceshi_xx1423"
oid = df_outfit_meta[df_outfit_meta["cn_Outfit_Name"] != outfit_name][
    "id"
].iloc[0]
print(len(df_outfit_meta))

# # %%
# df_outfit_items = df_outfit_items[df_outfit_items["outfit_id"] == oid]
# len(df_outfit_items)

# %% [markdown]
# ### Display outfits with their descriptions

# %%
n_sample = 1
show_original = True

# ids = df_outfit_meta[df_outfit_meta["en_Outfit_Fit"]=="Show breasts"]["id"].tolist()
# sample_outfits = df_outfit_items[df_outfit_items["outfit_id"].isin(ids)].sample(n_sample)
sample_outfits = df_outfit_items.sample(n_sample)

new_sizes = (224, 224)

sample_images = []
sample_outfit_titles = []
grey_images = []

for i, row in sample_outfits.iterrows():
    outfit_id = row["outfit_id"]

    item_images = []
    outfit_images = []

    outfit_info = df_outfit_meta[df_outfit_meta["id"] == outfit_id]
    outfit_dir = osp.join(outfits_dir, str(outfit_id))

    if show_original:
        outfit_title = outfit_info["en_Outfit_Name"].iloc[0]
        outfit_desc = outfit_info["en_Outfit_Description"].iloc[0]
        outfit_style = outfit_info["en_Outfit_Style"].iloc[0]
        outfit_occasion = outfit_info["en_Outfit_Occasion"].iloc[0]
        outfit_fit = outfit_info["outfit_fit"].iloc[0]

        if outfit_fit != outfit_fit:
            outfit_fit = "Unknown"

        outfit_text = f"Description: {outfit_desc}\nName: {outfit_title}\nStyle: {outfit_style}\nOccasion: {outfit_occasion}\nFit: {outfit_fit}"

    else:
        outfit_text = outfit_info["detailed_description"].iloc[0]

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
        image = cv2.putText(
            image, cate, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
        )
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

    outfit_images = cv2.resize(outfit_images, (int(ih * ow / oh), ih))
    combined_image = np.hstack((item_images, outfit_images))

    sample_images.append(combined_image)
    sample_outfit_titles.append(outfit_text)

# %%
# display_image_sets(sample_images, title=outfit_text)
plot.display_multiple_images(
    sample_images,
    grid_nrows=1,
    fig_size=12,
    # titles=sample_outfit_titles,
    titles=None,    
    fontsize=12,
    axes_pad=2.4,
    line_length=8,
)

# %%
print(sample_outfit_titles[0])

# %% [markdown]
# ### Check all the items in each category

# %%
df_item_cate = io.load_csv(
    osp.join(
        data_dir, "important", "processed_theme_aware_item_categories.csv"
    )
)
df_item_cate.head(5)

# %% [markdown]
# ## Category EDA
plot.plot_attribute_frequency(df_item_cate, "category", 10, 10)

# %%
for cate in df_item_cate["category"].unique():
    print("Category:", cate)
    sample_df = df_item_cate[df_item_cate["category"] == cate].sample(16)
    images = [
        osp.join(outfits_dir, osp.basename(img).split("_")[0], img)
        for img in sample_df["images"]
    ]
    plot.display_multiple_images(images)

# %% [markdown]
# ### Outlier images

# %% [markdown]
# #### Unknown category
outfit_id = "5858"

# %%
df_item_cate["category"].isna().sum()

# %%
df_item_cate_v1 = io.load_csv(
    osp.join(data_dir, "theme-aware_item_categories.csv")
)
df_item_cate_v1.head(5)

# %%
df_item_cate_v1.isna().sum()

# %%
img = df_item_cate_v1[df_item_cate_v1["category"].isna()]["image"].iloc[0]
df_item_cate[df_item_cate["images"] == img]

# %% [markdown]
# #### Nonexist
nonexist_images = io.load_txt(osp.join(data_dir, "nonexist_images.txt"))
len(nonexist_images)

# %% [markdown]
# Check again if these are not valid images
count = 0
get_dir = lambda x: osp.basename(x).split("_")[0]

for img in nonexist_images:
    img_path = osp.join(outfits_dir, get_dir(img), img)
    try:
        img = np.array(Image.open(img_path))
    except Exception:
        count += 1

assert count == len(nonexist_images)

# %%
df_outfit_items.head()

# %%
error_tpls = []

for img in nonexist_images:
    img_dir = get_dir(img)
    cate = df_item_cate[df_item_cate["images"] == img]["category"].iloc[0]
    outfit_img = df_outfit_items.loc[
        df_outfit_items["outfit_id"] == int(img_dir), cate
    ].iloc[0]
    if img == outfit_img:
        df_outfit_items.loc[
            df_outfit_items["outfit_id"] == int(img_dir), cate
        ] = "-1"
        print(
            df_outfit_items.loc[
                df_outfit_items["outfit_id"] == int(img_dir), cate
            ]
        )
    else:
        error_tpls.append((img, outfit_img))

# %%
len(error_tpls)

# %%
for img, real_img in error_tpls:
    cate = df_item_cate[df_item_cate["images"] == img]["sub_category"].iloc[0]
    real_cate = df_item_cate[df_item_cate["images"] == real_img][
        "sub_category"
    ].iloc[0]
    print(cate, real_cate)

# %% [markdown]
# ### Check missing items
(df_outfit_items != "-1").sum().sum() - len(df_outfit_items)

# %%
# %%
outfit_csv = osp.join(data_dir, "important", "clean_theme_outfit_items_v2.csv")
# io.to_csv(outfit_csv, df_outfit_items)
df_outfit_items = io.load_csv(outfit_csv)
df_outfit_items.head()

# %%
df_item_cate.head(5)

# %%
df_outfit_items_v1 = io.load_csv(
    osp.join(data_dir, "important", "clean_theme_outfit_items_v1.csv")
)
(df_outfit_items_v1 != "-1").sum().sum() - 14451

# %%
