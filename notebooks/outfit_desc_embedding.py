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

sys.path += ["../"]
from fashion_clip.fashion_clip import FashionCLIP
from reproducible_code.tools import io, plot

importlib.reload(plot)
importlib.reload(io)

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
    osp.join(data_dir, "important", "outfit_meta_v2.csv")
)
df_outfit_meta.head(5)

# %%
df_outfit_meta.sample(1)["additional_info"].iloc[0]

## TODO: remove name of people from description

# %% [markdown]
# some preprocessing and cleaning
df_outfit_meta["en_Outfit_Gender"].unique()

# %%
field = "en_Outfit_Description"
mask = df_outfit_meta[field].notna()
df_outfit_meta.loc[mask, "len_desc"] = df_outfit_meta.loc[
    mask, field
].progress_apply(lambda x: len(x.split(" ")))

sns.histplot(df_outfit_meta, x="len_desc")

# %%
row = df_outfit_meta.iloc[0]
desc, length = row[field], row["len_desc"]
assert len(desc.split(" ")) == length

# %%
df_outfit_meta.fillna("", inplace=True)

# %%
df_outfit_meta["en_Outfit_Description"] = df_outfit_meta[
    "en_Outfit_Description"
].str.lower()
df_outfit_meta["additional_info"] = df_outfit_meta.progress_apply(
    lambda row: (
        (
            row["en_Outfit_Name"] + ";"
            if len(row["en_Outfit_Name"]) != 0
            else ""
        )
        + (
            "For " + row["en_Outfit_Gender"] + ";"
            if len(row["en_Outfit_Gender"]) != 0
            else ""
        )
        + (
            "For " + row["en_Outfit_Occasion"] + " occasion;"
            if len(row["en_Outfit_Occasion"]) != 0
            else ""
        )
        + (
            row["en_Outfit_Occasion"] + " style;"
            if len(row["en_Outfit_Occasion"]) != 0
            else ""
        )
        + (row["outfit_fit"] + " fit" if len(row["outfit_fit"]) != 0 else "")
    ).lower(),
    axis=1,
)
df_outfit_meta["additional_info"] = df_outfit_meta["additional_info"].apply(
    lambda x: x[:-1] if x[-1] == ";" else x
)

# %%
df = df_outfit_meta.sample(1)
print("Description:", df["en_Outfit_Description"].iloc[0])
print("Additional info:", df["additional_info"].iloc[0])

# %%
df_outfit_meta.head(5)

# %%
field = "additional_info"
mask = df_outfit_meta[field].notna()
df_outfit_meta.loc[mask, "len_desc"] = df_outfit_meta.loc[
    mask, field
].progress_apply(lambda x: len(x.split(" ")))

sns.histplot(df_outfit_meta, x="len_desc")

# # %%
# path = osp.join(data_dir, "important", "outfit_meta_v2.csv")
# io.to_csv(
#     path,
#     df_outfit_meta
# )
# df_outfit_meta = io.load_csv(path)

# %% [markdown]
# ### Load model and embedding
model = FashionCLIP("fashion-clip")

# %%
outfit_descs = {}

for i, row in tqdm(df_outfit_meta.iterrows()):
    oid = str(row.id)
    desc = row.en_Outfit_Description
    add_info = row.additional_info
    embeddings = model.encode_text([desc, add_info], 2)

    outfit_descs[oid + "_1"] = embeddings[0][np.newaxis, ...]
    outfit_descs[oid + "_2"] = embeddings[1][np.newaxis, ...]

    if i % 1000 == 0:
        print("Row ", i)

# %%
print(len(outfit_descs))
oid = random.sample(list(outfit_descs.keys()), 1)[0]
emb = outfit_descs[oid]

# %%
len(df_outfit_meta)

# %%
outfit_descs

# %%
outfit_desc_pkl = osp.join(data_dir, "outfit_description.pkl")

# %%
io.save_pickle(outfit_descs, outfit_desc_pkl)

# %%
outfit_descs_load = io.load_pickle(outfit_desc_pkl)

# %%
print(len(outfit_descs_load))
emb_ = outfit_descs_load[oid]

# %%
(emb == emb_).all()

# %%
oid

# %%
outfit_descs_load[oid]

# %%
# oid_ = oid.replace('2', '1')
oid_ = oid.replace("1", "2")
print(oid_)
outfit_descs_load[oid_]

# %%
outfit_descs_load["10269_1"].shape

# %%
