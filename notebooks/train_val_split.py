# %%
import os
import os.path as osp

from tqdm import tqdm

tqdm.pandas()
import pandas as pd
from sklearn.model_selection import train_test_split

from reproducible_code.tools import image_io, io

# %%
data_dir = "/home/dungmaster/Datasets/Fashion-Outfits-Theme-Aware"
image_dir = osp.join(data_dir, "images")
train_dir = osp.join(data_dir, "train")

# %% [markdown]
# ### Get 200 first rows of train dataframe for training testing
n = 200
train_df = io.load_csv(
    osp.join(train_dir, "train_full.csv"),
)
train_df = train_df[:n]
print(len(train_df))
train_df.head()

# %%
io.save_csv(train_df, osp.join(train_dir, "train.csv"))

# %%
val_df = io.load_csv(
    osp.join(train_dir, "val_full.csv"),
)
val_df = val_df[:n]
print(len(val_df))
val_df.head()

# %%
io.save_csv(val_df, osp.join(train_dir, "val.csv"))

# %%
df_outfit_items = io.load_csv(
    osp.join(data_dir, "important", "clean_theme_outfit_items_v3.csv")
)
df_outfit_items["compatible"] = 1
df_outfit_items.head()

# %%
columns = df_outfit_items.columns.tolist()[1:-1]
columns

# %% [markdown]
# ### Preprocess
df_outfit_items_1 = df_outfit_items.copy()
df_outfit_items_1["outfit_id"] = df_outfit_items_1["outfit_id"].progress_apply(
    lambda x: str(x) + "_1"
)
df_outfit_items_1.head()

# %%
df_outfit_items_2 = df_outfit_items.copy()
df_outfit_items_2["outfit_id"] = df_outfit_items_2["outfit_id"].progress_apply(
    lambda x: str(x) + "_2"
)
df_outfit_items_2.head()

# %%
df_outfit_items = pd.concat([df_outfit_items_1, df_outfit_items_2])
print(len(df_outfit_items))
df_outfit_items.head(5)

# %%
df_outfit_items.tail(5)

# %% [markdown]
# ### Split
num_train = int(len(df_outfit_items) * 0.8)
print(num_train)
df_outfit_items = df_outfit_items.sample(frac=1).reset_index(drop=True)
df_outfit_items.head()

# %%
train_df, val_df = (
    df_outfit_items.iloc[:num_train],
    df_outfit_items.iloc[num_train:],
)
len(train_df), len(val_df)

# %%
num_val = int(len(val_df) * 0.5)
print(num_val)
val_df = val_df.sample(frac=1).reset_index(drop=True)
valid_df, test_df = val_df.iloc[:num_val], val_df.iloc[num_val:]
len(valid_df), len(test_df)

# %%
train_dir = osp.join(data_dir, "train_vse")
io.create_dir(train_dir, False)
io.save_csv(
    train_df,
    osp.join(train_dir, "train.csv"),
)
io.save_csv(
    valid_df,
    osp.join(train_dir, "val.csv"),
)
io.save_csv(
    test_df,
    osp.join(train_dir, "test.csv"),
)

# %%
