# %%
import yaml
import logging

from utils.param import FashionTrainParam
from utils.logger import Logger, config_log
from dataset.fashionset_v2 import FashionDataset
from dataset.transforms import get_img_trans

# %%
cfg_file = "../configs/train/FHN_VSE_T3_visual_new.yaml"
with open(cfg_file, "r") as f:
    kwargs = yaml.load(f, Loader=yaml.FullLoader)

# %%
config = FashionTrainParam(**kwargs)
config.add_timestamp()

logger = logging.getLogger("polyvore")
logger.info(f"Fashion param : {config}")

# %%
param = config.train_data_param
cate_selection = param.cate_selection

# %%
transforms = get_img_trans(param.phase, param.image_size)
dataset = FashionDataset(
    param, transforms, cate_selection.copy(), logger
)

# %%
dataset.num_posi, dataset.num_nega

# %%
dataset.nega_df

# %%
param.image_dir, param.data_mode

# %%
5 // 0.8

# %%
