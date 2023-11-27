import logging
import os
# import pickle5 as pickle
import pickle
import lmdb
import numpy as np
import pandas as pd
import six
import tqdm
from scipy.special import factorial
from PIL import Image
import cv2

import torch
from torch.utils.data import DataLoader, Dataset

import utils
import utils.config as cfg
from .transforms import get_img_trans

from icecream import ic


def count_pairwise(count_array: np.ndarray, num_pairwise: int):
    """
    Get number of pair according to num_pairwise in count_array input

    params:
        count_array: np.ndarray (N,)
        num_pairwise: int
    """
    clear_count_array = count_array[count_array >= num_pairwise]
    count_pairwise_array = factorial(clear_count_array)
    return int(count_pairwise_array.sum())


def open_lmdb(path):
    return lmdb.open(
        path,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

def load_semantic_data(semantic_fn):
    """Load semantic data."""
    data_fn = os.path.join(semantic_fn)
    with open(data_fn, "rb") as f:
        s2v = pickle.load(f)
    return s2v


class Datum(object):
    """
    Abstract class for Fashion Dataset.
    """

    def __init__(
        self,
        use_semantic=False,
        semantic=None,
        use_visual=False,
        image_dir="",
        lmdb_env=None,
        transforms=None,
    ):
        self.cate_dict = cfg.CateIdx
        self.cate_name = cfg.CateName

        self.use_semantic = use_semantic
        self.semantic = semantic
        self.use_visual = use_visual
        self.image_dir = image_dir
        self.lmdb_env = lmdb_env
        self.transforms = transforms

    def load_image(self, id_name):
        """
        PIL loader for loading image.

        Return
        ------
        img: The image of idx name in image directory, type of PIL.Image.
        """
        img_name = f"{id_name}"
        if self.lmdb_env:
            # Read with lmdb format
            with self.lmdb_env.begin(write=False) as txn:
                imgbuf = txn.get(img_name.encode())

            # if imgbuf is None:
            #     ic(img_name)
                
            # convert it to numpy
            image = np.frombuffer(imgbuf, dtype=np.uint8)  
            # decode image
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)  
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # buf = six.BytesIO()
            # buf.write(imgbuf)
            # buf.seek(0)
            # img = Image.open(buf).convert("RGB")
        else:
            # Read from raw image
            path = os.path.join(self.image_dir, img_name)
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_semantics(self, id_name):
        """Load semantic embedding.

        Return
        ------
        vec: the semantic vector of n-th item in c-the category,
            type of torch.FloatTensor.
        """
        img_name = f"{id_name}.jpg"
        vec = self.semantic[img_name]

        return torch.from_numpy(vec.astype(np.float32))

    def visual_data(self, indices):
        """Load image data of the outfit."""
        images = []
        for id_name in indices:
            if id_name == "-1":
                # why this array?
                img = (
                    np.ones((300, 300, 3), dtype=np.uint8) * 127
                )  # Gray image
                # img = Image.fromarray(img)
                if self.transforms:
                    img = self.transforms(image=img)["image"]
            else:
                img = self.load_image(id_name)
                if self.transforms:
                    img = self.transforms(image=img)["image"]
            images.append(img)
        return images

    def semantic_data(self, indices):
        """Load semantic data of one outfit."""
        vecs = []
        for id_name in indices:
            v = self.load_semantics(id_name)
            vecs.append(v)
        return vecs

    def get(self, tpl):
        """Convert a tuple to torch.FloatTensor
        
        Args:
           tpl: list of image ids
        
        Returns:
           list of item images and list of corresponding semantic vectors
        """
        if self.use_semantic and self.use_visual:
            return self.visual_data(tpl), self.semantic_data(tpl)
        if self.use_visual:
            return self.visual_data(tpl), []
        if self.use_semantic:
            return [], self.semantic_data(tpl)
        return tpl


##TODO: Merge with FashionDataset
class FashionExtractionDataset(Dataset):
    def __init__(
        self, param, transforms=None, cate_selection="all", logger=None
    ):
        self.param = param
        self.logger = logger

        self.df = pd.read_csv(self.param.data_csv)

        # After processing data
        if cate_selection == "all":
            cate_selection = list(self.df.columns)
        else:
            cate_selection = cate_selection + [
                "compatible",
            ]

        ##TODO: Simplify this later
        # -1 because the last column define
        # if the outift is compatible or not
        # self.cate_idxs = [cfg.CateIdx[col] for col in cate_selection[:-1]]

        # self.cate_idxs_to_tensor_idxs = {
        #     cate_idx: tensor_idx
        #     for cate_idx, tensor_idx in zip(
        #         self.cate_idxs, range(len(self.cate_idxs))
        #     )
        # }

        # self.tensor_idxs_to_cate_idxs = {
        #     v: k for k, v in self.cate_idxs_to_tensor_idxs.items()
        # }
        
        self.df = self.get_new_data_with_new_cate_selection(
            self.df, cate_selection
        )

        self.df = self.df.drop("compatible", axis=1)

        if param.use_semantic:
            semantic = load_semantic_data(param.semantic_fn)
        else:
            semantic = None

        ##TODO: Careful with lmdb
        lmdb_env = open_lmdb(param.lmdb_dir) if param.use_lmdb else None
        self.datum = Datum(
            use_semantic=param.use_semantic,
            semantic=semantic,
            use_visual=param.use_visual,
            image_dir=param.image_dir,
            lmdb_env=lmdb_env,
            transforms=transforms,
        )

    def get_new_data_with_new_cate_selection(self, df, cate_selection):
        """Get only outfits with nonzero number of items"""
        df = df.copy()
        df = df[cate_selection]
        df_count = (df.to_numpy()[..., :-1] != "-1").astype(int).sum(axis=-1)
        return df[df_count > 1]

    def get_tuple(self, idx):
        """Return item's cate ids and item ids of the outfit"""
        raw_tuple = self.df.iloc[idx]
        outfit_tuple = raw_tuple[raw_tuple != "-1"]
        outfit_idxs = [
            cfg.CateIdx[col] for col in outfit_tuple.index.to_list()
        ]
        return outfit_idxs, outfit_tuple.values.tolist()

    def __getitem__(self, index):
        """Get one tuple of examples by index."""
        idxs, tpl = self.get_tuple(index)
        return idxs, tpl, self.datum.get(tpl)

    def __len__(self):
        """Return the size of dataset."""
        return len(self.df)


class FashionDataset(Dataset):
    def __init__(
        self, param, transforms=None, cate_selection="all", logger=None
    ):
        self.param = param
        self.logger = logger

        self.df = pd.read_csv(param.data_csv)
        self.outfit_semantic = load_semantic_data(param.outfit_semantic)
        
        num_pairwise_list = param.num_pairwise

        self.logger.info("")
        self.logger.info("Dataframe processing...")

        # Before processing
        num_row_before = len(self.df)
        pairwise_count_before_list = self.get_pair_list(
            num_pairwise_list, self.df
        )
        self.logger.info(
            f"+ Before: Num row: {utils.colour(num_row_before)} - "
            + " - ".join(
                [
                    f"pairwise {num_pairwise}: {utils.colour(pairwise_count_before)}"
                    for num_pairwise, pairwise_count_before in zip(
                        num_pairwise_list, pairwise_count_before_list
                    )
                ]
            )
        )

        # After processing
        if cate_selection == "all":
            cate_selection = list(self.df.columns)
        else:
            cate_selection = cate_selection + [
                "outfit_id", "compatible"
            ]
            
        ##TODO: Simplify this later, Should we register this args?
        self.cate_idxs = [cfg.CateIdx[col] for col in cate_selection[:-2]]
        self.cate_idxs_to_tensor_idxs = {
            cate_idx: tensor_idx
            for cate_idx, tensor_idx in zip(
                self.cate_idxs, range(len(self.cate_idxs))
            )
        }
        self.tensor_idxs_to_cate_idxs = {
            v: k for k, v in self.cate_idxs_to_tensor_idxs.items()
        }

        self.df = self.get_new_data_with_new_cate_selection(
            self.df, cate_selection
        )

        self.df_drop = self.df.reset_index(drop=True).drop(
            ["outfit_id", "compatible"], axis=1
        )

        # Create item_list and remove -1 value
        self.item_list = [
            list(set(self.df_drop.iloc[:, i]))
            for i in range(len(self.df_drop.columns))
        ]

        for idx, _ in enumerate(self.item_list):
            try:
                self.item_list[idx].remove("-1")
            except Exception:
                continue

        num_row_after = len(self.df)
        pairwise_count_after_list = self.get_pair_list(
            num_pairwise_list, self.df
        )
        self.logger.info(
            f"+ After: Num row: {utils.colour(num_row_after)} - "
            + " - ".join(
                [
                    f"pairwise {num_pairwise}: {utils.colour(pairwise_count_after)}"
                    for num_pairwise, pairwise_count_after in zip(
                        num_pairwise_list, pairwise_count_after_list
                    )
                ]
            )
        )
        self.logger.info("")

        self.posi_df = self.df[self.df.compatible == 1]
        self.outfit_ids = self.posi_df["outfit_id"].tolist()
        
        self.posi_df = (
            # self.df[self.df.compatible == 1]
            self.posi_df
            .reset_index(drop=True)
            .drop(["outfit_id", "compatible"], axis=1)
        )
        self.nega_df = (
            self.df[self.df.compatible == 0]
            .reset_index(drop=True)
            .drop(["outfit_id", "compatible"], axis=1)
        )

        assert len(self.posi_df) + len(self.nega_df) == len(self.df)

        if param.use_semantic:
            ##TODO: Code this later
            semantic = load_semantic_data(param.semantic_fn)
        else:
            semantic = None

        ##TODO: Careful with lmdb
        lmdb_env = open_lmdb(param.lmdb_dir) if param.use_lmdb else None
        self.datum = Datum(
            use_semantic=param.use_semantic,
            semantic=semantic,
            use_visual=param.use_visual,
            image_dir=param.image_dir,
            lmdb_env=lmdb_env,
            transforms=transforms,
        )
        self.using_max_num_pairwise = param.using_max_num_pairwise

        # probability for hard negative samples
        self.hard_ratio = 0.8
        # the ratio between negative outfits and positive outfits
        self.ratio = self.ratio_fix = len(self.nega_df) / len(self.posi_df)
        self.set_data_mode(param.data_mode)
        self.set_nega_mode(param.nega_mode)

    ##TODO: Modify this, do we need this
    def set_nega_mode(self, mode):
        """Set negative outfits mode."""
        assert mode in [
            "ShuffleDatabase",  # Manh's method
            "RandomOnline",  # Hung's method
        ], "Unknown negative mode."
        if self.param.data_mode == "PosiOnly":
            self.logger.warning(
                f"Current data-mode is {utils.colour(self.param.data_mode)}."
                f"{utils.colour('The negative mode will be ignored!')}",
            )
        else:
            self.logger.info(f"Set negative mode to {utils.colour(mode)}")
            self.param.nega_mode = mode
            self.make_nega()

    def _shuffle_nega(
        self,
    ):
        return self.nega_df_ori.sample(frac=1).reset_index(drop=True)

    def _random_online(
        self,
    ):
        row, col = self.posi_df.shape
        df_nega = np.empty((row, col), dtype=object)        

        # item_list is the list of all items of each cate
        for i in range(col):
            df_nega[:, i] = np.random.choice(self.item_list[i], row)

        # df_drop is df after dropping 'compatible' column
        df_nega = pd.DataFrame(df_nega, columns=self.df_drop.columns)
        df_nega[self.posi_df == "-1"] = "-1"

        # Check each row of nega_df if it matches the row of posi_df
        for i, row in df_nega.iterrows():
            # if current row of posi_df 
            # match that of nega_df then
            # keep randomizing row of nega_df
            # TODO: what if current row of nega_df match other row of posi_df? 
            # TODO: add description embedding of each outfit
            while (self.posi_df.loc[i] == df_nega.loc[i]).all():
                outfit_id = row.outfit_id
                df_nega.loc[i] = [outfit_id] + list(
                    np.random.choice(self.item_list[i], 1)
                    for i in range(len(self.item_list))
                )
                df_nega.loc[i, self.posi_df.loc[i] == "-1"] = "-1"

        return df_nega

    def make_nega(self, ratio=1):
        """Make negative outfits according to its mode and ratio."""
        self.logger.info(
            f"Make negative outfit for mode {utils.colour(self.param.nega_mode)}"
        )
        if self.param.nega_mode == "ShuffleDatabase":
            self.nega_df = self._shuffle_nega()
            self.logger.info("Shuffle negative database")
        elif self.param.nega_mode == "RandomOnline":
            ##TODO: Random the negative dataframe from positive one
            self.nega_df = self._random_online()
            self.ratio = self.ratio_fix = len(self.nega_df) / len(self.posi_df)
            self.logger.info("Random online database")
        else:
            raise  ##TODO:
        self.logger.info("Done making negative outfits!")

    ##TODO: Modify this func
    def set_data_mode(self, mode):
        """Set data mode."""
        assert mode in [
            "TupleOnly",
            "PosiOnly",
            "NegaOnly",
            "PairWise",
            "PairWiseIncludeNull",
            "TripleWise",
        ], f"Unknown data mode: {mode}"
        self.logger.info(f"Set data mode to {utils.colour(mode)}")
        self.param.data_mode = mode

    ##TODO: What is it?
    def set_prob_hard(self, p):
        """Set the proportion for hard negative examples."""
        if self.param.data_mode == "PosiOnly":
            self.logger.warning(
                "Current data-mode is %s. " "The proportion will be ignored!",
                utils.colour(self.param.data_mode, "Red"),
            )
        elif self.param.nega_mode != "HardOnline":
            self.logger.warning(
                "Current negative-sample mode is %s. "
                "The proportion will be ignored!",
                utils.colour(self.param.nega_mode, "Red"),
            )
        else:
            self.phard = p
            self.logger.info(
                "Set the proportion of hard negative outfits to %s",
                utils.colour("%.3f" % p),
            )

    def get_new_data_with_new_cate_selection(self, df, cate_selection):
        df = df.copy()
        df = df[cate_selection]
        df_count = (df.to_numpy()[..., :-2] != "-1").astype(int).sum(axis=-1) # 2 last columns are outfit_id, compatible
        return df[df_count > 1]

    def get_pair_list(self, num_pairwise_list, df):
        # for i, row in df.iterrows()
        df_array = df.to_numpy()[..., :-1]  # Eliminate compatible
        df_count = (df_array != "-1").astype(int).sum(axis=-1)

        pairwise_count_list = []
        for num_pairwise in num_pairwise_list:
            pairwise_count_list.append(count_pairwise(df_count, num_pairwise))
        return pairwise_count_list

    ##TODO:
    # def get_tuple(self, df, idx):
    #     outfit_idxs_out = torch.zeros(len(df.columns))
    #     raw_tuple = df.iloc[idx]
    #     outfit_tuple = raw_tuple[raw_tuple != -1]
    #     outfit_idxs = [cfg.CateIdx[col] for col in outfit_tuple.index.to_list()]
    #     outfit_tensor_idxs = [self.cate_idxs_to_tensor_idxs[outfit_idx] for outfit_idx in outfit_idxs]
    #     outfit_idxs_out[outfit_tensor_idxs] = 1
    #     return outfit_idxs, raw_tuple.values.tolist()

    def get_tuple(self, df, idx, include_null=False):
        """Return item's cate ids and item ids of the outfit"""
        raw_tuple = df.iloc[idx]
        oid = self.outfit_ids[idx]

        if include_null:
            outfit_tuple = raw_tuple.copy()
        else:
            outfit_tuple = raw_tuple[raw_tuple != "-1"]

        outfit_idxs = [
            cfg.CateIdx[col] for col in outfit_tuple.index.to_list()
        ]
        return oid, outfit_idxs, outfit_tuple.values.tolist()

    def _PairWise(self, index):
        """Get a pair of outfits."""
        posi_oid, posi_idxs, posi_tpl = self.get_tuple(
            self.posi_df, int(index // self.ratio)
        )
        nega_oid, nega_idxs, nega_tpl = self.get_tuple(self.nega_df, index)

        assert posi_oid == nega_oid

        # Get semantic embedding of outfits
        outf_s = self.outfit_semantic[posi_oid]

        ## Mapping to tensor idxs for classification training
        posi_idxs = list(map(self.cate_idxs_to_tensor_idxs.get, posi_idxs))
        nega_idxs = list(map(self.cate_idxs_to_tensor_idxs.get, nega_idxs))

        ##TODO: Dynamic options for visual and semantic selections
        posi_v, posi_s = self.datum.get(posi_tpl)
        nega_v, nega_s = self.datum.get(nega_tpl)
        return (outf_s, (posi_idxs, posi_v, posi_s), (nega_idxs, nega_v, nega_s))

    def __getitem__(self, index):
        """Get one tuple of examples by index."""
        return dict(
            PairWise=self._PairWise,
        )[
            self.param.data_mode
        ](index)

    def __len__(self):
        """Return the size of dataset."""
        return dict(
            PairWise=int(self.ratio * self.num_posi),
        )[self.param.data_mode]

    @property
    def num_posi(self):
        """Number of positive outfit."""
        return len(self.posi_df)

    @property
    def num_nega(self):
        """Number of negative outfit."""
        return len(self.nega_df)


class FashionLoader(object):
    """Class for Fashion data loader"""

    def __init__(self, param, logger):
        self.logger = logger

        self.cate_selection = param.cate_selection
        self.cate_not_selection = [
            cate for cate in cfg.CateName if cate not in param.cate_selection
        ]

        self.logger.info(
            f"Loading data ({utils.colour(param.data_set)}) in phase ({utils.colour(param.phase)})"
        )
        self.logger.info(
            f"- Selected apparel: "
            + ", ".join([utils.colour(cate) for cate in self.cate_selection])
        )
        self.logger.info(
            f"- Not selected apparel: "
            + ", ".join(
                [utils.colour(cate, "Red") for cate in self.cate_not_selection]
            )
        )
        self.logger.info(
            f"- Data loader configuration: batch size ({utils.colour(param.batch_size)}), number of workers ({utils.colour(param.num_workers)})"
        )
        transforms = get_img_trans(param.phase, param.image_size)
        self.dataset = FashionDataset(
            param,
            transforms,
            # self.cate_selection.copy(),
            self.cate_selection,            
            logger
        )
        
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=param.batch_size,
            num_workers=param.num_workers,
            shuffle=param.shuffle,
            pin_memory=True,
            collate_fn=outfit_fashion_collate,
        )

    def __len__(self):
        """Return number of batches."""
        return len(self.loader)

    @property
    def num_batch(self):
        """Get number of batches."""
        return len(self.loader)

    @property
    def num_sample(self):
        """Get number of samples."""
        return len(self.dataset)

    def make_nega(self, ratio=1):
        """Prepare negative outfits."""
        self.dataset.make_nega(ratio)
        return self

    def set_nega_mode(self, mode):
        """Set the mode for generating negative outfits."""
        self.dataset.set_nega_mode(mode)
        return self

    def set_data_mode(self, mode):
        """Set the mode for data set."""
        self.dataset.set_data_mode(mode)
        return self

    def set_prob_hard(self, p):
        """Set the probability of negative outfits."""
        self.dataset.set_prob_hard(p)
        return self

    def __iter__(self):
        """Return generator."""
        for data in self.loader:
            yield data


def outfit_fashion_collate(batch):
    """Custom collate function for dealing with batch of fashion dataset
    Each sample will has following output from dataset:
        ((`posi_idxs`, `posi_imgs`), (`nega_idxs`, `nega_imgs`))
        ----------
        - Examples
            `posi_idxs`: [i1, i2, i3]
            `posi_imgs`: [(3, 300, 300), (3, 300, 300), (3, 300, 300)]
            `nega_idxs`: [i1, i2]
            `nega_imgs`: [(3, 300, 300), (3, 300, 300)]
        ----------
        The number of apparels in each list is different between different sample
        We need concatenate them wisely

    Outputs:
        ##TODO: Describe later
    --------
    """
    (
        outf_s_out,
        posi_mask,
        posi_idxs_out,
        posi_imgs_out,
        posi_s_out,
        nega_mask,
        nega_idxs_out,
        nega_imgs_out,
        nega_s_out,
    ) = ([], [], [], [], [], [], [], [], [])

    for i, sample in enumerate(batch):
        outf_s, (posi_idxs, posi_imgs, posi_s), (nega_idxs, nega_imgs, nega_s) = sample
        outf_s_out.extend(torch.from_numpy(outf_s).float())
        
        posi_mask.extend([i] * len(posi_idxs))
        posi_idxs_out.extend(posi_idxs)
        posi_imgs_out.extend(posi_imgs)
        posi_s_out.extend(posi_s)

        nega_mask.extend([i] * len(nega_idxs))
        nega_idxs_out.extend(nega_idxs)
        nega_imgs_out.extend(nega_imgs)
        nega_s_out.extend(nega_s)

    if len(posi_imgs) != 0 and len(posi_s) != 0:
        return (
            torch.stack(outf_s_out, 0),
            torch.Tensor(posi_mask).to(torch.long),
            torch.Tensor(posi_idxs_out).to(torch.long),
            torch.stack(posi_imgs_out, 0),
            torch.stack(posi_s_out, 0),
            torch.Tensor(nega_mask).to(torch.long),
            torch.Tensor(nega_idxs_out).to(torch.long),
            torch.stack(nega_imgs_out, 0),
            torch.stack(nega_s_out, 0),
        )
    elif len(posi_imgs) != 0:
        return (
            torch.stack(outf_s_out, 0),
            torch.Tensor(posi_mask).to(torch.long),
            torch.Tensor(posi_idxs_out).to(torch.long),
            torch.stack(posi_imgs_out, 0),
            torch.Tensor(nega_mask).to(torch.long),
            torch.Tensor(nega_idxs_out).to(torch.long),
            torch.stack(nega_imgs_out, 0),
        )
    else:
        return (
            torch.stack(outf_s_out, 0),
            torch.Tensor(posi_mask).to(torch.long),
            torch.Tensor(posi_idxs_out).to(torch.long),
            torch.stack(posi_s_out, 0),
            torch.Tensor(nega_mask).to(torch.long),
            torch.Tensor(nega_idxs_out).to(torch.long),
            torch.stack(nega_s_out, 0),
        )


# --------------------------
# Loader and Dataset Factory
# --------------------------


def get_dataloader(param, logger):
    name = param.__class__.__name__
    if name == "DataParam":
        return FashionLoader(param, logger)
    return None
