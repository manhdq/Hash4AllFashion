import logging
import os
import os.path as osp
import pickle
import lmdb
import numpy as np
import pandas as pd
import six
import tqdm
from scipy.special import factorial
from PIL import Image
import cv2
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset

import utils
import utils.config as cfg
from .transforms import get_img_trans

import snoop
from icecream import ic


LOGGER = logging.getLogger(__name__)


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
        visual_embedding=None,
        image_dir="",
        lmdb_env=None,
        transforms=None,
    ):
        self.cate_dict = cfg.CateIdx
        self.cate_name = cfg.CateName

        self.use_semantic = use_semantic
        self.semantic = semantic
        self.use_visual = use_visual
        self.visual_embedding = visual_embedding
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

        if self.visual_embedding is not None:
            img = self.visual_embedding[img_name]
        elif self.lmdb_env:
            # Read with lmdb format
            with self.lmdb_env.begin(write=False) as txn:
                imgbuf = txn.get(img_name.encode())

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
                img = np.zeros((300, 300, 3), dtype=np.float32)  # Gray image
                # img = Image.fromarray(img)
                if self.transforms:
                    img = self.transforms(image=img)["image"]
                else:
                    img = torch.from_numpy(cv2.resize(img, (224, 224)))
            else:
                img = self.load_image(id_name)
                if self.transforms:
                    img = self.transforms(image=img)["image"]
                else:
                    img = torch.from_numpy(cv2.resize(img, (224, 224)))
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
        tpl_data = defaultdict(list)

        if self.use_semantic:
            tpl_data["semantic"].extend(self.semantic_data(tpl))
        if self.use_visual:
            tpl_data["visual"].extend(self.visual_data(tpl)),

        return tpl_data


class FashionDataset(Dataset):
    def __init__(self, param, transforms=None, logger=None):
        self.param = param
        self.logger = logger

        self.df = pd.read_csv(param.data_csv)
        if param.use_outfit_semantic:
            self.outfit_semantic = load_semantic_data(param.outfit_semantic)
        else:
            self.outfit_semantic = None

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
        self.cate_selection = param.cate_selection
        if self.cate_selection == "all":
            cate_selection = list(self.df.columns)
        else:
            cate_selection = param.cate_selection + ["outfit_id", "compatible"]

        # Get cate-to-idx mapping
        self.cate_idxs = {
            cate: idx for idx, cate in enumerate(self.cate_selection)
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
            self.posi_df.reset_index(drop=True).drop(
                ["outfit_id", "compatible"], axis=1
            )
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

        if param.use_visual_embedding:
            ##TODO: Code this later
            visual_embedding = load_semantic_data(param.visual_embedding)
        else:
            visual_embedding = None

        ##TODO: Careful with lmdb
        lmdb_env = open_lmdb(param.lmdb_dir) if param.use_lmdb else None
        self.datum = Datum(
            use_semantic=param.use_semantic,
            semantic=semantic,
            use_visual=param.use_visual,
            visual_embedding=visual_embedding,
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
            "ShuffleDatabase",
            "RandomOnline",
            "HardOnline",
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
        n_rows, n_cols = self.posi_df.shape
        df_nega = np.empty((n_rows, n_cols), dtype=object)

        # item_list is the list of all items of each cate
        for i in range(n_cols):
            df_nega[:, i] = np.random.choice(self.item_list[i], n_rows)

        # df_drop is df after dropping 'compatible' column
        df_nega = pd.DataFrame(df_nega, columns=self.df_drop.columns)
        df_nega[self.posi_df == "-1"] = "-1"

        # Check each row of nega_df if it matches the row of posi_df
        for i, row in df_nega.iterrows():
            # if current row of posi_df
            # match that of nega_df then
            # keep randomizing row of nega_df
            # TODO: what if current row of nega_df match other row of posi_df?
            while (self.posi_df.loc[i] == df_nega.loc[i]).all():
                df_nega.loc[i] = list(
                    np.random.choice(self.item_list[i], 1)
                    for i in range(len(self.item_list))
                )
                df_nega.loc[i, self.posi_df.loc[i] == "-1"] = "-1"

        return df_nega

    def _hard_online(
        self,
    ):
        df_nega = self.posi_df.sample(frac=1).reset_index(drop=True)

        # Check each row of nega_df if it matches the row of posi_df
        for i, row in df_nega.iterrows():
            # sample other rows of posi_df
            while (self.posi_df.loc[i] == df_nega.loc[i]).all():
                df_nega.loc[i] = self.posi_df.sample(1).values.tolist()

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
        elif self.param.nega_mode == "HardOnline":
            ##TODO: Random the negative dataframe from positive one
            self.nega_df = self._hard_online()
            self.ratio = self.ratio_fix = len(self.nega_df) / len(self.posi_df)
            self.logger.info("Random hard online database")
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

    def get_pair_list(self, num_pairwise_list, df):
        # for i, row in df.iterrows()
        df_array = df.to_numpy()[..., :-1]  # Eliminate compatible
        df_count = (df_array != "-1").astype(int).sum(axis=-1)

        pairwise_count_list = []
        for num_pairwise in num_pairwise_list:
            pairwise_count_list.append(count_pairwise(df_count, num_pairwise))
        return pairwise_count_list

    def get_new_data_with_new_cate_selection(self, df, cate_selection):
        df = df.copy()
        df = df[cate_selection]
        df_count = (
            (df.to_numpy()[..., :-2] != "-1").astype(int).sum(axis=-1)
        )  # 2 last columns are outfit_id, compatible
        return df[df_count > 1]

    def get_tuple(self, df, idx, include_null=False):
        """Return item's cate ids and item ids of the outfit"""
        raw_tuple = df.iloc[idx]
        oid = self.outfit_ids[idx]

        if include_null:
            outfit_tuple = raw_tuple.copy()
        else:
            outfit_tuple = raw_tuple[raw_tuple != "-1"]

        outfit_cates = [
            self.cate_idxs[col] for col in outfit_tuple.index.to_list()
        ]
        return oid, outfit_cates, outfit_tuple.values.tolist()

    def _PairWise(self, index):
        """Get a pair of outfits."""
        posi_oid, posi_cates, posi_tpl = self.get_tuple(
            self.posi_df, int(index // self.ratio)
        )
        nega_oid, nega_cates, nega_tpl = self.get_tuple(self.nega_df, index)

        assert posi_oid == nega_oid

        # Get semantic embedding of outfits
        if self.outfit_semantic is not None:
            outf_s = [torch.from_numpy(self.outfit_semantic[posi_oid])]
        else:
            outf_s = []

        posi_tpl = self.datum.get(posi_tpl)
        posi_v, posi_s = posi_tpl["visual"], posi_tpl["semantic"]

        nega_tpl = self.datum.get(nega_tpl)
        nega_v, nega_s = nega_tpl["visual"], nega_tpl["semantic"]

        return {
            "outf_s": outf_s,
            "posi_tpl": (posi_cates, posi_v, posi_s),
            "nega_tpl": (nega_cates, nega_v, nega_s),
        }

    def _PairWiseIncludeNull(self, index):
        """Get a pair of outfits."""
        posi_oid, posi_cates, posi_tpl = self.get_tuple(
            self.posi_df, int(index // self.ratio), include_null=True
        )
        nega_oid, nega_cates, nega_tpl = self.get_tuple(
            self.nega_df, index, include_null=True
        )

        assert posi_oid == nega_oid

        # Get semantic embedding of outfits
        if self.outfit_semantic is not None:
            outf_s = [torch.from_numpy(self.outfit_semantic[posi_oid])]
        else:
            outf_s = []

        posi_tpl = self.datum.get(posi_tpl)
        posi_v, posi_s = posi_tpl["visual"], posi_tpl["semantic"]

        nega_tpl = self.datum.get(nega_tpl)
        nega_v, nega_s = nega_tpl["visual"], nega_tpl["semantic"]

        return {
            "outf_s": outf_s,
            "posi_tpl": (posi_cates, posi_v, posi_s),
            "nega_tpl": (nega_cates, nega_v, nega_s),
        }

    def _PosiOnly(self, index):
        """Get single outfit."""
        posi_oid, posi_cates, posi_tpl = self.get_tuple(
            self.posi_df, int(index // self.ratio)
        )

        # Get semantic embedding of outfits
        if self.outfit_semantic is not None:
            outf_s = [torch.from_numpy(self.outfit_semantic[posi_oid])]
        else:
            outf_s = []

        ## Mapping to tensor idxs for classification training
        # posi_cates = list(map(self.cate_idxs_to_tensor_idxs.get, posi_cates))

        posi_tpl = self.datum.get(posi_tpl)
        posi_v, posi_s = posi_tpl["visual"], posi_tpl["semantic"]

        return {
            "outf_s": outf_s,
            "posi_tpl": (posi_cates, posi_v, posi_s),
            "nega_tpl": ([], [], []),
        }

    def _NegaOnly(self, index):
        nega_oid, nega_cates, nega_tpl = self.get_tuple(self.nega_df, index)

        # Get semantic embedding of outfits
        if self.outfit_semantic is not None:
            outf_s = [torch.from_numpy(self.outfit_semantic[nega_oid])]
        else:
            outf_s = []

        ## Mapping to tensor idxs for classification training
        # nega_cates = list(map(self.cate_idxs_to_tensor_idxs.get, nega_cates))

        nega_tpl = self.datum.get(nega_tpl)
        nega_v, nega_s = nega_tpl["visual"], nega_tpl["semantic"]

        return {
            "outf_s": outf_s,
            "posi_tpl": ([], [], []),
            "nega_tpl": (nega_cates, nega_v, nega_s),
        }

    def __getitem__(self, index):
        """Get one tuple of examples by index."""
        return dict(
            PairWise=self._PairWise,
            PairWiseIncludeNull=self._PairWiseIncludeNull,
            PosiOnly=self._PosiOnly,
            NegaOnly=self._NegaOnly,
        )[self.param.data_mode](index)

    def __len__(self):
        """Return the size of dataset."""
        return dict(
            PairWise=int(self.ratio * self.num_posi),
            PairWiseIncludeNull=int(self.ratio * self.num_posi),
            PosiOnly=self.num_posi,  # all positive tuples
            NegaOnly=int(self.ratio * self.num_posi),  # all negative tuples
        )[self.param.data_mode]

    @property
    def num_posi(self):
        """Number of positive outfit."""
        return len(self.posi_df)

    @property
    def num_nega(self):
        """Number of negative outfit."""
        return len(self.nega_df)


# TODO: Check FITBDataset
class FITBDataset(Dataset):
    """Dataset for FITB task.

    Only test data has fitb file.
    """

    def __init__(self, param, transforms=None, logger=None):
        self.param = param
        self.logger = logger

        # Select item with selected cates from df
        self.cate_selection = param.cate_selection
        cate_selection = param.cate_selection + ["outfit_id"]
        self.cate_idxs = {
            cate: idx for idx, cate in enumerate(self.cate_selection)
        }

        self.df = pd.read_csv(param.data_csv)
        self.df = self.get_new_data_with_new_cate_selection(
            self.df, cate_selection
        )
        self.outfit_ids = self.df["outfit_id"].tolist()
        self.df = self.df.reset_index(drop=True).drop("outfit_id", axis=1)

        # Load semantic if any
        if param.use_outfit_semantic:
            self.outfit_semantic = load_semantic_data(param.outfit_semantic)
        else:
            self.outfit_semantic = None

        if param.use_semantic:
            semantic = load_semantic_data(param.semantic_fn)
        else:
            semantic = None

        # Get lmdb storage
        lmdb_env = open_lmdb(param.lmdb_dir) if param.use_lmdb else None
        self.datum = Datum(
            use_semantic=param.use_semantic,
            semantic=semantic,
            use_visual=param.use_visual,
            image_dir=param.image_dir,
            lmdb_env=lmdb_env,
            transforms=transforms,
        )

        # Create item_list and remove -1 value
        self.item_list = [
            list(set(self.df.iloc[:, i])) for i in range(len(self.df.columns))
        ]

        for idx, _ in enumerate(self.item_list):
            try:
                self.item_list[idx].remove("-1")
            except Exception:
                continue

        # Load fitb data
        if osp.exists(param.fitb_fn):
            self.fitb = pd.read_csv(param.fitb_fn)
            self.num_comparisons = len(self.fitb)
        else:
            self.fitb = self.make_fill_in_blank()

        self.summary()

    def get_new_data_with_new_cate_selection(self, df, cate_selection):
        df = df.copy()
        df = df[cate_selection]
        df_count = (
            (df.to_numpy()[..., :-1] != "-1").astype(int).sum(axis=-1)
        )  # last columns are outfit_id
        return df[df_count > 1]

    def get_tuple(self, df, idx, include_null=False):
        """Return item's cate ids and item ids of the outfit"""
        raw_tuple = df.iloc[idx]
        oid = self.outfit_ids[idx]

        if include_null:
            outfit_tuple = raw_tuple.copy()
        else:
            outfit_tuple = raw_tuple[raw_tuple != "-1"]

        outfit_cates = [
            cfg.CateIdx[col] for col in outfit_tuple.index.to_list()
        ]
        return oid, outfit_cates, outfit_tuple.values.tolist()

    def make_fill_in_blank(self, save_fn=None, num_cand=4, fix_cate=None):
        """Make fill-in-the-blank list.

        Parameters
        ----------
        num_cand: number of candidates.
        fix_cate: if given, then only generate list for this category.
        save_fn: save the fill-in-the-blank list.
        """
        if save_fn is None:
            save_fn = self.param.fitb_fn

        if os.path.isfile(save_fn):
            LOGGER.warning(
                "Old fill-in-the-blank file (%s) exists for %s. "
                "Delete it before override.",
                save_fn,
                self.param.phase,
            )
            return None

        # create candidate outfits
        outfits = self.df.copy()
        cand_tpl = [outfits.copy() for _ in range(num_cand)]
        num_posi = len(outfits)
        # num_cate = len(self.param.cate_map)
        num_cate = len(self.param.cate_selection)

        # randomly select items for each cate
        num_item = num_cand * 2  # why num_cand * 2?
        cand_item = []
        for cate in range(num_cate):
            cand_item.append(
                np.random.choice(self.item_list[cate], num_posi * num_item)
            )  # [num_cate, num_posi*num_item]

        # replace the items in outfits
        for idx in tqdm.tqdm(range(num_posi)):
            # Get nonempty cates of row
            raw_tpl = self.df.iloc[idx]
            outf_tpl = raw_tpl[raw_tpl != "-1"]
            cates = outf_tpl.index.to_list()

            # Get random cate in each candidate to fill with a random item
            cate = np.random.choice(cates, size=1)[0]
            cate_idx = self.cate_idxs[cate]
            cands = cand_item[cate_idx][num_item * idx : num_item * (idx + 1)]

            # remove the ground-truth item
            cands = list(set(cands) - {raw_tpl[cate]})

            # Replace the item
            for i in range(1, num_cand):
                cand_tpl[i].loc[idx, cate] = cands[i]

        # cand_tpl = np.concatenate(cand_tpl, axis=1)
        cols = (self.cate_selection) * num_cand
        cand_tpl = pd.concat(cand_tpl, axis=1)
        df = pd.DataFrame(cand_tpl, columns=cols)
        df.to_csv(save_fn, index=False)
        # return cand_tpl
        return df

    def summary(self):
        LOGGER.info("Summary for fill-in-the-blank data set")
        LOGGER.info("Number of outfits: %s", utils.colour(len(self.fitb)))
        LOGGER.info(
            "Number of candidates (include ground-truth): %s",
            utils.colour(self.param.num_cand),
        )

    def __getitem__(self, idx):
        """Get one tuple of examples by index."""
        n = int(idx // self.param.num_cand)
        i = idx % self.param.num_cand
        num_cate = len(self.cate_selection)

        # Load outfit semantic if any
        oid = self.outfit_ids[n]
        outf_smt = []
        if self.outfit_semantic is not None:
            outf_smt = [torch.from_numpy(self.outfit_semantic[oid])]

        # Load outfit tuple
        outf_tpls = self.fitb.iloc[n]
        outf_tpl = outf_tpls.iloc[num_cate * i : num_cate * (i + 1)]

        # Remove -1 and get cates
        outf_tpl = outf_tpl[outf_tpl != "-1"]
        outf_tpl = self.datum.get(outf_tpl)
        outf_v, outf_s = outf_tpl["visual"], outf_tpl["semantic"]

        return {
            "outf_s": outf_smt ,
            "imgs": outf_v,
            "s": outf_s,
        }

    def __len__(self):
        """Return the size of dataset."""
        return len(self.fitb) * self.param.num_cand


class FashionLoader(object):
    """Class for Fashion data loader"""

    def __init__(self, param, logger):
        self.logger = logger

        self.cate_selection = param.cate_selection
        # self.cate_not_selection = [
        #     cate for cate in self.cate_selection if cate not in cfg.CateName
        # ]

        self.logger.info(
            f"Loading data ({utils.colour(param.data_set)}) in phase ({utils.colour(param.phase)})"
        )
        # self.logger.info(
        #     f"- Selected apparel: "
        #     + ", ".join([utils.colour(cate) for cate in self.cate_selection])
        # )
        # self.logger.info(
        #     f"- Not selected apparel: "
        #     + ", ".join(
        #         [utils.colour(cate, "Red") for cate in self.cate_not_selection]
        #     )
        # )
        self.logger.info(
            f"- Data loader configuration: batch size ({utils.colour(param.batch_size)}), number of workers ({utils.colour(param.num_workers)})"
        )

        transforms = None
        if param.transforms:
            transforms = get_img_trans(param.phase, param.image_size)

        self.dataset = FashionDataset(param, transforms, logger)

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

    def get_outfits(self, index):
        """Get pair of outfits by index"""
        oid = self.dataset.outfit_ids[index]
        outfs = []

        for df in [self.dataset.posi_df, self.dataset.nega_df]:
            outf = df.iloc[index]
            outf = outf[outf != "-1"]
            outf_list = outf.values.tolist()
            cates_list = outf.index.tolist()
            outfs.append((outf_list, cates_list))

        return oid, outfs

    def __iter__(self):
        """Return generator."""
        for data in self.loader:
            yield data


class FITBLoader(object):
    """DataLoader warper FITB data."""

    def __init__(self, param):
        """Initialize a loader for FITBDataset."""
        LOGGER = logging.getLogger(__name__)
        LOGGER.info(
            "Loading data (%s) in phase (%s)",
            utils.colour(param.data_set),
            utils.colour(param.phase),
        )
        LOGGER.info(
            "Data loader configuration: batch size (%s) "
            "number of workers (%s)",
            utils.colour(param.num_cand),
            utils.colour(param.num_workers),
        )
        transforms = get_img_trans(param.phase, param.image_size)
        self.dataset = FITBDataset(param, transforms)
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=param.num_cand,
            num_workers=param.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=outfit_fitb_collate,
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

    def __iter__(self):
        """Return generator."""
        for data in self.loader:
            yield data

    def get_outfits(self, index):
        """Get pair of outfits by index"""
        oid = self.dataset.outfit_ids[index]

        outf = self.dataset.fitb.iloc[index]
        outf = outf[outf != "-1"]
        outf_list = outf.values.tolist()
        cates_list = outf.index.tolist()

        return oid, (outf_list, cates_list)


def outfit_fashion_collate(batch):
    """Custom collate function for dealing with batch of fashion dataset
    Each sample will has following output from dataset:
        ((`posi_cates`, `posi_imgs`), (`nega_cates`, `nega_imgs`))
        ----------
        - Examples
            `posi_cates`: [i1, i2, i3]
            `posi_imgs`: [(3, 300, 300), (3, 300, 300), (3, 300, 300)]
            `nega_cates`: [i1, i2]
            `nega_imgs`: [(3, 300, 300), (3, 300, 300)]
        ----------
        The number of apparels in each list is different between different sample
        We need concatenate them wisely

    Outputs:
        ##TODO: Describe later
    --------
    """
    batch_dict = dict()

    (
        outf_s_out,
        posi_mask,
        posi_cates_out,
        posi_imgs_out,
        posi_s_out,
        nega_mask,
        nega_cates_out,
        nega_imgs_out,
        nega_s_out,
    ) = ([], [], [], [], [], [], [], [], [])

    for i, sample in enumerate(batch):
        outf_s = sample["outf_s"]
        posi_cates, posi_imgs, posi_s = sample["posi_tpl"]
        nega_cates, nega_imgs, nega_s = sample["nega_tpl"]

        outf_s_out.extend(outf_s)

        posi_mask.extend([i] * len(posi_cates))
        posi_cates_out.extend(posi_cates)
        posi_imgs_out.extend(posi_imgs)
        posi_s_out.extend(posi_s)

        nega_mask.extend([i] * len(nega_cates))
        nega_cates_out.extend(nega_cates)
        nega_imgs_out.extend(nega_imgs)
        nega_s_out.extend(nega_s)

    if len(outf_s) != 0:
        outf_s_out = torch.cat(outf_s_out, 0)

    # Posi
    if len(posi_imgs_out) != 0:
        posi_imgs_out = torch.stack(posi_imgs_out, 0).squeeze()

    if len(posi_mask) != 0:
        posi_mask = torch.Tensor(posi_mask).to(torch.long)
        posi_cates_out = torch.Tensor(posi_cates_out).to(torch.long)

    if len(posi_s_out) != 0:
        posi_s_out = torch.cat(posi_s_out, 0)

    # Nega
    if len(nega_imgs_out) != 0:
        nega_imgs_out = torch.stack(nega_imgs_out, 0).squeeze()

    if len(nega_mask) != 0:
        nega_mask = torch.Tensor(nega_mask).to(torch.long)
        nega_cates_out = torch.Tensor(nega_cates_out).to(torch.long)

    if len(nega_s_out) != 0:
        nega_s_out = torch.cat(nega_s_out, 0)

    batch_dict["outf_s"] = outf_s_out

    batch_dict["cates"] = (posi_cates_out, nega_cates_out)

    batch_dict["mask"] = (posi_mask, nega_mask)

    batch_dict["imgs"] = (posi_imgs_out, nega_imgs_out)

    batch_dict["s"] = (posi_s_out, nega_s_out)

    return batch_dict


def outfit_fitb_collate(batch):
    """Custom collate function for dealing with batch of fashion dataset
    Each sample will has following output from dataset:
        ((`posi_cates`, `posi_imgs`), (`nega_cates`, `nega_imgs`))
        ----------
        - Examples
            `outf_s`: [(1, 512]), (1, 512), (1, 512)]
            `imgs`: [()]
        ----------

    Outputs:
        ##TODO: Describe later
    --------
    """
    batch_dict = dict()

    (
        outf_s_out,
        mask,
        imgs_out,
        s_out,
    ) = ([], [], [], [])

    for i, sample in enumerate(batch):
        outf_s = sample["outf_s"]
        imgs = sample["imgs"]
        s = sample["s"]

        outf_s_out.extend(outf_s)
        mask.extend([i] * len(imgs))
        imgs_out.extend(imgs)
        s_out.extend(s)

    if len(outf_s) != 0:
        outf_s_out = torch.cat(outf_s_out, 0)

    if len(mask) != 0:
        mask = torch.Tensor(mask).to(torch.long)

    if len(imgs_out) != 0:
        imgs_out = torch.stack(imgs_out, 0).squeeze()

    if len(s_out) != 0:
        s_out = torch.cat(s_out, 0)

    batch_dict["outf_s"] = outf_s_out

    batch_dict["cates"] = ((), ())

    batch_dict["mask"] = (mask, ())

    batch_dict["imgs"] = (imgs_out, ())

    batch_dict["s"] = (s_out, ())

    return batch_dict


# --------------------------
# Loader and Dataset Factory
# --------------------------


def get_dataloader(param, logger):
    name = param.__class__.__name__
    if name == "FITBDataParam":
        return FITBLoader(param)
    elif name == "DataParam":
        return FashionLoader(param, logger)
    return None
