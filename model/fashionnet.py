import threading
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import utils
from ..utils import config as cfg
from .losses import soft_margin_loss
from . import backbones as B
from . import basemodel as M

from icecream import ic

NAMED_MODEL = utils.get_named_function(B)


@utils.singleton
class RankMetric(threading.Thread):
    ##TODO: Need continue fix this class
    def __init__(self, num_users, args=(), kwargs=None):
        from queue import Queue

        threading.Thread.__init__(self, args=(), kwargs=None)
        self.daemon = True
        self.queue = Queue()
        self.daemon = True
        self.num_users = num_users
        self._scores = [
            [[] for _ in range(self.num_users)] for _ in range(4)
        ]  ##TODO: What is 4?

    def reset(self):
        self._scores = [[[] for _ in range(self.num_users)] for _ in range(4)]

    def put(self, data):
        self.queue.put(data)

    def process(self, data):
        with threading.Lock():
            scores = data
            # Assume user_id is 0
            ##TODO:
            u = 0
            for n, score in enumerate(scores):
                for s in score:
                    self._scores[n][u].append(s)

    def run(self):
        print(threading.currentThread().getName(), "RankMetric")
        while True:
            data = self.queue.get()
            if data is None:  # If you send `None`, the thread will exit.
                return
            self.process(data)

    def rank(self):
        auc = utils.metrics.calc_AUC(self._scores[0], self._scores[1])
        binary_auc = utils.metrics.calc_AUC(self._scores[2], self._scores[3])
        ndcg = utils.metrics.calc_NDCG(self._scores[0], self._scores[1])
        binary_ndcg = utils.metrics.calc_NDCG(self._scores[2], self._scores[3])
        return dict(
            auc=auc, binary_auc=binary_auc, ndcg=ndcg, binary_ndcg=binary_ndcg
        )


class FashionNet(nn.Module):
    """Base class for fashion net."""

    def __init__(self, param, cate_selection):
        """See NetParam for details."""
        super().__init__()
        self.param = param
        self.scale = 1.0
        self.shared_weight = param.shared_weight_network

        # Feature extractor
        self.features = None
        if self.param.use_visual and param.backbone is not None:
            self.features = NAMED_MODEL[param.backbone]()

        # Single encoder or multi-encoders, hashing codes
        if param.shared_weight_network:
            if self.param.use_visual:
                feat_dim = 512
                if self.features is not None:
                    feat_dim = self.features.dim
                self.encoder_v = M.ImgEncoder(feat_dim, param)
            if self.param.use_semantic:
                # feat_dim = 2400
                feat_dim = self.features.dim
                self.encoder_t = M.TxtEncoder(feat_dim, param)
        else:
            ##TODO: Code forward this later
            if self.param.use_visual:
                feat_dim = self.features.dim
                self.encoder_v = nn.ModuleDict(
                    {
                        cate_name: M.ImgEncoder(feat_dim, param)
                        for cate_name in cate_selection
                    }
                )
            if self.param.use_semantic:
                feat_dim = 2400
                self.encoder_t = nn.ModuleDict(
                    {
                        cate_name: M.TxtEncoder(feat_dim, param)
                        for cate_name in cate_selection
                    }
                )

        # Outfit semantic encoder block
        self.encoder_o = M.TxtEncoder(param.outfit_semantic_dim, param)

        # Classification block
        ##TODO: Simplify this later
        self.cate_idxs = [cfg.CateIdx[col] for col in cate_selection]
        self.cate_idxs_to_tensor_idxs = {
            cate_idx: tensor_idx
            for cate_idx, tensor_idx in zip(
                self.cate_idxs, range(len(self.cate_idxs))
            )
        }
        self.tensor_idxs_to_cate_idxs = {
            v: k for k, v in self.cate_idxs_to_tensor_idxs.items()
        }
        ##TODO: Code clssification module for text
        if self.param.use_visual:
            self.classifier_v = M.ImgClassifier(feat_dim, len(self.cate_idxs))
        if self.param.use_semantic:
            self.classifier_t = M.TxtClassifier(feat_dim, len(self.cate_idxs))

        # Matching block
        if self.param.hash_types == utils.param.NO_WEIGHTED_HASH:
            # Use learnable scale
            self.core = M.LearnableScale(1)
        elif self.param.hash_types == utils.param.WEIGHTED_HASH_BOTH:
            # two weighted hashing for both item-item and outfit semantic-item
            self.core = nn.ModuleList(
                [M.CoreMat(param.dim, param.pairwise_weight),
                 M.CoreMat(param.dim, param.outfit_semantic_weight)]
            )
        else:
            # Single weighed hashing for user-item or item-item, current only use for item
            ##TODO: Code this for very later
            self.core = M.CoreMat(param.dim)

        if self.param.use_semantic and self.param.use_visual:
            self.loss_weight = dict(
                rank_loss=1.0, binary_loss=None, vse_loss=0.1, cate_loss=0.8
            )
        else:
            ##TODO: Tune wweight for classification loss
            self.loss_weight = dict(
                rank_loss=1.0, binary_loss=None, cate_loss=0.8
            )
        self.configure_trace()
        ##TODO: Modify this and understanding later
        ##TODO: Assume num_users is 1
        self.rank_metric = RankMetric(num_users=1)
        if not self.rank_metric.is_alive():
            self.rank_metric.start()

    def configure_trace(self):
        self.tracer = dict()
        self.tracer["loss"] = {
            "train.loss": "Train Loss(*)",
            "train.binary_loss": "Train Loss",
            "test.loss": "Test Loss(*)",
            "test.binary_loss": "Test Loss",
        }
        if self.param.use_semantic and self.param.use_visual:
            # in this case, the overall loss dose not equal to rank loss
            vse = {
                "train.vse_loss": "Train VSE Loss",
                "test.vse_loss": "Test VSE Loss",
                "train.rank_loss": "Train Rank Loss",
                "test.rank_loss": "Test Rank Loss",
            }
            self.tracer["loss"].update(vse)
        self.tracer["accuracy"] = {
            "train.accuracy": "Train(*)",
            "train.binary_accuracy": "Train",
            "test.accuracy": "Test(*)",
            "test.binary_accuracy": "Test",
        }
        self.tracer["rank"] = {
            "test.auc": "AUC(*)",
            "test.binary_auc": "AUC",
            "test.ndcg": "NDCG(*)",
            "test.binary_ndcg": "NDCG",
        }

    def __repr__(self):
        return super().__repr__() + "\n" + self.param.__repr__()

    def set_scale(self, value):
        """Set scale of tanH layer."""
        if not self.param.scale_tanh:
            return
        self.scale = value
        # self.user_embedding.set_scale()  ##TODO:
        if self.param.use_visual:
            if not self.shared_weight:
                for encoder in self.encoder_v.values():
                    encoder.set_scale(value)
            else:
                self.encoder_v.set_scale(value)
        if self.param.use_semantic:
            if not self.shared_weight:
                for encoder in self.encoder_t.values():
                    encoder.set_scale(value)
            else:
                self.encoder_t.set_scale(value)

    def scores(self, olatent, ilatents, mask, scale=10.0):
        scores = []
        mask_idxs = torch.unique(mask).tolist()

        for idx in mask_idxs:
            # Get all feature vectors of a batch
            sub_olatent = olatent[idx]
            sub_ilatents = ilatents[mask == idx]
            size = len(sub_ilatents)
            indx, indy = np.triu_indices(size, k=1)

            # comb x D
            ##TODO: Rename
            pairwise = sub_ilatents[indx] * sub_ilatents[indy]
            semwise = sub_ilatents * sub_olatent

            # Get score
            if self.param.hash_types == utils.param.WEIGHTED_HASH_I:
                score_i = self.core(pairwise).mean()
            elif self.param.hash_types == utils.param.WEIGHTED_HASH_BOTH:
                score_i = self.core[0](pairwise).mean()
                score_o = self.core[1](semwise).mean()                
            else:
                ## TODO: Add user score
                raise

            ##TODO: Code for user score
            score = (score_i + score_o) * (scale * 2.0)
            scores.append(score)
        # Stack the scores, shape N x 1
        scores = torch.stack(scores, dim=0).view(-1, 1)
        return scores

    def sign(self, x):
        """Return hash code of x.

        if x is {1, -1} binary, then h(x) = sign(x)
        if x is {1, 0} binary, then h(x) = (sign(x - 0.5) + 1)/2
        """
        if self.param.binary01:
            return ((x.detach() - 0.5).sign() + 1) / 2.0
        return x.detach().sign()

    def latent_code(self, feat, idxs, encoder):
        """Return latent codes."""
        ##TODO: Code for multi-encoder
        ##TODO: idxs [2, 0, 3, 2, 4, 1]?? Code later
        latent_code = encoder(feat)
        # shaoe: N x D
        return latent_code

    ##TODO: Modify for not `shared weight` option, add user for very later
    def _pairwise_output(
        self,
        lco,
        bco,
        posi_mask,
        posi_idxs,
        pos_feat,
        nega_mask,
        nega_idxs,
        neg_feat,
        encoder,
    ):
        lcpi = self.latent_code(pos_feat, posi_idxs, encoder)
        lcni = self.latent_code(neg_feat, nega_idxs, encoder)
        # Score with relaxed features
        pscore = self.scores(lco, lcpi, posi_mask)  # list(): [0.33, 0.22], ...
        nscore = self.scores(lco, lcni, nega_mask)

        # Score with binary codes
        bcpi = self.sign(lcpi)
        bcni = self.sign(lcni)
        bpscore = self.scores(bco, bcpi, posi_mask)
        bnscore = self.scores(bco, bcni, nega_mask)

        # latents = torch.stack(lcpi + lcni, dim=1)
        # latents = latents.view(-1, self.param.dim)

        return (pscore, nscore, bpscore, bnscore), (lcpi, lcni)

    def visual_output(self, *inputs):
        if self.param.use_semantic:
            (
                lco,
                bco,
                posi_mask,
                posi_idxs,
                posi_imgs,
                _,
                nega_mask,
                nega_idxs,
                nega_imgs,
                _,
            ) = inputs
        else:
            (
                lco,
                bco,
                posi_mask,
                posi_idxs,
                posi_imgs,
                nega_mask,
                nega_idxs,
                nega_imgs,
            ) = inputs

        # Extract visual features
        if self.features is not None:
            posi_imgs = self.features(posi_imgs)
            nega_imgs = self.features(nega_imgs)

        feats = torch.cat([posi_imgs, nega_imgs], dim=0)

        scores, latents = self._pairwise_output(
            lco,
            bco,
            posi_mask,
            posi_idxs,
            posi_imgs,
            nega_mask,
            nega_idxs,
            nega_imgs,
            self.encoder_v,
        )

        return scores, latents, feats

    def semantic_output(self, *inputs):
        if self.param.use_visual:
            (
                posi_mask,
                posi_idxs,
                _,
                pos_feat,
                nega_mask,
                nega_idxs,
                _,
                neg_feat,
            ) = inputs
        else:
            (
                posi_mask,
                posi_idxs,
                pos_feat,
                nega_mask,
                nega_idxs,
                neg_feat,
            ) = inputs

        scores, latents = self._pairwise_output(
            posi_mask,
            posi_idxs,
            pos_feat,
            nega_mask,
            nega_idxs,
            neg_feat,
            self.encoder_t,
        )
        return scores, latents
    
    def outfit_semantic_output(self, outf_feat):
        lco = self.encoder_o(outf_feat)
        bco = self.sign(lco)
        return lco, bco

    def forward(self, *inputs):
        """Forward according to setting."""
        outf_s = inputs[0]
        outf_feats = self.outfit_semantic_output(outf_s)
        inputs = outf_feats + inputs[1:]

        # Pair-wise output
        ##TODO: Continue with this func code
        if self.param.use_semantic and self.param.use_visual:
            posi_idxs, nega_idxs = inputs[3], inputs[7]
        else:
            posi_idxs, nega_idxs = inputs[3], inputs[6]
            
        idxs = torch.cat([posi_idxs, nega_idxs])

        loss = dict()
        accuracy = dict()

        if self.param.use_semantic and self.param.use_visual:
            scores, latent_v, visual_feats = self.visual_output(*inputs)
            score_s, latent_s = self.semantic_output(*inputs)
            # scores = [0.5 * (v + s) for v, s in zip(score_v, score_s)]
            # visual-semantic similarity
            # vse_loss = contrastive_loss(self.param.margin, latent_v, latent_s)
            # loss.update(vse_loss=vse_loss)
            visual_fc = self.classifier_v(visual_feats)
            # visual_fc = self.classifier_s(semantic_feats)
        elif self.param.use_visual:
            scores, _, visual_feats = self.visual_output(*inputs)
            visual_fc = self.classifier_v(visual_feats)
        elif self.param.use_semantic:
            scores, _ = self.semantic_output(*inputs)
        else:
            raise ValueError

        # print(scores) ##TODO:
        data = [s.tolist() for s in scores]
        self.rank_metric.put(data)
        diff = scores[0] - scores[1]
        binary_diff = scores[2] - scores[3]

        ##### Calculate loss #####
        # Margin loss for ranking
        rank_loss = soft_margin_loss(diff)
        binary_loss = soft_margin_loss(binary_diff)
        cls_loss = F.cross_entropy(visual_fc, idxs, reduction="none")

        ##### Calculate accuracy #####
        acc = torch.gt(diff.data, 0)
        binary_acc = torch.gt(binary_diff, 0)
        cate_acc = self.calc_cate_acc(visual_fc.detach(), idxs)

        loss.update(
            rank_loss=rank_loss, binary_loss=binary_loss, cate_loss=cls_loss
        )
        accuracy.update(
            accuracy=acc, binary_accuracy=binary_acc, cate_acc=cate_acc
        )
        return loss, accuracy

    def calc_cate_acc(self, visual_fc, idxs):
        pred_idxs = torch.argmax(visual_fc, dim=1)
        return torch.eq(pred_idxs, idxs)

    ##TODO: Modify for not `shared weight` option, add user for very later
    def extract_features(self, inputs):
        if self.features is not None:
            feats = self.features(inputs)
        else:
            feats = inputs
            
        visual_fc = self.classifier_v(feats)

        lcis_v = self.encoder_v(feats)
        bcis_v = self.sign(lcis_v)

        if self.param.use_semantic:
            lcis_s = self.encoder_t(feats)
            bcis_s = self.sign(lcis_s)
        else:
            lcis_s, bcis_s = None, None

        return lcis_v, lcis_s, bcis_v, bcis_s, visual_fc

    def num_groups(self):
        """Size of sub-modules."""
        return len(self._modules)

    def init_weights(self):
        """Initialize net weights with pre-trained model.

        Each sub-module should has its own same methods.
        """
        for model in self.children():
            if isinstance(model, nn.ModuleList):
                for m in model:
                    m.init_weights()
            else:
                model.init_weights()
