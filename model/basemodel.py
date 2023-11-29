import math
from icecream import ic
import snoop

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import config as cfg


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if m.weight is not None:
            m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()


class LatentCode(nn.Module):
    """Basic class for learning latent code."""

    def __init__(self, param):
        """Latent code.

        Parameters:
        -----------
        See utils.param.NetParam
        """
        super().__init__()
        self.param = param
        self.register_buffer("scale", torch.ones(1))

    def set_scale(self, value):
        """Set the scale of tanh layer."""
        self.scale.fill_(value)

    def feat(self, x):
        """Compute the feature of all images."""
        raise NotImplementedError

    def forward(self, x):
        """Forward a feature from DeepContent."""
        x = self.feat(x)
        if self.param.without_binary:
            return x
        if self.param.scale_tanh:
            x = torch.mul(x, self.scale)
        if self.param.binary01:
            return 0.5 * (torch.tanh(x) + 1)
        # shape N x D
        return torch.tanh(x).view(-1, self.param.dim)


class CrossAttention(nn.Module):
    def __init__(
            self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    @snoop
    def forward(self, x, y):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)
        return output


class ImgEncoder(LatentCode):
    """Module for encode to learn the latent code."""

    def __init__(self, in_feature, param):
        """Initialize an encoder.

        Parameter
        ---------
        in_feature: feature dimension for image features
        param: see utils.param.NetParam for details

        """
        super().__init__(param)
        half = in_feature // 2
        self.encoder = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_feature, half),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(half, param.dim, bias=False),
        )

    def feat(self, x):
        return self.encoder(x)

    def init_weights(self):
        """Initialize weights for encoder with pre-trained model."""
        nn.init.normal_(self.encoder[1].weight.data, std=0.01)
        nn.init.constant_(self.encoder[1].bias.data, 0)
        nn.init.normal_(self.encoder[-1].weight.data, std=0.01)


class TxtEncoder(LatentCode):
    def __init__(self, in_feature, param):
        super().__init__(param)
        self.encoder = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, param.dim, bias=False),
        )

    def feat(self, x):
        return self.encoder(x)

    def init_weights(self):
        """Initialize weights for encoder with pre-trained model."""
        nn.init.normal_(self.encoder[0].weight.data, std=0.01)
        nn.init.constant_(self.encoder[0].bias.data, 0)
        nn.init.normal_(self.encoder[-1].weight.data, std=0.01)


class ImgClassifier(nn.Module):
    """Module for classification for visual features (image)"""

    def __init__(self, in_feature, num_classes):
        """Initialize an visual classification

        Parameter:
        ----------
        in_feature: feature_dimention for image features
        num_classes: Number of classes for classification

        """
        super().__init__()
        ##TODO: Change output_dim if in_feature is low
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_feature, in_feature // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_feature // 2, in_feature // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_feature // 4, in_feature // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_feature // 8, num_classes),
        )

    def forward(self, x):
        return self.fc(x)

    def init_weights(
        self,
    ):
        """Initialize weights for visual classifier with pre-trained model."""
        for name, param in self.named_parameters():
            if "weight" in name and param.requires_grad:
                nn.init.normal_(param.data, std=0.01)
            elif "bias" in name and param.requires_grad:
                nn.init.constant_(
                    param.data, 0
                )  ##TODO: Do we need zero for last bias?


class TxtClassifier(nn.Module):
    """Module for classification for visual features (image)"""

    def __init__(self, in_feature, num_classes):
        """Initialize an visual classification

        Parameter:
        ----------
        in_feature: feature_dimention for image features
        num_classes: Number of classes for classification

        """
        super().__init__()
        ##TODO: Change output_dim if in_feature is low
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_feature, in_feature // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_feature // 2, in_feature // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_feature // 4, in_feature // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_feature // 8, num_classes),
        )

    def forward(self, x):
        return self.fc(x)

    def init_weights(
        self,
    ):
        """Initialize weights for visual classifier with pre-trained model."""
        for name, param in self.named_parameters():
            if "weight" in name and param.requires_grad:
                nn.init.normal_(param.data, std=0.01)
            elif "bias" in name and param.requires_grad:
                nn.init.constant_(
                    param.data, 0
                )  ##TODO: Do we need zero for last bias?


class CoreMat(nn.Module):
    """Weighted hamming similarity."""

    def __init__(self, dim, weight=1.0):
        """Weights for this layer that is drawn from N(mu, std)."""
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.init_weights(weight)

    def init_weights(self, weight):
        """Initialize weights."""
        self.weight.data.fill_(weight)

    def forward(self, x):
        """Forward."""
        return torch.mul(x, self.weight)

    def __repr__(self):
        """Format string for module CoreMat."""
        return self.__class__.__name__ + "(dim=" + str(self.dim) + ")"


class LearnableScale(nn.Module):
    def __init__(self, init=1.0):
        super(LearnableScale, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1).fill_(init))

    def forward(self, inputs):
        return self.weight * inputs

    def init_weights(self):
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"
