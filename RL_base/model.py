import timm
import copy
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Any

from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.activation import MultiheadAttention

class policy_network(nn.Module):

    def __init__(self,
                 model_name="vit_large_patch14_clip_224.laion2b_ft_in12k_in1k",
                 add_linear=False,
                 embedding_size=128,
                 freeze_encoder=True) -> None:
        super().__init__()
        print("model_config:", model_name)
        self.model = timm.create_model(model_name, pretrained=True)

        # Freeze transformer encoder and only train the linear layer
        all_param = {}
        if freeze_encoder:
            for name, param in self.model.named_parameters():
                all_param[name] = param.shape
                param.requires_grad = False


        if add_linear:
            # Add an additional small, adjustable linear layer on top of BERT tuned through RL
            self.embedding_size = embedding_size
            self.projector = self._build_mlp(3, all_param['head.weight'][1], all_param['head.weight'][1], all_param['head.weight'][1])
            self.predictor = self._build_mlp(2, all_param['head.weight'][1], all_param['head.weight'][1], all_param['head.weight'][1])
        else:
            self.linear = None

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def forward(self, imgs):

        img_features = self.model.forward_features(imgs)
        img_features = self.model.forward_head(img_features, pre_logits=True) # len(input_list) x hidden_size

        return img_features

