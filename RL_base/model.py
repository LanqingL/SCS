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
                 checkpoint_path="./pytorch_model.bin",
                 add_linear=False,
                 embedding_size=128,
                 freeze_encoder=True) -> None:
        super().__init__()
        print("model_config:", model_name)
        self.model = timm.create_model(model_name,
                          pretrained=False,
                          checkpoint_path=checkpoint_path)
        # self.model = timm.create_model(model_name, pretrained=True)

        # Freeze transformer encoder and only train the linear layer
        all_param = {}
        if freeze_encoder:
            for name, param in self.model.named_parameters():
                all_param[name] = param.shape
                # if 'norm' in name and 'block' in name and '22' in name:
                #     param.requires_grad = True
                # else:
                #     param.requires_grad = False
                param.requires_grad = False


        if add_linear:
            # Add an additional small, adjustable linear layer on top of BERT tuned through RL
            self.embedding_size = embedding_size
            # self.linear = nn.Sequential(
            #     nn.Linear(all_param['head.weight'][1], embedding_size * 4),
            #     nn.GELU(),
            #     nn.Linear(embedding_size * 4, embedding_size * 2),
            #     nn.GELU(),
            #     nn.Linear(embedding_size * 2, embedding_size)
            # )
            # self.linear = self._build_mlp(2, all_param['head.weight'][1]*2, all_param['head.weight'][1]*2, all_param['head.weight'][1]*2)
            # self.norm = LayerNorm(all_param['head.weight'][1])
            # self.linear = TransformerDecoderLayer(all_param['head.weight'][1], 8)
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

    def preprocess_store(self, imgs):

        img_features = self.model.forward_features(imgs)

        img_features = self.model.forward_head(img_features, pre_logits=True) # len(input_list) x hidden_size

        # if self.linear:
        #     img_features = self.linear(img_features)  # len(input_list) x embedding_size

        return img_features

    def forward(self, imgs):

        img_features = self.model.forward_features(imgs)

        img_features = self.model.forward_head(img_features, pre_logits=True) # len(input_list) x hidden_size

        # if self.linear:
        #     img_features = self.linear(img_features)  # len(input_list) x embedding_size

        return img_features


class Multi_layer(nn.Module):

    def __init__(self,
                 embedding_size=128,
                 freeze_encoder=True) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.projector = self._build_mlp(3, embedding_size, embedding_size, embedding_size)
        self.predictor = self._build_mlp(2, embedding_size, embedding_size, embedding_size)

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

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )

    def forward(self, imgs):

        img_features = self.model.forward_features(imgs)

        img_features = self.model.forward_head(img_features, pre_logits=True) # len(input_list) x hidden_size

        if self.linear:
            img_features = self.linear(img_features)  # len(input_list) x embedding_size

        return img_features



class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class Mul_linear_model(nn.Module):

    def __init__(self, input_dim=512, output_dim=512, device=None) -> None:
        super().__init__()
        self.q = nn.Linear(input_dim, output_dim)
        # self.q = nn.Sequential(
        #         nn.Linear(input_dim, output_dim),
        #         nn.GELU(),
        #         nn.Linear(output_dim, output_dim),
        #     )


    def forward(self, input_feature):
        output_feature = self.q(input_feature)
        return output_feature
