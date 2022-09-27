# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import math

import torch.nn.functional as F

from fairseq.data.data_utils import compute_mask_indices
from fairseq.distributed import fsdp_wrap
from fairseq.models.wav2vec.wav2vec2 import (
    TransformerSentenceEncoderLayer,ConformerWav2Vec2EncoderLayer
)
from fairseq.utils import index_put
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    LayerNorm,
    SamePad,
    TransposeLast,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params


def pad_to_multiple(x, multiple, dim=-1, value=0):
    # Inspired from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py#L41
    if x is None:
        return None, 0
    tsz = x.size(dim)
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if m.is_integer():
        return x, 0
    pad_offset = (0,) * (-1 - dim) * 2

    return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder

def make_conv_pos(e, k, g):
    pos_conv = nn.Conv1d(
        e,
        e,
        kernel_size=k,
        padding=k // 2,
        groups=g,
    )
    dropout = 0
    std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
    nn.init.normal_(pos_conv.weight, mean=0, std=std)
    nn.init.constant_(pos_conv.bias, 0)

    pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
    pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.GELU())

    return pos_conv


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            
            x = conv(x)
        

        return x

class TransformerEncoder(nn.Module):
    def build_encoder_layer(self, args):
        if args.layer_type == "transformer":
            layer = TransformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=self.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                activation_fn=args.activation_fn,
                layer_norm_first=args.layer_norm_first,
            )
        elif args.layer_type == "conformer":
            layer = ConformerWav2Vec2EncoderLayer(
                embed_dim=self.embedding_dim,
                ffn_embed_dim=args.encoder_ffn_embed_dim,
                attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
                activation_fn="swish",
                attn_type=args.attn_type,
                use_fp16=args.fp16,
                pos_enc_type="abs",
            )
        layer = fsdp_wrap(layer)
        # if args.checkpoint_activations:
        #     layer = checkpoint_wrapper(layer)
        return layer

    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.required_seq_len_multiple = args.required_seq_len_multiple

        pos_conv_depth = getattr(args, "pos_conv_depth", 1)
        if pos_conv_depth > 1:
            num_layers = args.pos_conv_depth
            k = max(3, args.conv_pos // num_layers)

            def make_conv_block(e, k, g, l):
                return nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv1d(
                                e,
                                e,
                                kernel_size=k,
                                padding=k // 2,
                                groups=g,
                            ),
                            SamePad(k),
                            TransposeLast(),
                            LayerNorm(e, elementwise_affine=False),
                            TransposeLast(),
                            nn.GELU(),
                        )
                        for _ in range(l)
                    ]
                )

            self.pos_conv = make_conv_block(
                self.embedding_dim, k, args.conv_pos_groups, num_layers
            )

        else:
            self.pos_conv = make_conv_pos(
                self.embedding_dim,
                args.conv_pos,
                args.conv_pos_groups,
            )

        self.layers = nn.ModuleList(
            [self.build_encoder_layer(args) for _ in range(args.encoder_layers)]
        )
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None):
        
        x, layer_results = self.extract_features(x, padding_mask, layer)
        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(
        self,
        x,
        padding_mask=None,
        tgt_layer=None,
        min_layer=0,
    ):

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)
        # pad to the sequence length dimension
        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0
        )
        if pad_length > 0 and padding_mask is None:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_to_multiple(
                padding_mask, self.required_seq_len_multiple, dim=-1, value=True
            )
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random() if self.layerdrop > 0 else 1
            if not self.training or (dropout_probability > self.layerdrop):
                x, (z, lr) = layer(
                    x, self_attn_padding_mask=padding_mask, need_weights=False
                )
                if i >= min_layer:
                    layer_results.append((x, z, lr))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # undo paddding
        if pad_length > 0:
            x = x[:, :-pad_length]

            def undo_pad(a, b, c):
                return (
                    a[:-pad_length],
                    b[:-pad_length] if b is not None else b,
                    c[:-pad_length],
                )

            layer_results = [undo_pad(*u) for u in layer_results]

        return x, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class HubertModel(nn.Module):
    def __init__(
        self,
        args
    ) -> None:
        super().__init__()
        # logger.info(f"HubertModel Config: {cfg}")

        feature_enc_layers = eval(args.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode='default',
            conv_bias=False
        )
        # feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        # self.feat2tar_ratio = args.label_rate * feature_ds_rate / args.sample_rate

        self.post_extract_proj = (
            nn.Linear(self.embed, args.encoder_embed_dim)
            if self.embed != args.encoder_embed_dim
            else None
        )

        self.mask_prob = args.mask_prob
        self.mask_selection = args.mask_selection
        self.mask_other = args.mask_other
        self.mask_length = args.mask_length
        self.no_mask_overlap = args.no_mask_overlap
        self.mask_min_space = args.mask_min_space

        self.mask_channel_prob = args.mask_channel_prob
        self.mask_channel_selection = args.mask_channel_selection
        self.mask_channel_other = args.mask_channel_other
        self.mask_channel_length = args.mask_channel_length
        self.no_mask_channel_overlap = args.no_mask_channel_overlap
        self.mask_channel_min_space = args.mask_channel_min_space

        self.dropout_input = nn.Dropout(args.dropout_input)
        # self.dropout_features = nn.Dropout(args.dropout_features)

        self.feature_grad_mult = args.feature_grad_mult
        # self.logit_temp = cfg.logit_temp
        # self.skip_masked = cfg.skip_masked
        # self.skip_nomask = cfg.skip_nomask

        # final_dim = args.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(args.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(args)
        self.layer_norm = LayerNorm(self.embed)

        # self.target_glu = None
        # if cfg.target_glu:
        #     self.target_glu = nn.Sequential(
        #         nn.Linear(final_dim, final_dim * 2), nn.GLU()
        #     )

        # self.untie_final_proj = cfg.untie_final_proj
        # if self.untie_final_proj:
        #     self.final_proj = nn.Linear(
        #         cfg.encoder_embed_dim, final_dim * len(dictionaries)
        #     )
        # else:
        #     self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        # modules below are not needed during fine-tuning
        # if any([d is None for d in dictionaries]):
        #     logger.info("cannot find dictionary. assume will be used for fine-tuning")
        # else:
        #     self.num_classes = [len(d) for d in dictionaries]
        #     self.label_embs_concat = nn.Parameter(
        #         torch.FloatTensor(sum(self.num_classes), final_dim)
        #     )
        #     nn.init.uniform_(self.label_embs_concat)

    # def upgrade_state_dict_named(self, state_dict, name):
    #     """Upgrade a (possibly old) state dict for new versions of fairseq."""

    #     super().upgrade_state_dict_named(state_dict, name)
    #     return state_dict

    # @classmethod
    # def build_model(cls, cfg: HubertConfig, task: HubertPretrainingTask):
    #     """Build a new model instance."""

    #     model = HubertModel(cfg, task.cfg, task.dictionaries)
    #     return model

    def apply_mask(self, x, padding_mask, target_list):
        B, T, C = x.shape
        # x_dup = x.clone().detach()
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    # def compute_nce(self, x, pos, negs):
    #     neg_is_pos = (pos == negs).all(-1)
    #     pos = pos.unsqueeze(0)
    #     targets = torch.cat([pos, negs], dim=0)

    #     logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
    #     logits /= self.logit_temp
    #     if neg_is_pos.any():
    #         logits[1:][neg_is_pos] = float("-inf")
    #     logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
    #     return logits

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features

    def forward_targets(
        self,
        features: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, target_list

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        features = self.forward_features(source)
        
        # if target_list is not None:
        #     features, target_list = self.forward_targets(features, target_list)

        # features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        # unmasked_features = features.clone().detach()

        padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        # unmasked_features = self.dropout_features(unmasked_features)


        x, mask_indices = self.apply_mask(features, padding_mask, target_list)


        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool

        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None
        )
        
        return x, padding_mask, features

        # def compute_pred(proj_x, target, label_embs):
        #     # compute logits for the i-th label set
        #     y = torch.index_select(label_embs, 0, target.long())
        #     negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
        #     if self.target_glu:
        #         y = self.target_glu(y)
        #         negs = self.target_glu(negs)
        #     # proj_x: (S, D)
        #     # y: (S, D)
        #     # negs: (Neg, S, D)
        #     return self.compute_nce(proj_x, y, negs)

        # label_embs_list = self.label_embs_concat.split(self.num_classes, 0)

        # if not self.skip_masked:
        #     masked_indices = torch.logical_and(~padding_mask, mask_indices)
        #     proj_x_m = self.final_proj(x[masked_indices])
        #     if self.untie_final_proj:
        #         proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
        #     else:
        #         proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
        #     logit_m_list = [
        #         compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
        #         for i, (proj_x_m, t) in enumerate(zip(proj_x_m_list, target_list))
        #     ]
        # else:
        #     logit_m_list = [None for _ in target_list]

        # if not self.skip_nomask:
        #     nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
        #     proj_x_u = self.final_proj(x[nomask_indices])
        #     if self.untie_final_proj:
        #         proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
        #     else:
        #         proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

        #     logit_u_list = [
        #         compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
        #         for i, (proj_x_u, t) in enumerate(zip(proj_x_u_list, target_list))
        #     ]
        # else:
        #     logit_u_list = [None for _ in target_list]

        # result = {
        #     "logit_m_list": logit_m_list,
        #     "logit_u_list": logit_u_list,
        #     "padding_mask": padding_mask,
        #     "features_pen": features_pen,
        # }
        # return result

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    # def get_logits(self, net_output, is_masked=True):
    #     if is_masked:
    #         logits_list = net_output["logit_m_list"]
    #     else:
    #         logits_list = net_output["logit_u_list"]
    #     logits_list = [x.float() for x in logits_list if x is not None]
    #     return logits_list

    # def get_targets(self, net_output, is_masked=True):
    #     logits_list = self.get_logits(net_output, is_masked)
    #     targets_list = [x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list]
    #     return targets_list

    # def get_extra_losses(self, net_output):
    #     extra_losses = []
    #     names = []

    #     if "features_pen" in net_output:
    #         extra_losses.append(net_output["features_pen"])
    #         names.append("features_pen")

    #     return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None
