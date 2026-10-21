
# mobilevit_single.py
# Standalone MobileViT (v1-style) implementation whose state_dict key naming
# matches the common "block/exp_1x1/conv_3x3/red_1x1" + "local_rep/global_rep/conv_proj/fusion"
# checkpoints (e.g., Apple's MobileViT/CVNets naming convention).

from __future__ import annotations
import math
import argparse
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utils
# -------------------------
def getattr_nested(obj: Any, key: str, default: Any = None) -> Any:
    """
    Safe nested getattr for "a.b.c" style keys.
    Works with simple objects that store attributes.
    """
    cur = obj
    for k in key.split("."):
        if cur is None or not hasattr(cur, k):
            return default
        cur = getattr(cur, k)
    return cur


class logger:
    @staticmethod
    def error(msg: str):
        raise RuntimeError(msg)


# -------------------------
# Building blocks
# -------------------------
class ConvLayer2d(nn.Module):
    """
    Conv-Norm-Act packed under `.block` to match checkpoint keys:
    *.block.conv.weight, *.block.norm.weight, *.block.act (no params)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        use_norm: bool = True,
        use_act: bool = True,
        dilation: int = 1,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=(not use_norm) if bias is None else bias,
        )
        norm = nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()
        act = nn.SiLU(inplace=True) if use_act else nn.Identity()
        self.block = nn.Sequential(OrderedDict([
            ("conv", conv),
            ("norm", norm),
            ("act", act),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):
    """
    MBConv-like block with keys:
    layer_i.j.block.exp_1x1.block.{conv,norm}.*
    layer_i.j.block.conv_3x3.block.{conv,norm}.*
    layer_i.j.block.red_1x1.block.{conv,norm}.*
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        dilation: int = 1,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(round(in_channels * expand_ratio))

        block = OrderedDict()

        # expansion (point-wise)
        if expand_ratio != 1:
            block["exp_1x1"] = ConvLayer2d(
                in_channels, hidden_dim, kernel_size=1, stride=1, groups=1, use_norm=True, use_act=True
            )
            dw_in = hidden_dim
        else:
            # keep the name exp_1x1 OUT of the graph if not present in ckpt
            dw_in = in_channels

        # depth-wise
        block["conv_3x3"] = ConvLayer2d(
            dw_in,
            dw_in,
            kernel_size=3,
            stride=stride,
            groups=dw_in,
            use_norm=True,
            use_act=True,
            dilation=dilation,
        )

        # projection (point-wise), usually conv+bn w/o activation
        block["red_1x1"] = ConvLayer2d(
            dw_in, out_channels, kernel_size=1, stride=1, groups=1, use_norm=True, use_act=False
        )

        self.block = nn.Sequential(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        # exp may or may not exist
        if "exp_1x1" in self.block._modules:
            out = self.block.exp_1x1(out)
        out = self.block.conv_3x3(out)
        out = self.block.red_1x1(out)
        if self.use_res_connect:
            out = out + x
        return out


class MultiHeadSelfAttention(nn.Module):
    """
    Minimal MHSA with qkv_proj / out_proj naming.
    Keys:
      *.qkv_proj.weight/bias
      *.out_proj.weight/bias
    """
    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            logger.error(f"Transformer input dimension should be divisible by number of heads. Got {dim} and {num_heads}.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv_proj(x)  # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Module):
    """
    One block with:
      pre_norm_mha = Sequential(LayerNorm, MHSA)
      pre_norm_ffn = Sequential(LayerNorm, Linear, Act, Dropout, Linear)
    This matches keys shown in your ckpt error, e.g.:
      global_rep.0.pre_norm_mha.0.weight/bias
      global_rep.0.pre_norm_mha.1.qkv_proj.*
      global_rep.0.pre_norm_ffn.0.weight/bias
      global_rep.0.pre_norm_ffn.1.weight/bias
      global_rep.0.pre_norm_ffn.4.weight/bias
    """
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float, attn_dropout: float, ffn_dropout: float):
        super().__init__()
        self.pre_norm_mha = nn.Sequential(OrderedDict([
            ("0", nn.LayerNorm(dim)),
            ("1", MultiHeadSelfAttention(dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout)),
        ]))
        self.pre_norm_ffn = nn.Sequential(OrderedDict([
            ("0", nn.LayerNorm(dim)),
            ("1", nn.Linear(dim, ffn_dim)),
            ("2", nn.SiLU(inplace=True)),
            ("3", nn.Dropout(ffn_dropout)),
            ("4", nn.Linear(ffn_dim, dim)),
        ]))
        self.drop_path = nn.Identity()  # placeholder if you want stochastic depth later
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        x = x + self.dropout(self.pre_norm_mha(x))
        x = x + self.dropout(self.pre_norm_ffn(x))
        return x


class MobileViTBlock(nn.Module):
    """
    Keys match:
      layer_?.?.local_rep.conv_3x3.block.*
      layer_?.?.local_rep.conv_1x1.block.conv.weight (no BN params if use_norm=False)
      layer_?.?.global_rep.<i>.pre_norm_mha...
      layer_?.?.global_rep.<i>.pre_norm_ffn...
      layer_?.?.global_rep.<last>.weight/bias   (final LayerNorm)
      layer_?.?.conv_proj.block.*
      layer_?.?.fusion.block.*
    """
    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: int,
        patch_h: int,
        patch_w: int,
        num_heads: int,
        dropout: float,
        ffn_dropout: float,
        attn_dropout: float,
    ) -> None:
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w

        # local representation
        self.local_rep = nn.Sequential(OrderedDict([
            ("conv_3x3", ConvLayer2d(in_channels, in_channels, kernel_size=3, stride=1, groups=1, use_norm=True, use_act=True)),
            # conv_1x1 without BN to match "conv_1x1.block.conv.weight" only
            ("conv_1x1", ConvLayer2d(in_channels, transformer_dim, kernel_size=1, stride=1, groups=1, use_norm=False, use_act=False, bias=False)),
        ]))

        # global representation (transformer)
        blocks = []
        for i in range(n_transformer_blocks):
            blocks.append(TransformerBlock(
                dim=transformer_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout,
            ))
        blocks.append(nn.LayerNorm(transformer_dim))  # produces global_rep.<last>.weight/bias
        self.global_rep = nn.Sequential(*blocks)

        self.conv_proj = ConvLayer2d(transformer_dim, in_channels, kernel_size=1, stride=1, use_norm=True, use_act=True)
        self.fusion = ConvLayer2d(2 * in_channels, in_channels, kernel_size=3, stride=1, use_norm=True, use_act=True)

    def _unfold(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # x: (B, C, H, W) -> (B, N, C)
        B, C, H, W = x.shape
        ph, pw = self.patch_h, self.patch_w
        if H % ph != 0 or W % pw != 0:
            # pad to multiple of patch size (keeps runtime stable)
            new_h = int(math.ceil(H / ph) * ph)
            new_w = int(math.ceil(W / pw) * pw)
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            H, W = new_h, new_w

        nh, nw = H // ph, W // pw
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)  # (B, C, nh, nw, ph, pw)
        x = x.contiguous().permute(0, 2, 3, 4, 5, 1).contiguous()
        x = x.view(B, nh * nw * ph * pw, C)  # (B, N, C)
        return x, H, W

    def _fold(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, N, C) -> (B, C, H, W)
        B, N, C = x.shape
        ph, pw = self.patch_h, self.patch_w
        nh, nw = H // ph, W // pw
        expected = nh * nw * ph * pw
        if N != expected:
            logger.error(f"Token count mismatch in fold: got {N}, expected {expected}.")

        x = x.view(B, nh, nw, ph, pw, C).permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, H, W)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.local_rep(x)  # (B, transformer_dim, H, W) after conv_1x1
        x, H, W = self._unfold(x)   # (B, N, transformer_dim)
        x = self.global_rep(x)      # (B, N, transformer_dim)
        x = self._fold(x, H, W)     # (B, transformer_dim, H, W)
        x = self.conv_proj(x)       # back to in_channels
        x = torch.cat((x, res), dim=1)
        x = self.fusion(x)
        return x


# -------------------------
# Model definition
# -------------------------
@dataclass
class LayerCfg:
    out_channels: int
    expand_ratio: float
    num_blocks: int
    stride: int
    block_type: str  # "mv2" or "mobilevit"
    transformer_dim: int = 0
    ffn_dim: int = 0
    transformer_blocks: int = 0
    patch_h: int = 2
    patch_w: int = 2


def get_mobilevit_config(opts: Any) -> Dict[str, Any]:
    """Return MobileViT-v1 style configuration.

    Supports: xx_small, x_small, small.
    Uses LayerCfg objects for stage configs and an int 'last_exp' for the final 1x1 expansion.
    """
    mode = getattr_nested(opts, "model.classification.mit.mode", "small")
    mode = str(mode).lower()

    if mode == "xxs":
        # MobileViT-XXS (v1)
        mv2_exp = 2.0
        cfg = {
            "layer1": LayerCfg(out_channels=16, expand_ratio=mv2_exp, num_blocks=1, stride=1, block_type="mv2"),
            "layer2": LayerCfg(out_channels=24, expand_ratio=mv2_exp, num_blocks=3, stride=2, block_type="mv2"),
            "layer3": LayerCfg(out_channels=48, expand_ratio=mv2_exp, num_blocks=1, stride=2, block_type="mobilevit",
                               transformer_dim=64, ffn_dim=128, transformer_blocks=2, patch_h=2, patch_w=2),
            "layer4": LayerCfg(out_channels=64, expand_ratio=mv2_exp, num_blocks=1, stride=2, block_type="mobilevit",
                               transformer_dim=80, ffn_dim=160, transformer_blocks=4, patch_h=2, patch_w=2),
            "layer5": LayerCfg(out_channels=80, expand_ratio=mv2_exp, num_blocks=1, stride=2, block_type="mobilevit",
                               transformer_dim=96, ffn_dim=192, transformer_blocks=3, patch_h=2, patch_w=2),
            "last_exp": 320,
        }
        return cfg

    if mode == "xs":
        # MobileViT-XS (v1)
        mv2_exp = 4.0
        cfg = {
            "layer1": LayerCfg(out_channels=32, expand_ratio=mv2_exp, num_blocks=1, stride=1, block_type="mv2"),
            "layer2": LayerCfg(out_channels=48, expand_ratio=mv2_exp, num_blocks=3, stride=2, block_type="mv2"),
            "layer3": LayerCfg(out_channels=64, expand_ratio=mv2_exp, num_blocks=1, stride=2, block_type="mobilevit",
                               transformer_dim=96, ffn_dim=192, transformer_blocks=2, patch_h=2, patch_w=2),
            "layer4": LayerCfg(out_channels=80, expand_ratio=mv2_exp, num_blocks=1, stride=2, block_type="mobilevit",
                               transformer_dim=120, ffn_dim=240, transformer_blocks=4, patch_h=2, patch_w=2),
            "layer5": LayerCfg(out_channels=96, expand_ratio=mv2_exp, num_blocks=1, stride=2, block_type="mobilevit",
                               transformer_dim=144, ffn_dim=288, transformer_blocks=3, patch_h=2, patch_w=2),
            "last_exp": 384,
        }
        return cfg

    if mode == "s":
        # MobileViT-S (v1)
        mv2_exp = 4.0
        cfg = {
            "layer1": LayerCfg(out_channels=32, expand_ratio=mv2_exp, num_blocks=1, stride=1, block_type="mv2"),
            "layer2": LayerCfg(out_channels=64, expand_ratio=mv2_exp, num_blocks=3, stride=2, block_type="mv2"),
            "layer3": LayerCfg(out_channels=96, expand_ratio=mv2_exp, num_blocks=1, stride=2, block_type="mobilevit",
                               transformer_dim=144, ffn_dim=288, transformer_blocks=2, patch_h=2, patch_w=2),
            "layer4": LayerCfg(out_channels=128, expand_ratio=mv2_exp, num_blocks=1, stride=2, block_type="mobilevit",
                               transformer_dim=192, ffn_dim=384, transformer_blocks=4, patch_h=2, patch_w=2),
            "layer5": LayerCfg(out_channels=160, expand_ratio=mv2_exp, num_blocks=1, stride=2, block_type="mobilevit",
                               transformer_dim=240, ffn_dim=480, transformer_blocks=3, patch_h=2, patch_w=2),
            "last_exp": 640,
        }
        return cfg

    logger.error(f"Unsupported mode={mode}. Choose from: xxs, xs, s")



class MobileViT(nn.Module):
    def __init__(self, opts: Any) -> None:
        super().__init__()
        cfg = get_mobilevit_config(opts)

        self.num_classes = getattr_nested(opts, "model.classification.n_classes", 1000)

        # stem
        self.conv_1 = ConvLayer2d(3, 16, kernel_size=3, stride=2, use_norm=True, use_act=True)

        in_channels = 16
        self.layer_1, in_channels = self._make_layer(opts, in_channels, cfg["layer1"])
        self.layer_2, in_channels = self._make_layer(opts, in_channels, cfg["layer2"])
        self.layer_3, in_channels = self._make_layer(opts, in_channels, cfg["layer3"])
        self.layer_4, in_channels = self._make_layer(opts, in_channels, cfg["layer4"])
        self.layer_5, in_channels = self._make_layer(opts, in_channels, cfg["layer5"])

        self.conv_1x1_exp = ConvLayer2d(in_channels, cfg["last_exp"], kernel_size=1, stride=1, use_norm=True, use_act=True)
        in_channels = cfg["last_exp"]

        pool_type = getattr_nested(opts, "model.layer.global_pool", "mean")
        self.pool_type = pool_type

        # classifier naming to match ckpt: classifier.fc.weight/bias
        self.classifier = nn.Module()
        self.classifier.fc = nn.Linear(in_channels, self.num_classes)

    def _make_layer(self, opts: Any, in_channels: int, layer_cfg: LayerCfg) -> Tuple[nn.Sequential, int]:
        blocks: List[nn.Module] = []

        if layer_cfg.block_type == "mv2":
            # first block may have stride, others stride=1
            for i in range(layer_cfg.num_blocks):
                stride = layer_cfg.stride if i == 0 else 1
                blocks.append(InvertedResidual(
                    in_channels=in_channels,
                    out_channels=layer_cfg.out_channels,
                    stride=stride,
                    expand_ratio=layer_cfg.expand_ratio,
                    dilation=1
                ))
                in_channels = layer_cfg.out_channels
            return nn.Sequential(*blocks), in_channels

        if layer_cfg.block_type == "mobilevit":
            # One mv2 downsample + one MobileViTBlock (as in typical MobileViT v1)
            blocks.append(InvertedResidual(
                in_channels=in_channels,
                out_channels=layer_cfg.out_channels,
                stride=layer_cfg.stride,
                expand_ratio=layer_cfg.expand_ratio,
                dilation=1
            ))
            in_channels = layer_cfg.out_channels

            head_dim = getattr_nested(opts, "model.classification.mit.head_dim", 16)
            if layer_cfg.transformer_dim % head_dim != 0:
                logger.error(
                    f"Transformer input dimension should be divisible by head dimension. "
                    f"Got {layer_cfg.transformer_dim} and {head_dim}."
                )
            num_heads = layer_cfg.transformer_dim // head_dim

            dropout = getattr_nested(opts, "model.classification.mit.dropout", 0.0)
            ffn_dropout = getattr_nested(opts, "model.classification.mit.ffn_dropout", 0.0)
            attn_dropout = getattr_nested(opts, "model.classification.mit.attn_dropout", 0.0)

            blocks.append(MobileViTBlock(
                in_channels=in_channels,
                transformer_dim=layer_cfg.transformer_dim,
                ffn_dim=layer_cfg.ffn_dim,
                n_transformer_blocks=layer_cfg.transformer_blocks,
                patch_h=layer_cfg.patch_h,
                patch_w=layer_cfg.patch_w,
                num_heads=num_heads,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                attn_dropout=attn_dropout,
            ))
            return nn.Sequential(*blocks), in_channels

        logger.error(f"Unknown block_type: {layer_cfg.block_type}")
        return nn.Sequential(), in_channels  # unreachable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)

        if self.pool_type == "mean":
            x = x.mean(dim=(2, 3))
        elif self.pool_type == "max":
            x = x.amax(dim=(2, 3))
        else:
            # default mean
            x = x.mean(dim=(2, 3))

        x = self.classifier.fc(x)
        return x


# -------------------------
# CLI / quick test
# -------------------------
def load_pretrained_strict(model: nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    # remove DDP prefix if any
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="", help="Path to pretrained checkpoint (.pth/.pt)")
    parser.add_argument("--img_size", type=int, default=384)
    args = parser.parse_args()

    # Minimal opts container
    class DummyOpts:
        def __init__(self):
            self.model = type("model", (), {})()
            self.model.classification = type("classification", (), {})()
            self.model.classification.n_classes = 1000
            self.model.layer = type("layer", (), {})()
            self.model.layer.global_pool = "mean"
            self.model.classification.mit = type("mit", (), {})()
            self.model.classification.mit.mode = "s"
            # head_dim must divide transformer_dim for each stage
            if self.model.classification.mit.mode == "xxs":
                head_dim = 16
            else:  # x_small / small
                head_dim = 24
            self.model.classification.mit.head_dim = head_dim
            self.model.classification.mit.dropout = 0.0
            self.model.classification.mit.ffn_dropout = 0.0
            self.model.classification.mit.attn_dropout = 0.0

    opts = DummyOpts()
    model = MobileViT(opts)

    if args.pretrained:
        load_pretrained_strict(model, args.pretrained)
        print("strict=True checkpoint loaded.")

    x = torch.randn(1, 3, args.img_size, args.img_size)
    y = model(x)
    print("Output:", y.shape)
