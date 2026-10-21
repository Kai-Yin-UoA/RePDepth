import torch
import torch.nn as nn
import numpy as np
from .blocks import FeatureFusionBlock, _make_scratch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from zoedepth.models.base_models.dpt_dinov2.MobileNetWithProjector import *
# from zoedepth.models.base_models.dpt_dinov2.mobilevit import mobilevit_xs, mobilevit_s, mobilevit_xxs
# from zoedepth.models.base_models.dpt_dinov2.model import MobileViT_XXS, MobileViT_XS,MobileViT_S

from zoedepth.models.base_models.dpt_dinov2.mobileVitv1 import MobileViT
from zoedepth.models.base_models.dpt_dinov2.mobileVitv3 import MobileViT as MobileViTV3

def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class MobileViTFeatureExtractor(nn.Module):
    """Return the four stage maps expected by the CNN DPT head."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        x = self.encoder.conv_1(x)
        x = self.encoder.layer_1(x)
        features = []
        for layer in (self.encoder.layer_2, self.encoder.layer_3,
                      self.encoder.layer_4, self.encoder.layer_5):
            x = layer(x)
            features.append(x)
        return features


class DPTHeadCNN(nn.Module):
    def __init__(self, in_channels, features=256, use_bn=False, out_channels=[64, 96, 160, 320], use_clstoken=False, shrink_factor = 8):
        super(DPTHeadCNN, self).__init__()
        self.use_clstoken = use_clstoken

        self.shrink_factor = shrink_factor

        self.projector1 = nn.Conv2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0]//self.shrink_factor,
                kernel_size=1,
                stride=1,
                padding=0,
        )
        
        self.projector2 = nn.Conv2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1]//self.shrink_factor,
                kernel_size=1,
                stride=1,
                padding=0,
        )
        
        self.projector3 = nn.Conv2d(
                in_channels=out_channels[2],
                out_channels=out_channels[2]//self.shrink_factor,
                kernel_size=1,
                stride=1,
                padding=0,
        )
        
        self.projector4 = nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3]//self.shrink_factor,
                kernel_size=1,
                stride=1,
                padding=0,
        )

        self.projects = nn.ModuleList([
            self.projector1,
            self.projector2,
            self.projector3,
            self.projector4
        ])


        self.resize_layers = nn.ModuleList([
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(
                in_channels=out_channels[2]//self.shrink_factor,
                out_channels=out_channels[2]//self.shrink_factor,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.Conv2d(
                in_channels=out_channels[3]//self.shrink_factor,
                out_channels=out_channels[3]//self.shrink_factor,
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            [x//self.shrink_factor for x in out_channels],
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if x.dim() == 4:
                # mobilenet 输出: [B, C, H, W]
                x_in = x
            elif x.dim() == 3:
                # vits/vitl 输出: [B, N, C]，需要 permute/reshape
                x_in = x#.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            elif x.dim() == 1:
                print(f"[Warning] Skip feature with shape {x.shape} at index {i}")
                continue
            else:
                print(f"[Warning] Unexpected feature shape: {x.shape} at index {i}, skip.")
                continue

            x_in = self.projects[i](x_in.cuda())
            x_in = self.resize_layers[i](x_in)
            out.append(x_in)
            
        if len(out) != 4:
            raise RuntimeError(f"DPTHead expects 4 valid feature maps, got {len(out)}. Check feature extraction.")
        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)#.unsqueeze(3)
        layer_2_rn = self.scratch.layer2_rn(layer_2)#.unsqueeze(3)
        layer_3_rn = self.scratch.layer3_rn(layer_3)#.unsqueeze(3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)#.unsqueeze(3)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        return out

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

class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=32, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False, patch_size=2):

        super(DPT_DINOv2, self).__init__()

        torch.manual_seed(1)

        import os
        dim = 64

        # The backbone comes from config_zoedepth.json via DepthAnythingCore.
        # Do not read config.json here: that made the configured choice silently
        # disagree with the feature channels used to build the DPT head.
        encoder_name = encoder.lower()
        if encoder_name in ('mobilenetv2', 'mobilenetv3'):
            if encoder_name == 'mobilenetv2':
                torchvision_model = models.mobilenet_v2(weights=None)
                self.pretrained = MobileNetV2WithProjector(torchvision_model.features)
            else:
                torchvision_model = models.mobilenet_v3_large(weights=None)
                self.pretrained = MobileNetV3WithProjector(torchvision_model.features)
            self.pretrained = self.pretrained.to('cuda')
        elif encoder_name in ('mobilevit', 'mobilevitv1_xs', 'mobilevitv1_s', 'mobilevitv1_xxs', 'mobilevitv3v2'):
            mode = {
                'mobilevit': 'xs',
                'mobilevitv1_xs': 'xs',
                'mobilevitv1_s': 's',
                'mobilevitv1_xxs': 'xxs',
                'mobilevitv3v2': 'xs',
            }[encoder_name]
            opts.model.classification.mit.mode = mode
            opts.model.classification.mit.head_dim = 16 if mode == 'xxs' else 24
            mobilevit_constructor = MobileViTV3 if encoder_name == 'mobilevitv3v2' else MobileViT
            mobilevit_encoder = mobilevit_constructor(opts)
            self.pretrained = MobileViTFeatureExtractor(mobilevit_encoder)

            checkpoint_name = {
                'mobilevit': 'mobilevit_xs.pt',
                'mobilevitv1_xs': 'mobilevit_xs.pt',
                'mobilevitv1_s': 'mobilevit_s.pt',
                'mobilevitv1_xxs': 'mobilevit_xxs.pt',
                'mobilevitv3v2': 'mobilevit_xs.pt',
            }[encoder_name]
            if encoder_name != 'mobilevitv3v2':
                checkpoint_path = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), '../../../checkpoints', checkpoint_name))
                if os.path.isfile(checkpoint_path):
                    state_dict = torch.load(checkpoint_path, map_location='cpu')
                    self.pretrained.load_state_dict(state_dict)
            self.pretrained = self.pretrained.to('cuda')
        # elif ENCODER == 'mobilevitv1_s':
        #     self.pretrained = MobileViT_S() 
        #     all_params = []
        #     for p in self.pretrained.parameters():
        #         all_params.append(p.detach().cpu().numpy().ravel())
        #     all_params = np.concatenate(all_params)
        #     print(f"[MobileViT raw] mean: {all_params.mean():.6f}, std: {all_params.std():.6f}")

        #     state_dict = torch.load("./zoedepth/checkpoints/mobilevitv1_s.pt", map_location='cpu')['state_dict']
        #     # print("mobilevit_xs pretrained weights loaded")
        #     for key in list(state_dict.keys()):
        #         state_dict[key.replace('module.', '')] = state_dict.pop(key)
        #     # After loading MobileViT weights
        #     self.pretrained.load_state_dict(state_dict)
        #     self.pretrained = MobileViTv1SWithProjector(self.pretrained).to('cuda')
        #     # Print mean/std of MobileViTWithProjector encoder parameters
        #     all_enc_params = []
        #     for p in self.pretrained.encoder.parameters():
        #         all_enc_params.append(p.detach().cpu().numpy().ravel())
        #     all_enc_params = np.concatenate(all_enc_params)
        #     print(f"[MobileViTWithProjector encoder] mean: {all_enc_params.mean():.6f}, std: {all_enc_params.std():.6f}")

        if encoder_name in ['mobilenetv2', 'mobilenetv3', 'mobilevitv1_s', 'mobilevitv1_xs', 'mobilevit', 'mobilevitv3v2']:
            #self.depth_head = DPTHead(dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
            self.depth_head = DPTHeadCNN(dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken).cuda()
        elif encoder_name == 'mobilevitv1_xxs':
            self.depth_head = DPTHeadCNN(dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, shrink_factor=4).cuda()
        else:
            raise ValueError(f"Unsupported encoder {encoder!r} in DPT_DINOv2")

        self.patch_size = patch_size

        from zoedepth.utils.misc import count_parameters, colors
        param_count = count_parameters(self.pretrained, include_all=True)
        print(f"{colors.fg.red}pretrained parameters: {param_count/1e6:.2f}M{colors.reset}")
        param_count = count_parameters(self.depth_head, include_all=True)
        print(f"{colors.fg.red}depth_head parameters: {param_count/1e6:.2f}M{colors.reset}")



    def forward(self, x):
        # print("TinyDA Input shape:", x.shape)
        h, w = x.shape[-2:]

        features = self.pretrained(x)

        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1)
