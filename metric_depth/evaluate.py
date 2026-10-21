# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import argparse
from pprint import pprint
import cv2

import torch
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm

from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,
                        count_parameters)
import numpy as np
from torch import nn
from thop import profile
from thop.vision.basic_hooks import *
from fvcore.nn import FlopCountAnalysis, parameter_count

@torch.no_grad()
def infer(model, images, **kwargs):
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    # print("1", images.shape)
    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)

    # print(torch.flip(images, [3]).shape)
    pred2 = model(torch.flip(images, [3]), **kwargs)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    mean_pred = 0.5 * (pred1 + pred2)

    return mean_pred

def apply_color(depth):
    # depth[depth<0.5]=0
    depth_norm = cv2.normalize(depth.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)*255
    depth_uint8 = depth_norm.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)

@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):
    model.eval()
    metrics = RunningAverageDict()
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        image, depth = sample['image'], sample['depth']
        image, depth = image.cuda(), depth.cuda()
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get('focal', torch.Tensor(
            [715.0873]).cuda())  # This magic number (focal) is only used for evaluating BTS model
        
        # print(image.shape, depth.shape)
        pred = infer(model, image, dataset=sample['dataset'][0], focal=focal) 

        # print(depth.shape, pred.shape)
        metrics.update(compute_metrics(depth, pred, config=config))
        
        # Save image, depth, pred for visualization
        if "save_images" in config and config.save_images:
            import os
            # print("Saving images ...")
            from PIL import Image
            import torchvision.transforms as transforms
            from zoedepth.utils.misc import colorize

            os.makedirs(config.save_images, exist_ok=True)
            # def save_image(img, path):
            d = depth.squeeze().cpu().numpy()
            p = pred.squeeze().cpu().numpy()

            p[0,0] = 0

            # maxd = max(d.max(), p.max())
            # p = colorize(p, vmin=0, vmax=d.max(), cmap='viridis')
            # d = colorize(d, vmin=0, vmax=d.max(), cmap='viridis')
            # print(f"\033[33m GT  {d.max()} {d.min()} \033[0m")
            # print(f"\033[33m PRD {p.max()} {p.min()} \033[0m")

            # d = apply_color(d)
            # p = apply_color(p)
            # p = colorize(p, vmin=0, vmax=p.max(), cmap='viridis')
            import matplotlib.pyplot as plt

            vmax = np.max(d[d != 0.0])
            vmin = np.min(d[d != 0.0])
            cmap = 'viridis'

            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(d, vmin=vmin, vmax=vmax, cmap=cmap)
            fig.savefig(os.path.join(config.save_images, f"{i}_depth.png"))

            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(p, vmin=vmin, vmax=vmax, cmap=cmap)
            fig.savefig(os.path.join(config.save_images, f"{i}_pred.png" ))


            # im = transforms.ToPILImage()(image.squeeze().cpu())
            # im.save(os.path.join(config.save_images, f"{i}_img.png"))
            # Image.fromarray(d).save(os.path.join(config.save_images, f"{i}_depth.png"))
            # Image.fromarray(p).save(os.path.join(config.save_images, f"{i}_pred.png" ))
            
            depth = torch.nn.functional.interpolate(depth, size=pred.shape[-2:], mode='bilinear', align_corners=True)
            
            d_np = depth.squeeze().cpu().numpy()
            p_np = pred.squeeze().cpu().numpy()
            diff = p_np - d_np
            diff_img = np.zeros((*diff.shape, 3), dtype=np.uint8)
            # 归一化到0-1
            max_val = np.max(np.abs(diff))
            if max_val < 1e-6:
                max_val = 1.0
            norm = np.clip(np.abs(diff) / max_val, 0, 1)
            # pred > depth: 蓝色
            diff_img[..., 2] = (diff > 0) * (norm * 255)
            # pred < depth: 红色
            diff_img[..., 0] = (diff < 0) * (norm * 255)
            # pred ≈ depth: 黑色（已默认）
            diff_img = diff_img.astype(np.uint8)
            Image.fromarray(diff_img).save(os.path.join(config.save_images, f"{i}_diff.png"))

            # print("difference: ",diff.max(), diff.min(), np.mean(np.abs(diff)))

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics

# -----------------------------
# 辅助函数：计算模型参数量
# -----------------------------
def count_parameters(model, include_all=False):
    """
    统计模型参数数量
    Args:
        model: torch.nn.Module
        include_all: 若为 True，则包括所有参数（训练+冻结）
    """
    if include_all:
        params = sum(p.numel() for p in model.parameters())
    else:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params


def main(config):
    model = build_model(config)

    from zoedepth.utils.misc import count_parameters
    param_count = count_parameters(model, include_all=True)
    print(f"{colors.fg.red}Model parameters: {param_count/1e6:.2f}M{colors.reset}")

    from fvcore.nn import FlopCountAnalysis, parameter_count
    dummy = torch.randn(1, 3, 480, 640).cuda()
    flops = FlopCountAnalysis(model, dummy)
    params = parameter_count(model)
    print(f"\033[31mFLOPs: {flops.total() / 1e9:.2f} G\033[0m")
    print(f"\033[31mParams: {params[''] / 1e6:.2f} M\033[0m")

    # 红色字体输出
    test_loader = DepthDataLoader(config, 'online_eval').data
    model = model.cuda()
    metrics = evaluate(model, test_loader, config)
    print(f"{colors.fg.green}")
    print(metrics)
    print(f"{colors.reset}")
    metrics['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"
    return metrics


def eval_model(model_name, pretrained_resource, dataset='nyu', **kwargs):

    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    # config = change_dataset(config, dataset)  # change the dataset
    pprint(config)
    print(f"Evaluating {model_name} on {dataset}...")
    metrics = main(config)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=True, help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default="", help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='nyu', help="Dataset to evaluate on")

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    if "ALL_INDOOR" in args.dataset:
        datasets = ALL_INDOOR
    elif "ALL_OUTDOOR" in args.dataset:
        datasets = ALL_OUTDOOR
    elif "ALL" in args.dataset:
        datasets = ALL_EVAL_DATASETS
    elif "," in args.dataset:
        datasets = args.dataset.split(",")
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        eval_model(args.model, pretrained_resource=args.pretrained_resource,
                    dataset=dataset, **overwrite_kwargs)
