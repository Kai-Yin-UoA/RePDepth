import torch
import torch.nn as nn

class HCBinGenerator(nn.Module):
    def __init__(self, n_bins=64, n_bins_fine=64, min_depth=0.0, max_depth=10.0):
        super().__init__()
        self.n_bins = n_bins
        self.n_bins_fine = n_bins_fine
        self.min_depth = min_depth + 1e-6
        self.max_depth = max_depth + 1

        # coarse bin：log spacing bin
        log_min = torch.tensor(self.min_depth)
        log_max = torch.tensor(self.max_depth)
        self.coarse_bins = torch.linspace(log_min, log_max, n_bins).to('cuda')

        # fine bin：fine-grade equal distance
        self.fine_width = (self.max_depth - self.min_depth) / n_bins_fine

        print(self.coarse_bins)
        print(self.fine_width)

    def forward(self, logits_coarse, logits_fine):
        """
        logits_coarse: [B, n_bins, H, W]       → coarse prob
        logits_fine:   [B, n_bins_fine, H, W]  → fine prob
        """

        # 概率化
        probs_coarse = torch.softmax(logits_coarse, dim=1)   # [B, n_bins, H, W] maximum bin index prob = 1
        probs_fine   = nn.functional.normalize(logits_fine, dim=1)   # [B, n_bins_fine, H, W] distribution of each bin width
        
        # print(logits_coarse.shape, logits_fine.shape) 
        # print( self.coarse_bins.shape, self.fine_bins.shape)

        # ---- Step 1: coarse depth expectation ----
        coarse_depth = torch.sum(probs_coarse * self.coarse_bins.view(1,-1,1,1), dim=1, keepdim=True)
        # shape = [B,1,H,W]

        # ---- Step 2: fine depth expectation ----
        fine_depth = torch.sum(probs_fine * self.fine_width, dim=1, keepdim=True)

        # ---- Step 3: fusion ----
        depth = coarse_depth + fine_depth - self.min_depth
        print("output: ", depth.max().item(), depth.min().item(), coarse_depth.max().item(), coarse_depth.min().item(), fine_depth.max().item(), fine_depth.min().item())

        # print(depth.max(), depth.min())

        return depth, coarse_depth, fine_depth
