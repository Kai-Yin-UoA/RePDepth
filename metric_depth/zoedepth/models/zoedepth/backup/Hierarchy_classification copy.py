import torch
import torch.nn as nn

class HCBinGenerator(nn.Module):
    def __init__(self, n_bins=256, n_bins_fine=256, min_depth=0.0, max_depth=10.0):
        super().__init__()
        self.n_bins = n_bins
        self.n_bins_fine = n_bins_fine
        self.min_depth = min_depth
        self.max_depth = max_depth

        # coarse bin：log spacing bin
        log_min = torch.log(torch.tensor(min_depth + 1e-6))
        log_max = torch.log(torch.tensor(max_depth))
        log_bins = torch.linspace(log_min, log_max, n_bins).to('cuda')
        self.coarse_bins = torch.exp(log_bins)   # [n_bins]
        # self.register_buffer("coarse_bins", coarse_bins)

        # fine bin：fine-grade equal distance
        coarse_diff = self.coarse_bins[1:] - self.coarse_bins[:-1]  # [n_bins-1]
        last_width = coarse_diff[-1].unsqueeze(0)  # [1]
        self.fine_width = torch.cat([coarse_diff, last_width], dim=0)  # [n_bins]

        print(self.coarse_bins)
        print(self.fine_width)

    def forward(self, logits_coarse, logits_fine):
        """
        logits_coarse: [B, n_bins, H, W]       → coarse prob
        logits_fine:   [B, n_bins_fine, H, W]  → fine prob
        """

        # 概率化
        probs_coarse = torch.softmax(logits_coarse, dim=1)   # [B, n_bins, H, W] maximum bin index prob = 1
        probs_fine   = logits_fine / logits_fine.sum(dim=1, keepdim=True)   # [B, n_bins_fine, H, W] distribution of each bin width
        # print(logits_coarse.shape, logits_fine.shape) 
        # print( self.coarse_bins.shape, self.fine_bins.shape)

        # ---- Step 1: coarse depth expectation ----
        coarse_depth = torch.sum(probs_coarse * self.coarse_bins.view(1,-1,1,1), dim=1, keepdim=True)
        # shape = [B,1,H,W]

        # ---- Step 2: fine depth expectation ----
        fine_depth = torch.sum(probs_fine * self.fine_width.view(1,-1,1,1), dim=1, keepdim=True)
        # shape = [B,1,H,W]

        # ---- Step 3: fusion ----
        depth = coarse_depth + fine_depth - self.min_depth
        print(depth.max().item(), depth.min().item(), self.coarse_bins.max().item(), self.coarse_bins.min().item(), probs_fine.max().item(), probs_fine.min().item())

        # print(depth.max(), depth.min())

        return depth, coarse_depth, fine_depth
