# max pooling
# 64 dim MLP
# small adjust range 0.1

import torch
import torch.nn as nn

class HCBinGenerator(nn.Module):
    def __init__(self, n_bins=256, n_bins_fine=256, stride=4, fusion_dim=4, min_depth=0.0, max_depth=10.0):
        super().__init__()
        self.n_bins = n_bins
        self.n_bins_fine = n_bins_fine
        self.min_depth = min_depth + 1e-6
        self.max_depth = max_depth + 1

        # coarse bin：log spacing bin
        log_min = torch.log(torch.tensor(self.min_depth))
        log_max = torch.log(torch.tensor(self.max_depth))
        log_bins = torch.linspace(log_min, log_max, self.n_bins).to('cuda')
        self.coarse_bins = torch.exp(log_bins)   # [n_bins]
        self.coarse_bins = nn.Parameter(torch.exp(log_bins), requires_grad=True).to('cuda')

        # fine bin：fine-grade equal distance
        coarse_diff = self.coarse_bins[1:] - self.coarse_bins[:-1]  # [n_bins-1]
        last_width = coarse_diff[-1].unsqueeze(0)  # [1]
        self.fine_width = torch.cat([coarse_diff, last_width], dim=0)  # [n_bins]
        self.fine_width = self.fine_width[::stride]  # [n_bins//stride]

        print("coarse bin:",self.coarse_bins)
        print("fine   bin:",self.fine_width)
        print("coarse bin size:",self.coarse_bins.size())
        print("fine   bin size:",self.fine_width.size())

        self.adjustors = nn.ModuleList([
                            CoarseBinAdjustor(c, self.n_bins)
                            for c in fusion_dim
                        ])

    def forward(self, logits_coarse, logits_fine, fusion_feats):
        """
        logits_coarse: [B, n_bins, H, W]       → coarse prob
        logits_fine:   [B, n_bins_fine, H, W]  → fine prob
        """

        b_prev = self.coarse_bins  # 初始 bin
        for adjustor, x in zip(self.adjustors, fusion_feats):
            b_prev = adjustor(x, b_prev)

        new_coarse_bins = b_prev  # [B, n_bins]


        probs_coarse = torch.softmax(logits_coarse, dim=1)   # [B, n_bins, H, W] maximum bin index prob = 1
        probs_fine   = nn.functional.normalize(logits_fine, dim=1)   # [B, n_bins_fine, H, W] distribution of each bin width
        
        # print(logits_coarse.shape, logits_fine.shape) 
        # print( self.coarse_bins.shape, self.fine_bins.shape)

        # ---- Step 1: coarse depth expectation ----
        # print(new_coarse_bins.size(), probs_coarse.size())
        coarse_depth = torch.sum(probs_coarse * new_coarse_bins.view(1,-1,1,1), dim=1, keepdim=True)
        # shape = [B,1,H,W]

        # ---- Step 2: fine depth expectation ----
        fine_depth = torch.sum(probs_fine * self.fine_width.view(1,-1,1,1), dim=1, keepdim=True)

        # ---- Step 3: fusion ----
        depth = coarse_depth + fine_depth - self.min_depth
        print("output: ", depth.max().item(), depth.min().item(), coarse_depth.max().item(), coarse_depth.min().item(), fine_depth.max().item(), fine_depth.min().item())

        return depth, coarse_depth, fine_depth

class CoarseBinAdjustor(nn.Module):
    def __init__(self, in_channels, n_bins, mlp_dim=64):
        super().__init__()
        self.n_bins = n_bins

        self.global_pool = nn.MaxPool2d(1)  # [B, C, 1, 1]

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, n_bins)
        )

    def forward(self, feat, coarse_bins):
        """
        feat: [B, C, H, W]   (e.g., B*16*37*59)
        coarse_bins: [n_bins] (1D tensor, precomputed log-spaced bins)
        """
        B, C, H, W = feat.shape
        x = self.global_pool(feat).view(B, C) 

        adjustor = self.mlp(x)                
        adjustor = torch.tanh(adjustor) * 0.1

        # print(coarse_bins.shape, adjustor.shape)
        # coarse_bins = coarse_bins.unsqueeze(0)
        adjusted_bins = torch.mean(coarse_bins + coarse_bins * adjustor, dim=0)
        # print(adjusted_bins.size())

        return adjusted_bins