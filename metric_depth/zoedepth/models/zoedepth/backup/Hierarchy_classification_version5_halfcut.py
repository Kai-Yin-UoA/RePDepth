import torch
import torch.nn as nn

class HCBinGenerator(nn.Module):
    def __init__(self, in_channel=32, n_bins=256, n_bins_fine=256, min_depth=0.0, max_depth=10.0):
        super().__init__()
        self.n_bins = n_bins
        self.n_bins_fine = n_bins_fine
        self.min_depth = min_depth + 1e-6
        self.max_depth = max_depth+1 # extend the max depth a bit to except the edge case of 10m
        self.in_channel = in_channel

        # coarse bin：log spacing bin
        log_min = torch.log(torch.tensor(self.min_depth))
        log_max = torch.log(torch.tensor(self.max_depth))
        log_bins = torch.linspace(log_min, log_max, n_bins*2)[n_bins:].to('cuda')
        log_bins[0] = torch.log(torch.tensor(1e-3))
        self.coarse_bins = torch.exp(log_bins)   # [n_bins]
        
        # fine bin：fine-grade equal distance
        coarse_diff = self.coarse_bins[1:] - self.coarse_bins[:-1]  # [n_bins-1]
        last_width = coarse_diff[-1].unsqueeze(0)  # [1]
        self.fine_width = torch.cat([coarse_diff, last_width], dim=0)  # [n_bins]

        print(self.coarse_bins)
        print(self.fine_width)

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, n_bins//2, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_bins//2, n_bins+1, 1, 1, 0),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        """
        x: [B, n_bins, H, W]       → midas out
        """
        x = self.conv(x)
        logits_coarse = x[1:].unsqueeze(1)
        logits_fine = x[0].unsqueeze(1)

        # probality distribution
        probs_coarse = torch.softmax(logits_coarse, dim=1)   # [B, n_bins, H, W] maximum bin index prob = 1
        # ---- Step 1: coarse depth expectation ----
        coarse_depth = torch.sum(probs_coarse * self.coarse_bins.view(1,-1,1,1), dim=1, keepdim=True)

        # ---- Step 2: fine depth expectation ----
        probs_fine   = nn.functional.normalize(logits_fine, dim=1)   # [B, n_bins_fine, H, W] distribution of each bin width
        # probs_coarse * self.fine_width.view(1,-1,1,1) selects the bin width according to the coarse depth prob
        # probs_fine * ... choose the width within the selected bin
        fine_depth = torch.sum(probs_fine * probs_coarse * self.fine_width.view(1,-1,1,1), dim=1, keepdim=True)

        # ---- Step 3: fusion ----
        depth = coarse_depth + fine_depth - self.min_depth
        print("output max, output min: ",depth.max().item(), depth.min().item())
        print("coarse max, coarse min: ", coarse_depth.max().item(), coarse_depth.min().item())
        print(fine_depth.max().item(), fine_depth.min().item())

        # print(depth.max(), depth.min())

        return depth, coarse_depth, fine_depth
