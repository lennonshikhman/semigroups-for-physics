import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def time_embedding(t, dim=32):
    half = dim // 2
    freqs = torch.exp(torch.linspace(0, math.log(1000), half, device=t.device))
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock1D(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1), nn.GELU(), nn.Conv1d(width, width, 3, padding=1)
        )

    def forward(self, x):
        return x + self.net(x)


class TimeConditionedConvNet1D(nn.Module):
    def __init__(self, width=32, n_blocks=3, t_dim=32):
        super().__init__()
        self.inp = nn.Conv1d(1, width, 1)
        self.blocks = nn.ModuleList([ResBlock1D(width) for _ in range(n_blocks)])
        self.t_proj = nn.Linear(t_dim, width)
        self.out = nn.Conv1d(width, 1, 1)
        self.t_dim = t_dim

    def forward(self, u, t):
        x = self.inp(u[:, None, :])
        te = self.t_proj(time_embedding(t, self.t_dim))[:, :, None]
        x = x + te
        for blk in self.blocks:
            x = blk(x)
        return u + self.out(x)[:, 0, :]


class SpectralConv1D(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) / (width * width))

    def forward(self, x):
        b, c, n = x.shape
        xh = torch.fft.rfft(x, dim=-1)
        out = torch.zeros_like(xh)
        m = min(self.modes, xh.shape[-1])
        out[:, :, :m] = torch.einsum('bcm,com->bom', xh[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out, n=n, dim=-1)


class FNO1D(nn.Module):
    def __init__(self, width=32, modes=12, n_blocks=2, t_dim=32):
        super().__init__()
        self.inp = nn.Conv1d(1, width, 1)
        self.spec = nn.ModuleList([SpectralConv1D(width, modes) for _ in range(n_blocks)])
        self.w = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_blocks)])
        self.t_proj = nn.Linear(t_dim, width)
        self.out = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))
        self.t_dim = t_dim

    def forward(self, u, t):
        x = self.inp(u[:, None, :])
        x = x + self.t_proj(time_embedding(t, self.t_dim))[:, :, None]
        for s, w in zip(self.spec, self.w):
            x = F.gelu(s(x) + w(x))
        return u + self.out(x)[:, 0, :]
