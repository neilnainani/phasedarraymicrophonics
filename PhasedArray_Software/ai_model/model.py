#######################################
#
#      Phased Array Microphonics
# Array-response model: a neural field that maps
# (array position, mic, frequency, time) to the
# log-magnitude spectrogram of that mic's room
# impulse response.
#
# Rendering "what the array hears at (x, y, z)" is then
#   predicted IR  (*)  any dry source audio
#
#######################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    """x -> [x, sin(2^k pi x), cos(2^k pi x)] for k < n_freqs. Inputs are expected in ~[-1, 1]."""

    def __init__(self, n_freqs):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freqs)) * math.pi, persistent=False)

    def out_dim(self, in_dim):
        return in_dim * (1 + 2 * len(self.freqs))

    def forward(self, x):
        xf = x.unsqueeze(-1) * self.freqs
        return torch.cat([x, torch.sin(xf).flatten(-2), torch.cos(xf).flatten(-2)], dim=-1)


class IRFieldNet(nn.Module):
    def __init__(self, num_mics, pos_freqs=4, tf_freqs=10, mic_embed_dim=16, hidden=256, layers=6, skip_at=3):
        super().__init__()
        self.pos_enc = FourierFeatures(pos_freqs)
        self.tf_enc = FourierFeatures(tf_freqs)
        self.mic_embed = nn.Embedding(num_mics, mic_embed_dim)
        in_dim = self.pos_enc.out_dim(3) + self.tf_enc.out_dim(2) + mic_embed_dim
        self.skip_at = skip_at
        self.layers = nn.ModuleList()
        for i in range(layers):
            d_in = in_dim if i == 0 else hidden + (in_dim if i == skip_at else 0)
            self.layers.append(nn.Linear(d_in, hidden))
        self.head = nn.Linear(hidden, 1)

    def forward(self, pos, mic_idx, ft):
        """
        pos:     [B, 3] normalized array position
        mic_idx: [B]    index into the mic list (0..num_mics-1)
        ft:      [B, 2] normalized (frequency, time) of the spectrogram bin, each in [-1, 1]
        returns  [B]    normalized log-magnitude
        """
        x0 = torch.cat([self.pos_enc(pos), self.tf_enc(ft), self.mic_embed(mic_idx)], dim=-1)
        h = x0
        for i, layer in enumerate(self.layers):
            if i == self.skip_at:
                h = torch.cat([h, x0], dim=-1)
            h = F.silu(layer(h))
        return self.head(h).squeeze(-1)

    @torch.no_grad()
    def predict_spectrogram(self, pos, mic_idx, n_freq, n_time):
        """Full [n_freq, n_time] normalized log-magnitude grid for one position/mic."""
        dev = next(self.parameters()).device
        f = torch.linspace(-1, 1, n_freq, device=dev)
        t = torch.linspace(-1, 1, n_time, device=dev)
        ff, tt = torch.meshgrid(f, t, indexing="ij")
        ft = torch.stack([ff.flatten(), tt.flatten()], dim=-1)
        p = pos.to(dev).view(1, 3).expand(len(ft), 3)
        m = torch.full((len(ft),), int(mic_idx), device=dev, dtype=torch.long)
        return self(p, m, ft).view(n_freq, n_time)
