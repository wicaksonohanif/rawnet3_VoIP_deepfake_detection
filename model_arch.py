# model_arch.py
# Arsitektur RawNet3 + Gated ConvNext untuk Deepfake Audio Detection
# File ini harus diletakkan di folder yang sama dengan streamlit_app.py
# dan inference.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_SR = 16000
FIXED_LEN = 3 * TARGET_SR  # 64.000 sampel (4 detik)


class SincConv(nn.Module):
    """
    Sinc-based Conv1d — belajar filter bank langsung dari raw waveform.
    Diinisialisasi pada mel-scale; frekuensi batas adalah parameter yang
    dapat dilatih.
    """
    def __init__(self, out_channels=128, kernel_size=251,
                 sample_rate=16000, min_low_hz=50, min_band_hz=50):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size  = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate  = sample_rate
        self.min_low_hz   = min_low_hz
        self.min_band_hz  = min_band_hz

        # Inisialisasi frekuensi pada mel-scale
        low_hz  = 30.0
        high_hz = sample_rate / 2.0 - (min_low_hz + min_band_hz)
        mel_pts = np.linspace(
            2595.0 * np.log10(1.0 + low_hz  / 700.0),
            2595.0 * np.log10(1.0 + high_hz / 700.0),
            out_channels + 1
        )
        hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)

        self.low_hz_  = nn.Parameter(torch.Tensor(hz_pts[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz_pts)).view(-1, 1))

        # Buffer — ikut .to(device) otomatis, tidak dilatih
        half   = (self.kernel_size - 1) // 2
        n_left = torch.arange(1, half + 1, dtype=torch.float32).view(1, -1)
        self.register_buffer('n_left', n_left)
        self.register_buffer(
            'window_',
            torch.hamming_window(self.kernel_size, periodic=False)
        )

    def forward(self, x):
        low  = self.min_low_hz  + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2.0
        )
        band = (high - low).squeeze(1)

        f_low  = 2.0 * np.pi * low  * self.n_left / self.sample_rate
        f_high = 2.0 * np.pi * high * self.n_left / self.sample_rate

        band_left   = (torch.sin(f_high) - torch.sin(f_low)) / \
                      (self.n_left * np.pi / self.sample_rate + 1e-8)
        band_center = 2.0 * band.unsqueeze(1)
        band_right  = torch.flip(band_left, dims=[1])

        band_pass = torch.cat([band_left, band_center, band_right], dim=1)
        band_pass = band_pass / (band.unsqueeze(1) + 1e-8) * self.window_

        return F.conv1d(x, band_pass.unsqueeze(1),
                        stride=1, padding=self.kernel_size // 2)


class GatedConvNextBlock(nn.Module):
    """
    Gated ConvNext Block:
    Depthwise Conv → LayerNorm → [Pointwise × Sigmoid Gate] → Residual
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=7, expansion=4, drop_path=0.1):
        super().__init__()
        mid_ch = in_channels * expansion

        self.dwconv   = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,   # jaga panjang T
            groups=in_channels
        )
        self.norm     = nn.LayerNorm(in_channels)
        self.pwconv1  = nn.Linear(in_channels, mid_ch)
        self.act      = nn.GELU()
        self.gate     = nn.Linear(in_channels, mid_ch)
        self.pwconv2  = nn.Linear(mid_ch, out_channels)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )
        self.drop_path_rate = drop_path

    @staticmethod
    def _drop_path(x: torch.Tensor, drop_prob: float, training: bool):
        if drop_prob == 0.0 or not training:
            return x
        keep  = 1.0 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x / keep * torch.rand(shape, dtype=x.dtype,
                                     device=x.device).floor_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.dwconv(x).permute(0, 2, 1)   # (B, T, C)
        out = self.norm(out)
        gate_val = torch.sigmoid(self.gate(out))
        out = self.act(self.pwconv1(out)) * gate_val
        out = self.pwconv2(out).permute(0, 2, 1) # (B, C, T)
        return self._drop_path(out, self.drop_path_rate, self.training) + residual


class AttentiveStatisticsPooling(nn.Module):
    """Merangkum dimensi temporal T menjadi statistik (mean + std) berbobot."""
    def __init__(self, in_dim: int, bottleneck: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck, 1),
            nn.Tanh(),
            nn.Conv1d(bottleneck, in_dim, 1),
            nn.Softmax(dim=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.attention(x)                           # (B, C, T)
        mu    = (alpha * x).sum(dim=2)                     # (B, C)
        sigma = torch.sqrt(
            (alpha * (x - mu.unsqueeze(2)) ** 2).sum(dim=2) + 1e-8
        )                                                   # (B, C)
        return torch.cat([mu, sigma], dim=1)                # (B, 2C)


class RawNet3(nn.Module):
    """
    RawNet3 + Gated ConvNext untuk Deepfake Audio Detection.

    Input : (B, 1, T)  — raw waveform (float32, normalized)
    Output: (B, 2)     — logits [bonafide, spoofed]

    Cara load:
        model = RawNet3()
        state = torch.load('rawnet3_voip_weights.pt', map_location='cpu')
        # Jika disimpan sebagai full checkpoint:
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        model.eval()
    """
    def __init__(
        self,
        sinc_out         : int       = 128,
        sinc_kernel      : int       = 251,
        channels         : list      = None,
        blocks_per_stage : list      = None,
        kernel_size      : int       = 7,
        expansion        : int       = 4,
        emb_dim          : int       = 256,
        num_classes      : int       = 2,
        sr               : int       = TARGET_SR,
        drop_path        : float     = 0.1,
        dropout          : float     = 0.2,
    ):
        super().__init__()
        if channels         is None: channels         = [128, 256, 256, 512, 512]
        if blocks_per_stage is None: blocks_per_stage = [1, 2, 3, 2, 1]

        # ── Sinc Filterbank ──
        self.sinc_conv = SincConv(sinc_out, sinc_kernel, sr)
        self.bn_sinc   = nn.BatchNorm1d(sinc_out)
        self.init_pool = nn.MaxPool1d(kernel_size=3, stride=3)

        # ── Gated ConvNext Stages ──
        stage_modules, prev_ch = [], sinc_out
        for i, (out_ch, n_blk) in enumerate(zip(channels, blocks_per_stage)):
            stage_modules.append(nn.Sequential(*[
                GatedConvNextBlock(
                    prev_ch if b == 0 else out_ch,
                    out_ch, kernel_size, expansion, drop_path
                )
                for b in range(n_blk)
            ]))
            if i < len(channels) - 1:
                stage_modules.append(nn.MaxPool1d(kernel_size=2, stride=2))
            prev_ch = out_ch
        self.stages = nn.ModuleList(stage_modules)

        # ── Head ──
        self.asp        = AttentiveStatisticsPooling(prev_ch)
        self.fc_emb     = nn.Linear(prev_ch * 2, emb_dim)
        self.bn_emb     = nn.BatchNorm1d(emb_dim)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(emb_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) and not isinstance(m, SincConv):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if m.weight is not None: nn.init.ones_(m.weight)
                if m.bias   is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.abs(self.sinc_conv(x))
        x = self.init_pool(self.bn_sinc(x))
        for layer in self.stages:
            x = layer(x)
        return F.relu(self.bn_emb(self.fc_emb(self.asp(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self._encode(x)))

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x)
