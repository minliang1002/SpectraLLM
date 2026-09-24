"""
Complex-Valued Frequency-Aware Modules V12 - Improved Frequency Positional Encoding
改进的频率位置编码：在投影之后添加，保留完整的位置信息

V11 问题：在 FFT 输出上添加标量位置编码，信息损失大
V12 改进：在 ComplexMLP 投影之后添加向量位置编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# Complex-Valued Building Blocks (from V9)
# =============================================================================

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.W_real = nn.Linear(in_features, out_features, bias=bias)
        self.W_imag = nn.Linear(in_features, out_features, bias=False)
        if bias:
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_imag', None)
    
    def forward(self, x_real, x_imag):
        out_real = self.W_real(x_real) - self.W_imag(x_imag)
        out_imag = self.W_real(x_imag) + self.W_imag(x_real)
        if self.bias_imag is not None:
            out_imag = out_imag + self.bias_imag
        return out_real, out_imag


class ComplexDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
    
    def forward(self, x_real, x_imag):
        if self.training and self.p > 0:
            mask = torch.ones_like(x_real).bernoulli_(1 - self.p) / (1 - self.p)
            return x_real * mask, x_imag * mask
        return x_real, x_imag


class ComplexGELU(nn.Module):
    def forward(self, x_real, x_imag):
        return F.gelu(x_real), F.gelu(x_imag)


class ComplexMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = ComplexLinear(in_features, hidden_features)
        self.act = ComplexGELU()
        self.dropout = ComplexDropout(dropout)
        self.fc2 = ComplexLinear(hidden_features, out_features)
    
    def forward(self, x_real, x_imag):
        h_real, h_imag = self.fc1(x_real, x_imag)
        h_real, h_imag = self.act(h_real, h_imag)
        h_real, h_imag = self.dropout(h_real, h_imag)
        return self.fc2(h_real, h_imag)


# =============================================================================
# Improved Frequency Positional Encoding
# =============================================================================

class ScalePositionalEncoding(nn.Module):
    """
    尺度位置编码：给不同尺度的特征添加位置信息
    
    在投影之后添加，维度与投影输出一致
    每个尺度有独立的可学习位置编码
    """
    def __init__(self, embed_dim, scale_name='coarse'):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale_name = scale_name
        
        # 可学习的尺度位置编码
        # 实部和虚部分别有独立的编码
        self.pe_real = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pe_imag = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 初始化
        self._init_pe()
    
    def _init_pe(self):
        """基于尺度的初始化"""
        # 不同尺度用不同的初始化策略
        scale_factors = {
            'fine': 0.01,      # 细粒度：小扰动
            'medium': 0.02,    # 中粒度：中等扰动
            'coarse': 0.03     # 粗粒度：较大扰动
        }
        scale = scale_factors.get(self.scale_name, 0.02)
        nn.init.normal_(self.pe_real, std=scale)
        nn.init.normal_(self.pe_imag, std=scale)
    
    def forward(self, x_real, x_imag):
        """
        Args:
            x_real: [B, M, embed_dim]
            x_imag: [B, M, embed_dim]
        """
        return x_real + self.pe_real, x_imag + self.pe_imag


class FrequencyBandEncoding(nn.Module):
    """
    频带编码：根据频率的物理意义添加编码
    
    将特征维度分成多个频带，每个频带有不同的编码
    - 低频带：对应慢变化的趋势
    - 中频带：对应周期性模式
    - 高频带：对应快速变化/噪声
    """
    def __init__(self, embed_dim, n_bands=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_bands = n_bands
        self.band_dim = embed_dim // n_bands
        
        # 每个频带的可学习编码
        self.band_pe_real = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, self.band_dim))
            for _ in range(n_bands)
        ])
        self.band_pe_imag = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, self.band_dim))
            for _ in range(n_bands)
        ])
        
        # 频带混合权重 (可学习)
        self.band_weights = nn.Parameter(torch.ones(n_bands) / n_bands)
        
        self._init_pe()
    
    def _init_pe(self):
        """频带感知的初始化"""
        for i in range(self.n_bands):
            # 低频带用较小的值，高频带用较大的值
            scale = 0.01 * (i + 1)
            nn.init.normal_(self.band_pe_real[i], std=scale)
            nn.init.normal_(self.band_pe_imag[i], std=scale)
    
    def forward(self, x_real, x_imag):
        """
        Args:
            x_real: [B, M, embed_dim]
            x_imag: [B, M, embed_dim]
        """
        B, M, D = x_real.shape
        
        # 分割成频带
        bands_real = x_real.split(self.band_dim, dim=-1)
        bands_imag = x_imag.split(self.band_dim, dim=-1)
        
        # 对每个频带添加位置编码
        out_real_bands = []
        out_imag_bands = []
        
        weights = F.softmax(self.band_weights, dim=0)
        
        for i in range(min(self.n_bands, len(bands_real))):
            out_real_bands.append(bands_real[i] + self.band_pe_real[i] * weights[i])
            out_imag_bands.append(bands_imag[i] + self.band_pe_imag[i] * weights[i])
        
        # 处理剩余维度（如果有）
        if len(bands_real) > self.n_bands:
            out_real_bands.extend(bands_real[self.n_bands:])
            out_imag_bands.extend(bands_imag[self.n_bands:])
        
        out_real = torch.cat(out_real_bands, dim=-1)
        out_imag = torch.cat(out_imag_bands, dim=-1)
        
        return out_real, out_imag


# =============================================================================
# Stage 1: Complex Frequency-Aware Multi-Resolution Encoding with Improved PE
# =============================================================================

class ComplexFrequencyAwareMultiResolutionExtractorV12(nn.Module):
    """
    Stage 1 with Improved Frequency Positional Encoding
    
    改进点：
    1. 在 ComplexMLP 投影之后添加位置编码
    2. 每个尺度有独立的尺度位置编码
    3. 添加频带编码，区分低/中/高频特征
    """
    def __init__(self, n_features=37, d_model=768, input_len=24):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.input_len = input_len
        
        self.sub_dim = d_model // 3  # 256
        self.complex_sub_dim = self.sub_dim // 2  # 128
        
        # FFT output dimensions
        self.n_freq_coarse = 13
        self.n_freq_medium = 7
        self.n_freq_fine = 4
        
        # Complex Scale Projectors
        self.proj_coarse = ComplexMLP(
            in_features=self.n_freq_coarse,
            hidden_features=self.complex_sub_dim,
            out_features=self.complex_sub_dim,
            dropout=0.1
        )
        self.proj_medium = ComplexMLP(
            in_features=self.n_freq_medium,
            hidden_features=self.complex_sub_dim,
            out_features=self.complex_sub_dim,
            dropout=0.1
        )
        self.proj_fine = ComplexMLP(
            in_features=self.n_freq_fine,
            hidden_features=self.complex_sub_dim,
            out_features=self.complex_sub_dim,
            dropout=0.1
        )
        
        # 尺度位置编码 (在投影之后添加)
        self.scale_pe_coarse = ScalePositionalEncoding(self.complex_sub_dim, 'coarse')
        self.scale_pe_medium = ScalePositionalEncoding(self.complex_sub_dim, 'medium')
        self.scale_pe_fine = ScalePositionalEncoding(self.complex_sub_dim, 'fine')
        
        # 频带编码 (在最终拼接之后添加)
        self.freq_band_encoding = FrequencyBandEncoding(
            embed_dim=self.complex_sub_dim * 3,  # 384
            n_bands=3  # 对应 fine/medium/coarse
        )

    def forward(self, x):
        B, L, M = x.shape
        x = x.permute(0, 2, 1)  # [B, M, 24]
        
        # FFT at multiple scales
        fft_coarse = torch.fft.rfft(x, dim=-1)  # [B, M, 13]
        fft_med_1 = torch.fft.rfft(x[:, :, 0:12], dim=-1)  # [B, M, 7]
        fft_med_2 = torch.fft.rfft(x[:, :, 12:24], dim=-1)
        fft_fine_list = [torch.fft.rfft(x[:, :, i*6:(i+1)*6], dim=-1) for i in range(4)]
        
        # Complex Projection
        emb_coarse_real, emb_coarse_imag = self.proj_coarse(fft_coarse.real, fft_coarse.imag)
        
        emb_med_1_real, emb_med_1_imag = self.proj_medium(fft_med_1.real, fft_med_1.imag)
        emb_med_2_real, emb_med_2_imag = self.proj_medium(fft_med_2.real, fft_med_2.imag)
        emb_medium_real = (emb_med_1_real + emb_med_2_real) / 2
        emb_medium_imag = (emb_med_1_imag + emb_med_2_imag) / 2
        
        emb_fine_reals, emb_fine_imags = [], []
        for fft_fine in fft_fine_list:
            emb_r, emb_i = self.proj_fine(fft_fine.real, fft_fine.imag)
            emb_fine_reals.append(emb_r)
            emb_fine_imags.append(emb_i)
        emb_fine_real = torch.stack(emb_fine_reals, dim=0).mean(dim=0)
        emb_fine_imag = torch.stack(emb_fine_imags, dim=0).mean(dim=0)
        
        # 添加尺度位置编码 (在投影之后)
        emb_coarse_real, emb_coarse_imag = self.scale_pe_coarse(emb_coarse_real, emb_coarse_imag)
        emb_medium_real, emb_medium_imag = self.scale_pe_medium(emb_medium_real, emb_medium_imag)
        emb_fine_real, emb_fine_imag = self.scale_pe_fine(emb_fine_real, emb_fine_imag)
        
        # Energy-based Weighting
        power = fft_coarse.abs().pow(2)
        energy_low = power[:, :, 0:4].sum(dim=-1)
        energy_mid = power[:, :, 4:9].sum(dim=-1)
        energy_high = power[:, :, 9:13].sum(dim=-1)
        
        energies = torch.stack([energy_high, energy_mid, energy_low], dim=-1)
        freq_weights = F.softmax(torch.log(energies + 1e-6), dim=-1)
        
        w_fine = freq_weights[:, :, 0:1]
        w_mid = freq_weights[:, :, 1:2]
        w_low = freq_weights[:, :, 2:3]
        
        # Weighted Fusion
        feat_fine_real = emb_fine_real * w_fine
        feat_fine_imag = emb_fine_imag * w_fine
        feat_medium_real = emb_medium_real * w_mid
        feat_medium_imag = emb_medium_imag * w_mid
        feat_coarse_real = emb_coarse_real * w_low
        feat_coarse_imag = emb_coarse_imag * w_low
        
        # Concat real parts and imag parts
        all_real = torch.cat([feat_fine_real, feat_medium_real, feat_coarse_real], dim=-1)  # [B, M, 384]
        all_imag = torch.cat([feat_fine_imag, feat_medium_imag, feat_coarse_imag], dim=-1)  # [B, M, 384]
        
        # 添加频带编码 (在拼接之后)
        all_real, all_imag = self.freq_band_encoding(all_real, all_imag)
        
        # Final concat
        multi_res_feat = torch.cat([all_real, all_imag], dim=-1)  # [B, M, 768]
        
        return multi_res_feat, freq_weights


# =============================================================================
# Stage 2: Feature Denoising Attention (same as V9)
# =============================================================================

class ComplexFeatureDenoisingAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.complex_dim = d_model // 2
        self.head_dim = self.complex_dim // n_heads
        
        self.q_proj = ComplexLinear(self.complex_dim, self.complex_dim)
        self.k_proj = ComplexLinear(self.complex_dim, self.complex_dim)
        self.v_proj = ComplexLinear(self.complex_dim, self.complex_dim)
        self.out_proj = ComplexLinear(self.complex_dim, self.complex_dim)
        self.ln = nn.LayerNorm(d_model)
        
    def forward(self, x, target_idx):
        B, M, D = x.shape
        
        x_real = x[:, :, :self.complex_dim]
        x_imag = x[:, :, self.complex_dim:]
        
        target_real = x_real[:, target_idx:target_idx+1, :]
        target_imag = x_imag[:, target_idx:target_idx+1, :]
        
        q_real, q_imag = self.q_proj(target_real, target_imag)
        k_real, k_imag = self.k_proj(x_real, x_imag)
        v_real, v_imag = self.v_proj(x_real, x_imag)
        
        q_real = q_real.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        q_imag = q_imag.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k_real = k_real.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        k_imag = k_imag.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v_real = v_real.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v_imag = v_imag.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores_real = (torch.matmul(q_real, k_real.transpose(-2, -1)) + 
                       torch.matmul(q_imag, k_imag.transpose(-2, -1)))
        scores = scores_real / (self.head_dim ** 0.5)
        attn_gate = torch.sigmoid(scores)
        
        context_real = torch.matmul(attn_gate, v_real)
        context_imag = torch.matmul(attn_gate, v_imag)
        
        context_real = context_real.transpose(1, 2).contiguous().view(B, 1, self.complex_dim)
        context_imag = context_imag.transpose(1, 2).contiguous().view(B, 1, self.complex_dim)
        
        out_real, out_imag = self.out_proj(context_real, context_imag)
        output = torch.cat([out_real, out_imag], dim=-1)
        
        out = x.clone()
        out[:, target_idx:target_idx+1, :] = x[:, target_idx:target_idx+1, :] + output
        
        return self.ln(out)
