"""
Multi-Scale Frequency Extractor for Paper Implementation
输出多尺度嵌入以支持论文中的频域注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFrequencyExtractor(nn.Module):
    """
    论文Stage 1实现：Energy-Aware Multi-Resolution Spectral Encoding
    
    输出多尺度嵌入：
    - fine: 高频细节 (L/4窗口)
    - medium: 中频模式 (L/2窗口)
    - coarse: 低频趋势 (L窗口)
    
    每个尺度输出: [B, C, 2*d_e] (实部+虚部拼接)
    """
    
    def __init__(self, n_features=37, d_embed=384, input_len=24):
        """
        Args:
            n_features: 特征数量 (C)
            d_embed: 每个尺度的嵌入维度 (d_e)
            input_len: 输入序列长度 (L)
        """
        super().__init__()
        self.n_features = n_features
        self.d_embed = d_embed
        self.input_len = input_len
        
        # 论文中的三个尺度
        self.coarse_len = input_len  # W_coarse = L
        self.medium_len = input_len // 2  # W_medium = L/2
        self.fine_len = input_len // 4  # W_fine = L/4
        
        # FFT输出维度 (K_s = ⌊W_s/2⌋ + 1)
        self.n_freq_coarse = self.coarse_len // 2 + 1
        self.n_freq_medium = self.medium_len // 2 + 1
        self.n_freq_fine = self.fine_len // 2 + 1
        
        # Complex-valued MLP for each scale
        # 论文公式: E^(s)_i = ComplexMLP^(s)(F^(s)_i)
        self.complex_mlp_coarse = ComplexMLP(self.n_freq_coarse, d_embed)
        self.complex_mlp_medium = ComplexMLP(self.n_freq_medium, d_embed)
        self.complex_mlp_fine = ComplexMLP(self.n_freq_fine, d_embed)
        
        # Scale positional encoding
        # 论文公式: \tilde{E}^(s) = E^(s) + (P^(s)_r + jP^(s)_i)
        self.scale_pe_coarse_real = nn.Parameter(torch.randn(1, 1, d_embed) * 0.1)
        self.scale_pe_coarse_imag = nn.Parameter(torch.randn(1, 1, d_embed) * 0.1)
        self.scale_pe_medium_real = nn.Parameter(torch.randn(1, 1, d_embed) * 0.05)
        self.scale_pe_medium_imag = nn.Parameter(torch.randn(1, 1, d_embed) * 0.05)
        self.scale_pe_fine_real = nn.Parameter(torch.randn(1, 1, d_embed) * 0.02)
        self.scale_pe_fine_imag = nn.Parameter(torch.randn(1, 1, d_embed) * 0.02)
        
        # Energy-based adaptive weighting
        # 论文公式: w_s ∝ exp(log E_{b(s)})
        self.register_buffer('low_freq_mask', self._create_freq_mask(self.n_freq_coarse, 'low'))
        self.register_buffer('mid_freq_mask', self._create_freq_mask(self.n_freq_coarse, 'mid'))
        self.register_buffer('high_freq_mask', self._create_freq_mask(self.n_freq_coarse, 'high'))
        
    def _create_freq_mask(self, n_freq, band):
        """创建频段掩码"""
        mask = torch.zeros(n_freq)
        if band == 'low':
            mask[:n_freq//3] = 1.0
        elif band == 'mid':
            mask[n_freq//3:2*n_freq//3] = 1.0
        elif band == 'high':
            mask[2*n_freq//3:] = 1.0
        return mask
    
    def _compute_energy_weights(self, fft_coarse):
        """
        计算基于能量的自适应权重
        论文公式: E_b[c] = Σ_{k∈B_b} |F^(coarse)[c,k]|^2
        """
        B, C, K = fft_coarse.shape
        
        # 计算功率谱
        power_spectrum = torch.abs(fft_coarse) ** 2  # [B, C, K]
        
        # 计算各频段能量
        E_low = (power_spectrum * self.low_freq_mask.view(1, 1, -1)).sum(dim=-1)  # [B, C]
        E_mid = (power_spectrum * self.mid_freq_mask.view(1, 1, -1)).sum(dim=-1)
        E_high = (power_spectrum * self.high_freq_mask.view(1, 1, -1)).sum(dim=-1)
        
        # 论文公式: w_s ∝ exp(log E_{b(s)})
        # 添加小常数避免log(0)
        eps = 1e-8
        log_E_low = torch.log(E_low + eps)
        log_E_mid = torch.log(E_mid + eps)
        log_E_high = torch.log(E_high + eps)
        
        # Softmax归一化
        energies = torch.stack([log_E_high, log_E_mid, log_E_low], dim=-1)  # [B, C, 3]
        weights = F.softmax(energies, dim=-1)  # [B, C, 3]
        
        # 分离各尺度权重
        w_fine = weights[..., 0:1]  # [B, C, 1] - 对应高频
        w_medium = weights[..., 1:2]  # [B, C, 1] - 对应中频
        w_coarse = weights[..., 2:3]  # [B, C, 1] - 对应低频
        
        return w_coarse, w_medium, w_fine
    
    def forward(self, x):
        """
        Args:
            x: [B, L, C] 输入时间序列
            
        Returns:
            multi_scale_embeddings: Dict with keys ['fine', 'medium', 'coarse']
                Each: [B, C, 2*d_embed] (实部和虚部拼接)
            freq_weights: [B, C, 3] 能量权重
        """
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L]
        
        # ===== Coarse Scale (全长) =====
        # 论文公式: F^(coarse)_i = rFFT(X^(coarse)_i)
        fft_coarse = torch.fft.rfft(x, dim=-1)  # [B, C, n_freq_coarse]
        
        # Complex-valued MLP
        # 论文公式: E^(coarse) = ComplexMLP^(coarse)(F^(coarse))
        emb_coarse_real, emb_coarse_imag = self.complex_mlp_coarse(
            fft_coarse.real, fft_coarse.imag
        )  # [B, C, d_embed]
        
        # ===== Medium Scale (L/2窗口) =====
        num_medium_windows = L // self.medium_len
        emb_medium_real_list = []
        emb_medium_imag_list = []
        
        for i in range(num_medium_windows):
            start = i * self.medium_len
            end = start + self.medium_len
            x_window = x[:, :, start:end]
            fft_window = torch.fft.rfft(x_window, dim=-1)
            
            emb_real, emb_imag = self.complex_mlp_medium(
                fft_window.real, fft_window.imag
            )
            emb_medium_real_list.append(emb_real)
            emb_medium_imag_list.append(emb_imag)
        
        # 论文：对多个窗口取平均
        emb_medium_real = torch.stack(emb_medium_real_list, dim=0).mean(dim=0)
        emb_medium_imag = torch.stack(emb_medium_imag_list, dim=0).mean(dim=0)
        
        # ===== Fine Scale (L/4窗口) =====
        num_fine_windows = L // self.fine_len
        emb_fine_real_list = []
        emb_fine_imag_list = []
        
        for i in range(num_fine_windows):
            start = i * self.fine_len
            end = start + self.fine_len
            x_window = x[:, :, start:end]
            fft_window = torch.fft.rfft(x_window, dim=-1)
            
            emb_real, emb_imag = self.complex_mlp_fine(
                fft_window.real, fft_window.imag
            )
            emb_fine_real_list.append(emb_real)
            emb_fine_imag_list.append(emb_imag)
        
        emb_fine_real = torch.stack(emb_fine_real_list, dim=0).mean(dim=0)
        emb_fine_imag = torch.stack(emb_fine_imag_list, dim=0).mean(dim=0)
        
        # ===== Scale Positional Encoding =====
        # 论文公式: \tilde{E}^(s) = E^(s) + (P^(s)_r + jP^(s)_i)
        emb_coarse_real = emb_coarse_real + self.scale_pe_coarse_real
        emb_coarse_imag = emb_coarse_imag + self.scale_pe_coarse_imag
        emb_medium_real = emb_medium_real + self.scale_pe_medium_real
        emb_medium_imag = emb_medium_imag + self.scale_pe_medium_imag
        emb_fine_real = emb_fine_real + self.scale_pe_fine_real
        emb_fine_imag = emb_fine_imag + self.scale_pe_fine_imag
        
        # ===== Energy-Based Adaptive Weighting =====
        # 论文公式: \hat{E}^(s) = w_s ⊙ \tilde{E}^(s)
        w_coarse, w_medium, w_fine = self._compute_energy_weights(fft_coarse)
        
        emb_coarse_real = emb_coarse_real * w_coarse
        emb_coarse_imag = emb_coarse_imag * w_coarse
        
        emb_medium_real = emb_medium_real * w_medium
        emb_medium_imag = emb_medium_imag * w_medium
        
        emb_fine_real = emb_fine_real * w_fine
        emb_fine_imag = emb_fine_imag * w_fine
        
        # ===== Concatenate Real and Imaginary Parts =====
        # 论文：实部和虚部在每个尺度内拼接
        emb_coarse = torch.cat([emb_coarse_real, emb_coarse_imag], dim=-1)  # [B, C, 2*d_embed]
        emb_medium = torch.cat([emb_medium_real, emb_medium_imag], dim=-1)
        emb_fine = torch.cat([emb_fine_real, emb_fine_imag], dim=-1)
        
        multi_scale_embeddings = {
            'coarse': emb_coarse,
            'medium': emb_medium,
            'fine': emb_fine
        }
        
        # 返回能量权重用于可视化
        freq_weights = torch.cat([w_coarse, w_medium, w_fine], dim=-1)  # [B, C, 3]
        
        return multi_scale_embeddings, freq_weights


class ComplexMLP(nn.Module):
    """
    Complex-valued MLP
    论文公式: 遵循复数乘法规则
    
    对于复数输入 z = z_r + jz_i 和权重 W = W_r + jW_i:
    Wz = (W_r z_r - W_i z_i) + j(W_r z_i + W_i z_r)
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        
        # 第一层
        self.fc1_real = nn.Linear(input_dim, hidden_dim)
        self.fc1_imag = nn.Linear(input_dim, hidden_dim)
        
        # 第二层
        self.fc2_real = nn.Linear(hidden_dim, output_dim)
        self.fc2_imag = nn.Linear(hidden_dim, output_dim)
        
        self.activation = nn.GELU()
        
    def complex_linear(self, x_real, x_imag, fc_real, fc_imag):
        """
        复数线性变换
        (W_r + jW_i)(x_r + jx_i) = (W_r x_r - W_i x_i) + j(W_r x_i + W_i x_r)
        """
        out_real = fc_real(x_real) - fc_imag(x_imag)
        out_imag = fc_real(x_imag) + fc_imag(x_real)
        return out_real, out_imag
    
    def forward(self, x_real, x_imag):
        """
        Args:
            x_real: [B, C, input_dim]
            x_imag: [B, C, input_dim]
            
        Returns:
            out_real: [B, C, output_dim]
            out_imag: [B, C, output_dim]
        """
        # 第一层 + 激活
        h_real, h_imag = self.complex_linear(x_real, x_imag, self.fc1_real, self.fc1_imag)
        h_real = self.activation(h_real)
        h_imag = self.activation(h_imag)
        
        # 第二层
        out_real, out_imag = self.complex_linear(h_real, h_imag, self.fc2_real, self.fc2_imag)
        
        return out_real, out_imag


if __name__ == '__main__':
    # 测试代码
    print("=" * 80)
    print("Testing Multi-Scale Frequency Extractor")
    print("=" * 80)
    
    B, L, C = 32, 24, 37
    d_embed = 384
    
    x = torch.randn(B, L, C)
    
    model = MultiScaleFrequencyExtractor(
        n_features=C,
        d_embed=d_embed,
        input_len=L
    )
    
    multi_scale_embeddings, freq_weights = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"\nMulti-scale embeddings:")
    for scale, emb in multi_scale_embeddings.items():
        print(f"  {scale}: {emb.shape}")
    print(f"\nFrequency weights: {freq_weights.shape}")
    
    print("\n" + "=" * 80)
    print("Test passed!")
    print("=" * 80)
