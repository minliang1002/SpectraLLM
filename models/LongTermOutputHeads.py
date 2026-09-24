"""
Long-Term Forecasting Output Heads with Matched Loss Functions
用于长时预测的多种输出头设计，每种设计配套专属的 loss 函数

设计理念：
- Head A: 直接映射 → 时域 MSE Loss
- Head B: 频域→时域高维→输出 → 时域 MSE Loss
- Head C: 频域压缩→IRFFT → 时域 + 频域联合 Loss
- Head D: 多尺度频率合成 → 时域 + 多尺度频域 Loss
- Head E: 粗到细渐进预测 → 粗粒度 + 细粒度联合 Loss
- Head F: 双路径融合 → 时域 + 频域 + 融合 Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class HeadOutput:
    """输出头的统一输出格式，包含中间结果用于计算 loss"""
    pred: torch.Tensor                          # 最终时域预测 [B, pred_len]
    freq_coeffs: Optional[torch.Tensor] = None  # 预测的频域系数 (复数)
    coarse_pred: Optional[torch.Tensor] = None  # 粗粒度预测
    freq_path_pred: Optional[torch.Tensor] = None   # 频域路径预测
    time_path_pred: Optional[torch.Tensor] = None   # 时域路径预测
    scale_preds: Optional[Dict[str, torch.Tensor]] = None  # 多尺度预测


# ============================================================
# Head A: Direct Linear Projection (Baseline)
# ============================================================

class HeadA_Direct(nn.Module):
    """
    Head A: Direct Linear Projection (Baseline)
    
    freq_feat [B, d_model] → Linear → pred [B, pred_len]
    
    Loss: 纯时域 MSE
    """
    def __init__(self, d_model, pred_len):
        super().__init__()
        self.proj = nn.Linear(d_model, pred_len)
        
    def forward(self, x) -> HeadOutput:
        pred = self.proj(x)
        return HeadOutput(pred=pred)


class LossA_TimeDomain(nn.Module):
    """Head A 的 Loss: 纯时域 MSE"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, output: HeadOutput, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss_time = self.mse(output.pred, target)
        return {
            'total': loss_time,
            'time': loss_time,
        }


# ============================================================
# Head B: Frequency → Time Domain MLP
# ============================================================

class HeadB_FreqToTimeMLP(nn.Module):
    """
    Head B: Frequency → Time Domain MLP
    
    freq_feat → MLP → time_high_dim → Linear → pred
    
    Loss: 时域 MSE + 时域平滑正则
    """
    def __init__(self, d_model, pred_len, hidden_time_dim=None):
        super().__init__()
        self.hidden_time_dim = hidden_time_dim or (pred_len * 2)
        
        self.freq_to_time = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.hidden_time_dim),
            nn.GELU(),
        )
        
        self.time_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.output_proj = nn.Linear(self.hidden_time_dim, pred_len)
        
    def forward(self, x) -> HeadOutput:
        time_feat = self.freq_to_time(x)
        time_feat = time_feat.unsqueeze(1)
        time_feat = self.time_conv(time_feat).squeeze(1)
        pred = self.output_proj(time_feat)
        return HeadOutput(pred=pred)


class LossB_TimeSmooth(nn.Module):
    """Head B 的 Loss: 时域 MSE + 平滑正则"""
    def __init__(self, smooth_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.smooth_weight = smooth_weight
        
    def forward(self, output: HeadOutput, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss_time = self.mse(output.pred, target)
        
        # 平滑正则：惩罚相邻点的剧烈变化
        diff_pred = output.pred[:, 1:] - output.pred[:, :-1]
        diff_target = target[:, 1:] - target[:, :-1]
        loss_smooth = self.mse(diff_pred, diff_target)
        
        total = loss_time + self.smooth_weight * loss_smooth
        return {
            'total': total,
            'time': loss_time,
            'smooth': loss_smooth,
        }


# ============================================================
# Head C: Frequency Compression + IRFFT
# ============================================================

class HeadC_FreqCompressIRFFT(nn.Module):
    """
    Head C: Frequency Compression + IRFFT
    
    freq_feat → ComplexMLP → freq_coeffs [K complex] → IRFFT → pred
    
    Loss: 时域 MSE + 频域 MSE (幅度 + 相位)
    """
    def __init__(self, d_model, pred_len, bottleneck_dim=128):
        super().__init__()
        self.pred_len = pred_len
        self.n_freqs = pred_len // 2 + 1
        
        self.freq_compress = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.GELU(),
        )
        
        self.real_head = nn.Linear(bottleneck_dim, self.n_freqs)
        self.imag_head = nn.Linear(bottleneck_dim, self.n_freqs)
        self.amplitude_scale = nn.Parameter(torch.ones(self.n_freqs))
        
    def forward(self, x) -> HeadOutput:
        compressed = self.freq_compress(x)
        
        real = self.real_head(compressed) * self.amplitude_scale
        imag = self.imag_head(compressed) * self.amplitude_scale
        
        freq_coeffs = torch.complex(real, imag)
        pred = torch.fft.irfft(freq_coeffs, n=self.pred_len)
        
        return HeadOutput(pred=pred, freq_coeffs=freq_coeffs)


class LossC_TimeFreq(nn.Module):
    """
    Head C 的 Loss: 时域 + 频域联合 Loss
    
    L = L_time + α * L_freq_amplitude + β * L_freq_phase
    """
    def __init__(self, freq_amp_weight=0.5, freq_phase_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.freq_amp_weight = freq_amp_weight
        self.freq_phase_weight = freq_phase_weight
        
    def forward(self, output: HeadOutput, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 时域 loss
        loss_time = self.mse(output.pred, target)
        
        # 计算 target 的频域表示
        target_freq = torch.fft.rfft(target, n=output.pred.shape[-1])
        
        # 频域幅度 loss
        pred_amp = torch.abs(output.freq_coeffs)
        target_amp = torch.abs(target_freq)
        loss_freq_amp = self.mse(pred_amp, target_amp)
        
        # 频域相位 loss (使用 cos 相似度避免相位缠绕问题)
        pred_phase = torch.angle(output.freq_coeffs)
        target_phase = torch.angle(target_freq)
        # 相位差的 cos，越接近 1 越好
        phase_diff = torch.cos(pred_phase - target_phase)
        loss_freq_phase = 1 - phase_diff.mean()
        
        total = (loss_time + 
                 self.freq_amp_weight * loss_freq_amp + 
                 self.freq_phase_weight * loss_freq_phase)
        
        return {
            'total': total,
            'time': loss_time,
            'freq_amp': loss_freq_amp,
            'freq_phase': loss_freq_phase,
        }


# ============================================================
# Head D: Multi-Scale Frequency Synthesis
# ============================================================

class HeadD_MultiScaleFreqSynthesis(nn.Module):
    """
    Head D: Multi-Scale Frequency Synthesis
    
    freq_feat → [low_freq_head, mid_freq_head, high_freq_head]
             → weighted_sum → IRFFT → pred
    
    Loss: 时域 + 各频段独立 Loss
    """
    def __init__(self, d_model, pred_len, bottleneck_dim=64):
        super().__init__()
        self.pred_len = pred_len
        self.n_freqs = pred_len // 2 + 1
        
        # 频段划分
        self.low_freq_end = max(1, self.n_freqs // 4)
        self.mid_freq_end = max(2, self.n_freqs * 3 // 4)
        
        self.n_low = self.low_freq_end
        self.n_mid = self.mid_freq_end - self.low_freq_end
        self.n_high = self.n_freqs - self.mid_freq_end
        
        # 各频段预测头
        self.low_freq_head = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.n_low * 2)
        )
        
        self.mid_freq_head = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.n_mid * 2)
        )
        
        self.high_freq_head = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.n_high * 2)
        )
        
        self.scale_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.1]))
        
    def forward(self, x) -> HeadOutput:
        B = x.shape[0]
        
        low_out = self.low_freq_head(x)
        mid_out = self.mid_freq_head(x)
        high_out = self.high_freq_head(x)
        
        weights = F.softmax(self.scale_weights, dim=0)
        
        low_real, low_imag = low_out.chunk(2, dim=-1)
        mid_real, mid_imag = mid_out.chunk(2, dim=-1)
        high_real, high_imag = high_out.chunk(2, dim=-1)
        
        # 保存各尺度的频域系数（用于 loss）
        low_coeffs = torch.complex(low_real, low_imag)
        mid_coeffs = torch.complex(mid_real, mid_imag)
        high_coeffs = torch.complex(high_real, high_imag)
        
        # 应用权重
        low_real, low_imag = low_real * weights[0], low_imag * weights[0]
        mid_real, mid_imag = mid_real * weights[1], mid_imag * weights[1]
        high_real, high_imag = high_real * weights[2], high_imag * weights[2]
        
        full_real = torch.cat([low_real, mid_real, high_real], dim=-1)
        full_imag = torch.cat([low_imag, mid_imag, high_imag], dim=-1)
        
        freq_coeffs = torch.complex(full_real, full_imag)
        pred = torch.fft.irfft(freq_coeffs, n=self.pred_len)
        
        return HeadOutput(
            pred=pred,
            freq_coeffs=freq_coeffs,
            scale_preds={
                'low': low_coeffs,
                'mid': mid_coeffs,
                'high': high_coeffs,
                'weights': weights,
                'boundaries': (self.low_freq_end, self.mid_freq_end, self.n_freqs)
            }
        )


class LossD_MultiScale(nn.Module):
    """
    Head D 的 Loss: 时域 + 多尺度频域 Loss
    
    L = L_time + α_low * L_low + α_mid * L_mid + α_high * L_high
    
    低频权重大（趋势重要），高频权重小（细节次要）
    """
    def __init__(self, low_weight=1.0, mid_weight=0.5, high_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.low_weight = low_weight
        self.mid_weight = mid_weight
        self.high_weight = high_weight
        
    def forward(self, output: HeadOutput, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 时域 loss
        loss_time = self.mse(output.pred, target)
        
        # 计算 target 的频域
        target_freq = torch.fft.rfft(target, n=output.pred.shape[-1])
        
        # 获取频段边界
        low_end, mid_end, n_freqs = output.scale_preds['boundaries']
        
        # 各频段的 target
        target_low = target_freq[:, :low_end]
        target_mid = target_freq[:, low_end:mid_end]
        target_high = target_freq[:, mid_end:]
        
        # 各频段 loss (幅度)
        loss_low = self.mse(torch.abs(output.scale_preds['low']), torch.abs(target_low))
        loss_mid = self.mse(torch.abs(output.scale_preds['mid']), torch.abs(target_mid))
        loss_high = self.mse(torch.abs(output.scale_preds['high']), torch.abs(target_high))
        
        total = (loss_time + 
                 self.low_weight * loss_low + 
                 self.mid_weight * loss_mid + 
                 self.high_weight * loss_high)
        
        return {
            'total': total,
            'time': loss_time,
            'low_freq': loss_low,
            'mid_freq': loss_mid,
            'high_freq': loss_high,
            'scale_weights': output.scale_preds['weights'].detach(),
        }


# ============================================================
# Head E: Coarse-to-Fine Progressive Prediction
# ============================================================

class HeadE_CoarseToFine(nn.Module):
    """
    Head E: Coarse-to-Fine Progressive Prediction
    
    Stage 1: freq_feat → coarse_pred (低分辨率)
    Stage 2: coarse_pred + freq_feat → residual → refined_pred
    
    Loss: 粗粒度 Loss + 细粒度 Loss + 残差正则
    """
    def __init__(self, d_model, pred_len, coarse_ratio=4):
        super().__init__()
        self.pred_len = pred_len
        self.coarse_len = max(1, pred_len // coarse_ratio)
        
        # Stage 1: 粗粒度预测
        self.coarse_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.coarse_len)
        )
        
        self.upsample = nn.Upsample(size=pred_len, mode='linear', align_corners=True)
        
        # Stage 2: 残差细化
        self.refine_encoder = nn.Linear(pred_len, d_model // 2)
        
        self.refine_head = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len)
        )
        
        self.residual_gate = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x) -> HeadOutput:
        # Stage 1
        coarse = self.coarse_head(x)
        coarse_up = self.upsample(coarse.unsqueeze(1)).squeeze(1)
        
        # Stage 2
        coarse_feat = self.refine_encoder(coarse_up)
        combined = torch.cat([x, coarse_feat], dim=-1)
        residual = self.refine_head(combined)
        
        pred = coarse_up + self.residual_gate * residual
        
        return HeadOutput(pred=pred, coarse_pred=coarse_up)


class LossE_CoarseToFine(nn.Module):
    """
    Head E 的 Loss: 粗粒度 + 细粒度 + 残差正则
    
    L = L_fine + α * L_coarse + β * L_residual_reg
    
    粗粒度 loss 确保趋势正确，残差正则防止过拟合
    """
    def __init__(self, coarse_weight=0.5, residual_reg_weight=0.01):
        super().__init__()
        self.mse = nn.MSELoss()
        self.coarse_weight = coarse_weight
        self.residual_reg_weight = residual_reg_weight
        
    def forward(self, output: HeadOutput, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 细粒度 loss (最终预测)
        loss_fine = self.mse(output.pred, target)
        
        # 粗粒度 loss (下采样 target 后比较)
        coarse_len = output.coarse_pred.shape[-1]
        # 使用平均池化下采样 target
        target_coarse = F.adaptive_avg_pool1d(
            target.unsqueeze(1), coarse_len
        ).squeeze(1)
        # 再上采样回原长度比较
        target_coarse_up = F.interpolate(
            target_coarse.unsqueeze(1), size=target.shape[-1], mode='linear', align_corners=True
        ).squeeze(1)
        loss_coarse = self.mse(output.coarse_pred, target_coarse_up)
        
        # 残差正则 (残差应该是小的高频修正)
        residual = output.pred - output.coarse_pred
        loss_residual_reg = (residual ** 2).mean()
        
        total = (loss_fine + 
                 self.coarse_weight * loss_coarse + 
                 self.residual_reg_weight * loss_residual_reg)
        
        return {
            'total': total,
            'fine': loss_fine,
            'coarse': loss_coarse,
            'residual_reg': loss_residual_reg,
        }


# ============================================================
# Head F: Hybrid Frequency-Time Dual Path
# ============================================================

class HeadF_HybridFreqTime(nn.Module):
    """
    Head F: Hybrid Frequency-Time Dual Path
    
    Path 1 (Freq): freq_feat → freq_coeffs → IRFFT → pred_freq
    Path 2 (Time): freq_feat → MLP → pred_time
    Output: learnable_weight * pred_freq + (1 - weight) * pred_time
    
    Loss: 时域总 Loss + 频域路径 Loss + 时域路径 Loss + 一致性 Loss
    """
    def __init__(self, d_model, pred_len, bottleneck_dim=128):
        super().__init__()
        self.pred_len = pred_len
        self.n_freqs = pred_len // 2 + 1
        
        # Path 1: Frequency path
        self.freq_path = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
        )
        self.freq_real = nn.Linear(bottleneck_dim, self.n_freqs)
        self.freq_imag = nn.Linear(bottleneck_dim, self.n_freqs)
        
        # Path 2: Time path
        self.time_path = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, pred_len)
        )
        
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x) -> HeadOutput:
        # Path 1: Frequency
        freq_feat = self.freq_path(x)
        real = self.freq_real(freq_feat)
        imag = self.freq_imag(freq_feat)
        freq_coeffs = torch.complex(real, imag)
        pred_freq = torch.fft.irfft(freq_coeffs, n=self.pred_len)
        
        # Path 2: Time
        pred_time = self.time_path(x)
        
        # Fusion
        w = torch.sigmoid(self.fusion_weight)
        pred = w * pred_freq + (1 - w) * pred_time
        
        return HeadOutput(
            pred=pred,
            freq_coeffs=freq_coeffs,
            freq_path_pred=pred_freq,
            time_path_pred=pred_time
        )


class LossF_Hybrid(nn.Module):
    """
    Head F 的 Loss: 多路径联合 Loss
    
    L = L_total + α * L_freq_path + β * L_time_path + γ * L_consistency
    
    一致性 loss 鼓励两条路径的预测接近
    """
    def __init__(self, freq_path_weight=0.3, time_path_weight=0.3, 
                 consistency_weight=0.1, freq_domain_weight=0.2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.freq_path_weight = freq_path_weight
        self.time_path_weight = time_path_weight
        self.consistency_weight = consistency_weight
        self.freq_domain_weight = freq_domain_weight
        
    def forward(self, output: HeadOutput, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 总体时域 loss
        loss_total = self.mse(output.pred, target)
        
        # 频域路径 loss
        loss_freq_path = self.mse(output.freq_path_pred, target)
        
        # 时域路径 loss
        loss_time_path = self.mse(output.time_path_pred, target)
        
        # 一致性 loss (两条路径应该预测相似的结果)
        loss_consistency = self.mse(output.freq_path_pred, output.time_path_pred)
        
        # 频域 loss (针对频域路径)
        target_freq = torch.fft.rfft(target, n=output.pred.shape[-1])
        loss_freq_domain = self.mse(torch.abs(output.freq_coeffs), torch.abs(target_freq))
        
        total = (loss_total + 
                 self.freq_path_weight * loss_freq_path +
                 self.time_path_weight * loss_time_path +
                 self.consistency_weight * loss_consistency +
                 self.freq_domain_weight * loss_freq_domain)
        
        return {
            'total': total,
            'time_total': loss_total,
            'freq_path': loss_freq_path,
            'time_path': loss_time_path,
            'consistency': loss_consistency,
            'freq_domain': loss_freq_domain,
        }


# ============================================================
# 统一的多任务输出头包装器
# ============================================================

class MultiTaskLongTermOutputHead(nn.Module):
    """
    多任务长时预测输出头包装器
    """
    def __init__(self, d_model, pred_len, num_tasks, head_type='A', **kwargs):
        super().__init__()
        self.num_tasks = num_tasks
        self.head_type = head_type
        
        head_class = {
            'A': HeadA_Direct,
            'B': HeadB_FreqToTimeMLP,
            'C': HeadC_FreqCompressIRFFT,
            'D': HeadD_MultiScaleFreqSynthesis,
            'E': HeadE_CoarseToFine,
            'F': HeadF_HybridFreqTime,
        }[head_type]
        
        self.heads = nn.ModuleList([
            head_class(d_model, pred_len, **kwargs) for _ in range(num_tasks)
        ])
        
    def forward(self, last_hidden_state, target_indices) -> List[HeadOutput]:
        outputs = []
        if last_hidden_state.dim() == 4:
            # Paper path: one backbone sequence per task, with task i's
            # forecasting head reading that task's target token.
            for i, idx in enumerate(target_indices):
                feat = last_hidden_state[:, i, idx, :]
                output = self.heads[i](feat)
                outputs.append(output)
            return outputs

        for i, idx in enumerate(target_indices):
            feat = last_hidden_state[:, idx, :]
            output = self.heads[i](feat)
            outputs.append(output)
        return outputs


class MultiTaskLongTermLoss(nn.Module):
    """
    多任务长时预测 Loss 包装器
    """
    def __init__(self, head_type='A', num_tasks=4, **kwargs):
        super().__init__()
        self.head_type = head_type
        
        loss_class = {
            'A': LossA_TimeDomain,
            'B': LossB_TimeSmooth,
            'C': LossC_TimeFreq,
            'D': LossD_MultiScale,
            'E': LossE_CoarseToFine,
            'F': LossF_Hybrid,
        }[head_type]
        
        self.losses = nn.ModuleList([
            loss_class(**kwargs) for _ in range(num_tasks)
        ])
        
    def forward(self, outputs: List[HeadOutput], targets: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: List of HeadOutput from each task
            targets: List of target tensors [B, pred_len] for each task
        """
        total_loss = 0
        all_losses = {}
        
        for i, (output, target) in enumerate(zip(outputs, targets)):
            task_losses = self.losses[i](output, target)
            total_loss = total_loss + task_losses['total']
            
            # 记录每个任务的详细 loss
            for key, value in task_losses.items():
                all_losses[f'task{i}_{key}'] = value
        
        all_losses['total'] = total_loss
        return all_losses


# ============================================================
# 工厂函数：快速创建输出头和对应的 Loss
# ============================================================

def create_output_head_and_loss(head_type, d_model, pred_len, num_tasks, **kwargs):
    """
    工厂函数：创建输出头和对应的 Loss
    
    Args:
        head_type: 'A', 'B', 'C', 'D', 'E', 'F'
        d_model: 模型隐藏维度
        pred_len: 预测长度
        num_tasks: 任务数量
        **kwargs: 传递给输出头和 Loss 的额外参数
    
    Returns:
        output_head: MultiTaskLongTermOutputHead
        loss_fn: MultiTaskLongTermLoss
    """
    # 分离输出头参数和 loss 参数
    head_kwargs = {}
    loss_kwargs = {}
    
    # Head 特定参数
    head_params = {
        'B': ['hidden_time_dim'],
        'C': ['bottleneck_dim'],
        'D': ['bottleneck_dim'],
        'E': ['coarse_ratio'],
        'F': ['bottleneck_dim'],
    }
    
    # Loss 特定参数
    loss_params = {
        'B': ['smooth_weight'],
        'C': ['freq_amp_weight', 'freq_phase_weight'],
        'D': ['low_weight', 'mid_weight', 'high_weight'],
        'E': ['coarse_weight', 'residual_reg_weight'],
        'F': ['freq_path_weight', 'time_path_weight', 'consistency_weight', 'freq_domain_weight'],
    }
    
    for key, value in kwargs.items():
        if head_type in head_params and key in head_params[head_type]:
            head_kwargs[key] = value
        if head_type in loss_params and key in loss_params[head_type]:
            loss_kwargs[key] = value
    
    output_head = MultiTaskLongTermOutputHead(
        d_model=d_model,
        pred_len=pred_len,
        num_tasks=num_tasks,
        head_type=head_type,
        **head_kwargs
    )
    
    loss_fn = MultiTaskLongTermLoss(
        head_type=head_type,
        num_tasks=num_tasks,
        **loss_kwargs
    )
    
    return output_head, loss_fn


# ============================================================
# 参数统计和比较
# ============================================================

def count_head_parameters(head_type, d_model, pred_len, **kwargs):
    """统计单个输出头的参数量"""
    head_class = {
        'A': HeadA_Direct,
        'B': HeadB_FreqToTimeMLP,
        'C': HeadC_FreqCompressIRFFT,
        'D': HeadD_MultiScaleFreqSynthesis,
        'E': HeadE_CoarseToFine,
        'F': HeadF_HybridFreqTime,
    }[head_type]
    
    head = head_class(d_model, pred_len, **kwargs)
    return sum(p.numel() for p in head.parameters())


def compare_all_heads(d_model=1024, pred_len=96):
    """比较所有输出头的参数量"""
    print(f"\n{'='*60}")
    print(f"Output Head Parameter Comparison")
    print(f"d_model={d_model}, pred_len={pred_len}")
    print(f"{'='*60}")
    
    heads = ['A', 'B', 'C', 'D', 'E', 'F']
    for h in heads:
        params = count_head_parameters(h, d_model, pred_len)
        print(f"Head {h}: {params:,} parameters")
    
    print(f"{'='*60}\n")


# ============================================================
# 测试代码
# ============================================================

if __name__ == '__main__':
    import torch
    
    B, d_model, pred_len, num_tasks = 4, 1024, 96, 4
    
    print("=" * 70)
    print("Testing Long-Term Output Heads with Matched Losses")
    print("=" * 70)
    
    # 模拟输入
    x = torch.randn(B, num_tasks, d_model)  # [B, M, d_model]
    target_indices = list(range(num_tasks))
    targets = [torch.randn(B, pred_len) for _ in range(num_tasks)]
    
    heads = ['A', 'B', 'C', 'D', 'E', 'F']
    
    for head_type in heads:
        print(f"\n--- Head {head_type} ---")
        
        # 创建输出头和 loss
        output_head, loss_fn = create_output_head_and_loss(
            head_type=head_type,
            d_model=d_model,
            pred_len=pred_len,
            num_tasks=num_tasks
        )
        
        # 前向传播
        outputs = output_head(x, target_indices)
        
        # 计算 loss
        losses = loss_fn(outputs, targets)
        
        # 打印结果
        print(f"  Output shape: {outputs[0].pred.shape}")
        print(f"  Loss components:")
        for key, value in losses.items():
            if isinstance(value, torch.Tensor) and value.dim() == 0:
                print(f"    {key}: {value.item():.4f}")
    
    # 参数量比较
    compare_all_heads(d_model, pred_len)
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
