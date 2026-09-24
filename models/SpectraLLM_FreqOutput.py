"""
SpectraLLM_FreqOutput: 频域输入+频域输出版本
输入和输出都在频域，最后通过逆FFT转换回时域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.FreqQwen2_V12 import Qwen2CoupledAdapter
from transformers import AutoModel, AutoConfig
from models.FreqModules_MultiScale import MultiScaleFrequencyExtractor


class FrequencyAwareTaskOrientedAttention(nn.Module):
    """
    Frequency-Aware Task-Oriented Attention (NoValue版本)
    """
    
    def __init__(self, d_embed=384, num_scales=3, dropout=0.1, use_value_proj=False):
        super().__init__()
        self.d_embed = d_embed * 2  # 实部+虚部
        self.num_scales = num_scales
        self.scale_names = ['fine', 'medium', 'coarse']
        self.use_value_proj = use_value_proj
        
        # Pre-LayerNorm
        self.pre_ln_q = nn.ModuleDict({
            s: nn.LayerNorm(self.d_embed) for s in self.scale_names
        })
        self.pre_ln_k = nn.ModuleDict({
            s: nn.LayerNorm(self.d_embed) for s in self.scale_names
        })
        
        # Q和K投影
        self.q_projs = nn.ModuleDict({
            s: nn.Linear(self.d_embed, self.d_embed) for s in self.scale_names
        })
        self.k_projs = nn.ModuleDict({
            s: nn.Linear(self.d_embed, self.d_embed) for s in self.scale_names
        })
        
        # 融合权重
        self.fusion_weights = nn.Parameter(torch.ones(num_scales, num_scales) / num_scales)
        
        # Temperature
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_embed * num_scales, self.d_embed * num_scales)
        
        # Post-LayerNorm
        self.post_ln = nn.LayerNorm(self.d_embed * num_scales)
        
        # Residual gate
        self.residual_gate = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, multi_scale_embs, target_idx):
        B = multi_scale_embs['fine'].shape[0]
        C = multi_scale_embs['fine'].shape[1]
        
        original_embs = [multi_scale_embs[s] for s in self.scale_names]
        original_concat = torch.cat(original_embs, dim=-1)
        
        modulated_scales = []
        
        for s_idx, src_scale in enumerate(self.scale_names):
            src_emb = multi_scale_embs[src_scale]
            
            target_feat = src_emb[:, target_idx:target_idx+1, :]
            mask = torch.ones(C, dtype=torch.bool, device=src_emb.device)
            mask[target_idx] = False
            context_feats = src_emb[:, mask, :]
            
            target_feat_norm = self.pre_ln_q[src_scale](target_feat)
            Q = self.q_projs[src_scale](target_feat_norm)
            
            attn_maps = []
            value_outputs = []
            
            for tgt_scale in self.scale_names:
                tgt_emb = multi_scale_embs[tgt_scale]
                tgt_context = tgt_emb[:, mask, :]
                
                tgt_context_norm = self.pre_ln_k[tgt_scale](tgt_context)
                K = self.k_projs[tgt_scale](tgt_context_norm)
                
                scores = torch.matmul(Q, K.transpose(-2, -1))
                scores = scores / (self.d_embed ** 0.5) / torch.clamp(self.temperature, min=0.1)
                
                attn = torch.sigmoid(scores)
                attn = self.dropout(attn)
                attn_maps.append(attn)
                value_outputs.append(tgt_context)
            
            fusion_w = F.softmax(self.fusion_weights[s_idx], dim=0)
            fused_attn = sum(w * a for w, a in zip(fusion_w, attn_maps))
            
            # NoValue: 直接调制原始特征
            modulated_context = context_feats * fused_attn.transpose(-2, -1)
            
            modulated_full = src_emb.clone()
            modulated_full[:, mask, :] = modulated_context
            modulated_scales.append(modulated_full)
        
        output = torch.cat(modulated_scales, dim=-1)
        output = self.out_proj(output)
        output = self.dropout(output)
        
        # 残差连接
        output = original_concat + self.residual_gate * output
        output = self.post_ln(output)
        
        return output


class FrequencyDomainOutputHead(nn.Module):
    """
    频域输出头：预测频域系数，然后通过逆FFT转换回时域
    """
    
    def __init__(self, d_model, pred_len, num_tasks=4):
        super().__init__()
        self.pred_len = pred_len
        self.num_tasks = num_tasks
        
        # 计算频域长度 (pred_len // 2 + 1 for rfft)
        self.freq_len = pred_len // 2 + 1
        
        # 为每个任务创建独立的频域预测头
        self.freq_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, self.freq_len * 2)  # *2 for real and imag
            )
            for _ in range(num_tasks)
        ])
        
    def forward(self, hidden_states, target_indices):
        """
        Args:
            hidden_states: [B, seq_len, d_model]
            target_indices: list of target indices
        Returns:
            time_domain_preds: list of [B, pred_len, 1] tensors
            freq_domain_preds: list of complex tensors for analysis
        """
        B = hidden_states.shape[0]
        
        # 使用最后一个时间步的特征
        last_hidden = hidden_states[:, -1, :]  # [B, d_model]
        
        time_domain_preds = []
        freq_domain_preds = []
        
        for i, predictor in enumerate(self.freq_predictors):
            # 预测频域系数
            freq_flat = predictor(last_hidden)  # [B, freq_len * 2]
            freq_flat = freq_flat.view(B, self.freq_len, 2)  # [B, freq_len, 2]
            
            # 构建复数频域表示
            freq_real = freq_flat[:, :, 0]  # [B, freq_len]
            freq_imag = freq_flat[:, :, 1]  # [B, freq_len]
            freq_complex = torch.complex(freq_real, freq_imag)  # [B, freq_len]
            
            # 逆FFT转换回时域
            time_domain = torch.fft.irfft(freq_complex, n=self.pred_len, dim=-1)  # [B, pred_len]
            time_domain = time_domain.unsqueeze(-1)  # [B, pred_len, 1]
            
            time_domain_preds.append(time_domain)
            freq_domain_preds.append(freq_complex)
        
        return time_domain_preds, freq_domain_preds


class SpectraLLM_FreqOutput(nn.Module):
    """
    SpectraLLM with Frequency Domain Output
    
    Architecture:
    - Stage 1: Multi-Scale Frequency Extraction (输入频域)
    - Stage 2: Frequency-Aware Task-Oriented Attention
    - Stage 3: Qwen2 Transformer + Adapters
    - Stage 4: Frequency Domain Output Head (输出频域 + 逆FFT)
    """
    
    def __init__(self, configs, target_indices=[0, 1, 2, 3]):
        super().__init__()
        self.configs = configs
        self.target_indices = target_indices
        self.num_tasks = len(target_indices)
        self.qwen2_model_name = getattr(configs, 'qwen2_path', 'Qwen/Qwen1.5-0.5B')
        
        print(f"Initializing SpectraLLM_FreqOutput")
        print(f"  ✓ Stage 1: Multi-Scale Frequency Extraction (Input)")
        print(f"  ✓ Stage 2: Frequency-Aware Attention (NoValue)")
        print(f"  ✓ Stage 3: Qwen2 Transformer (LayerNorm trainable)")
        print(f"  ✓ Stage 4: Frequency Domain Output Head + IRFFT")
        
        qwen2_config = AutoConfig.from_pretrained(self.qwen2_model_name, trust_remote_code=True)
        self.d_model = qwen2_config.hidden_size
        
        # Stage 1
        self.freq_extractor = MultiScaleFrequencyExtractor(
            n_features=configs.enc_in,
            d_embed=256,
            input_len=configs.seq_len
        )
        
        self.input_projection = nn.Linear(256 * 2 * 3, self.d_model)
        
        # Stage 2
        self.freq_attentions = nn.ModuleList([
            FrequencyAwareTaskOrientedAttention(
                d_embed=256, 
                num_scales=3,
                dropout=0.1,
                use_value_proj=False
            )
            for _ in range(self.num_tasks)
        ])
        
        # Stage 3
        self.qwen2 = AutoModel.from_pretrained(
            self.qwen2_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        num_layers = getattr(configs, 'qwen2_layers', 6)
        if num_layers < len(self.qwen2.layers):
            self.qwen2.layers = self.qwen2.layers[:num_layers]
        
        self.adapters = nn.ModuleList([
            Qwen2CoupledAdapter(self.d_model, configs.adapter_dim, self.num_tasks)
            for _ in range(len(self.qwen2.layers))
        ])
        
        self._freeze_qwen2_except_layernorm()
        
        # Stage 4: 频域输出头
        self.output_head = FrequencyDomainOutputHead(
            d_model=self.d_model,
            pred_len=configs.pred_len,
            num_tasks=self.num_tasks
        )
    
    def _freeze_qwen2_except_layernorm(self):
        for param in self.qwen2.parameters():
            param.requires_grad = False
        
        trainable_ln = 0
        for layer in self.qwen2.layers:
            for param in layer.input_layernorm.parameters():
                param.requires_grad = True
                trainable_ln += param.numel()
            for param in layer.post_attention_layernorm.parameters():
                param.requires_grad = True
                trainable_ln += param.numel()
        
        for param in self.qwen2.norm.parameters():
            param.requires_grad = True
            trainable_ln += param.numel()
        
        print(f"  ✓ Qwen2 LayerNorm trainable: {trainable_ln:,}")
    
    def forward(self, x):
        x = x.float()
        
        # Stage 1: 输入频域提取
        multi_scale_embs, freq_weights = self.freq_extractor(x)
        
        # Stage 2: 频域注意力
        attended_embs = []
        for i, attn_mod in enumerate(self.freq_attentions):
            target_idx = self.target_indices[i]
            attended = attn_mod(multi_scale_embs, target_idx)
            attended_embs.append(attended)
        
        x_embed = torch.stack(attended_embs, dim=0).mean(dim=0)
        x_embed = self.input_projection(x_embed)
        
        hidden_states = x_embed
        
        # Stage 3: Qwen2 Transformer
        for layer_idx, layer in enumerate(self.qwen2.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            
            attn_output, _, _ = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = residual + attn_output
            
            adapter_out = self.adapters[layer_idx](hidden_states, self.target_indices)
            hidden_states = hidden_states + adapter_out
            
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
        
        hidden_states = self.qwen2.norm(hidden_states)
        
        # Stage 4: 频域输出 + 逆FFT
        time_domain_preds, freq_domain_preds = self.output_head(hidden_states, self.target_indices)
        
        return time_domain_preds, freq_weights, freq_domain_preds
    
    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100*trainable/total:.2f}")
