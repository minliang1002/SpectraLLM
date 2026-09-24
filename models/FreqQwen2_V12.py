"""
FreqQwen2 V12 - Improved Frequency Positional Encoding
改进的频率位置编码：在投影之后添加，保留完整的位置信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModel, AutoConfig

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.FreqModules_V12_FreqPE_Improved import (
    ComplexFrequencyAwareMultiResolutionExtractorV12,
    ComplexFeatureDenoisingAttention
)


class Qwen2CoupledAdapter(nn.Module):
    def __init__(self, d_model, bottle_dim=64, num_tasks=4):
        super().__init__()
        self.d_model = d_model
        self.num_tasks = num_tasks
        
        self.shared_down = nn.Linear(d_model * num_tasks, bottle_dim * num_tasks)
        self.shared_up = nn.Linear(bottle_dim * num_tasks, d_model * num_tasks)
        
        self.specific_downs = nn.ModuleList([
            nn.Linear(d_model, bottle_dim) for _ in range(num_tasks)
        ])
        self.specific_ups = nn.ModuleList([
            nn.Linear(bottle_dim, d_model) for _ in range(num_tasks)
        ])
        
        self.gate = nn.Parameter(torch.ones(1) * 1e-3)
        self.act = nn.SiLU()

    def forward(self, hidden_states, target_indices):
        """Apply shared and task-specific updates to task target tokens.

        The paper model carries one sequence per forecasting task.  Therefore
        the main implementation uses ``[B, M, C, D]`` and lets the shared
        adapter mix the M target tokens at each transformer layer.  The
        original ``[B, C, D]`` path is retained for the older standalone
        FreqQwen2 model in this repository.
        """
        if hidden_states.dim() == 4:
            B, M, C, D = hidden_states.shape
            if M != self.num_tasks or len(target_indices) != M:
                raise ValueError(
                    f"Expected {self.num_tasks} task sequences and target indices, "
                    f"got shape {tuple(hidden_states.shape)} and {target_indices}."
                )

            target_feats = torch.stack(
                [
                    hidden_states[:, task_idx, channel_idx, :]
                    for task_idx, channel_idx in enumerate(target_indices)
                ],
                dim=1,
            )  # [B, M, D]
            flat_targets = target_feats.reshape(B, -1)
            shared_latents = self.act(self.shared_down(flat_targets))
            shared_out = self.shared_up(shared_latents).view(B, M, D)

            specific_outs = []
            for task_idx in range(M):
                task_feat = target_feats[:, task_idx:task_idx + 1, :]
                task_out = self.specific_ups[task_idx](
                    self.act(self.specific_downs[task_idx](task_feat))
                ).squeeze(1)
                specific_outs.append(task_out)
            specific_out = torch.stack(specific_outs, dim=1)

            total_delta = shared_out + specific_out
            full_delta = torch.zeros_like(hidden_states)
            for task_idx, channel_idx in enumerate(target_indices):
                full_delta[:, task_idx, channel_idx, :] = total_delta[:, task_idx, :]
            return self.gate * full_delta

        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must have shape [B, C, D] or [B, M, C, D], "
                f"got {tuple(hidden_states.shape)}"
            )

        B, _, D = hidden_states.shape
        target_feats = hidden_states[:, target_indices, :]
        flat_targets = target_feats.reshape(B, -1)
        shared_latents = self.act(self.shared_down(flat_targets))
        shared_out = self.shared_up(shared_latents).view(B, self.num_tasks, D)

        specific_outs = []
        for i in range(self.num_tasks):
            t_in = target_feats[:, i:i + 1, :]
            t_out = self.specific_ups[i](self.act(self.specific_downs[i](t_in)))
            specific_outs.append(t_out)
        specific_out = torch.cat(specific_outs, dim=1)

        total_delta = shared_out + specific_out
        full_delta = torch.zeros_like(hidden_states)
        full_delta[:, target_indices, :] = total_delta
        return self.gate * full_delta


class Qwen2DenoisingAttention(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        
    def forward(self, x, target_idx):
        B, M, D = x.shape
        
        q = self.q_proj(x[:, target_idx:target_idx+1, :])
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_gate = torch.sigmoid(scores)
        context = torch.matmul(attn_gate, v)
        
        context = context.transpose(1, 2).contiguous().view(B, 1, D)
        output = self.out_proj(context)
        
        out = x.clone()
        out[:, target_idx:target_idx+1, :] = x[:, target_idx:target_idx+1, :] + output
        
        return self.ln(out)


class Qwen2OutputHead(nn.Module):
    def __init__(self, d_model, pred_len, num_tasks):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(d_model, pred_len) for _ in range(num_tasks)
        ])
        
    def forward(self, last_hidden_state, target_indices):
        preds = []
        for i, idx in enumerate(target_indices):
            feat = last_hidden_state[:, idx, :]
            pred = self.heads[i](feat)
            preds.append(pred)
        return preds


class FreqQwen2_V12(nn.Module):
    """
    FreqQwen2 V12 - Improved Frequency Positional Encoding
    
    改进点：
    1. 尺度位置编码：在 ComplexMLP 投影之后添加
    2. 频带编码：区分 fine/medium/coarse 频带
    """
    def __init__(self, configs, target_indices=[0, 1, 2, 3]):
        super().__init__()
        self.configs = configs
        self.target_indices = target_indices
        self.num_tasks = len(target_indices)
        
        self.qwen2_model_name = getattr(configs, 'qwen2_path', 'Qwen/Qwen1.5-0.5B')
        
        print(f"Initializing FreqQwen2 V12 (Improved FreqPE) with {self.num_tasks} tasks")
        print(f"Loading Qwen2 from: {self.qwen2_model_name}")
        
        qwen2_config = AutoConfig.from_pretrained(self.qwen2_model_name, trust_remote_code=True)
        self.d_model = qwen2_config.hidden_size
        print(f"Qwen2 hidden size: {self.d_model}")
        
        # Stage 1: Complex Frequency Encoder with Improved PE
        self.freq_extractor = ComplexFrequencyAwareMultiResolutionExtractorV12(
            n_features=configs.enc_in,
            d_model=768,
            input_len=configs.seq_len
        )
        
        # 维度适配
        if self.d_model != 768:
            self.input_projection = nn.Linear(768, self.d_model)
        else:
            self.input_projection = nn.Identity()
        
        # Stage 2: Denoising Attention
        self.denoising_modules = nn.ModuleList([
            Qwen2DenoisingAttention(d_model=self.d_model, n_heads=8)
            for _ in range(self.num_tasks)
        ])
        
        # Stage 3: Qwen2 + Adapters
        print("Loading Qwen2 model...")
        self.qwen2 = AutoModel.from_pretrained(
            self.qwen2_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        num_layers = getattr(configs, 'qwen2_layers', 6)
        if num_layers < len(self.qwen2.layers):
            self.qwen2.layers = self.qwen2.layers[:num_layers]
            print(f"Using first {num_layers} layers of Qwen2")
        
        self.adapters = nn.ModuleList([
            Qwen2CoupledAdapter(
                d_model=self.d_model,
                bottle_dim=configs.adapter_dim,
                num_tasks=self.num_tasks
            )
            for _ in range(len(self.qwen2.layers))
        ])
        
        print("Freezing Qwen2 backbone...")
        for param in self.qwen2.parameters():
            param.requires_grad = False
        
        # Stage 4: Output Head
        self.output_head = Qwen2OutputHead(
            d_model=self.d_model,
            pred_len=configs.pred_len,
            num_tasks=self.num_tasks
        )

    def forward(self, x):
        x = x.float()
        B, L, M = x.shape
        
        # Stage 1: Complex Frequency Encoding with Improved PE
        x_embed, freq_weights = self.freq_extractor(x)
        x_embed = self.input_projection(x_embed)
        
        # Stage 2: Denoising
        for i, mod in enumerate(self.denoising_modules):
            target_idx = self.target_indices[i]
            x_embed = mod(x_embed, target_idx=target_idx)
        
        # Stage 3: Qwen2 + Adapters
        hidden_states = x_embed
        
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
        
        # Stage 4: Output
        preds = self.output_head(hidden_states, self.target_indices)
        preds = [p.unsqueeze(-1) for p in preds]
        
        return preds, freq_weights

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || "
            f"trainable%: {100 * trainable_params / all_param:.2f}"
        )
