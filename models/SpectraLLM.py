"""SpectraLLM: spectral large language model for multi-load forecasting."""

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
from models.LongTermOutputHeads import MultiTaskLongTermOutputHead
from models.FreqModules_MultiScale import MultiScaleFrequencyExtractor


class FrequencyAwareTaskOrientedAttention(nn.Module):
    """
    Frequency-Aware Task-Oriented Attention (FATA).

    For one forecasting task, FATA computes target queries and context keys
    at every resolution, fuses intra-scale and cross-scale sigmoid gates,
    projects context values, and restores the target/context rows to the
    original channel order.
    """

    def __init__(self, d_embed=384, num_scales=3, output_dim=None, dropout=0.1):
        super().__init__()
        self.d_embed = d_embed * 2  # real + imaginary parts
        self.d_k = self.d_embed
        self.num_scales = num_scales
        # Keep the concatenation order identical to the manuscript:
        # coarse, medium, fine.
        self.scale_names = ['coarse', 'medium', 'fine']
        self.output_dim = output_dim

        # The normalization layers are applied before the learned Q/K/V
        # projections.  A separate V projection is part of the paper model.
        self.pre_ln_q = nn.ModuleDict({
            s: nn.LayerNorm(self.d_embed) for s in self.scale_names
        })
        self.pre_ln_k = nn.ModuleDict({
            s: nn.LayerNorm(self.d_embed) for s in self.scale_names
        })
        self.pre_ln_v = nn.ModuleDict({
            s: nn.LayerNorm(self.d_embed) for s in self.scale_names
        })

        self.q_projs = nn.ModuleDict({
            s: nn.Linear(self.d_embed, self.d_k) for s in self.scale_names
        })
        self.k_projs = nn.ModuleDict({
            s: nn.Linear(self.d_embed, self.d_k) for s in self.scale_names
        })
        self.v_projs = nn.ModuleDict({
            s: nn.Linear(self.d_embed, self.d_k) for s in self.scale_names
        })
        self.target_projs = nn.ModuleDict({
            s: nn.Linear(self.d_embed, self.d_k) for s in self.scale_names
        })

        # Row-wise softmax starts with a larger same-scale logit, as described
        # by the diagonal-favoring initialization in the manuscript.
        fusion_init = torch.zeros(num_scales, num_scales)
        fusion_init.fill_diagonal_(1.0)
        self.fusion_weights = nn.Parameter(fusion_init)

        self.dropout = nn.Dropout(dropout)
        # W_O maps the concatenated resolution features to the LLM hidden
        # size.  Keeping output_dim=None retains the standalone extractor API.
        out_dim = output_dim or (self.d_k * num_scales)
        self.out_proj = nn.Linear(self.d_k * num_scales, out_dim)

    def forward(self, multi_scale_embs, target_idx):
        C = multi_scale_embs['fine'].shape[1]

        task_scales = []
        for s_idx, src_scale in enumerate(self.scale_names):
            src_emb = multi_scale_embs[src_scale]
            target_feat = src_emb[:, target_idx:target_idx+1, :]
            mask = torch.ones(C, dtype=torch.bool, device=src_emb.device)
            mask[target_idx] = False

            target_feat_norm = self.pre_ln_q[src_scale](target_feat)
            Q = self.q_projs[src_scale](target_feat_norm)

            attn_maps = []
            for context_scale in self.scale_names:
                context = multi_scale_embs[context_scale][:, mask, :]
                K = self.k_projs[context_scale](
                    self.pre_ln_k[context_scale](context)
                )
                scores = torch.matmul(Q, K.transpose(-2, -1))
                scores = scores / (self.d_k ** 0.5)
                attn = torch.sigmoid(scores)
                attn = self.dropout(attn)
                attn_maps.append(attn)

            fusion_w = F.softmax(self.fusion_weights[s_idx], dim=0)
            fused_attn = sum(w * a for w, a in zip(fusion_w, attn_maps))

            # Values come from the source/output resolution.  The target row
            # is projected separately so every row has the same d_k width.
            context = src_emb[:, mask, :]
            V = self.v_projs[src_scale](self.pre_ln_v[src_scale](context))
            modulated_context = V * fused_attn.transpose(-2, -1)
            target_repr = self.target_projs[src_scale](target_feat)

            restored = target_repr.new_zeros(
                target_repr.shape[0], C, self.d_k
            )
            restored[:, target_idx:target_idx+1, :] = target_repr
            restored[:, mask, :] = modulated_context
            task_scales.append(restored)

        concatenated = torch.cat(task_scales, dim=-1)
        return self.dropout(self.out_proj(concatenated))


class SpectraLLM(nn.Module):
    """
    SpectraLLM: Spectral Large Language Model
    
    Architecture:
    - Stage 1: Multi-Scale Frequency Extraction
    - Stage 2: Frequency-Aware Task-Oriented Attention
    - Stage 3: Qwen2 Transformer + Adapters (LayerNorm trainable)
    - Stage 4: Multi-Task Output Head
    """
    
    def __init__(self, configs, target_indices=[0, 1, 2, 3]):
        super().__init__()
        self.configs = configs
        self.target_indices = target_indices
        self.num_tasks = len(target_indices)
        self.qwen2_model_name = getattr(configs, 'qwen2_path', 'Qwen/Qwen1.5-0.5B')
        
        print(f"Initializing SpectraLLM")
        print(f"  ✓ Stage 1: Multi-Scale Frequency Extraction")
        print(f"  ✓ Stage 2: Frequency-Aware Task-Oriented Attention (Q/K/V)")
        print(f"  ✓ Stage 3: Qwen2 Transformer (LayerNorm trainable)")
        print(f"  ✓ Stage 4: Multi-Task Output Head")
        
        qwen2_config = AutoConfig.from_pretrained(self.qwen2_model_name, trust_remote_code=True)
        self.d_model = qwen2_config.hidden_size
        
        # Stage 1
        self.freq_extractor = MultiScaleFrequencyExtractor(
            n_features=configs.enc_in,
            d_embed=256,
            input_len=configs.seq_len
        )
        
        # Stage 2
        self.freq_attentions = nn.ModuleList([
            FrequencyAwareTaskOrientedAttention(
                d_embed=256,
                num_scales=3,
                output_dim=self.d_model,
                dropout=0.1,
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
        
        # Stage 4
        self.output_head = MultiTaskLongTermOutputHead(
            d_model=self.d_model,
            pred_len=configs.pred_len,
            num_tasks=self.num_tasks,
            head_type='A'
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
        
        # Stage 1
        multi_scale_embs, freq_weights = self.freq_extractor(x)
        
        # Stage 2: keep one FATA representation per forecasting task.
        attended_embs = []
        for i, attn_mod in enumerate(self.freq_attentions):
            target_idx = self.target_indices[i]
            attended = attn_mod(multi_scale_embs, target_idx)
            attended_embs.append(attended)

        # [B, M, C, d]: each task enters the same frozen backbone.  The
        # adapter later couples the M target tokens at every layer.
        hidden_states = torch.stack(attended_embs, dim=1)
        batch_size, num_tasks, channel_count, hidden_size = hidden_states.shape

        # Stage 3
        for layer_idx, layer in enumerate(self.qwen2.layers):
            flat_states = hidden_states.reshape(
                batch_size * num_tasks, channel_count, hidden_size
            )
            residual = flat_states
            flat_states = layer.input_layernorm(flat_states)

            attn_output, _, _ = layer.self_attn(
                hidden_states=flat_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = (residual + attn_output).reshape(
                batch_size, num_tasks, channel_count, hidden_size
            )

            adapter_out = self.adapters[layer_idx](
                hidden_states, self.target_indices
            )
            hidden_states = hidden_states + adapter_out

            flat_states = hidden_states.reshape(
                batch_size * num_tasks, channel_count, hidden_size
            )
            residual = flat_states
            flat_states = layer.post_attention_layernorm(flat_states)
            flat_states = layer.mlp(flat_states)
            hidden_states = (residual + flat_states).reshape(
                batch_size, num_tasks, channel_count, hidden_size
            )

        hidden_states = self.qwen2.norm(
            hidden_states.reshape(batch_size * num_tasks, channel_count, hidden_size)
        ).reshape(batch_size, num_tasks, channel_count, hidden_size)

        # Stage 4
        outputs = self.output_head(hidden_states, self.target_indices)
        preds = [out.pred.unsqueeze(-1) for out in outputs]
        
        return preds, freq_weights, outputs
    
    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100*trainable/total:.2f}")
