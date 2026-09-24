"""
时序大模型Adapter微调实验
支持: Timer, Chronos, Sundial, SimpleTransformer
用于与SpectraLLM对比

Timer: 使用patch embedding，期望1D时序输入 [B, seq_len]
Chronos: 使用T5 encoder，期望tokenized输入
Sundial: 使用patch embedding + diffusion，期望1D时序输入
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys
import time

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from data_provider.data_factory import data_provider
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Configs:
    def __init__(self, args):
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.enc_in = args.enc_in


def calculate_metrics_original_scale(pred, true, scaler, target_idx):
    """在原始尺度上计算指标"""
    N, pred_len = pred.shape
    pred_full = np.zeros((N * pred_len, scaler.mean_.shape[0]))
    true_full = np.zeros((N * pred_len, scaler.mean_.shape[0]))
    pred_full[:, target_idx] = pred.reshape(-1)
    true_full[:, target_idx] = true.reshape(-1)
    pred_original = scaler.inverse_transform(pred_full)[:, target_idx]
    true_original = scaler.inverse_transform(true_full)[:, target_idx]
    
    mae = mean_absolute_error(true_original, pred_original)
    rmse = np.sqrt(mean_squared_error(true_original, pred_original))
    r2 = r2_score(true_original, pred_original)
    return {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)}


# ============================================================
# Adapter模块
# ============================================================

class BottleneckAdapter(nn.Module):
    """通用Bottleneck Adapter"""
    def __init__(self, d_model, bottleneck_dim=64, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        residual = x
        x = self.down(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up(x)
        return residual + self.gate * x


class MultiTaskOutputHead(nn.Module):
    """多任务输出头"""
    def __init__(self, d_model, pred_len, num_tasks=4):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, pred_len)
            )
            for _ in range(num_tasks)
        ])
        
    def forward(self, x, target_indices):
        """x: [B, seq_len, d_model] or [B, d_model]"""
        if x.dim() == 3:
            x = x[:, -1, :]  # 取最后一个时间步
        preds = [head(x) for head in self.heads]
        return preds


# ============================================================
# Timer Adapter Model
# ============================================================

class TimerAdapter(nn.Module):
    """
    Timer + Adapter微调
    Timer期望输入: [B, seq_len] 1D时序
    Timer使用patch embedding (patch_len=96)
    """
    def __init__(self, configs, target_indices=[3, 4, 5, 6], 
                 model_path='pretrained_models/timer-base-84m'):
        super().__init__()
        self.target_indices = target_indices
        self.num_tasks = len(target_indices)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # 使用AutoModelForCausalLM加载Timer (因为Timer注册在CausalLM下)
        from transformers import AutoConfig, AutoModelForCausalLM
        
        timer_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.d_model = timer_config.hidden_size  # 1024
        self.patch_len = timer_config.input_token_len  # 96
        
        # 加载完整模型 (TimerForPrediction)
        full_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        
        # Timer的结构是 TimerForPrediction.model = TimerModel
        self.timer = full_model.model
        
        # 冻结Timer
        for param in self.timer.parameters():
            param.requires_grad = False
            
        # 解冻LayerNorm
        for name, param in self.timer.named_parameters():
            if 'norm' in name.lower():
                param.requires_grad = True
        
        # 输入投影: 将多变量时序投影到单变量
        self.input_proj = nn.Linear(self.enc_in, 1)
        
        # 由于Timer的patch_len=96，我们需要padding
        self.padded_len = ((self.seq_len + self.patch_len - 1) // self.patch_len) * self.patch_len
        
        # Adapter层 (在每个decoder layer后)
        num_layers = timer_config.num_hidden_layers  # 8
        self.adapters = nn.ModuleList([
            BottleneckAdapter(self.d_model, bottleneck_dim=64)
            for _ in range(num_layers)
        ])
        
        # 输出头
        self.output_head = MultiTaskOutputHead(self.d_model, self.pred_len, self.num_tasks)
        
    def forward(self, x):
        """x: [B, seq_len, n_features]"""
        B, L, C = x.shape
        
        # 投影到单变量: [B, seq_len, n_features] -> [B, seq_len]
        x = self.input_proj(x).squeeze(-1)  # [B, seq_len]
        
        # Padding到patch_len的倍数
        if L < self.padded_len:
            x = torch.nn.functional.pad(x, (self.padded_len - L, 0))  # 左边padding
        
        # 通过Timer的embed_layer
        hidden_states = self.timer.embed_layer(x)  # [B, num_patches, d_model]
        
        # 通过Timer的decoder layers + adapters
        for i, layer in enumerate(self.timer.layers):
            layer_output = layer(hidden_states)
            hidden_states = layer_output[0]
            # 应用adapter
            hidden_states = self.adapters[i](hidden_states)
        
        # 最后的norm
        hidden_states = self.timer.norm(hidden_states)
        
        # 输出: 取最后一个patch的表示
        preds = self.output_head(hidden_states, self.target_indices)
        preds = [p.unsqueeze(-1) for p in preds]
        
        return preds
    
    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100*trainable/total:.2f}")


# ============================================================
# Chronos Adapter Model
# ============================================================

class ChronosAdapter(nn.Module):
    """
    Chronos + Adapter微调
    Chronos使用T5 encoder-decoder架构
    我们只使用encoder部分提取特征
    """
    def __init__(self, configs, target_indices=[3, 4, 5, 6],
                 model_path='pretrained_models/chronos-t5-base'):
        super().__init__()
        self.target_indices = target_indices
        self.num_tasks = len(target_indices)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        from transformers import T5EncoderModel, T5Config
        
        # 加载T5配置
        t5_config = T5Config.from_pretrained(model_path)
        self.d_model = t5_config.d_model  # 768
        
        # 输入投影: 将多变量时序投影到T5的embedding维度
        self.input_proj = nn.Linear(self.enc_in, self.d_model)
        
        # 加载T5 encoder
        self.encoder = T5EncoderModel.from_pretrained(model_path)
        
        # 冻结encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # 解冻LayerNorm
        for name, param in self.encoder.named_parameters():
            if 'layer_norm' in name.lower():
                param.requires_grad = True
        
        # Adapter层
        num_layers = t5_config.num_layers  # 12
        self.adapters = nn.ModuleList([
            BottleneckAdapter(self.d_model, bottleneck_dim=64)
            for _ in range(num_layers)
        ])
        
        # 输出头
        self.output_head = MultiTaskOutputHead(self.d_model, self.pred_len, self.num_tasks)
        
    def forward(self, x):
        """x: [B, seq_len, n_features]"""
        B = x.shape[0]
        
        # 投影到T5维度
        x = self.input_proj(x)  # [B, seq_len, d_model]
        
        # 通过T5 encoder (使用inputs_embeds)
        outputs = self.encoder(inputs_embeds=x, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state
        
        # 应用adapters (在最后的hidden states上)
        for adapter in self.adapters:
            hidden_states = adapter(hidden_states)
        
        # 输出
        preds = self.output_head(hidden_states, self.target_indices)
        preds = [p.unsqueeze(-1) for p in preds]
        
        return preds
    
    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100*trainable/total:.2f}")


# ============================================================
# Sundial Adapter Model
# ============================================================

class SundialAdapter(nn.Module):
    """
    Sundial + Adapter微调
    Sundial使用patch embedding (patch_len=16) + diffusion
    我们只使用transformer backbone部分
    """
    def __init__(self, configs, target_indices=[3, 4, 5, 6],
                 model_path='pretrained_models/sundial-base-128m'):
        super().__init__()
        self.target_indices = target_indices
        self.num_tasks = len(target_indices)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # 使用AutoModelForCausalLM加载Sundial
        from transformers import AutoConfig, AutoModelForCausalLM
        
        sundial_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.d_model = sundial_config.hidden_size  # 768
        self.patch_len = sundial_config.input_token_len  # 16
        
        # 加载完整模型 (SundialForPrediction)
        full_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        
        # Sundial的结构是 SundialForPrediction.model = SundialModel
        self.sundial = full_model.model
        
        # 冻结Sundial
        for param in self.sundial.parameters():
            param.requires_grad = False
            
        # 解冻LayerNorm
        for name, param in self.sundial.named_parameters():
            if 'norm' in name.lower():
                param.requires_grad = True
        
        # 输入投影: 将多变量时序投影到单变量
        self.input_proj = nn.Linear(self.enc_in, 1)
        
        # Adapter层
        num_layers = sundial_config.num_hidden_layers  # 12
        self.adapters = nn.ModuleList([
            BottleneckAdapter(self.d_model, bottleneck_dim=64)
            for _ in range(num_layers)
        ])
        
        # 输出头
        self.output_head = MultiTaskOutputHead(self.d_model, self.pred_len, self.num_tasks)
        
    def forward(self, x):
        """x: [B, seq_len, n_features]"""
        B, L, C = x.shape
        
        # 投影到单变量: [B, seq_len, n_features] -> [B, seq_len]
        x = self.input_proj(x).squeeze(-1)  # [B, seq_len]
        
        # 通过Sundial的embed_layer
        hidden_states = self.sundial.embed_layer(x)  # [B, num_patches, d_model]
        
        # 通过Sundial的decoder layers + adapters
        for i, layer in enumerate(self.sundial.layers):
            layer_output = layer(hidden_states)
            hidden_states = layer_output[0]
            hidden_states = self.adapters[i](hidden_states)
        
        # 最后的norm
        hidden_states = self.sundial.norm(hidden_states)
        
        # 输出
        preds = self.output_head(hidden_states, self.target_indices)
        preds = [p.unsqueeze(-1) for p in preds]
        
        return preds
    
    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100*trainable/total:.2f}")


# ============================================================
# 简单Transformer Baseline (用于对比)
# ============================================================

class SimpleTransformerAdapter(nn.Module):
    """简单Transformer + Adapter (不依赖预训练模型)"""
    def __init__(self, configs, target_indices=[3, 4, 5, 6], d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.target_indices = target_indices
        self.num_tasks = len(target_indices)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = d_model
        
        # 输入投影
        self.input_proj = nn.Linear(configs.enc_in, d_model)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, configs.seq_len, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出头
        self.output_head = MultiTaskOutputHead(d_model, self.pred_len, self.num_tasks)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.transformer(x)
        preds = self.output_head(x, self.target_indices)
        preds = [p.unsqueeze(-1) for p in preds]
        return preds
    
    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100*trainable/total:.2f}")


# ============================================================
# 训练函数
# ============================================================

def train_model(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"Using device: {device}")
    
    print("="*70)
    print(f"Training {args.model_type} on Spring Dataset")
    print("="*70)
    print(f"Model: {args.model_type}")
    print(f"seq_len: {args.seq_len}, pred_len: {args.pred_len}")
    print(f"Epochs: {args.train_epochs}")
    print("="*70)
    
    # 加载数据
    train_data, train_loader = data_provider(args, flag='train')
    val_data, val_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # 初始化模型
    configs = Configs(args)
    configs.enc_in = train_data.data_x.shape[-1]
    
    if args.model_type == 'timer':
        model = TimerAdapter(configs, args.target_indices, args.model_path)
    elif args.model_type == 'chronos':
        model = ChronosAdapter(configs, args.target_indices, args.model_path)
    elif args.model_type == 'sundial':
        model = SundialAdapter(configs, args.target_indices, args.model_path)
    elif args.model_type == 'transformer':
        model = SimpleTransformerAdapter(configs, args.target_indices)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    model.print_trainable_parameters()
    
    # 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, weight_decay=0.01
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.train_epochs, eta_min=1e-6
    )
    
    # 训练
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.model_type}_best.pth')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("\nTraining...")
    for epoch in range(args.train_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            optimizer.zero_grad()
            
            preds = model(batch_x)
            
            total_loss = sum(
                criterion(preds[i], batch_y[:, -args.pred_len:, idx].unsqueeze(-1))
                for i, idx in enumerate(args.target_indices)
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y, _, _ in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                preds = model(batch_x)
                total_loss = sum(
                    criterion(preds[i], batch_y[:, -args.pred_len:, idx].unsqueeze(-1))
                    for i, idx in enumerate(args.target_indices)
                )
                val_loss += total_loss.item()
        val_loss /= len(val_loader)
        
        scheduler.step()
        print(f"Epoch {epoch+1}/{args.train_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {time.time()-start_time:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print("  🏆 Best model saved!")
    
    # 测试
    print("\n" + "="*70)
    print("Testing on Original Scale...")
    print("="*70)
    
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    test_preds = {idx: [] for idx in args.target_indices}
    test_trues = {idx: [] for idx in args.target_indices}
    
    with torch.no_grad():
        for batch_x, batch_y, _, _ in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            preds = model(batch_x)
            for i, idx in enumerate(args.target_indices):
                test_preds[idx].append(preds[i].cpu().numpy())
                test_trues[idx].append(batch_y[:, -args.pred_len:, idx].unsqueeze(-1).cpu().numpy())
    
    # 计算指标
    names = {3: 'Tamp_E', 4: 'Tamp_C', 5: 'Tamp_H', 6: 'Tamp_Comb'}
    print(f"\n{'Task':<10} | {'MAE':<10} | {'RMSE':<10} | {'R²':<10}")
    print("-" * 50)
    
    results = {}
    for idx in args.target_indices:
        preds = np.concatenate(test_preds[idx], axis=0).squeeze(-1)
        trues = np.concatenate(test_trues[idx], axis=0).squeeze(-1)
        metrics = calculate_metrics_original_scale(preds, trues, train_data.scaler, idx)
        results[names.get(idx, str(idx))] = metrics
        print(f"{names.get(idx, str(idx)):<10} | {metrics['mae']:<10.3f} | {metrics['rmse']:<10.3f} | {metrics['r2']:<10.4f}")
    
    avg_mae = np.mean([m['mae'] for m in results.values()])
    avg_rmse = np.mean([m['rmse'] for m in results.values()])
    avg_r2 = np.mean([m['r2'] for m in results.values()])
    
    print("-" * 50)
    print(f"{'Average':<10} | {avg_mae:<10.3f} | {avg_rmse:<10.3f} | {avg_r2:<10.4f}")
    
    print(f"\n✅ Training completed! Best val loss: {best_val_loss:.6f}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='transformer', 
                        choices=['timer', 'chronos', 'sundial', 'transformer'])
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--root_path', type=str, default='./dataset/spring_cleaned')
    parser.add_argument('--data_path', type=str, default='train_spring_final.csv')
    parser.add_argument('--target_indices', type=list, default=[3, 4, 5, 6])
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--enc_in', type=int, default=37)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/tsfm')
    # data_provider需要的参数
    parser.add_argument('--data', type=str, default='Multi_Target')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--percent', type=int, default=100)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--target', type=str, default='Tamp_E')
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()
    
    train_model(args)
