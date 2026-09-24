"""
训练 SpectraLLM_FreqOutput: 频域输入+频域输出版本
- 训练损失在频域计算
- 评价指标在时域计算（便于比较）
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from data_provider.data_factory import data_provider
from models.SpectraLLM_FreqOutput import SpectraLLM_FreqOutput
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Configs:
    def __init__(self, args):
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.enc_in = args.enc_in
        self.qwen2_layers = args.qwen2_layers
        self.adapter_dim = args.adapter_dim
        self.qwen2_path = args.qwen2_path


def calculate_metrics_original_scale(pred, true, scaler, target_idx):
    """
    在原始尺度上计算指标
    """
    N, pred_len = pred.shape
    
    # 创建完整的特征矩阵
    pred_full = np.zeros((N * pred_len, scaler.mean_.shape[0]))
    true_full = np.zeros((N * pred_len, scaler.mean_.shape[0]))
    
    # 填充目标列
    pred_full[:, target_idx] = pred.reshape(-1)
    true_full[:, target_idx] = true.reshape(-1)
    
    # 反归一化到原始尺度
    pred_original = scaler.inverse_transform(pred_full)[:, target_idx]
    true_original = scaler.inverse_transform(true_full)[:, target_idx]
    
    # 计算指标
    mae = mean_absolute_error(true_original, pred_original)
    rmse = np.sqrt(mean_squared_error(true_original, pred_original))
    r2 = r2_score(true_original, pred_original)
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2)
    }


def frequency_domain_loss(pred_freq, true_freq):
    """
    频域损失：计算频域系数的MSE
    pred_freq, true_freq: complex tensors [B, freq_len]
    """
    # 分别计算实部和虚部的损失
    loss_real = torch.mean((pred_freq.real - true_freq.real) ** 2)
    loss_imag = torch.mean((pred_freq.imag - true_freq.imag) ** 2)
    return loss_real + loss_imag


def train_model(args):
    use_cuda = torch.cuda.is_available() and args.gpu >= 0
    device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    print("="*70)
    print("SpectraLLM_FreqOutput Training on Spring Dataset")
    print("="*70)
    print(f"Dataset: {args.root_path}")
    print(f"Training: Frequency Domain Loss")
    print(f"Evaluation: Time Domain Metrics (Original Scale)")
    print(f"Qwen2 Model: {args.qwen2_path}")
    print(f"Epochs: {args.train_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print("="*70)
    
    # 加载数据
    train_data, train_loader = data_provider(args, flag='train')
    val_data, val_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # 初始化模型
    configs = Configs(args)
    configs.enc_in = train_data.data_x.shape[-1]
    
    model = SpectraLLM_FreqOutput(configs, target_indices=args.target_indices).to(device)
    model.print_trainable_parameters()
    
    # 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.train_epochs, eta_min=1e-6
    )
    
    # 任务权重
    task_weights = {
        args.target_indices[i]: getattr(args, f'weight_{["e","c","h","comb"][i]}')
        for i in range(4)
    }
    
    # 训练
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth') if args.checkpoint_dir else 'checkpoint_spectrallm_freqout.pth'
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Training with Frequency Domain Loss...")
    print("="*70)
    
    for epoch in range(args.train_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            optimizer.zero_grad()
            
            # 前向传播：得到时域预测和频域预测
            time_preds, freq_weights, freq_preds = model(batch_x)
            
            # 将标签转换到频域
            total_loss = 0.0
            for i, idx in enumerate(args.target_indices):
                # 获取时域标签
                true_time = batch_y[:, -args.pred_len:, idx]  # [B, pred_len]
                
                # 转换到频域
                true_freq = torch.fft.rfft(true_time, dim=-1)  # [B, freq_len]
                
                # 频域损失
                loss = frequency_domain_loss(freq_preds[i], true_freq)
                total_loss += loss * task_weights[idx]
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证（也在频域计算损失）
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y, _, _ in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                time_preds, freq_weights, freq_preds = model(batch_x)
                
                total_loss = 0.0
                for i, idx in enumerate(args.target_indices):
                    true_time = batch_y[:, -args.pred_len:, idx]
                    true_freq = torch.fft.rfft(true_time, dim=-1)
                    loss = frequency_domain_loss(freq_preds[i], true_freq)
                    total_loss += loss * task_weights[idx]
                
                val_loss += total_loss.item()
        
        val_loss /= len(val_loader)
        
        scheduler.step()
        print(f"Epoch {epoch+1}/{args.train_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {time.time()-start_time:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print("  🏆 Best model saved!")
    
    # 测试（在原始尺度上评价）
    print("\n" + "="*70)
    print("Testing on Original Scale (Time Domain Metrics)...")
    print("="*70)
    
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    test_preds = {idx: [] for idx in args.target_indices}
    test_trues = {idx: [] for idx in args.target_indices}
    
    with torch.no_grad():
        for batch_x, batch_y, _, _ in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            # 前向传播：使用时域预测（已经通过IRFFT转换）
            time_preds, freq_weights, freq_preds = model(batch_x)
            
            for i, idx in enumerate(args.target_indices):
                test_preds[idx].append(time_preds[i].cpu().numpy())
                test_trues[idx].append(batch_y[:, -args.pred_len:, idx].unsqueeze(-1).cpu().numpy())
    
    # 计算原始尺度指标
    names = {3: 'Tamp_E', 4: 'Tamp_C', 5: 'Tamp_H', 6: 'Tamp_Comb'}
    print(f"\n{'Task':<10} | {'MAE':<10} | {'RMSE':<10} | {'R²':<10}")
    print("-" * 50)
    
    results = {}
    for idx in args.target_indices:
        preds = np.concatenate(test_preds[idx], axis=0).squeeze(-1)
        trues = np.concatenate(test_trues[idx], axis=0).squeeze(-1)
        
        # 在原始尺度上计算
        metrics = calculate_metrics_original_scale(preds, trues, train_data.scaler, idx)
        results[names.get(idx, str(idx))] = metrics
        print(f"{names.get(idx, str(idx)):<10} | {metrics['mae']:<10.3f} | {metrics['rmse']:<10.3f} | {metrics['r2']:<10.4f}")
    
    # 计算平均
    avg_mae = np.mean([m['mae'] for m in results.values()])
    avg_rmse = np.mean([m['rmse'] for m in results.values()])
    avg_r2 = np.mean([m['r2'] for m in results.values()])
    
    print("-" * 50)
    print(f"{'Average':<10} | {avg_mae:<10.3f} | {avg_rmse:<10.3f} | {avg_r2:<10.4f}")
    
    print("\n✅ Training completed!")
    print(f"Best validation loss (freq domain): {best_val_loss:.6f}")
    print(f"Test MAE (time domain, original scale): {avg_mae:.3f}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./dataset/spring_cleaned')
    parser.add_argument('--data_path', type=str, default='train_spring_final.csv')
    parser.add_argument('--target_indices', type=list, default=[3, 4, 5, 6])
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=37)
    parser.add_argument('--qwen2_path', type=str, default='./pretrained_models/qwen2-0.5b-local')
    parser.add_argument('--qwen2_layers', type=int, default=6)
    parser.add_argument('--adapter_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--weight_e', type=float, default=1.0)
    parser.add_argument('--weight_c', type=float, default=1.0)
    parser.add_argument('--weight_h', type=float, default=1.0)
    parser.add_argument('--weight_comb', type=float, default=1.0)
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
    parser.add_argument('--checkpoint_dir', type=str, default='')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--augment_factor', type=int, default=4)
    parser.add_argument('--noise_std', type=float, default=0.05)
    args = parser.parse_args()
    train_model(args)
