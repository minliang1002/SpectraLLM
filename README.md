# SpectraLLM: Spectral Large Language Model for Time Series Forecasting

## 项目结构

```
SpectraLLM/
├── models/                          # 模型文件
│   ├── SpectraLLM.py               # 主模型 (时域输出)
│   ├── SpectraLLM_FreqOutput.py    # 频域输出版本
│   ├── FreqQwen2_V12.py            # Qwen2适配器
│   ├── FreqModules_MultiScale.py   # 多尺度频率提取器
│   ├── FreqModules_V12_FreqPE_Improved.py  # 频率位置编码
│   └── LongTermOutputHeads.py      # 输出头
├── data_provider/                   # 数据加载
│   ├── data_factory.py
│   ├── data_loader.py
│   ├── data_loader_multi_target.py
│   └── ...
├── dataset/                         # 数据集
│   ├── spring_cleaned/
│   ├── summer_cleaned/
│   ├── autumn_cleaned/
│   └── winter_cleaned/
├── pretrained_models/               # 预训练模型
│   ├── qwen2-0.5b-local/           # Qwen2-0.5B
│   ├── gpt2-base-uncased/          # GPT2
│   ├── timer-base-84m/             # Timer
│   ├── chronos-t5-base/            # Chronos
│   └── sundial-base-128m/          # Sundial
├── checkpoints/                     # 训练好的权重
│   ├── spectrallm_spring_best.pth
│   └── spectrallm_spring_long_best.pth
├── logs/                            # 训练日志
│   ├── SPECTRALLM_EXPERIMENTS_LOG.md
│   ├── spectrallm_spring.log
│   └── spectrallm_spring_long.log
├── train_spectrallm_spring.py       # 训练脚本
├── train_spectrallm_freqout.py      # 频域输出训练脚本
├── run_spectrallm_spring.sh         # 短期预测运行脚本
└── run_spectrallm_spring_long.sh    # 长期预测运行脚本
```



