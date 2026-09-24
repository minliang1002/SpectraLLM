SpectraLLM/
├── models/                          # Model files
│   ├── SpectraLLM.py               # Main model (time-domain output)
│   ├── SpectraLLM_FreqOutput.py    # Frequency-domain output version
│   ├── FreqQwen2_V12.py            # Qwen2 adapter
│   ├── FreqModules_MultiScale.py   # Multi-scale frequency extractor
│   ├── FreqModules_V12_FreqPE_Improved.py  # Frequency positional encoding
│   └── LongTermOutputHeads.py      # Output heads
├── data_provider/                   # Data loading
│   ├── data_factory.py
│   ├── data_loader.py
│   ├── data_loader_multi_target.py
│   └── ...
├── dataset/                         # Datasets
│   ├── spring_cleaned/
│   ├── summer_cleaned/
│   ├── autumn_cleaned/
│   └── winter_cleaned/
├── pretrained_models/               # Pretrained models
│   ├── qwen2-0.5b-local/           # Qwen2-0.5B
│   ├── gpt2-base-uncased/          # GPT2
│   ├── timer-base-84m/             # Timer
│   ├── chronos-t5-base/            # Chronos
│   └── sundial-base-128m/          # Sundial
├── checkpoints/                     # Trained weights
│   ├── spectrallm_spring_best.pth
│   └── spectrallm_spring_long_best.pth
├── logs/                            # Training logs
│   ├── SPECTRALLM_EXPERIMENTS_LOG.md
│   ├── spectrallm_spring.log
│   └── spectrallm_spring_long.log
├── train_spectrallm_spring.py       # Training script
├── train_spectrallm_freqout.py      # Frequency-domain output training script
├── run_spectrallm_spring.sh         # Short-term forecasting run script
└── run_spectrallm_spring_long.sh    # Long-term forecasting run script
