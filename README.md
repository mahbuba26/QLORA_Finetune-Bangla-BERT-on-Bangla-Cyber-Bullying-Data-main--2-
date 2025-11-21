# BanglaBERT Cyberbullying Detection Fine-Tuning

![BanglaBERT Logo](https://img.shields.io/badge/Model-BanglaBERT-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

## Project Overview

This project provides a modular Python framework for fine-tuning the BanglaBERT model (from `sagorsarker/bangla-bert-base`) on a Bangla cyberbullying dataset for multi-label classification. The dataset consists of comments labeled with five binary categories: `bully`, `sexual`, `religious`, `threat`, and `spam`.

The framework uses PyTorch for training, Hugging Face Transformers for model handling, and MLflow for experiment tracking. It supports K-Fold cross-validation (default: 5 folds), class weighting for imbalanced labels, early stopping, and optional freezing of base layers during fine-tuning.

Key features:
- **Modular Structure**: Separate files for configuration (`config.py`), data handling (`data.py`), model definition (`model.py`), training logic (`train.py`), and entry point (`main.py`).
- **Experiment Tracking**: All runs are logged to MLflow, including parameters, per-fold metrics, per-epoch metrics, and average results.
- **Customization**: Easily experiment with hyperparameters via command-line arguments.
- **Multi-Label Support**: Uses BCEWithLogitsLoss with sigmoid for binary predictions per label.

This setup is designed for reproducibility and collaboration—ideal for researchers or contributors experimenting with different hyperparameters on Bangla NLP tasks.

## Usage

Run fine-tuning via `main.py` with command-line arguments. All experiments log to MLflow under `./mlruns`.

### Running in Google Colab (Recommended for Free GPU)
1. Open Colab: [colab.research.google.com](https://colab.research.google.com).
2. Enable GPU: Runtime > Change runtime type > T4 GPU.
3. Clone repo: `!git clone https://github.com/SaifSiddique009/Finetune-Bangla-BERT-on-Bangla-Cyber-Bullying-Data.git`
4. `%cd Finetune-Bangla-BERT-on-Bangla-Cyber-Bullying-Data`
5. Install deps: `!!pip install -q torch transformers scikit-learn pandas numpy tqdm mlflow iterative-stratification peft`
6. Pick an experiment from `experiments.py` (open the file, copy a dict's values).
7. Run command (replace with your values):
   ```
   !python main.py \
    --author_name "your_name" \
    --dataset_path "your_dataset_path" \
    --batch 32 \
    --lr 5e-5 \
    --epochs 2 \
    --model_path "model_name" \
    --stratification_type multilabel \
    --seed 42 \
    --dropout 0.1 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --gradient_clip_norm 1.0 \
    --early_stopping_patience 5 \
    --num_folds 5\
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1
   ```
   - Full args:
     - `--batch`: Batch size (e.g., 16, 32, 64).
     - `--lr`: Learning rate (e.g., 2e-5).
     - `--epochs`: Number of epochs (e.g., 10-30).
     - `--author_name`: Your name (tags the MLflow run).
     - `--dataset_path`: Path to CSV (required).
     - `--model_path`: Pre-trained model (default: 'sagorsarker/bangla-bert-base').
     - `--num_folds`: Folds for CV (default: 5).
     - `--max_length`: Token max length (default: 128).
     - `--stratification_type`: Stratification type (default: 'multilable').
     - `--seed`: Random seed (default: 42).
     - `--dropout`: Dropout rate for classification head (default: 0.1).
     - `--weight_decay`: Weight decay for AdamW (default: 0.01).
     - `warmup_ratio`: Ratio of total steps for learning rate warmup (default: 0.1).
     - `--gradient_clip_norm`: Maximum norm for gradient clipping (default: 1.0).
     - `--early_stopping_patience`: Number of epochs without improvement before early stopping (default: 5).
     - `--freeze_base`: Freeze BERT base layers.
     - `--mlflow_experiment_name`: Experiment name (default: 'Bangla-BERT-Cyberbullying').

8. After run: Zip and download MLflow logs:
   ```
   !zip -r mlruns_saif_batch_16_lr_2e-5_epochs_10_dropout_0.1.zip ./mlruns
   ```
   - Download `mlruns_saif_batch_16_lr_2e-5_epochs_10_dropout_0.1.zip` from Colab's files sidebar.

### Viewing Results Locally
1. Unzip `mlruns_saif_batch_16_lr_2e-5_epochs_10_dropout_0.1.zip` to a local directory (e.g., `experiments/mlruns_saif_batch_16_lr_2e-5_epochs_10_dropout_0.1.zip`).
2. In VS Code or terminal: Navigate to the dir, activate venv, run:
   ```
   mlflow ui
   ```
3. Open `http://localhost:5000` in browser to view experiments, metrics, params, and models.

### Running Locally (No Colab)
Same as above, but use `python main.py ...` instead of `!python`.

## Collaboration Guide

To collaborate with minimal effort:
1. Fork the repo and clone to Colab/local.
2. Open `experiments.txt`: Select an experiment config (e.g., `{'batch_size': 32, 'learning_rate': 2e-5, 'num_epochs': 20, 'freeze_base': True}`).
3. Run with your name and dataset: `!python main.py --batch 32 --lr 2e-5 --epochs 20 --author_name 'collaborator' --dataset_path 'your/path.csv' --freeze_base...`
4. Zip/download `mlruns.zip`, view locally as above.
5. Add new experiments? Edit `experiments.txt` with your configs, commit, and PR.
6. Issues/PRs: Welcome! Describe your changes (e.g., "Added new LR 1e-5 for better F1").

For questions, open an issue.
