# Structured Over Scale: Learning Spatial Reasoning from Educational Video

[![arXiv](https://img.shields.io/badge/arXiv-2601.23251-b31b1b.svg)](https://arxiv.org/abs/2601.23251)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://ostadabbas.github.io/SoS.github.io/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qween-blue)](https://huggingface.co/bishoygaloaa/Qween)
[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/bishoygaloaa/Qween-Space)

Fine-tuning vision-language models on Dora the Explorer Q&A videos for educational reasoning tasks.

## Overview

This project fine-tunes QWEN-VL models to answer questions from Dora videos using GRPO (Group Relative Policy Optimization). The model learns to combine visual frames and transcript context to provide accurate, specific answers.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Dataset Generation

Validate labels and generate dataset from SRT files and videos:

```bash
# Validate and filter labels
python scripts/validate_all_seasons.py

# Generate dataset
python scripts/generate_grpo_dataset.py \
    --labels-dir outputs/filtered_labels \
    --videos-dir /path/to/videos \
    --srt-dir /path/to/srt \
    --output-dir outputs/dataset \
    --seasons 1 2 3 4 5 6 7 8 \
    --use-full-video 0.3 \
    --full-video-fps 0.2 \
    --full-video-max-frames 20 \
    --max-collages 2 \
    --num-workers 4
```

See `shell_scripts/dora-dataset.sh` for reference.

### 2. Training

Train with GRPO (recommended):

```bash
python scripts/train_grpo_dora_vl_clean.py /path/to/dataset \
    --output-dir ./outputs/visual-only \
    --use-lora \
    --use-frames \
    --visual-only \
    --num-generations 8 \
    --max-steps 5000 \
    --batch-size 16 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --max-prompt-length 512 \
    --max-response-length 256 \
    --save-steps 50 \
    --dataloader-num-workers 4 \
    --kl-beta 0.01 \
    --reward-weights 2.0
```

See `shell_scripts/dora-train.sh` for reference.

Alternative supervised fine-tuning:

```bash
python scripts/train_finetune.py /path/to/dataset \
    --output-dir ./outputs/train_sft \
    --use-lora
```

See `shell_scripts/dora-train-sft.sh` for reference.

### 3. Evaluation

```bash
python scripts/eval_mcq.py
```

See `shell_scripts/dora-eval.sh` for reference.

## Project Structure

```
dora/
├── scripts/
│   ├── generate_grpo_dataset.py
│   ├── validate_all_seasons.py
│   ├── train_grpo_dora_vl_clean.py
│   ├── train_finetune.py
│   └── eval_mcq.py
├── shell_scripts/
│   ├── dora-dataset.sh
│   ├── dora-train.sh
│   ├── dora-train-sft.sh
│   └── dora-eval.sh
└── config/
    ├── dataset_config.yaml
    └── train_config.yaml
```

## Configuration

Edit `config/dataset_config.yaml` for dataset extraction settings and `config/train_config.yaml` for training hyperparameters.

## Citation

If you use this work, please cite:

```bibtex
@article{structured2026,
  title={Structured Over Scale: Learning Spatial Reasoning from Educational Video},
  author={Galoaa, Bishoy and Bai, Xiangyu and Ostadabbas, Sarah},
  journal={arXiv preprint arXiv:2601.23251},
  year={2026}
}
```

## License

MIT License
