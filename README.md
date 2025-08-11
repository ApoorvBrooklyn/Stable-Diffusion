---
title: Stable Diffusion Implementation
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
python_version: 3.9
---

# Stable Diffusion Implementation

A custom implementation of Stable Diffusion from scratch.

## Setup

```bash
pip install -r requirements.txt

# Download the model weights
wget -O v1-5-pruned-emaonly.ckpt "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
```

## Usage

```python
python demo.py
```
