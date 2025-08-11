---
language:
  - en
tags:
  - stable-diffusion
  - pytorch
  - text-to-image
  - image-to-image
  - diffusion-models
  - computer-vision
  - generative-ai
license: mit
library_name: pytorch
pipeline_tag: text-to-image
base_model: stable-diffusion-v1-5
model-index:
  - name: pytorch-stable-diffusion
    results:
      - task:
          type: text-to-image
          name: Text-to-Image Generation
        dataset:
          type: custom
          name: Stable Diffusion v1.5
        metrics:
          - type: inference_steps
            value: 50
          - type: cfg_scale
            value: 8
          - type: image_size
            value: 512x512
---

# pytorch-stable-diffusion
PyTorch implementation of Stable Diffusion from scratch

## Download weights and tokenizer files:

1. Download `vocab.json` and `merges.txt` from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer and save them in the `data` folder
2. Download `v1-5-pruned-emaonly.ckpt` from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main and save it in the `data` folder

## Tested fine-tuned models:

Just download the `ckpt` file from any fine-tuned SD (up to v1.5).

1. InkPunk Diffusion: https://huggingface.co/Envvi/Inkpunk-Diffusion/tree/main
2. Illustration Diffusion (Hollie Mengert): https://huggingface.co/ogkalu/Illustration-Diffusion/tree/main