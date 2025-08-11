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
  - deep-learning
  - neural-networks
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

# PyTorch Stable Diffusion Implementation

A complete, from-scratch PyTorch implementation of Stable Diffusion v1.5, featuring both text-to-image and image-to-image generation capabilities. This project demonstrates the inner workings of diffusion models by implementing all components without relying on pre-built libraries.

## 🚀 Features

- **Text-to-Image Generation**: Create high-quality images from text descriptions
- **Image-to-Image Generation**: Transform existing images using text prompts
- **Complete Implementation**: All components built from scratch in PyTorch
- **Flexible Sampling**: Configurable inference steps and CFG scale
- **Model Compatibility**: Support for various fine-tuned Stable Diffusion models
- **Clean Architecture**: Modular design with separate components for each part of the pipeline

## 🏗️ Architecture

This implementation includes all the core components of Stable Diffusion:

- **CLIP Text Encoder**: Processes text prompts into embeddings
- **VAE Encoder/Decoder**: Handles image compression and reconstruction
- **U-Net Diffusion Model**: Core denoising network with attention mechanisms
- **DDPM Sampler**: Implements the denoising diffusion probabilistic model
- **Pipeline Orchestration**: Coordinates all components for generation

## 📁 Project Structure

```
├── main/
│   ├── attention.py      # Multi-head attention implementation
│   ├── clip.py           # CLIP text encoder
│   ├── ddpm.py           # DDPM sampling algorithm
│   ├── decoder.py        # VAE decoder for image reconstruction
│   ├── diffusion.py      # U-Net diffusion model
│   ├── encoder.py        # VAE encoder for image compression
│   ├── model_converter.py # Converts checkpoint files to PyTorch format
│   ├── model_loader.py   # Loads and manages model weights
│   ├── pipeline.py       # Main generation pipeline
│   └── demo.py           # Example usage and demonstration
├── data/                 # Model weights and tokenizer files
└── images/               # Input/output images
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- Transformers library
- PIL (Pillow)
- NumPy
- tqdm

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/https://github.com/ApoorvBrooklyn/Stable-Diffusion
   cd pytorch-stable-diffusion
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision torchaudio
   pip install transformers pillow numpy tqdm
   ```

4. **Download required model files:**
   - Download `vocab.json` and `merges.txt` from [Stable Diffusion v1.5 tokenizer](https://huggingface.co/ApoorvBrooklyn/stable-diffusion-implementation/tree/main/data)
   - Download `v1-5-pruned-emaonly.ckpt` from [Stable Diffusion v1.5](https://huggingface.co/ApoorvBrooklyn/stable-diffusion-implementation/tree/main/data)
   - Place all files in the `data/` folder

## 🎯 Usage

### Basic Text-to-Image Generation

```python
import model_loader
import pipeline
from transformers import CLIPTokenizer

# Initialize tokenizer and load models
tokenizer = CLIPTokenizer("data/vocab.json", merges_file="data/merges.txt")
models = model_loader.preload_models_from_standard_weights("data/v1-5-pruned-emaonly.ckpt", "cpu")

# Generate image from text
output_image = pipeline.generate(
    prompt="A beautiful sunset over mountains, highly detailed, 8k resolution",
    uncond_prompt="",  # Negative prompt
    do_cfg=True,
    cfg_scale=8,
    sampler_name="ddpm",
    n_inference_steps=50,
    seed=42,
    models=models,
    device="cpu",
    tokenizer=tokenizer
)
```

### Image-to-Image Generation

```python
from PIL import Image

# Load input image
input_image = Image.open("images/input.jpg")

# Generate transformed image
output_image = pipeline.generate(
    prompt="Transform this into a watercolor painting",
    input_image=input_image,
    strength=0.8,  # Controls how much to change the input
    # ... other parameters
)
```

### Advanced Configuration

- **CFG Scale**: Controls how closely the image follows the prompt (1-14)
- **Inference Steps**: More steps = higher quality but slower generation
- **Strength**: For image-to-image, controls transformation intensity (0-1)
- **Seed**: Set for reproducible results

## 🔧 Model Conversion

The `model_converter.py` script converts Stable Diffusion checkpoint files to PyTorch format:

```bash
python main/model_converter.py --checkpoint_path data/v1-5-pruned-emaonly.ckpt --output_dir converted_models/
```

## 🎨 Supported Models

This implementation is compatible with:
- **Stable Diffusion v1.5**: Base model
- **Fine-tuned Models**: Any SD v1.5 compatible checkpoint
- **Custom Models**: Models trained on specific datasets or styles

### Tested Fine-tuned Models:
- **InkPunk Diffusion**: Artistic ink-style images
- **Illustration Diffusion**: Hollie Mengert's illustration style

## 🚀 Performance Tips

- **Device Selection**: Use CUDA for GPU acceleration, MPS for Apple Silicon
- **Batch Processing**: Process multiple prompts simultaneously
- **Memory Management**: Use `idle_device="cpu"` to free GPU memory
- **Optimization**: Adjust inference steps based on quality vs. speed needs

## 🔬 Technical Details

### Diffusion Process
- Implements DDPM (Denoising Diffusion Probabilistic Models)
- Uses U-Net architecture with cross-attention for text conditioning
- VAE handles 512x512 image compression to 64x64 latents

### Attention Mechanisms
- Multi-head self-attention in U-Net
- Cross-attention between text embeddings and image features
- Efficient attention implementation for memory optimization

### Sampling
- Configurable number of denoising steps
- Classifier-free guidance (CFG) for prompt adherence
- Deterministic generation with seed control

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Performance improvements
- New sampling algorithms
- Additional model support
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stability AI** for the original Stable Diffusion model
- **OpenAI** for the CLIP architecture
- **CompVis** for the VAE implementation
- **Hugging Face** for the transformers library

## 📚 References

- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the existing documentation
- Review the demo code for examples

---

**Note**: This is a research and educational implementation. For production use, consider using the official Stable Diffusion implementations or cloud-based APIs.