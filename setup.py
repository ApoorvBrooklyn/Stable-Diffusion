from setuptools import setup, find_packages

setup(
    name="stable-diffusion-web",
    version="1.0.0",
    description="Stable Diffusion Web Application",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
        "tqdm>=4.64.0",
        "gradio>=4.0.0",
        "accelerate>=0.20.0",
        "safetensors>=0.3.0",
    ],
    python_requires=">=3.8",
)
