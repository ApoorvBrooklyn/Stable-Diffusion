import gradio as gr
import torch
import numpy as np
from PIL import Image
import model_loader
import pipeline
from transformers import CLIPTokenizer
import os
import tempfile

# Global variables for model and tokenizer
models = None
tokenizer = None
device = "cpu"

def load_models():
    """Load models once at startup"""
    global models, tokenizer, device
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.has_mps or torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = CLIPTokenizer("data/vocab.json", merges_file="data/merges.txt")
    
    # Load models
    model_file = "data/v1-5-pruned-emaonly.ckpt"
    models = model_loader.preload_models_from_standard_weights(model_file, device)
    
    print("Models loaded successfully!")

def generate_image(prompt, negative_prompt, cfg_scale, steps, seed, input_image=None, strength=0.8):
    """Generate image from text prompt"""
    global models, tokenizer, device
    
    if models is None or tokenizer is None:
        return None, "Models not loaded. Please wait and try again."
    
    try:
        # Set seed
        if seed == -1:
            seed = None
        
        # Generate image
        output_image = pipeline.generate(
            prompt=prompt,
            uncond_prompt=negative_prompt,
            input_image=input_image,
            strength=strength,
            do_cfg=True,
            cfg_scale=cfg_scale,
            sampler_name="ddpm",
            n_inference_steps=steps,
            seed=seed,
            models=models,
            device=device,
            idle_device="cpu",
            tokenizer=tokenizer,
        )
        
        # Convert to PIL Image
        result_img = Image.fromarray(output_image)
        return result_img, "Image generated successfully!"
        
    except Exception as e:
        return None, f"Error generating image: {str(e)}"

def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Stable Diffusion Web App", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 Stable Diffusion Text-to-Image Generator")
        gr.Markdown("Generate high-quality images from text descriptions using Stable Diffusion v1.5")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A beautiful sunset over mountains, highly detailed, 8k resolution",
                    lines=3
                )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt (optional)",
                    placeholder="blurry, low quality, distorted",
                    lines=2
                )
                
                with gr.Row():
                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=14.0,
                        value=8.0,
                        step=0.5,
                        label="CFG Scale"
                    )
                    
                    steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=5,
                        label="Inference Steps"
                    )
                
                with gr.Row():
                    seed = gr.Number(
                        value=42,
                        label="Seed (-1 for random)"
                    )
                    
                    strength = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.8,
                        step=0.1,
                        label="Strength (for img2img)"
                    )
                
                # Image input for img2img
                input_image = gr.Image(
                    label="Input Image (optional - for image-to-image)",
                    type="pil"
                )
                
                generate_btn = gr.Button("Generate Image", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                # Output
                output_image = gr.Image(label="Generated Image")
                status = gr.Textbox(label="Status", interactive=False)
        
        # Examples
        gr.Examples(
            examples=[
                ["A majestic lion in the savanna, highly detailed, 8k resolution", "", 8.0, 50, 42],
                ["A futuristic city skyline at sunset, cyberpunk style, neon lights", "blurry, low quality", 7.5, 40, 123],
                ["A cute cat wearing a space helmet, digital art", "", 9.0, 60, 456],
                ["Abstract painting with vibrant colors, modern art style", "realistic, photograph", 6.0, 30, 789],
            ],
            inputs=[prompt, negative_prompt, cfg_scale, steps, seed]
        )
        
        # Event handlers
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, negative_prompt, cfg_scale, steps, seed, input_image, strength],
            outputs=[output_image, status]
        )
        
        # Load models on startup
        demo.load(load_models, outputs=[])
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
