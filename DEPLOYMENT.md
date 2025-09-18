# 🚀 Free Deployment Guide for Stable Diffusion Web App

This guide will help you deploy your Stable Diffusion project for free on various platforms.

## 📋 Prerequisites

1. **GitHub Repository**: Push your code to GitHub
2. **Model Files**: Ensure your model files are in the `data/` directory
3. **Account Setup**: Create accounts on your chosen deployment platform

## 🎯 Deployment Options

### Option 1: Hugging Face Spaces (Recommended - Easiest)

**Best for**: Quick deployment with minimal setup
**Limitations**: 16GB RAM, 2 CPU cores, 50GB storage

#### Steps:
1. Go to [Hugging Face Spaces](https://huggingface.co/new-space)
2. Create a new Space with:
   - Name: `your-username/stable-diffusion-web`
   - SDK: **Gradio**
   - Hardware: **CPU basic** (free)
3. Upload your files to the Space
4. Your app will be available at: `https://huggingface.co/spaces/your-username/stable-diffusion-web`

#### Space Configuration:
Create `README.md` in your Space:
```markdown
---
title: Stable Diffusion Web App
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
short_description: Generate images with Stable Diffusion
---

# Stable Diffusion Web App

Generate high-quality images from text descriptions using Stable Diffusion v1.5.
```

### Option 2: Railway (Recommended - Most Flexible)

**Best for**: Full control, custom domains
**Limitations**: $5/month after free tier (500 hours/month)

#### Steps:
1. Go to [Railway](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will automatically detect the Dockerfile and deploy
6. Your app will be available at: `https://your-app-name.railway.app`

### Option 3: Render (Good Alternative)

**Best for**: Reliable hosting with good performance
**Limitations**: 750 hours/month free, sleeps after 15 minutes of inactivity

#### Steps:
1. Go to [Render](https://render.com)
2. Sign up with GitHub
3. Click "New" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Environment**: Python 3
6. Deploy!

### Option 4: Heroku (Limited Free Tier)

**Best for**: Simple deployment
**Limitations**: No free tier anymore, but good for learning

#### Steps:
1. Install Heroku CLI
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Deploy: `git push heroku main`

### Option 5: Google Colab (For Testing)

**Best for**: Testing and demos
**Limitations**: Session-based, not permanent

#### Steps:
1. Upload your code to Google Colab
2. Install dependencies in a cell:
```python
!pip install gradio torch torchvision transformers pillow numpy tqdm
```
3. Run your app in a cell:
```python
!python app.py
```

## 🔧 Model File Setup

### For Hugging Face Spaces:
1. Upload model files to your Space's "Files" tab
2. Update paths in `app.py` to use relative paths

### For Other Platforms:
1. Ensure model files are in the `data/` directory
2. Files will be included in deployment

## 🐳 Docker Deployment (Advanced)

If you want to use Docker locally or on other platforms:

```bash
# Build the image
docker build -t stable-diffusion-web .

# Run locally
docker run -p 7860:7860 stable-diffusion-web

# Push to Docker Hub (for other platforms)
docker tag stable-diffusion-web yourusername/stable-diffusion-web
docker push yourusername/stable-diffusion-web
```

## 🌐 Custom Domain (Railway/Render)

1. **Railway**: Go to your project → Settings → Domains
2. **Render**: Go to your service → Settings → Custom Domains

## 📊 Performance Optimization

### For CPU-only deployment:
- Reduce `n_inference_steps` to 20-30
- Use smaller CFG scale (6-8)
- Consider using smaller models

### For GPU deployment (paid tiers):
- Use full inference steps (50+)
- Higher CFG scale (8-12)
- Better quality results

## 🔒 Environment Variables

Set these in your deployment platform:

```bash
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
PYTHONUNBUFFERED=1
```

## 🐛 Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce batch size or use CPU
2. **Model Loading Errors**: Check file paths and permissions
3. **Slow Generation**: Reduce inference steps
4. **App Not Starting**: Check logs for missing dependencies

### Debug Commands:

```bash
# Check if models load
python -c "import model_loader; print('Models OK')"

# Test Gradio
python -c "import gradio as gr; print('Gradio OK')"

# Check device
python -c "import torch; print(f'Device: {torch.cuda.is_available()}')"
```

## 📈 Monitoring

- **Railway**: Built-in metrics and logs
- **Render**: Dashboard with usage stats
- **Hugging Face**: Space metrics in dashboard

## 💡 Tips for Success

1. **Start Small**: Deploy with minimal features first
2. **Test Locally**: Ensure everything works before deploying
3. **Monitor Usage**: Keep track of resource consumption
4. **Optimize**: Reduce model size if possible
5. **Backup**: Keep your code in version control

## 🎉 Going Live

Once deployed, your Stable Diffusion web app will be accessible worldwide! Share the link with friends, family, or the community.

### Example URLs:
- Hugging Face: `https://huggingface.co/spaces/yourusername/stable-diffusion-web`
- Railway: `https://your-app-name.railway.app`
- Render: `https://your-app-name.onrender.com`

## 🔄 Updates

To update your deployed app:
1. Push changes to GitHub
2. Platform will automatically redeploy (Railway, Render)
3. For Hugging Face, manually trigger rebuild

---

**Need Help?** Check the platform documentation or open an issue in your repository!
