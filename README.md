# DDPM Diffusion Lab 🎨

A Denoising Diffusion Probabilistic Model (DDPM) implementation with a Streamlit web interface for image generation and reconstruction.

## Features
- **✨ Image Generation**: Generate synthetic images from pure Gaussian noise.
- **🔄 Image Reconstruction**: Upload an image, add noise, and see how the model denoises it back.
- **⚙️ Interactive UI**: Adjust timesteps, noise intensity, and model settings on the fly.

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Model Checkpoint
The app expects a trained PyTorch checkpoint (`.pt`) in the root directory or in `ddpm_checkpoints/`.

## License
MIT
