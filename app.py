import streamlit as st
import torch
import numpy as np
from PIL import Image
from model_utils import UNet, DiffusionScheduler
import matplotlib.pyplot as plt
from torchvision import transforms
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DDPM Diffusion Lab",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium feel
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4f8ef7;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3b6fcb;
        border: 1px solid #4f8ef7;
    }
    .reportview-container {
        color: #e0e0e0;
    }
    .header-text {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f8ef7, #a061ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_view_for_change=True, unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")
    
    img_size = st.selectbox("Image Size", [64, 128, 256], index=1)
    timesteps = st.slider("Diffusion Timesteps (T)", 100, 1000, 300)
    schedule = st.selectbox("Beta Schedule", ["cosine", "linear"], index=0)
    
    st.markdown("---")
    st.markdown("### 📦 Checkpoint")
    ckpt_files = [f for f in os.listdir(".") if f.endswith(".pt")]
    if not ckpt_files:
        st.warning("No .pt files found in root. Searching subdirectories...")
        # Check ddpm_checkpoints
        if os.path.exists("ddpm_checkpoints"):
            ckpt_files += [os.path.join("ddpm_checkpoints", f) for f in os.listdir("ddpm_checkpoints") if f.endswith(".pt")]
    
    selected_ckpt = st.selectbox("Select Model Weight", ckpt_files if ckpt_files else ["No weights found"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"Running on: **{device.type.upper()}**")

# ── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path, size, t, sched):
    if not os.path.exists(path):
        return None, None
    
    # Initialize model with notebook parameters
    model = UNet(in_ch=3, base_dim=64, dim_mults=(1, 2, 4), image_size=size).to(device)
    try:
        ckpt = torch.load(path, map_location=device)
        # Handle cases where model is wrapped in DataParallel or saved with dict keys
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        # Strip 'module.' prefix if it exists
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    except Exception as e:
        st.error(f"Error loading checkpoint: {e}")
        return None, None
    
    model.eval()
    scheduler = DiffusionScheduler(T=t, schedule=sched)
    return model, scheduler

# ── Utility Functions ────────────────────────────────────────────────────────
def to_pil(tensor):
    """Tensor [-1, 1] -> PIL Image [0, 255]"""
    img = (tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))

# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown('<p class="header-text">DDPM Diffusion Lab</p>', unsafe_allow_html=True)
st.markdown("Experience the magic of Denoising Diffusion Probabilistic Models.")

if selected_ckpt == "No weights found":
    st.error("Please ensure your `.pt` model checkpoint is in the same directory as this app.")
    st.stop()

model, scheduler = load_model(selected_ckpt, img_size, timesteps, schedule)

if model is None:
    st.warning("Waiting for model to load...")
    st.stop()

tab1, tab2 = st.tabs(["✨ Generate", "🔄 Reconstruct"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("### New Samples")
        num_samples = st.number_input("Number of images", 1, 8, 4)
        if st.button("Generate Images"):
            with st.spinner("Generating from pure noise..."):
                # Run sampling
                # Note: We simulate progress by yielding or just showing a bar
                progress_bar = st.progress(0)
                
                # Manual sampling loop to show progress
                img = torch.randn((num_samples, 3, img_size, img_size), device=device)
                for i in reversed(range(scheduler.T)):
                    img = scheduler.p_sample(model, img, i)
                    if i % (scheduler.T // 10) == 0:
                        progress_bar.progress((scheduler.T - i) / scheduler.T)
                
                progress_bar.progress(1.0)
                generated_imgs = img.cpu()
                
                cols = st.columns(min(num_samples, 4))
                for i, img_tensor in enumerate(generated_imgs):
                    cols[i % 4].image(to_pil(img_tensor), use_container_width=True, caption=f"Sample {i+1}")

with tab2:
    st.write("### Image Reconstruction")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        img_input = Image.open(uploaded_file).convert("RGB")
        st.image(img_input, caption="Original Image", width=256)
        
        noise_level = st.slider("Noise Intensity (Timestep t)", 1, timesteps-1, timesteps//2)
        
        if st.button("Run Reconstruction"):
            with st.spinner("Adding noise and denoising..."):
                # Transform to tensor
                tf = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5]*3, [0.5]*3)
                ])
                x0 = tf(img_input).unsqueeze(0).to(device)
                
                # Noise it
                t_tensor = torch.tensor([noise_level]).to(device)
                x_noisy = scheduler.q_sample(x0, t_tensor)
                
                # Denoise it
                x = x_noisy.clone()
                progress_bar = st.progress(0)
                for i in reversed(range(noise_level)):
                    x = scheduler.p_sample(model, x, i)
                    if i % max(1, noise_level // 10) == 0:
                        progress_bar.progress((noise_level - i) / noise_level)
                progress_bar.progress(1.0)
                
                # Results
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.image(to_pil(x0[0]), caption="Input (Resized)", use_container_width=True)
                res_col2.image(to_pil(x_noisy[0]), caption=f"Noised (t={noise_level})", use_container_width=True)
                res_col3.image(to_pil(x[0]), caption="Reconstructed", use_container_width=True)

st.markdown("---")
st.caption("Built with ❤️ using PyTorch and Streamlit")
