import gradio as gr
import torch
import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from io import BytesIO
from unidepth.models import UniDepthV2

# --- Global Setup ---
model = None

def load_model_once(config_path, model_path, device):
    global model
    if model is None:
        with open(config_path) as f:
            config = json.load(f)
        model = UniDepthV2(config)
        # Load weights safely
        checkpoint = torch.load(model_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
        model = model.to(device).eval()
    return model

def depth_estimation(image, progress=gr.Progress()):
    if image is None:
        return None

    try:
        progress(0, desc="Initializing...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config_path = 'configs/config_v2_vits14.json'
        model_path = 'checkpoint/latest.pth'
        
        if not os.path.exists(config_path):
            return "Configuration file not found."
        if not os.path.exists(model_path):
            return "Model checkpoint not found."
            
        progress(0.2, desc="Loading model...")
        model = load_model_once(config_path, model_path, device)
        
        progress(0.4, desc="Processing image...")
        # Ensure image is RGB (removes Alpha channel if present)
        if image.shape[-1] == 4:
            image = image[..., :3]
            
        rgb = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
        
        # Validation
        if rgb.shape[2] < 10 or rgb.shape[3] < 10:
            return "Image is too small."

        progress(0.6, desc="Generating depth map...")
        with torch.no_grad():
            predictions = model.infer(rgb)
        
        depth = predictions["depth"].squeeze().cpu().numpy()

        progress(0.8, desc="Creating visualization...")
        # Normalize depth for visualization
        min_depth = depth.min()
        max_depth = depth.max()
        depth_normalized = (depth - min_depth) / (max_depth - min_depth)
        
        # Colorize depth map
        cmap = matplotlib.colormaps.get_cmap('Spectral')
        depth_color = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
        
        # --- COLORBAR GENERATION ---
        # 1. Create the figure
        # Note: We keep a fixed figsize height to ensure the bar doesn't get too tall/short,
        # but we will resize the result to fit the image width exactly.
        fig, ax = plt.subplots(figsize=(8, 1))
        fig.subplots_adjust(bottom=0.5)
        
        norm = matplotlib.colors.Normalize(vmin=min_depth, vmax=max_depth)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=10) 
        cbar.set_label('Depth (meters)', size=12)
        
        # 2. Save to buffer
        buf = BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        
        # 3. Load and RESIZE the colorbar
        colorbar_img = Image.open(buf)
        target_width = depth_color.shape[1] # The width of the depth map
        
        # We resize the colorbar to match the depth map width exactly.
        # We keep the colorbar's original height to prevent it from distorting vertically.
        colorbar_img = colorbar_img.resize((target_width, colorbar_img.height), Image.Resampling.LANCZOS)
        
        # 4. Combine images
        new_height = depth_color.shape[0] + colorbar_img.height
        new_img = Image.new('RGB', (target_width, new_height), (255, 255, 255))
        
        new_img.paste(Image.fromarray(depth_color), (0, 0))
        new_img.paste(colorbar_img, (0, depth_color.shape[0]))
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        progress(1.0, desc="Done!")
        return new_img

    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Return a blank image with error text or just raise for Gradio to show
        raise gr.Error(f"Error occurred: {str(e)}")

def main():
    iface = gr.Interface(
        fn=depth_estimation,
        inputs=[
            gr.Image(type="numpy", label="Input Image"),
        ],
        outputs=[
            gr.Image(type="pil", label="Predicted Depth")
        ],
        title="Metric Depth Estimation",
        description="Upload an image to get its estimated depth map."
    )
    iface.launch()

if __name__ == "__main__":
    main()