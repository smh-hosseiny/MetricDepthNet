import gradio as gr
import torch
import cv2
import numpy as np
import json
from unidepth.models import UniDepthV2
import os
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from io import BytesIO

# Global model variable
model = None

def load_model_once(config_path, model_path, device):
    global model
    if model is None:
        with open(config_path) as f:
            config = json.load(f)
        model = UniDepthV2(config)
        model.load_state_dict(torch.load(model_path, map_location=device)['model'], strict=True)
        model = model.to(device).eval()
    return model

def depth_estimation(image, progress=gr.Progress()):
    if image is None:
        return "Please provide an input image."
        
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
        rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(device)
        
        if rgb.shape[1] < 10 or rgb.shape[2] < 10:
            return "Image is too small. Please provide a larger image."

        progress(0.6, desc="Generating depth map...")
        predictions = model.infer(rgb)
        depth = predictions["depth"].squeeze().to('cpu').numpy()

        progress(0.8, desc="Creating visualization...")
        min_depth = depth.min()
        max_depth = depth.max()
        depth_normalized = (depth - min_depth) / (max_depth - min_depth)
        
        cmap = matplotlib.colormaps.get_cmap('Spectral')
        depth_color = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)

        fig, ax = plt.subplots(figsize=(8, 1))
        fig.subplots_adjust(bottom=0.5)

        norm = matplotlib.colors.Normalize(vmin=min_depth, vmax=max_depth)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax, orientation='horizontal', label='Depth (meters)')

        cbar.ax.tick_params(labelsize=10)  
        cbar.set_label('Depth (meters)', size=12)  
        plt.tight_layout()


        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        buf.seek(0)

        colorbar_img = Image.open(buf)
        new_height = depth_color.shape[0] + colorbar_img.size[1] 
        new_img = Image.new('RGB', (depth_color.shape[1], new_height), (255, 255, 255))
        new_img.paste(Image.fromarray(depth_color), (0, 0))
        new_img.paste(colorbar_img, (0, depth_color.shape[0]))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        progress(1.0, desc="Done!")
        return new_img

    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"Error occurred: {str(e)}"

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
        description="Upload an image to get its estimated depth map.",
        examples=[
            ["examples/example1.jpg"],
        ]
    )

    iface.launch()

if __name__ == "__main__":
    main()
