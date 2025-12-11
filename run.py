import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import json
import torch
from unidepth.models import UniDepthV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', default=True, help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    with open('configs/config_v2_vits14.json') as f:
        config = json.load(f)
    model = UniDepthV2(config)
    model.load_state_dict(torch.load(args.load_from, map_location='cpu')['model'], strict=True)
    model = model.to(DEVICE).eval()
 
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Create separate directory for numpy outputs
    npy_outdir = args.outdir
    os.makedirs(npy_outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        rgb = torch.from_numpy(np.array(raw_image)).permute(2, 0, 1) # C, H, W        
        predictions = model.infer(rgb)
        depth = predictions["depth"]
        depth = depth.squeeze().to('cpu').numpy()
        
        # Save raw depth as .npy file
        if args.save_numpy:
            npy_filename = os.path.splitext(os.path.basename(filename))[0] + '_depth.npy'
            npy_path = os.path.join(npy_outdir, npy_filename)
            np.save(npy_path, depth)
            print(f'  Saved raw depth to: {npy_path}')
                
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        output_path = os.path.join(args.outdir, 'pred_' + os.path.splitext(os.path.basename(filename))[0] + '.png')
        if args.pred_only:
            cv2.imwrite(output_path, depth)
        else:
            scale_factor = 2.0

            # Resize images
            h, w = raw_image.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            raw_image_scaled = cv2.resize(raw_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            depth_scaled = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Create proportionally scaled split region
            split_region = np.ones((new_h, int(50 * scale_factor), 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image_scaled, split_region, depth_scaled])
            
            cv2.imwrite(output_path, combined_result)