import cv2
import h5py
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop


def hypersim_distance_to_depth(npyDistance):
    intWidth, intHeight, fltFocal = 1024, 768, 886.81

    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(
        1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5,
                                 intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate(
        [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return npyDepth


class NYU_Depth_V2(Dataset):
    def __init__(self, filelist_path, mode, size=(476, 476)):
        
        self.mode = mode
        self.size = size

        self.base_dir = '../../NYU/official_splits'
        
        with open(filelist_path, 'r') as f:
            filelist = f.read().splitlines()

        self.filelist = []
        for file in filelist:
            img_path = os.path.join(self.base_dir, file.split(' ')[1].replace('depth', 'sync_depth'))
            if os.path.isfile(img_path):
                self.filelist.append(file)


        fx, fy = 525.0, 525.0  # focal lengths
        cx, cy = 319.5, 239.5  # principal points

        # Intrinsic matrix
        self.K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ],  dtype=torch.float32)

        original_width, original_height = 640, 480
        new_width, new_height = size

        # Calculate scale factors
        scale_width = new_width / original_width
        scale_height = new_height / original_height

        # Adjust the intrinsic matrix
        self.K[0, 0] *= scale_width   # Scale focal length x
        self.K[1, 1] *= scale_height  # Scale focal length y
        self.K[0, 2] *= scale_width   # Scale principal point x
        self.K[1, 2] *= scale_height  # Scale principal point y

                
        # print(f'number of images: {len(self.filelist)}')
        
        net_h, net_w = self.size 
              
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True,
                # if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]  + ([Crop(self.size)]))
        # if self.mode == 'train' else []
        
    def __getitem__(self, item):
        img_path = os.path.join(self.base_dir, self.filelist[item].split(' ')[0])
        depth_path = os.path.join(self.base_dir, self.filelist[item].split(' ')[1].replace('depth', 'sync_depth'))

        # print(img_path, depth_path)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        depth = np.array(Image.open(depth_path)) 
        # Convert to meters
        depth = depth / 1000.0
                
        
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        
        sample['valid_mask'] = (sample['depth']) > 0.0
        sample['depth'][sample['valid_mask'] == 0] = 0
        
        sample['image_path'] = self.filelist[item].split(' ')[0]

        sample['K'] = self.K

        # print(f'listen: {sample["image"].shape}')
        return sample

    def __len__(self):
        return len(self.filelist)