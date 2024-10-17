import torch
import tarfile
import os
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop


class SYNSDataset(Dataset):
    HEIGHT, WIDTH = 376, 1242

    def __init__(self,
            filelist_path,
            mode,
            size=(518,518)
        ):

        self.mode = mode
        self.size = size
        self.base_dir = '../../Our_SYNSPatches'

        KITTI_FOV = (25.46, 84.10)
        KITTI_SHAPE = (376, 1242)
        
        Fy, Fx = KITTI_FOV
        h, w = KITTI_SHAPE

        cx, cy = w//2, h//2
        fx = cx / np.tan(np.deg2rad(Fx)/2)
        fy = cy / np.tan(np.deg2rad(Fy)/2)

        self.K = torch.tensor([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype=torch.float32)
        
        self.R = np.eye(3)

        self.T = np.array([0, -1.65, 0])

        self.min_depth=1e-2
        self.max_depth=120


        # Original and new dimensions
        original_width, original_height = 1240, 375
        new_width, new_height = size

        # Calculate scale factors
        scale_width = new_width / original_width
        scale_height = new_height / original_height

        # Adjust the intrinsic matrix
        self.K[0, 0] *= scale_width   # Scale focal length x
        self.K[1, 1] *= scale_height  # Scale focal length y
        self.K[0, 2] *= scale_width   # Scale principal point x
        self.K[1, 2] *= scale_height  # Scale principal point y
        


        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
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
        ]+ ([Crop(self.size)]))
        # if self.mode == 'train' else []
    
    def __getitem__(self, item):
        img_path = os.path.join(self.base_dir, self.filelist[item].split(' ')[0])
        depth_path = os.path.join(self.base_dir, self.filelist[item].split(' ')[2])
        
        # print(img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth = np.load(depth_path)
        depth[np.isnan(depth)] = 0.0

        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['K'] =  self.K 
        
        sample['valid_mask'] = (sample['depth'] <= 120) & (sample['depth'] > 0.0)
        
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        
        return sample

    def __len__(self):
        return len(self.filelist)