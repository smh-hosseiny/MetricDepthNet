import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
import numpy as np

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop


class VKITTI2(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        
        self.mode = mode
        self.size = size
        self.base_dir = '../../../../mnt/data/vkitti'

        self.K = torch.tensor([[725.0087, 0, 620.5],
                       [0, 725.0087, 187],
                       [0, 0, 1]], dtype=torch.float32)

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
        

        # # resized to (518,518) intrinsic matrix
        # self.K = torch.tensor([[302.7086, 0, 258.9735],
        #                [0, 1002.0089, 258.247],
        #                [0, 0, 1]], dtype=torch.float32)

        
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
        depth_path = os.path.join(self.base_dir, self.filelist[item].split(' ')[1])
        
        # print(img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.0  # cm to m
        
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['K'] =  self.K 
        
        sample['valid_mask'] = (sample['depth'] <= 80)
        
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        
        return sample

    def __len__(self):
        return len(self.filelist)