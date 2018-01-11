"""
Project:    m3n_fast_pd
File:       KittiDataset_test.py
Created by: louise
On:         02/01/18
At:         5:53 PM
"""

from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.KittiDataset import KITTI, indexer_KITTI

# Transform dataset
patch_size = 200
transformations = transforms.Compose([transforms.CenterCrop((patch_size, patch_size)), transforms.ToTensor()])
transformations_d = transforms.Compose([transforms.CenterCrop((patch_size, patch_size))])
dd = KITTI("/media/louise/data/datasets/KITTI/stereo/training", indexer=indexer_KITTI, transform=transformations,
           depth_transform=transformations_d)

train_loader = DataLoader(dd, batch_size=1, num_workers=4, shuffle=False)
print(train_loader.dataset[5])