import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop
)
from albumentations.pytorch.functional import img_to_tensor


def get_filenames(root, split):
    
    if split == 'train':
        file_names = []
        for instrument_id in range(1, 7):
            file_names += list(Path(root + '/instrument_dataset_' + str(instrument_id) + '/images').glob('*'))
            
    elif split == 'val':
        file_names = []
        for instrument_id in range(7, 9):
            file_names += list(Path(root + '/instrument_dataset_' + str(instrument_id) + '/images').glob('*'))
    
    elif split == 'pred':
        file_names = []
        for instrument_id in range(7, 9):
            file_names += list(Path(root + '/instrument_dataset_' + str(instrument_id) + '/images').glob('*'))
        
    return file_names        
    
        
def train_transform(p=1):
    return Compose([
        PadIfNeeded(min_height=1024, min_width=1280, p=1),
        RandomCrop(height=1024, width=1280, p=1),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        Normalize(p=1)
    ], p=p)

def val_transform(p=1):
    return Compose([
        PadIfNeeded(min_height=1024, min_width=1280, p=1),
        CenterCrop(height=1024, width=1280, p=1),
        Normalize(p=1)
    ], p=p)      

def pred_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)  


class RoboticsDataset(Dataset):
    IN_CHANNELS = 3
    NUM_CLASS = 1
    CLASS_WEIGHTS = None    
    def __init__(self, root, split='train', mode='train', to_augment=False, problem_type='binary'):
        

        self.root = root
        self.split = split
        self.file_names = get_filenames(self.root, split)
        self.to_augment = to_augment
        if split == 'train':
            self.transform = train_transform(p=1)
        elif split == 'val':
            self.transform = val_transform(p=1)
        elif split == 'pred':
            self.transform = pred_transform(p=1)   
        self.mode = mode
        self.problem_type = problem_type

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name, self.problem_type)

        data = {"image": image, "mask": mask}
        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]

        if self.mode == 'train' or self.mode == 'val':
            if self.problem_type == 'binary':
                return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
        else:
            return img_to_tensor(image), str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, problem_type):
    if problem_type == 'binary':
        mask_folder = 'binary_masks'
        factor = 255

    mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)

    return (mask / factor).astype(np.uint8)




