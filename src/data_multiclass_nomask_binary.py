import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import pandas as pd
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
from torch.utils.data import IterableDataset

class CustomDataset(Dataset): 
    def __init__(self, split="test", transforms=None): 
        
        self.class_map = {
            "real": 0,
            "tampered": 1, #was 2
        }

        self.images = []

        for class_name, label in self.class_map.items():
            #class_dir = f"/home/mbrigo/RINE/data/{split}/{class_name}"
            #mask_dir = f"/home/mbrigo/RINE/data/{split}/masks"
            class_dir = f"/home/mbrigo/imd2020_ds/{class_name}" # imd2020
            mask_dir = f"/home/mbrigo/imd2020_ds/masks/masks_imd"
            #class_dir = f"/home/mbrigo/CASIA2.0/CASIA2.0_revised/{class_name}/" #CASIA2.0
            #mask_dir = f"/home/mbrigo/CASIA2.0/CASIA2.0_Groundtruth/"
            #class_dir = f"/home/mbrigo/inpainted_samecat/{class_name}" # B-free
            #mask_dir = f"/home/mbrigo/inpainted_samecat/masks"

            if not os.path.exists(class_dir):
                continue

            for fname in os.listdir(class_dir):

                image_path = os.path.join(class_dir, fname)
                mask_path = None
                if label == 1:  
                    base, _ = os.path.splitext(fname)
                    #mask_fname = f"{base}_mask.png"
                    mask_fname = f"{base}_mask.jpg" #imd2020
                    #mask_fname = f"{base}_gt.png" #CASIA2.0
                    #mask_fname = f"{base}.png" #B-free
                    mask_path = os.path.join(mask_dir, mask_fname)

                self.images.append((image_path, label, mask_path))

        random.shuffle(self.images)  # shuffle loaded images 
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, label, mask_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")

        mask = None
        if label == 1 and mask_path and os.path.exists(mask_path): 
            mask = Image.open(mask_path).convert("L")  

        image, mask = mask_unaware_crop(image,mask)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label, mask


class HuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, transforms=None,augment=None):
        """
        hf_dataset: a Hugging Face Dataset split (train/validation/test)
        transforms: torchvision transforms to apply
        """
        self.dataset = hf_dataset
        self.transforms = transforms
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.dataset[idx]["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        label = self.dataset[idx]["label"]

        mask = None
        if(label==1):
            mask = self.dataset[idx]["mask"]

        image,mask = mask_unaware_crop(image,mask)

        if self.augment is not None:
            image, mask = self.augment(image,mask,label)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label, mask

class StreamingHFDataset(IterableDataset):
    def __init__(self, hf_dataset, transforms=None):
        self.dataset = hf_dataset
        self.transforms = transforms

    def __iter__(self):
        for example in self.dataset:
            image = example["image"]

            if isinstance(image, dict):
                # Handle HF streaming format
                image = Image.open(BytesIO(image["bytes"]))
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            image = image.convert("RGB")

            label = example["label"]
            mask = example.get("mask", None)

            if isinstance(mask, dict):
                mask = Image.open(BytesIO(mask["bytes"]))

            if label == 2:
                image, mask = mask_unaware_crop(image, mask)

            if self.transforms is not None:
                image = self.transforms(image)

            if mask is not None:
                mask = np.array(mask, dtype=np.float32)
                mask = torch.from_numpy(mask).unsqueeze(0) / 255.0

            yield image, label, mask


def mask_unaware_crop(img, mask=None, crop_size=224, crop_big=518): #was 512 and not 518 that is only for DINOv2

    trans_img = transforms.Compose([
        transforms.CenterCrop(crop_big),
    ])

    img_crop = trans_img(img)

    if mask is None:
        mask_tensor = torch.zeros((crop_big, crop_big), dtype=torch.uint8) 
    else:
        if isinstance(mask, torch.Tensor):
            mask = transforms.ToPILImage()(mask)

        trans_mask = transforms.Compose([
            transforms.CenterCrop(crop_big),
            transforms.ToTensor()
        ])

        mask_tensor = trans_mask(mask)[0]  # remove channel dimension
        mask_tensor = (mask_tensor > 0.5).to(torch.uint8)

    return img_crop, mask_tensor
        
