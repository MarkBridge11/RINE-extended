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

class CustomDataset(Dataset): # can be used also for testing (split="test") and validation (split="validation")
    def __init__(self, split="test", transforms=None): 
        # Map folder names to class labels
        self.class_map = {
            "real": 0,
            "full_synthetic": 1,
            "tampered": 2,
        }

        self.images = []
        #count = 0  # counter for loaded images

        for class_name, label in self.class_map.items():
            class_dir = f"/home/mbrigo/RINE/data/{split}/{class_name}"
            mask_dir = f"/home/mbrigo/RINE/data/{split}/masks"

            if not os.path.exists(class_dir):
                continue

            for fname in os.listdir(class_dir):

                image_path = os.path.join(class_dir, fname)
                mask_path = None
                if label == 2:  # tampered → mask exists
                    base, _ = os.path.splitext(fname)
                    mask_fname = f"{base}_mask.png"
                    mask_path = os.path.join(mask_dir, mask_fname)

                self.images.append((image_path, label, mask_path))
            #     count += 1
            #     if count >= 100:  # stop after 100 images
            #        break

            # if count >= 100:  # stop outer loop as well
            #    break

        random.shuffle(self.images)  # shuffle loaded images 
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, label, mask_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")

        # Load mask if available
        mask = None
        if label == 2 and mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")  # grayscale mask

        # crop image + mask together
        image, mask = mask_aware_crop(image,mask)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label, mask


class HuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, transforms=None):
        """
        hf_dataset: a Hugging Face Dataset split (train/validation/test)
        transforms: torchvision transforms to apply
        """
        self.dataset = hf_dataset
        
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Hugging Face image objects can be PIL Images or numpy arrays
        image = self.dataset[idx]["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        label = self.dataset[idx]["label"]

        mask = None
        if(label==2):
            mask = self.dataset[idx]["mask"]

        image,mask = mask_aware_crop(image,mask)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label, mask

"""
If mask is None (so the image is real or full synthetic) the image get randomcropped as in the original method.
If the image is actually tampered we do the same pre-processing to the mask and the image (center-crop 512x512 and then resize 224x224). The centercrop is designed in such a way that
the crop is taking the majority of the area that it can of the bounding box plus a margin (not only take the bounding box)
"""
def mask_aware_crop(img, mask=None, crop_size=224, crop_big=512, margin=20):

    if mask is None:
        # simple random crop for images without mask
        trans = transforms.Compose([
            transforms.CenterCrop(crop_big),
            transforms.Resize((224, 224))
        ])
        img_crop = trans(img)
        mask = torch.zeros((crop_size, crop_size), dtype=torch.uint8)
        return img_crop, mask

    # Convert mask to tensor if it's a PIL Image
    if isinstance(mask, Image.Image):
        mask = TF.to_tensor(mask)  # float tensor 0-1

    mask = mask[0]
    mask = (mask > 0.5).squeeze(0).to(torch.uint8)

    H, W = mask.shape

    ch, cw = crop_big, crop_big   # big crop first (512x512)

    # find all the coordinates where pixels are non zero (before we had converted them to just 0-1)
    ys, xs = torch.where(mask > 0)

    ymin, ymax = ys.min().item(), ys.max().item() # find the lowest and highest coordinate in y axis in the mask
    xmin, xmax = xs.min().item(), xs.max().item() # find the lowest and highest cooridnate in x axis in the mask

    # Here we expand the bounding box by a margin
    xmin = max(0, xmin - margin) # we subtract to the lowest coordinate in x axis for the margin otherwise 0
    xmax = min(W, xmax + margin) # the same but we take care to not go over the width
    ymin = max(0, ymin - margin) # ...
    ymax = min(H, ymax + margin)

    # This ensure that the crop fully contains the mask region
    x_min_crop = max(0, xmax - cw) 
    # this says that in order to catch the rightmost pixel in x axis, we have to start at least from max(0,xmax-cw), 
    # in a sense that if cw=512 and xmax is 500 the difference would be -12 so it will be clamped to 0 in order to 
    # take the rightmost pixel of the mask
    y_min_crop = max(0, ymax - ch) # ...
    x_max_crop = min(xmin, W - cw)
    y_max_crop = min(ymin, H - ch)

    # safeguard code for corner cases: this ensures that the code never crashes, even for masks at the edges or with unusual sizes.
    x_start = x_min_crop if x_max_crop < x_min_crop else random.randint(x_min_crop, x_max_crop)
    y_start = y_min_crop if y_max_crop < y_min_crop else random.randint(y_min_crop, y_max_crop)

    # crop both image and mask
    img_crop = TF.crop(img, y_start, x_start, ch, cw)
    mask_crop = mask[y_start:y_start+ch, x_start:x_start+cw]

    # resize both
    img_crop = TF.resize(img_crop, (crop_size, crop_size))
    mask_crop = TF.resize(mask_crop.unsqueeze(0).float(), (crop_size, crop_size))
    mask_crop = (mask_crop > 0.5).squeeze(0).to(torch.uint8)

    return img_crop, mask_crop
