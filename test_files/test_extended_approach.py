import os
import time
import copy
import json
import random
import pickle
from io import BytesIO
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.utils_nomask import get_transforms_multiclass
from src.data_multiclass_nomask import CustomDataset
from src.models import DINOv2Model
from src.segmentation_head import TamperSegHead
from src.utils_segmentation import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################################################################
def save_tensor_image_overlay(tensor, mask_prob, path, alpha=0.5):
    """
    Save a single RGB image with the predicted mask overlaid.
    tensor: torch.Tensor of shape [3,H,W] or [1, H, W]
    mask_prob: numpy array [H,W] with values [0,1]
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    if tensor.ndim == 3 and tensor.shape[0] in (1,3):
        tensor = tensor.permute(1,2,0)
    img = tensor.numpy()
    img = (img * 255).clip(0,255).astype(np.uint8)
    
    mask_color = (mask_prob*255).astype(np.uint8)
    mask_color = cv2.applyColorMap(mask_color, cv2.COLORMAP_JET)
    
    # overlay
    overlay = cv2.addWeighted(img[:, :, ::-1], 1-alpha, mask_color, alpha, 0)  # RGB->BGR
    cv2.imwrite(path, overlay)

############################################# DATASET LOADING ###########
_, _, transforms_test = get_transforms_multiclass()
testing_dataset = CustomDataset(split="test", transforms=transforms_test)

test = DataLoader(
    testing_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)

ckpt_cls_path = "/home/mbrigo/RINE/ckpt_DINO/RINE_DINO_SIDA_dataset_0_of_1_2_512_0.4_[6, 11]_mean.pth" # change paths accordingly
ckpt_seg_path = "/home/mbrigo/RINE/ckpt_seg/SegBranch_05913_1763898060_superDPTfaithful_10.pth"

proj_dim = 512
nproj = 2
frozen_backbone = "dinov2_vits14"
selected_layers = [2, 5, 8, 11]

cls_model = get_model(device, frozen_backbone, ckpt_cls_path, selected_layers,
                      nproj=nproj, proj_dim=proj_dim)
cls_model.eval()

seg_head = TamperSegHead().to(device)
ckpt = torch.load(ckpt_seg_path, map_location="cpu")
seg_head.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
seg_head.eval()

#########################################################################

patch_size = 14

# Classification accumulators
y_true_cls = []
y_pred_cls = []
y_probs_cls = []

# Segmentation accumulators
seg_ious = []
seg_aucs = []
seg_f1s = []
seg_indices = []

pred_masks_all = []
mask_probs_all = []

softmax = F.softmax
sigmoid = torch.sigmoid

# Directory for failure cases
save_dir = "./failure_cases_bfree"
os.makedirs(save_dir, exist_ok=True)
saved_count = 0
max_saved = 20

with torch.no_grad():
    for images, labels, masks in tqdm(test, desc="Testing"):

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float().unsqueeze(1)

        B, _, H, W = images.shape
        patch_h, patch_w = H // patch_size, W // patch_size

        # -----------------------------------------
        # CLASSIFICATION STEP
        # -----------------------------------------
        logits, _ = cls_model(images)
        probs = softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        probs_np = probs.cpu().numpy()
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # accumulate
        y_true_cls.extend(labels_np.tolist())
        y_pred_cls.extend(preds_np.tolist())
        y_probs_cls.extend(probs_np)

        base_idx = len(y_true_cls) - B

        # -----------------------------------------
        # SEGMENTATION (only for class == 2)
        # -----------------------------------------
        for b in range(B):

            if preds_np[b] != 2:
                continue

            fmaps = []
            missing = False

            for layer_idx in cls_model.selected_layers:
                ptoks = cls_model.get_patch_tokens(layer_idx)
                if ptoks is None:
                    missing = True
                    break

                try:
                    tok = ptoks[b]
                except:
                    if ptoks.shape[0] == B:
                        tok = ptoks[b]
                    else:
                        missing = True
                        break

                fmaps.append(tok.unsqueeze(0).float().to(device))

            if missing or not fmaps:
                continue

            mask_pred = seg_head(fmaps, patch_h, patch_w)
            mask_pred = F.interpolate(mask_pred, size=(H, W),
                                      mode='bilinear', align_corners=True)
            mask_prob = sigmoid(mask_pred)

            # CPU numpy conversion
            mask_pred_np = mask_pred.squeeze().cpu().numpy()
            mask_prob_np = mask_prob.squeeze().cpu().numpy()

            pred_masks_all.append(mask_pred_np)
            mask_probs_all.append(mask_prob_np)

            # metrics
            y_true_mask = masks[b].cpu().numpy().ravel()
            y_score_mask = mask_prob_np.ravel()
            y_pred_mask = (y_score_mask > 0.5).astype(np.float32)

            # IoU
            inter = (y_true_mask * y_pred_mask).sum()
            union = np.clip(y_true_mask + y_pred_mask, 0, 1).sum()
            iou = float(inter / (union + 1e-6))
            seg_ious.append(iou)

            seg_indices.append(base_idx + b)

            # Save failure cases efficiently
            if 0.10 <= iou <= 0.20 and saved_count < max_saved:
                global_idx = base_idx + b
                image_path = testing_dataset.images[global_idx][0]
                filename = os.path.basename(image_path)
                filename_no_ext, _ = os.path.splitext(filename)
                path = os.path.join(save_dir,f"{filename_no_ext}_{saved_count:02d}_iou_{iou:.3f}.png")
                #path = os.path.join(save_dir, f"{images.name}_{saved_count:02d}_iou_{iou:.3f}.png")
                save_tensor_image_overlay(images[b], mask_prob_np, path)
                saved_count += 1

            # AUC only when valid
            if y_true_mask.sum() > 0 and (len(y_true_mask) - y_true_mask.sum()) > 0:
                seg_aucs.append(float(roc_auc_score(y_true_mask, y_score_mask)))

            seg_f1s.append(float(f1_score(y_true_mask, y_pred_mask, zero_division=0)))

# ------------------- Classification metrics -------------------

y_true_cls = np.array(y_true_cls)
y_pred_cls = np.array(y_pred_cls)
y_probs_cls = np.vstack(y_probs_cls) if len(y_probs_cls) > 0 else np.zeros((0,3))

class_ids = [0,1,2]
target_names = ['real','fake','tampered']

print("\nClassification accuracy:", accuracy_score(y_true_cls, y_pred_cls))

print(classification_report(
    y_true_cls, y_pred_cls,
    labels=class_ids, target_names=target_names,
    zero_division=0
))

# AP (macro)
try:
    y_true_onehot = label_binarize(y_true_cls, classes=class_ids)
    test_ap_macro = average_precision_score(y_true_onehot, y_probs_cls, average="macro")
except:
    test_ap_macro = None

print("Macro AP:", test_ap_macro)

# ------------------- Segmentation metrics -------------------

mean_iou = float(np.mean(seg_ious)) if seg_ious else 0.0
mean_auc = float(np.mean(seg_aucs)) if seg_aucs else 0.0
mean_f1 = float(np.mean(seg_f1s)) if seg_f1s else 0.0

print("\nSegmentation results:")
print("Mean IoU:", mean_iou)
print("Mean AUC:", mean_auc)
print("Mean F1:", mean_f1)
