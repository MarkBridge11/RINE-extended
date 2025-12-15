import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn.functional as F
import cv2

from src.utils_nomask import get_transforms_multiclass
from src.data_multiclass_nomask_binary import CustomDataset
from src.models_binary import DINOv2Model
from src.segmentation_head import TamperSegHead
from src.get_model_binary import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################################################################
# Helpers
#########################################################################
def save_tensor_image(tensor, path):
    #Save a torch tensor image (C,H,W or H,W,C) as PNG.
    #Supports tensors in [-1,1] or [0,1].
   
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.array(tensor)

    # If channel-first
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        if arr.min() < 0:
            arr = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)

    # assume arr is RGB
    img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def save_overlay_only(image_tensor, mask_prob_np, save_path, alpha=0.5):
    """
    Save ONLY the overlay (original RGB + red predicted mask blend).
    image_tensor: torch tensor [3,H,W] in [-1,1] or [0,1]
    mask_prob_np: numpy array [H,W]
    """
    # Prepare image
    img = image_tensor.detach().cpu()
    if img.min() < 0:
        img = (img + 1) / 2
    img = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    # Binary mask
    mask_bin = (mask_prob_np > 0.5).astype(np.uint8)

    # Red overlay
    red = np.zeros_like(img)
    red[..., 0] = 255

    overlay = (img * (1 - alpha) + red * mask_bin[..., None] * alpha).astype(np.uint8)

    Image.fromarray(overlay).save(save_path)

#########################################################################
# DATASET
#########################################################################
_, _, transforms_test = get_transforms_multiclass()
testing_dataset = CustomDataset(split="test", transforms=transforms_test)
test_loader = torch.utils.data.DataLoader(
    testing_dataset, batch_size=4, shuffle=False, num_workers=1, pin_memory=False
)

#########################################################################
# MODEL
#########################################################################
ckpt_cls_path = "/home/mbrigo/RINE/ckpt_DINO/RINE_DINO_SIDA_binary_0_of_1_2_512_0.4_bin_aug.pth" # change paths accordingly
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
# RUN SETTINGS
#########################################################################
patch_size = 14

# Classification accumulators
y_true_cls = []
y_pred_cls = []
y_probs_cls = []

# Segmentation streamed metrics
seg_count = 0
seg_iou_sum = 0.0
seg_f1_sum = 0.0

# Streaming global ROC-AUC buffer
auc_buffer_size = 1000  # number of masks to buffer
true_buffer = []
score_buffer = []

sigmoid = torch.sigmoid

save_dir = "./failure_cases_cheat_bfree"
os.makedirs(save_dir, exist_ok=True)
K = 20
saved_iou_count = 0

base_idx = 0

#########################################################################
# INFERENCE LOOP (CHEAT: perfect classifier)
#########################################################################
with torch.no_grad():
    for images, labels, masks in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device).float().unsqueeze(1)  # [B,1,H,W]

        B, _, H, W = images.shape
        patch_h = H // patch_size
        patch_w = W // patch_size

        logits, _ = cls_model(images)
        probs = sigmoid(logits)  
        preds = labels.long().view(-1)

        y_true_cls.extend(labels.cpu().numpy().tolist())
        y_pred_cls.extend(preds.cpu().numpy().tolist())

        y_probs_cls.extend(probs.cpu().numpy())

        # ---------------- Segmentation ----------------
        for b in range(B):
            # run segmentation only on predicted tampered (cheat -> equals ground truth tampered)
            if preds[b] != 1:  # only tampered
                continue

            # Extract patch tokens
            fmaps = []
            valid = True
            for layer_idx in cls_model.selected_layers:
                patch_tokens = cls_model.get_patch_tokens(layer_idx)
                if patch_tokens is None or patch_tokens.shape[0] != B:
                    valid = False
                    break
                fmaps.append(patch_tokens[b].unsqueeze(0).float().to(device))
            if not valid or len(fmaps) == 0:
                continue

            mask_pred = seg_head(fmaps, patch_h, patch_w)
            mask_pred = F.interpolate(mask_pred, size=(H, W), mode="bilinear", align_corners=True)
            mask_prob = sigmoid(mask_pred)  # tensor [1,H,W] or [H,W] after squeeze

            y_true_mask = masks[b].cpu().numpy().ravel()
            y_score_mask = mask_prob.cpu().numpy().ravel()
            y_pred_mask = (y_score_mask > 0.5).astype(np.float32)

            inter = (y_pred_mask * y_true_mask).sum()
            union = ((y_pred_mask + y_true_mask) > 0).sum()
            seg_iou_sum += float(inter / (union + 1e-6))
            seg_f1_sum += float(f1_score(y_true_mask, y_pred_mask, zero_division=0))
            seg_count += 1

            true_buffer.append(y_true_mask)
            score_buffer.append(y_score_mask)

            if len(true_buffer) >= auc_buffer_size:
                y_true_flat = np.concatenate(true_buffer)
                y_score_flat = np.concatenate(score_buffer)
                try:
                    batch_auc = roc_auc_score(y_true_flat, y_score_flat)
                except:
                    batch_auc = 0.0
                true_buffer, score_buffer = [], []

            iou = float(inter / (union + 1e-6))
            if 0.10 <= iou <= 0.20 and saved_iou_count < K:
                saved_iou_count += 1
                try:
                    img_tensor = images[b].detach().cpu()            # [3,H,W]
                    mask_prob_np = mask_prob.squeeze().cpu().numpy()  # [H,W]
                    global_idx = base_idx + b
                    image_path = testing_dataset.images[global_idx][0]
                    filename = os.path.basename(image_path)
                    filename_no_ext, _ = os.path.splitext(filename)
                    save_path = os.path.join(save_dir, f"cheat_{filename_no_ext}_{saved_iou_count}_iou.png")
                    save_overlay_only(img_tensor, mask_prob_np, save_path)
                except Exception as e:
                    print(f"Warning: failed to save overlay image: {e}")
                    pass
        base_idx += B
#########################################################################
# CLASSIFICATION METRICS
#########################################################################
y_true_cls = np.array(y_true_cls)
y_pred_cls = np.array(y_pred_cls)
y_probs_cls = np.vstack(y_probs_cls) if len(y_probs_cls) > 0 else np.zeros((0, 1))

print("Classification results (CHEAT - perfect classifier):")
print(f"Accuracy: {accuracy_score(y_true_cls, y_pred_cls):.4f}")
print(classification_report(y_true_cls, y_pred_cls, labels=[0,1], target_names=["real","tampered"], zero_division=0))

cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0,1])
for i, acc in enumerate(cm.diagonal() / (cm.sum(axis=1) + 1e-12)):
    print(f"Class {i} accuracy: {acc:.4f}")

# Macro AP (binary)
try:
    y_true_onehot = label_binarize(y_true_cls, classes=[0,1])
    test_ap_macro = average_precision_score(y_true_onehot, y_probs_cls, average="macro")
except Exception:
    test_ap_macro = None

print(f"Classification macro AP: {test_ap_macro}")

#########################################################################
# SEGMENTATION METRICS
#########################################################################
if seg_count > 0:
    mean_iou = seg_iou_sum / seg_count
    mean_f1 = seg_f1_sum / seg_count
else:
    mean_iou = 0.0
    mean_f1 = 0.0

# Final global ROC-AUC
if true_buffer:
    y_true_flat = np.concatenate(true_buffer)
    y_score_flat = np.concatenate(score_buffer)
    try:
        global_auc = roc_auc_score(y_true_flat, y_score_flat)
    except:
        global_auc = 0.0
else:
    global_auc = 0.0

print(f"\nSegmentation results (predicted tampered only; CHEAT classifier):")
print(f"Mean IoU: {mean_iou:.4f} (n={seg_count})")
print(f"Mean F1: {mean_f1:.4f} (n={seg_count})")
print(f"Global ROC-AUC: {global_auc:.4f}")
print(f"Saved {saved_iou_count} images with IoU in [0.10,0.20] to: {save_dir}")
