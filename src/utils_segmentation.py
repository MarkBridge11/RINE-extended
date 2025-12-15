import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import os
from io import BytesIO
import pickle
import copy
import json
import random
import time

import cv2
import numpy as np
from PIL import Image, ImageFilter
import math

from src.models import Model #,DINOv2Model
from src.models_binary import DINOv2Model

from src.data_multiclass_nomask import CustomDataset, HuggingFaceDataset
from datasets import load_dataset, load_from_disk 
import torchvision.transforms.functional as TF 
from tqdm import tqdm 
import wandb 
from src.segmentation_head import TamperSegHead
import timm 
from sklearn.metrics import average_precision_score, f1_score 

def get_model(device,backbone,ckpt_path,selected_layers,nproj,proj_dim):
    if backbone=="ViT-L/14":
        model = Model(
            backbone=("ViT-L/14", 1024),
            nproj=nproj,
            proj_dim=proj_dim,
            device=device,
        )
    else:
        model = DINOv2Model(
            backbone=("vit_base_patch14_dinov2",768),
            nproj=nproj,
            proj_dim=proj_dim,
            device=device,
            selected_layers=selected_layers,
        )
    #state_dict = torch.load(ckpt_path, map_location=device)
    #for name in state_dict:
    #    exec(
    #        f'model.{name.replace(".", "[", 1).replace(".", "].", 1)} = torch.nn.Parameter(state_dict["{name}"])'
    #    )

    state_dict = torch.load(ckpt_path, map_location=device)

    missing_keys = []
    for name, value in state_dict.items():
        try:
            parts = name.split('.')
            obj = model
            for p in parts[:-1]:
                obj = getattr(obj, p)
            if isinstance(value, torch.Tensor):
                setattr(obj, parts[-1], torch.nn.Parameter(value))
            else:
                setattr(obj, parts[-1], value)
        except AttributeError:
            missing_keys.append(name)
    
    if missing_keys:
        print(f"⚠️ Warning: some keys from checkpoint were not found in model ({len(missing_keys)}):")
        for k in missing_keys[:10]:
            print(f"  - {k}")
        if len(missing_keys) > 10:
            print("  ...")


    return model

def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2,3))
    loss = 1 - ((2. * intersection + smooth) / 
                (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth))
    return loss.mean()

def get_transforms_multiclass():
    transforms_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    transforms_mask = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]   
    )
    transforms_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    return transforms_train, transforms_mask, transforms_test

############################################## DINO V2 HELPER FUNCTIONS
def interpolate_pos_embed(pos_embed, grid_h, grid_w):
    """
    Interpolates the DINOv2 positional embeddings (1, N+1, C)
    to a new grid size (grid_h × grid_w).
    """
    cls_pos = pos_embed[:, 0:1, :]        # [1, 1, C]
    patch_pos = pos_embed[:, 1:, :]       # [1, H*W, C]

    # original grid size (square)
    num_patches = patch_pos.shape[1]
    orig_h = orig_w = int(num_patches ** 0.5)

    # reshape to 2D grid
    patch_pos = patch_pos.reshape(1, orig_h, orig_w, -1).permute(0, 3, 1, 2)  # [1, C, H, W]

    # interpolate to new grid size
    patch_pos = F.interpolate(
        patch_pos,
        size=(grid_h, grid_w),
        mode="bicubic",
        align_corners=False
    )

    # reshape back to tokens
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)

    # concatenate CLS + interpolated patch embeddings
    return torch.cat([cls_pos, patch_pos], dim=1)


def extract_dinov2_features(model, images, layers):
    B, _, H, W = images.shape
    # 1. Patch embedding
    x_patches = model.patch_embed(images)   # [B, N, C]
    cls_token = model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_token, x_patches), dim=1)

    # 2. Compute new patch grid (important!)
    ph, pw = model.patch_embed.patch_size
    grid_h = H // ph
    grid_w = W // pw

    # 3. Interpolate pos embeddings
    pos_embed = interpolate_pos_embed(
         model.pos_embed,
         grid_h, grid_w
    )
    x = x + pos_embed

    fmaps = []
    #x = model.patch_embed(images)
    #cls_token = model.cls_token.expand(x.shape[0], -1, -1)
    #x = torch.cat((cls_token, x), dim=1)
    #x = x + model.pos_embed
    x = model.pos_drop(x)

    for i, blk in enumerate(model.blocks):
        x = blk(x)
        if i in layers:
            fmaps.append(x[:, 1:, :].detach())
    return fmaps
############################################################################

class RINEAugment:
    def __init__(self, p_flip=0.5, p_blur=0.5, p_jpeg=0.5):
        self.p_flip = p_flip
        self.p_blur = p_blur
        self.p_jpeg = p_jpeg

    def jpeg_compress(self, img: Image.Image, qfactor: float):
        buffer = BytesIO()
        # convert quality factor [0.3,1] → JPEG quality [30,100]
        q = int(30 + 70 * qfactor)
        img.save(buffer, format="JPEG", quality=q)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def __call__(self, img, mask, label):

        if random.random() < self.p_flip: # random flip also applied to masks
            img = TF.hflip(img)
            if label == 2 and mask is not None:
               mask = TF.hflip(mask)

        if random.random() < self.p_blur: # random blur
            img = img.filter(ImageFilter.GaussianBlur(radius=1 + random.random() * 2))

        if random.random() < self.p_jpeg: # random jpeg compression
            qf = random.uniform(0.30, 1.0)
            img = self.jpeg_compress(img, qf)

        return img, mask

def train_segmentation_branch(num_epochs,run_setting,workers,device):

    unique_id = f"{random.randint(0, 99999):05d}_{int(time.time())}"

    wandb.init(
        project="RINE-segmentation",   # name of your project
        entity="vision-team-unipd",
        name=f"experiment-{unique_id}",            # optional: name of this run
        config={
            "epochs": num_epochs,
            **run_setting,
            "model": "RINE-segmentation-branch",
        }
    )

    ################################# VIT-L/14
    #model = get_model(device,backbone="ViT-L/14",ckpt_path="/home/mbrigo/RINE-extended/ckpt/RINE_multiclass_SIDA_best.pth",nproj=4,proj_dim=512) 
    ################################# 
    backbone_name, embed_dim = run_setting["backbone"]
    model = timm.create_model(backbone_name,pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()

    seg_head = TamperSegHead().to(device) # or SegDINOHead() for the lightweight version of ~2M parameters instead of ~8/9M 

    transforms_train , transforms_mask , transforms_test = get_transforms_multiclass()
    
    dataset = load_from_disk("/home/mbrigo/SID_Set_local")
    dataset = {
        split: d.filter(lambda example: example["label"] == 2)
        for split, d in dataset.items()
    }
    train_hf = dataset["train"]
    validation_hf = dataset["validation"]

    augment = RINEAugment(p_flip=0.5,p_blur=0.5,p_jpeg=0.5)

    # Remember to modify the preprocessing of the images when using DINOv2 (518 input size and not anymore 224)
    training_dataset = HuggingFaceDataset(train_hf,transforms=transforms_train,augment=augment) 
    validation_dataset = HuggingFaceDataset(validation_hf,transforms=transforms_train)

    testing_dataset = CustomDataset(split="test",transforms=transforms_test)
    testing_dataset.images = [img for img in testing_dataset.images if img[1] == 2]

    train = DataLoader(training_dataset, batch_size=run_setting['batch_size'], 
                       shuffle=True, num_workers=workers, 
                       pin_memory=True, drop_last=False)
    val = DataLoader(validation_dataset, batch_size=run_setting['batch_size'], 
                     shuffle=False, num_workers=workers, 
                     pin_memory=True, drop_last = False)
    test = DataLoader(testing_dataset, batch_size=run_setting['batch_size'], 
                      shuffle=False, num_workers=workers, 
                      pin_memory=True, drop_last = False)

    optimizer = torch.optim.Adam(seg_head.parameters(),lr=run_setting["lr"])

    print(json.dumps(run_setting,indent=2))
    results = {"train_loss": [], "val_loss": [], "val_iou": [], "test": {}}

    patch_size = 14  # ViT-L/14 and also for DINOv2

    lambda_bce = 2.0
    lambda_dice = 0.5

    # training loop
    for epoch in range(num_epochs):
        seg_head.train()
        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_dice = 0.0
        num_batches = 0

        for batch in tqdm(train, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            fmaps = []
            images, _ , masks = batch
            images = images.to(device)
            masks = masks.to(device).float().unsqueeze(1)
            patch_h = images.shape[2] // patch_size
            patch_w = images.shape[3] // patch_size

            #######################################################################################
            # DINOv2 version
            with torch.no_grad():
                fmaps = extract_dinov2_features(model, images, run_setting["layers"])
            #######################################################################################
            fmaps = [f.float() for f in fmaps]

            optimizer.zero_grad()
            
            mask_pred = seg_head(fmaps,patch_h,patch_w)
            mask_pred = F.interpolate(mask_pred, size=(images.shape[2], images.shape[3]),
                          mode='bilinear', align_corners=True)

            bce = F.binary_cross_entropy_with_logits(mask_pred, masks)

            pred_probs = torch.sigmoid(mask_pred)
            dice = dice_loss(pred_probs, masks)

            loss = lambda_bce*bce + lambda_dice*dice

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bce += bce.item()
            epoch_dice += dice.item()

            num_batches += 1

            wandb.log({
                "batches": num_batches,
                "batch_loss": loss.item(),
                "batch_loss_bce": bce.item(),
                "batch_loss_dice": dice.item(),
            })

        avg_train_loss = epoch_loss / num_batches
        avg_bce_loss = epoch_bce / num_batches
        avg_dice_loss = epoch_dice / num_batches
        results["train_loss"].append(avg_train_loss)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss_epoch": avg_train_loss,
            "bce_loss_epoch": avg_bce_loss,
            "dice_loss_epoch": avg_dice_loss,
        })
    
        # inner validation loop
        seg_head.eval()
        val_loss = 0.0
        ious = []
        with torch.no_grad():
            for batch in tqdm(val, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                fmaps = []
                images, _, masks = batch
                images = images.to(device)
                masks = masks.to(device).float().unsqueeze(1)

                #######################################################################################
                # DINOv2 version
                with torch.no_grad():
                    fmaps = extract_dinov2_features(model, images, run_setting["layers"])
                #######################################################################################

                mask_pred = seg_head(fmaps,patch_h,patch_w)
                mask_pred = F.interpolate(mask_pred, size=(images.shape[2], images.shape[3]),
                          mode='bilinear', align_corners=True)

                bce = F.binary_cross_entropy_with_logits(mask_pred, masks)
                
                pred_probs = torch.sigmoid(mask_pred)
                dice = dice_loss(pred_probs, masks)
                loss = bce + dice

                val_loss += loss.item()

                # IoU metric
                preds = (pred_probs > 0.5).float()
                intersection = (preds * masks).sum(dim=(1,2,3))
                union = (preds + masks).clamp(max=1).sum(dim=(1,2,3))
                iou = (intersection / (union + 1e-6)).mean().item()

                ious.append(iou)

                wandb.log({
                    "batch_val_loss": loss.item(),
                    "batch_val_loss_bce": bce.item(),
                    "batch_val_loss_dice": dice.item(),
                    "val_ious": iou,
                })
        
        avg_val_loss = val_loss / len(val)
        avg_val_iou = np.mean(ious)
        results["val_loss"].append(avg_val_loss)
        results["val_iou"].append(avg_val_iou)

        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "val_loss": avg_val_loss,
            "val_iou": avg_val_iou,
        })
    
    # TESTING PHASE
    seg_head.eval()
    test_ious, test_aucs, test_f1s = [], [], []
    with torch.no_grad():
        for batch in tqdm(test, desc="Testing"):
            fmaps = []
            images, _, masks = batch
            images = images.to(device)
            masks = masks.to(device).float().unsqueeze(1)

            #######################################################################################
            # DINOv2 version
            with torch.no_grad():
                fmaps = extract_dinov2_features(model, images, run_setting["layers"])
            #######################################################################################

            mask_pred = seg_head(fmaps,patch_h,patch_w)
            mask_pred = F.interpolate(mask_pred, size=(images.shape[2], images.shape[3]), mode='bilinear', align_corners=True)
            pred_probs = torch.sigmoid(mask_pred)

            preds = (pred_probs > 0.5).float()
            intersection = (preds * masks).sum(dim=(1,2,3))
            union = (preds + masks).clamp(max=1).sum(dim=(1,2,3))
            iou = (intersection / (union + 1e-6)).mean().item()
            test_ious.append(iou)

            # --- Flatten tensors for AUC & F1 ---
            y_true = masks.cpu().numpy().ravel()
            y_score = pred_probs.cpu().numpy().ravel()
            y_pred = preds.cpu().numpy().ravel()

            # --- AUC (pixel-wise precision-recall AUC = average precision) ---
            if np.any(y_true == 1):  # skip if no positives in mask
                auc = average_precision_score(y_true, y_score)
                test_aucs.append(auc)

            # --- F1 Score (pixel-level binary classification) ---
            f1 = f1_score(y_true, y_pred, zero_division=0)
            test_f1s.append(f1)

    results["test"]["mean_iou"] = float(np.mean(test_ious))
    results["test"]["mean_auc"] = float(np.mean(test_aucs)) if len(test_aucs) > 0 else 0.0
    results["test"]["mean_f1"] = float(np.mean(test_f1s))

    print(f"Test Mean IoU: {results['test']['mean_iou']:.4f}")
    print(f"Test Mean AUC: {results['test']['mean_auc']:.4f}")
    print(f"Test Mean F1 : {results['test']['mean_f1']:.4f}")

    wandb.log({
        "test_mean_iou": results["test"]["mean_iou"],
        "test_mean_auc": results["test"]["mean_auc"],
        "test_mean_f1": results["test"]["mean_f1"]
    })

    # ---- SAVE CKPT ----
    ckpt_name = f"/home/mbrigo/RINE/ckpt_seg/SegBranch_{unique_id}_superDPTfaithful_{num_epochs}_SIDAweighted.pth"
    torch.save(seg_head.state_dict(), ckpt_name)
    wandb.finish()

