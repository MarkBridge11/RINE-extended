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
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import accuracy_score, average_precision_score

from src.models import Model, DINOv2Model

from src.data_multiclass_nomask import CustomDataset, HuggingFaceDataset
from sklearn.preprocessing import label_binarize 
from sklearn.metrics import classification_report 
from datasets import load_from_disk 
import torchvision.transforms.functional as TF 
import wandb 
from sklearn.metrics import confusion_matrix


def get_transforms():
    transforms_train = transforms.Compose(
        [
            transforms.Lambda(lambda img: data_augment(img)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    transforms_test_1 = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    transforms_test_2 = transforms.Compose(
        [
            transforms.TenCrop(224),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [transforms.PILToTensor()(crop) for crop in crops]
                )
            ),
            transforms.Lambda(lambda x: x / 255),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    return transforms_train, transforms_test_1, transforms_test_2

######################## MULTICLASS METHODS
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


    return transforms_train,transforms_mask,transforms_test


def train_multiclass(model_setting,device,epochs_reduce_lr,workers=12,epochs=1): # CUSTOM TRAINING
    
    unique_id = f"{random.randint(0, 99999):05d}_{int(time.time())}" # just to generate a random number to change the logging 
    
    wandb.init(
        project="RINE-multiclass-1epoch-nomask",   # name of your project
        entity="vision-team-unipd",
        name=f"experiment-{unique_id}",            
        config={
            "learning_rate": model_setting["lr"],
            "batch_size": model_setting["batch_size"],
            "epochs": epochs,
            "nproj": model_setting["nproj"],
            "proj_dim": model_setting["proj_dim"],
            "factor": model_setting["factor"],
            "model": "RINE-multiclass-extended",
        }
    )

    dataset = load_from_disk("/home/mbrigo/SID_Set_local")

    train_hf = dataset["train"]
    validation_hf = dataset["validation"]

    transforms_train, _, transforms_test = get_transforms_multiclass()

    training_dataset = HuggingFaceDataset(train_hf,transforms=transforms_train)
    validation_dataset = HuggingFaceDataset(validation_hf,transforms=transforms_train)
    testing_dataset = CustomDataset(split="test",transforms=transforms_test)

    train = DataLoader(training_dataset, batch_size=model_setting["batch_size"],
                       shuffle=True, num_workers=workers, 
                       pin_memory=True, drop_last=False)
    val = DataLoader(validation_dataset, batch_size=model_setting["batch_size"], 
                     shuffle=False, num_workers=workers, 
                     pin_memory=True, drop_last = False)
    test = DataLoader(testing_dataset, batch_size=model_setting["batch_size"], 
                      shuffle=False, num_workers=workers, 
                      pin_memory=True, drop_last = False)
    #model = Model(backbone=("ViT-L/14", 1024),
    #              nproj=model_setting["nproj"],
    #              proj_dim=model_setting["proj_dim"],
    #              device=device)
    model = DINOv2Model(backbone=model_setting["backbone"],
                  nproj=model_setting["nproj"],
                  proj_dim=model_setting["proj_dim"],
                  device=device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=model_setting["lr"])
    cce = nn.CrossEntropyLoss(reduction="mean") 
    supcon = SupConLoss()

    print(json.dumps(model_setting,indent=2))
    results = {"val_loss": [], "val_acc": [], "test": {}}
    rlr = 0 # reduce learning rate
    training_time = 0
    # TRAINING LOOP
    for epoch in range(epochs):
        training_epoch_start = time.time()
        epoch_train_loss = 0
        epoch_loss_cls = 0
        epoch_loss_supcon = 0
        
        if epoch + 1 in epochs_reduce_lr:
            rlr += 1
            optimizer.param_groups[0]["lr"] = model_setting["lr"] / 10**rlr
            
        current_lr = optimizer.param_groups[0]['lr']
        
        model.train()
        # inner training loop that takes images from dataloader
        for i,data in enumerate(train):
            images, labels, _ = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss_cls = cce(outputs[0], labels)

            # supervised contrastive loss
            loss_supcon = model_setting["factor"] * supcon( # factor is the weight for the supervised contrastive loss
                F.normalize(outputs[1]).unsqueeze(1), labels
            )
            
            # total loss
            loss_ = loss_cls + loss_supcon
            
            loss_.backward()
            optimizer.step()
            
            epoch_train_loss += loss_.item()
            epoch_loss_cls += loss_cls.item()
            epoch_loss_supcon += loss_supcon.item()

            global_step = epoch * len(train) + i  # unique global step for each batch

            wandb.log({
                "step": global_step,
                "batch_loss": loss_.item(),
                "batch_loss_cls": loss_cls.item(),
                "batch_loss_supcon": loss_supcon.item(),
                "lr": current_lr,
            })

        print(
            f"\r[Epoch {epoch + 1:02d}/{epochs:02d} | Batch {i + 1:04d}/{len(train):04d} | Time {training_time + time.time() - training_epoch_start:1.1f}s] loss: {loss_.item():1.4f}",
            end="",
        )

        epoch_train_loss /= len(train)
        epoch_loss_cls /= len(train)
        epoch_loss_supcon /= len(train)

        wandb.log({
            "epoch": epoch,
            "train_loss_epoch": epoch_train_loss,
            "loss_cls_epoch": epoch_loss_cls,
            "loss_supcon_epoch": epoch_loss_supcon,
            "lr": current_lr,
        })

        training_time += time.time() - training_epoch_start

        #Validation
        model.eval()
        y_true = []
        y_score = []
        val_loss = 0
        with torch.no_grad():
            #Validation loop that takes images from val dataloader
            print(f"Validation for epoch: {epoch}")
            for data in val:
                images, labels, _ = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss_cls = cce(outputs[0],labels)
                loss_supcon = model_setting["factor"]*supcon(
                    F.normalize(outputs[1]).unsqueeze(1),labels
                )
                loss_ = loss_cls + loss_supcon
                val_loss += loss_.item()
                y_true.extend(labels.cpu().numpy().tolist())
                y_score.extend(outputs[0].argmax(dim=1).cpu().numpy().tolist()) #change to softmax since we have multiclass training

        val_acc = accuracy_score(np.array(y_true), np.array(y_score))
        target_names = ['real','fake','tampered']
        report = classification_report(np.array(y_true),np.array(y_score),labels=[0,1,2],target_names=target_names,output_dict=True)
        print(report)
        results["val_loss"].append(val_loss / len(val))
        results["val_acc"].append(val_acc)
        print(f", val_loss: {val_loss / len(val):1.4f}, val_acc: {val_acc:1.4f}")

        wandb.log({
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "LR": current_lr,
            **{f"ClassAcc/{cls_name}": report[cls_name]["recall"] for cls_name in target_names},
        })

        cm = confusion_matrix(np.array(y_true), np.array(y_score))
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(class_accuracies):
            print(f"Validation: Class {i}: {acc:.2f}")

        # Testing, here done after the final epoch 
        if epoch + 1 == epochs:
            print("SIDA testing set: ACC/AP")
            test_acc = 0
            test_ap = 0
            model.eval()
            y_true = []
            y_score = []
            y_probs = []  # <-- collect probabilities here
            with torch.no_grad():
                print(f"Testing with SIDA test:")
                for data in test:
                    images, labels, _ = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)

                    # true labels
                    y_true.extend(labels.cpu().numpy().tolist())

                    # hard predictions (for accuracy/classification_report)
                    y_score.extend(outputs[0].argmax(dim=1).cpu().numpy().tolist())

                    # softmax probabilities (for AP)
                    batch_probs = torch.softmax(outputs[0], dim=1).cpu().numpy()
                    y_probs.extend(batch_probs)

            y_true = np.array(y_true)
            y_score = np.array(y_score)
            y_probs = np.array(y_probs)

            test_acc = accuracy_score(y_true, y_score) #Accuracy
            print(classification_report(y_true, y_score, labels=[0,1,2], target_names=target_names))

            y_true_onehot = label_binarize(y_true, classes=[0,1,2]) #Average precision
            test_ap = average_precision_score(y_true_onehot, y_probs, average="macro")

            results["test"]["SIDA"] = {
                "acc": test_acc,
                "ap": test_ap
            }

            print(f"SIDA test set: {100*test_acc:1.1f}/{100*test_ap:1.1f}")

            wandb.log({
                "epoch": epoch,
                "test_acc": test_acc,
                "test_ap": test_ap,
            })


            cm = confusion_matrix(np.array(y_true), np.array(y_score))
            class_accuracies = cm.diagonal() / cm.sum(axis=1)
            for i, acc in enumerate(class_accuracies):
                print(f"Testing: Class {i}: {acc:.2f}")


            ckpt_name = f"/home/mbrigo/RINE/ckpt_DINO/RINE_DINO_SIDA_multiclass_{epoch}_of_{epochs}_{model_setting['nproj']}_{model_setting['proj_dim']}_{model_setting['factor']}.pth"
            # torch.save(
            #     {
            #         k:model.state_dict()[k]
            #         for k in model.state_dict() # save everything that is not CLIP
            #         if "clip" not in k
            #    },
            #     ckpt_name
            # )
            torch.save(
                {
                    k: v
                    for k, v in model.state_dict().items()
                    if "clip" not in k and "dinov2" not in k
                },
                ckpt_name
            )
            
    wandb.finish()

##################################################################################

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
