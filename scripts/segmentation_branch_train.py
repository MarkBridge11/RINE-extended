import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils_segmentation import train_segmentation_branch


device = "cuda:0"
workers = 1
epochss = [1]
epochs_reduce_lr = [6, 11]
backbones = [("vit_base_patch14_dinov2", 768)]
layers = [(2,5,8,11)] # correspond to 3,6,9,12 because 0-indexed
proj_dims = [512]
batch_sizes = [16]
lrs = [1e-3]
experiments = []
for backbone in backbones:
    for layer in layers:
        for proj_dim in proj_dims:
            for batch_size in batch_sizes:
                for lr in lrs:
                    experiments.append(
                        {
                            "backbone": backbone,
                            "layers": layer,
                            "proj_dim": proj_dim,
                            "batch_size": batch_size,
                            "lr": lr,
                        }
                    )

for experiment in experiments:
    for epoch in epochss:
        train_segmentation_branch(
            num_epochs=epoch,
            run_setting=experiment,
            workers=workers,
            device=device
        )
