import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import train_multiclass

device = "cuda:0"
workers = 1
epochss = 1
epochs_reduce_lr = [6, 11]
backbones = [("ViT-L/14", 1024)]
factors = [0.2, 0.4]
nprojs = [2, 4]
proj_dims = [512, 1024]
batch_sizes = [128]
lrs = [1e-3]
experiments = []
for backbone in backbones:
    for factor in factors:
        for nproj in nprojs:
            for proj_dim in proj_dims:
                for batch_size in batch_sizes:
                    for lr in lrs:
                        experiments.append(
                            {
                                "backbone": backbone,
                                "factor": factor,
                                "nproj": nproj,
                                "proj_dim": proj_dim,
                                "batch_size": batch_size,
                                "lr": lr,
                                "savpath": f"/home/mbrigo/results_RINE_SIDA/{backbone[0].replace('/', '-')}_{factor}_{nproj}_{proj_dim}_{batch_size}_{lr}",
                            }
                        )

for experiment in experiments:
    train_multiclass(
        model_setting=experiment,
        epochs=epochss,
        epochs_reduce_lr=epochs_reduce_lr,
        workers=workers,
        device=device
    )
