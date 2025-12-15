# Paper
This repository contains the a modified implementation code for the Koutlis *et al.* ([arXiv:2402.19091](https://arxiv.org/abs/2402.19091)) approach.

# About the work
In the following work an enhanced version of the RINE approach proposed by Koutlis *et al.* is designed. Minimal changes were made to the classification head in order to permit multi-class classification on real, synthetic and tampered datasets.


Futhermore a segmentation branch is attached to the frozen backbone that exploit the forward phase of the classification part to take the patch tokens to generate the mask of the tampered region in tampered images. This branch is activated if and only the classification label the input image as tampered.

# Files
In the `scripts/` directory you can find files to train the different part of the architecture performing a grid search on hyperparameters that were tryed.

In `src/` directory you can find:
- base RINE architecture with DINOv2 ([arXiv:2304.07193](https://arxiv.org/abs/2304.07193)) as frozen backbone instead of CLIP:ViT-L/14, because as shown in my thesis, it lead to improvement results in terms of generalization capability and feature discrimination in the multi-class setting of the task as studied by [Wang *et al.*](https://arxiv.org/abs/2505.04410);
- segmentation architecture designed getting highly inspired by [Ranftl *et al.*](https://arxiv.org/abs/2103.13413) and [Yang *et al.*](https://arxiv.org/abs/2509.00833);
- many utility files with code retrieved from Koutlis *et al.* repository (https://github.com/mever-team/rine).

In `single_inference.ipynb` you can perform an inference with a single image. Paths must be changed accordingly to make the code work.


# Checkpoint link
https://drive.google.com/drive/folders/1B18BmLRLUJsqxuKdwokyCGZQopshmXCQ?usp=drive_link


