# Enhancing Survival Outcomes in Head and Neck Cancer through Joint HPV Classification and Tumor Segmentation

This work was developed and evaluated as part of our participation in the MICCAI 2025 HEad and neCK TumOR (HECKTOR 2025) segmentation and outcome prediction challenge as the SIMS-LIFE team.

In this work, we employed a multi-task learning framework that simultaneously performs Human Papillomavirus (HPV) classification, tumor segmentation, and survival prediction based on Positron Emission Tomography, Computed Tomography imaging, and clinical features. 


### Paper:
link will be added after publication


## Packages:
This project is built with **TensorFlow 2.x (GPU-compatible)** to leverage recent NVIDIA GPUs for training, python 3.8+ is required.
1. pip install --no-cache-dir tensorflow[and-cuda]
2. pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url http://download.pytorch.org/whl/cu126 --trusted-host download.pytorch.org


## Pre-processing Hecktor dataset:
1. Modify preprocessing_hecktor.py, by changing the path to your dataset directory
3. Run python preprocessing_hecktor.py


## To run the Multi-task framework code:
python train_CV_cont.py
