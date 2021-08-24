# CNN-ViT-for-medical-image-diagnosis

This repository is Keras implementation of hybrid CNN and Vision Transformer models for medical image based diagnostic solutions.  Codes are verified on python3.8 with tensorflow version '2.4.1'. Other dependencies are NumPy, cv2, sklearn, matplotlib, random, os, etc.


For training and visualization:
Use train_models.py to train the models.
Run view_gradients_and_attention_maps.py  for visualization of attention maps and gradient maps generation.

Data Resources:
1. https://www.kaggle.com/c/diabetic-retinopathy-detection/data
2. https://www.kaggle.com/kmader/colorectal-histology-mnist
3. https://challenge2018.isic-archive.com/task3/
4. https://www.tensorflow.org/datasets/catalog/curated_breast_imaging_ddsm
5. https://github.com/lindawangg/COVID-Net
6.  https://www.kaggle.com/paultimothymooney/breast-histopathology-images
7. https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Data preparation:
For data preparation use data_prep.py.

