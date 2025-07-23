# Brain Tumor Segmentation App using Deep Learning & Streamlit

This is a web-based application that performs brain tumor segmentation from MRI scans using a deep learning model (U-Net architecture). The model takes in `.nii` format MRI volumes and returns segmentation masks highlighting tumor regions.

---

## Features

- ✅ Upload `.nii` (NIfTI) files for brain MRI scans
- ✅ Visualize input MRI slices
- ✅ View predicted tumor segmentation masks
- ✅ Compare predictions side-by-side with original
- ✅ Web interface using Streamlit

---

## Model Overview

- Model: **U-Net**
- Framework: **TensorFlow / Keras**
- Input Modalities: T1, T1ce, T2, FLAIR (preprocessed as needed)
- Output: Binary segmentation mask per slice

---

