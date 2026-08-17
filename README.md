# 🧠 Brain Tumor Classification (MRI)

Classify brain MRI scans into four categories — **glioma**, **meningioma**, **pituitary tumor**, and **no tumor** — using three deep learning approaches: a custom CNN built from scratch, a fine-tuned **VGG16**, and a fine-tuned **ResNet101**. The best model reaches **~98.2% test accuracy**.

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Original Notebook](#original-notebook)
- [License](#license)

## Overview

Brain tumors are abnormal growths of cells in or around the brain that can be benign or malignant. Early, accurate detection from MRI scans is critical for treatment planning. This project trains and compares three convolutional architectures on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) to classify scans into four classes:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

## Results

| Model       | Test Accuracy | Test Loss | Notes                                   |
|-------------|:-------------:|:---------:|------------------------------------------|
| Custom CNN  | 97.10%        | 0.132     | 4-block CNN trained from scratch          |
| VGG16       | 98.17%        | 0.066     | ImageNet weights, last 10 layers fine-tuned |
| ResNet101   | **98.25%**    | **0.054** | ImageNet weights, fully fine-tuned        |

*(Numbers reflect a single training run with the hyperparameters in `src/config.py`; expect minor variation across runs/hardware.)*

## Dataset

This project uses the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle (~7,000 T1-weighted contrast MRI images, already split into `Training/` and `Testing/` folders, each containing one sub-folder per class).

Download it either via the Kaggle CLI/`kagglehub`, or manually, and place it so the folder layout looks like:

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Using `kagglehub` (recommended):

```python
import kagglehub
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
print(path)  # symlink/copy this into ./data
```

Or via the Kaggle CLI:

```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data --unzip
```

## Project Structure

```
brain-tumor-classification/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── brain_tumor_classification_mri_3_models.ipynb   # original exploratory notebook
├── src/
│   ├── config.py         # paths, hyperparameters, class names
│   ├── data_loader.py     # ImageDataGenerator pipelines
│   ├── models.py           # CNN / VGG16 / ResNet101 architectures
│   ├── train.py             # training entry point (CLI)
│   ├── evaluate.py           # evaluation entry point (CLI)
│   ├── predict.py             # single-image inference (CLI)
│   └── utils.py                 # plotting & metrics helpers
├── saved_models/            # trained .h5 models (git-ignored)
└── assets/                   # generated plots (git-ignored)
```

## Setup

```bash
git clone https://github.com/<your-username>/brain-tumor-classification.git
cd brain-tumor-classification

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> A GPU (CUDA-enabled) is strongly recommended for training — the notebook was originally run on a Tesla T4.

## Usage

### 1. Train a model

```bash
# Train the custom CNN (30 epochs by default)
python -m src.train --model cnn

# Train VGG16 (fine-tuned)
python -m src.train --model vgg16

# Train ResNet101 (fine-tuned)
python -m src.train --model resnet101 --epochs 20 --batch-size 32
```

Each run saves the best checkpoint and final model to `saved_models/`, and training curves to `assets/`.

### 2. Evaluate a saved model

```bash
python -m src.evaluate --model-path saved_models/resnet101_final.h5
```

Prints a classification report and saves a confusion matrix to `assets/confusion_matrix.png`.

### 3. Predict on a single image

```bash
python -m src.predict --model-path saved_models/resnet101_final.h5 --image path/to/scan.jpg
```

```
Predicted class: pituitary (97.42% confidence)
All class scores:
  glioma      :  0.812%
  meningioma  :  1.203%
  notumor     :  0.565%
  pituitary   : 97.420%
```

## Model Architectures

**Custom CNN** — 4 convolutional blocks (32 → 64 → 128 → 128 filters), each followed by max-pooling, flattened into a 512-unit dense layer with dropout (0.5), and a 4-way softmax output. Trained from scratch with Adam.

**VGG16 (transfer learning)** — ImageNet-pretrained VGG16 backbone with the last 10 layers unfrozen for fine-tuning, topped with global average pooling → Dense(512, relu) → Dropout(0.5) → softmax. Trained with a low learning rate (1e-4).

**ResNet101 (transfer learning)** — ImageNet-pretrained ResNet101 backbone, fully fine-tuned end-to-end, with the same classification head as VGG16.

All models use `categorical_crossentropy` loss, images resized to `150×150×3`, and training data augmented with rotation, shear, brightness jitter, and horizontal flips.

## Original Notebook

The full, exploratory Kaggle notebook — including EDA, data visualization, per-model training runs, learning curves, and confusion matrices — is preserved at [`notebooks/brain_tumor_classification_mri_3_models.ipynb`](notebooks/brain_tumor_classification_mri_3_models.ipynb). The `src/` package is a cleaned-up, modular, and reproducible version of the same pipeline, suitable for CLI use and further development.

## License

This project is released under the [MIT License](LICENSE). The dataset itself is subject to its own license/terms on Kaggle — please review those before redistributing data.

---

**Disclaimer:** This project is for educational/research purposes only and is **not** a certified medical diagnostic tool. Do not use it for real clinical decision-making.
