# 🛰️ EuroSAT Satellite Image Classification
### Deep Learning | Computer Vision | PyTorch

## 📌 Project Overview

This project explores **land use and land cover classification using deep learning** on the EuroSAT dataset, which contains **27,000 Sentinel-2 satellite images across 10 geographical categories**.

The objective was to develop, optimize, and compare Convolutional Neural Networks (CNNs) trained from scratch with pretrained ResNet-18 models using transfer learning.

The project demonstrates an end-to-end deep learning workflow, from data preprocessing and model development to hyperparameter optimization and performance evaluation.

## 🚀 Key Implementations

- **Custom CNN Development:** Designed and trained CNN architectures from scratch, experimenting with network depth and architectural improvements.
- **Hyperparameter Optimization:** Investigated different optimizers, learning rates, and training configurations to improve model performance.
- **Transfer Learning:** Implemented pretrained ResNet-18 using three strategies:
  - Frozen Feature Extraction
  - Partial Fine-Tuning
  - Full Fine-Tuning
- **Model Evaluation:** Compared architectures using accuracy, loss curves, confusion matrices, and class-level performance metrics.
- **Performance Analysis:** Analyzed classification errors and the impact of different training strategies on model generalization.

## 🛠️ Technologies & Tools

- **Language:** Python
- **Deep Learning:** PyTorch, Torchvision
- **Data Processing:** NumPy
- **Visualization:** Matplotlib
- **Evaluation:** Scikit-learn
- **Development Environment:** Jupyter Notebook

## 📊 Dataset

**EuroSAT RGB Dataset**
- 27,000 satellite images
- 10 land use and land cover categories
- 64 × 64 RGB images
- Source: Sentinel-2 satellite imagery

Dataset reference: [EuroSAT Official Repository](https://github.com/phelber/EuroSAT)

## 📈 Project Outcomes

The experiments provided a comparative analysis of CNN architectures trained from scratch and ResNet-18 transfer learning approaches.

The project highlights the importance of architecture selection, hyperparameter tuning, and fine-tuning strategies in satellite image classification.

Detailed experimental results, performance comparisons, and visualizations are available in the project notebook and report.

## 📁 Repository Structure

- `notebooks/` — Model development, training, experiments, and evaluation.
- `report&ppt/` — Final project report and presentation.
- `docs/` — Additional project documentation.

## 👥 Project Team

Developed as a collaborative academic deep learning project by:

- **Amar Farshoukh**
- **Aya Mrad**
- **Zahraa Tarhini**

---

**Keywords:** Deep Learning · Computer Vision · Satellite Image Classification · CNN · ResNet-18 · Transfer Learning · PyTorch · Image Classification
