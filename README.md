# Domain Adaptation with Deep Feature Clustering for Pseudo-Label Denoising in Cross-Source SAR Image Classification

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-red.svg)](https://pytorch.org/)

Official implementation of our SAR-ATR method that achieves state-of-the-art performance on the SAMPLE dataset through innovative domain adaptation and pseudo-label denoising techniques.

## Key Features

- **Linear-kernel MMD Domain Adaptation**: Novel domain adaptation framework using linear-kernel Maximum Mean Discrepancy (MMD) for effective cross-domain feature alignment
- **Deep Feature Clustering**: t-SNE-based clustering of adapter layer features for robust pseudo-label denoising
- **End-to-End Training**: Single-stage training pipeline without complex multi-phase optimization
- **SOTA Performance**: Achieves 97.87% accuracy on Scenario I (standard cross-domain) and 98.54% on Scenario III (generalization) of SAMPLE dataset

## Method Overview

Our approach addresses cross-source SAR image classification through:
1. **Domain-Invariant Feature Learning**: Linear-kernel MMD loss minimizes distribution discrepancy between simulated and measured SAR images
2. **Adapter FC Layer**: 512×256 dimensional bottleneck layer that learns domain-shared representations
3. **Progressive Pseudo-Label Refinement**: Multi-stage denoising via t-SNE clustering of deep features with dynamic thresholding

![Framework](assets/framework.png) *(Architecture diagram from paper)*

## Installation

```bash
git clone https://github.com/TheGreatTreatsby/SAR-Domain-Adaptation.git
cd SAR-Domain-Adaptation
conda env create -f environment.yml
conda activate sar-da
