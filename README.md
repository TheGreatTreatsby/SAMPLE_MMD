🚀 ## Domain Adaptation with Deep Feature Clustering for Pseudo-Label Denoising in Cross-Source SAR Image Classification
This repository contains the implementation of the paper titled "Domain Adaptation with Deep Feature Clustering for Pseudo-Label Denoising in Cross-Source SAR Image Classification". The proposed method addresses the domain shift problem in Synthetic Aperture Radar (SAR) image classification by combining domain adaptation with deep feature clustering for pseudo-label denoising. The approach achieves state-of-the-art (SOTA) performance on the SAMPLE dataset across three challenging evaluation scenarios.
🔍 ##Key Contributions
Linear Kernel Maximum Mean Discrepancy (MMD): A domain adaptation framework that minimizes the distribution gap between synthetic and real SAR images using a linear kernel MMD loss, eliminating the need for complex data augmentation or multi-stage training.
Deep Feature Clustering for Pseudo-Label Denoising: A novel pseudo-label denoising strategy based on deep feature clustering, which leverages t-SNE for dimensionality reduction and clustering to filter out noisy pseudo-labels.
End-to-End Training: The proposed method is fully data-driven and does not rely on handcrafted features or noise models, achieving superior performance in cross-domain SAR image classification tasks.
📂 ##Dataset
The experiments are conducted on the SAMPLE dataset, which contains paired synthetic and real SAR images of 10 military vehicle targets. The dataset is divided into three evaluation scenarios:
Scenario I: Training on synthetic data (14-17°) and testing on real data (14-17°).
Scenario II: Training on synthetic data (14-16°) and testing on real data (17°).
Scenario III: Training on limited real data (14-16°) and testing on real data (17°).
📊 ##Results
The proposed method achieves the following results:
Scenario I: 97.87% accuracy.
Scenario II: 97.69% accuracy.
Scenario III: 98.54% accuracy.
These results outperform existing methods, demonstrating the effectiveness of the proposed approach in addressing domain shift and pseudo-label noise.
🛠️ ##Code Structure
