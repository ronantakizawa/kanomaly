# Time Series Anomaly Detection Using Kolmogorov-Arnold Networks

<img width="868" alt="Screenshot 2024-10-23 at 10 43 40 PM" src="https://github.com/user-attachments/assets/d495f18d-3641-42e2-8e72-1d4185ce03dc">
<img width="835" alt="Screenshot 2024-10-23 at 10 43 55 PM" src="https://github.com/user-attachments/assets/9faf03fd-98ed-432c-aed6-11599c1a6ed9">

This is an implementation of Kolmogorov-Arnold Networks (KANs) for detecting anomalies in time series data. KANs offer a robust framework for modeling intricate temporal patterns with enhanced precision, making them particularly effective for anomaly detection tasks.

**Google Colab**: https://colab.research.google.com/drive/1IO6wNfNIG7Oln8OsugBsZlxxwtbD8jud?usp=sharing

**Article**: https://medium.com/@ronantech/time-series-anomaly-detection-using-kolmogorov-arnold-networks-abd9bbeaa9af

## Overview

Kolmogorov-Arnold Networks, inspired by the Kolmogorov-Arnold representation theorem, provide a powerful alternative by approximating complex multivariate functions through the composition and summation of univariate functions. This approach enables KANs to capture subtle temporal dependencies and identify deviations from expected patterns with high precision.

## Key Features

- Implementation of Kolmogorov-Arnold Networks (KANs) for time series analysis for synthetic data and real-world datasets (ECG5000)
- Synthetic data generation with three types of anomalies:
  - **Point anomalies**: Large random noise added to individual data points
  - **Contextual anomalies**: Specific sine values amplified by a random factor
  - **Collective anomalies**: Windows of consecutive data points altered with noise
- Advanced data preprocessing techniques:
  - Data normalization with StandardScaler
  - Overlapping window creation for capturing temporal context
  - Stratified dataset splitting to maintain class distribution
- Class balancing techniques for imbalanced datasets:
  - SMOTE for synthetic data
  - SMOTEENN (combined oversampling and undersampling) for ECG5000 dataset
- Focal Loss implementation for handling class imbalance during training
- Comprehensive metrics evaluation and visualization:
  - Precision, recall, F1 score, ROC AUC
  - Precision-Recall curves
  - ROC curves
  - Visual anomaly identification

## Theoretical Background

### Kolmogorov-Arnold Representation Theorem

The Kolmogorov-Arnold representation theorem states that any multivariate continuous function can be represented as a finite sum and composition of univariate continuous functions and additions:

For any continuous function, there exist continuous functions φ and ψ such that:

```
f(x₁, x₂, ..., xₙ) = ∑ᵢ₌₁ᵖ φᵢ(∑ⱼ₌₁ⁿ ψᵢⱼ(xⱼ))
```

This theorem provides the theoretical underpinning for KANs, which approximate complex multivariate functions by composing and summing univariate functions. The activation functions in KANs are executed on the edges, making them "learnable" and enhancing performance.

### Advantages of KANs for Anomaly Detection

1. **Universal Function Approximation**: KANs can approximate any continuous function, enabling them to model the underlying patterns in normal time series data accurately.
2. **Sensitivity to Anomalies**: Their precise modeling of normal behavior makes them highly sensitive to deviations from expected patterns.
3. **Hierarchical Feature Learning**: By stacking layers, KANs can capture both local and global patterns, essential for detecting different types of anomalies.
4. **Fourier-based Transformations**: The use of Fourier features enables KANs to efficiently capture periodic patterns common in time series data.

## Results and Analysis

### Synthetic Data Results

The model achieves the following performance on synthetic data:
- **Precision**: 1.0 (all predicted anomalies are true anomalies)
- **Recall**: 0.57 (model detects 57% of all anomalies)
- **F1 Score**: 0.73 (harmonic mean of precision and recall)
- **ROC AUC**: 0.88 (strong overall discrimination ability)

These results indicate that the KAN model excels at precision (no false positives) but has room for improvement in recall. The high AUC score demonstrates strong overall performance.

### ECG5000 Dataset Results

On the ECG5000 dataset, the model demonstrates:
- **Accuracy**: 82%
- **Precision**: 72%
- **Recall**: 93%
- **F1 Score**: 81%

The high recall (93%) indicates that the model successfully detects almost all anomalies in the ECG data, making it particularly suitable for medical applications where missing an anomaly could have severe consequences.
