---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Supervised Feature Selection Using Mutual Information: A Comparison with PCA

## 🌟 Motivation

Feature selection is a crucial step in machine learning that can improve model performance and interpretability. Principal Component Analysis (PCA) and Mutual Information (MI) are two powerful but fundamentally different methods for selecting informative features.

This project explores:

- How PCA, a linear unsupervised method, reduces dimensionality based on variance.
- How Mutual Information, a nonlinear supervised metric, selects features based on dependency with the target.
- Their comparative effectiveness in classification tasks.

---

## 📌 What You Will Do in This Project

> **Implement PCA and Mutual Information from scratch for supervised feature selection on the UCI Breast Cancer dataset, then compare their impact on classification accuracy.**

You will:

- Estimate probabilities and calculate Mutual Information manually.
- Select top features using both methods.
- Train a classifier on the reduced feature sets.
- Evaluate and analyze classification accuracy.

---

## 🔍 Key Concepts You'll Master

- Probability estimation and Mutual Information calculation
- Supervised vs unsupervised feature selection
- Impact of feature selection on classification

---

## 🚧 Core Tasks (Implementation Details)

- Load the UCI Breast Cancer dataset from sklearn or UCI repository.
- Compute covariance matrix from training data.
- Perform eigen decomposition to find principal components.
- Estimate joint and marginal probabilities to calculate Mutual Information between each feature and the class label.
- Select top-k features according to PCA loadings and MI scores.
- Train a classifier (e.g., logistic regression or SVM) on reduced datasets.
- Compare and interpret classification results.

---

## 📝 Reporting: Analysis and Insights

Your report (~2 pages) should cover:

- Mathematical derivation of PCA and Mutual Information.
- Visual comparison of selected features.
- Classification accuracy and robustness analysis.
- Discussion on strengths and limitations of each method.

---

## ✅ Summary Table

| Component          | Description                                                         |
|--------------------|---------------------------------------------------------------------|
| **Dataset**        | UCI Breast Cancer Wisconsin dataset                                 |
| **Goal**           | Implement PCA and MI for feature selection and compare their effect |
| **Key Concepts**   | Covariance, eigen decomposition, probability estimation, MI         |
| **Core Tasks**     | Calculate PCA components, estimate MI, select features, classify    |
| **Evaluation**     | Classification accuracy comparison on reduced features              |
| **Tools**          | NumPy, scikit-learn (for classifier), matplotlib                     |

---
