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
# Normalizing Flows with Coupling Layers

```{code-cell} ipython3
:tags: [remove-cell]

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from torch.distributions import MultivariateNormal

# Install and import FrEIA (if available)
try:
    from FrEIA.framework import SequenceINN
    from FrEIA.modules import AffineCoupling, PermuteRandom
    FREIA_AVAILABLE = True
except ImportError:
    FREIA_AVAILABLE = False

# Generate two-moons data
X, _ = make_moons(n_samples=500, noise=0.05)
X = torch.tensor(X, dtype=torch.float32)

if FREIA_AVAILABLE:
    # Define simple RealNVP with 4 coupling layers using FrEIA
    inn = SequenceINN(2)
    for i in range(4):
        inn.append(AffineCoupling, subnet_constructor=lambda c_in, c_out: nn.Sequential(
            nn.Linear(c_in, 128), nn.ReLU(), nn.Linear(128, c_out)))
        inn.append(PermuteRandom, seed=i)

    # Forward pass: map data to latent space
    z, log_jac_det = inn(X)
    # Sample from standard Gaussian and invert
    z_sample = torch.randn_like(z)
    x_sample = inn.inverse(z_sample)
    x_sample = x_sample.detach().numpy()
    z = z.detach().numpy()
else:
    z = np.zeros_like(X)
    x_sample = np.zeros_like(X)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

axs[0].set_title("Original Two-Moons Data")
axs[0].scatter(X[:, 0], X[:, 1], s=10, alpha=0.6)
axs[0].set_xlim(-2, 3)
axs[0].set_ylim(-1.5, 2)

axs[1].set_title("Latent Space (After Flow)")
axs[1].scatter(z[:, 0], z[:, 1], s=10, alpha=0.6)
axs[1].set_xlim(-3, 3)
axs[1].set_ylim(-3, 3)

axs[2].set_title("Generated Samples (Inverse Flow)")
axs[2].scatter(x_sample[:, 0], x_sample[:, 1], s=10, alpha=0.6)
axs[2].set_xlim(-2, 3)
axs[2].set_ylim(-1.5, 2)

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.suptitle("Normalizing Flows: Two-Moons ↔ Latent Gaussian", fontsize=14, y=1.05)
plt.show()

```


## 🌟 Motivation

Imagine you have data from a complicated distribution—such as images, speech signals, or intricate 2-D patterns like spirals, checkerboards, or moons. How could we generate new, realistic samples from this data? Or evaluate how "likely" a new sample is under this complex distribution?

One elegant solution is to use **Normalizing Flows**, a powerful class of generative models that:

* **Transform complex data** into simple, well-known distributions (like a Gaussian).
* Let you easily sample new data by transforming simple Gaussian samples back into the complex space.
* Allow precise computation of the likelihood of any given data point.

This makes Normalizing Flows extremely useful for density estimation, generative modeling, and even anomaly detection.



---

## 📌 What You Will Do in This Project

In this project, you'll implement and experiment with **Real Non-Volume-Preserving (RealNVP)** flows, specifically using **Affine Coupling Layers**.

Your main goal is:

> **Implement a density estimator** using Normalizing Flows to transform a complex 2-D distribution into a simple Gaussian distribution, and then back.

You'll:

* Build **affine coupling layers** from scratch in PyTorch.
* Chain these layers together (at least four) using alternating masking patterns.
* Train your model on well-known synthetic datasets (such as the two-moons or checkerboard data).
* Visualize how your model learns to map complex data to Gaussian space and vice versa.

---

## 🔍 Key Concepts You'll Master

You'll dive into several fascinating and practically relevant concepts:

* **Change of Variables**:
  How can we map a complex distribution to a simpler one in a differentiable way, while keeping track of probabilities?

* **Log-Determinant of the Jacobian**:
  Every transformation in a flow changes volumes in the data space. You'll derive and compute how much "volume" changes under affine coupling layers.

* **Invertible Neural Networks**:
  Coupling layers are carefully designed neural networks that can be easily inverted. You'll understand why invertibility is guaranteed and critical.

* **Monte-Carlo Log-Likelihood Estimation**:
  You'll use samples to estimate and optimize the likelihood directly.

---

## 🚧 Core Tasks (Implementation Details)

Your implementation will involve the following steps:

* Implement **one affine coupling layer** (forward and inverse passes, plus log-det Jacobian) in PyTorch.
* Stack at least **four coupling layers** into a Normalizing Flow, ensuring masks alternate.
* Train on a simple yet rich dataset (**two-moons or checkerboard**).
* Plot and analyze **forward and inverse samples** as well as training curves of the **log-likelihood**.

Your final result will showcase:

* Complex-to-simple and simple-to-complex transformations visually.
* Clear, well-documented Python code (NumPy and PyTorch) that others can easily follow.

---

## 📝 Reporting: Derivations and Insights

Your short (\~2 pages) report should clearly present:

* **A derivation** of the log-determinant Jacobian for affine coupling layers.
* **A conceptual explanation** for why the coupling layer is invertible by construction.
* Visual evidence of your model's capability and learning progress.

---

## 🚀 Stretch Goals (Optional, for Extra Insight)

If you're eager to go deeper, you might:

* Add **ActNorm layers or 1×1 convolutions (Glow)** to enhance flow expressivity.
* Experiment with a direct **NumPy implementation** and compare speed and performance to PyTorch.

---

## 📚 Resources and Support

* You are encouraged to leverage **open-source resources** and **AI tools** to assist you, as long as you clearly document their use.
* Use provided starter notebooks for datasets and visualizations to focus your effort effectively.

---

## ✅ Why This Matters

Beyond the project grade, this work equips you with practical experience in modern generative modeling, a core skill in today's AI landscape. Normalizing Flows appear in research and real-world applications from image generation to anomaly detection, finance, physics, and bioinformatics.

This is your chance to:

* Gain a deep, intuitive understanding of Normalizing Flows.
* Showcase practical coding skills highly sought-after in industry and academia.
* Produce a polished, impressive piece for your portfolio.

## Project Summary: Normalizing Flows with Coupling Layers

*(NumPy + **PyTorch** for autodiff)*

| Item                        | Details                                                                                                                                                                                                                                               |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Goal**                    | Build a density estimator that maps complex 2-D data to a unit Gaussian via affine coupling layers (RealNVP).                                                                                                                                         |
| **Key ideas**               | Change of variables, log-det Jacobian, invertible nets, Monte-Carlo log-likelihood.                                                                                                                                                                   |
| **Core tasks**              | <ul><li>Implement one affine coupling layer from scratch (PyTorch).</li><li>Stack ≥ 4 layers with alternating masks.</li><li>Train on a two-moon or checkerboard dataset.</li><li>Plot forward / inverse samples and log-likelihood curves.</li></ul> |
| **Report focus (≈2 pages)** | Derive the log-det Jacobian for coupling layers; discuss why invertibility is guaranteed.                                                                                                                                                             |
| **Stretch ideas**           | Add an ActNorm layer or glow-style 1×1 convolution; compare NumPy vs. PyTorch speed.                                                                                                                                                                  |

---
