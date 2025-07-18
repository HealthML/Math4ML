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

# Variational Autoencoder (VAE)

## 🌟 Motivation

Variational Autoencoders (VAEs) are neural networks designed to learn how to generate new handwritten digits by modeling the underlying data distribution. Unlike traditional autoencoders that only compress and reconstruct inputs, VAEs incorporate probabilistic principles to enable meaningful data generation.

**Variational Autoencoders (VAEs)** introduce a probabilistic twist:

* They learn a **distribution over the latent space**—not just fixed encodings.
* They allow for **sampling and generation** of new data points.
* They fuse deep learning with **Bayesian reasoning** through **variational inference**.
* They're optimized via the **Evidence Lower Bound (ELBO)**, balancing reconstruction quality and latent regularization.

---

## 📌 What You Will Do in This Project

Your primary goal is:

> **Implement a Variational Autoencoder (VAE) using PyTorch and apply it to the MNIST dataset.**

Specifically, you'll:

* Train an encoder to output mean and log-variance of latent Gaussian.
* Sample latent vectors via the reparameterization trick.
* Decode latent vectors to reconstruct images.
* Derive and minimize the negative **ELBO** loss.
* Visualize latent space, reconstructions, and generated samples.

---

## 🔍 Key Concepts You'll Master

You'll gain intuition and implementation skills in:

* **Latent variable models**:  
  Learn how data can be represented via probabilistic latent spaces.

* **KL Divergence**:  
  Understand how KL measures divergence from a prior distribution, and why regularization is essential.

* **Reparameterization Trick**:  
  Make stochastic sampling differentiable for backpropagation.

* **Evidence Lower Bound (ELBO)**:  
  The central loss function that balances reconstruction loss and regularization.

* **Generative Modeling**:  
  Use trained VAEs to **sample new digits**, perform **interpolations**, and understand how deep models learn distributions.

---

## 🚧 Core Tasks (Implementation Details)

Your VAE implementation should follow these key steps:

### 1. **Data Preparation**

Load and preprocess the MNIST dataset using torchvision.

### 2. **Model Architecture**

* Encoder outputs mean ($\mu$) and log-variance ($\log \sigma^2$).
* Decoder reconstructs input from latent vector $z$.
* Use `nn.Sequential` or custom `nn.Module` classes.

### 3. **Sampling via Reparameterization**

```python
z = mu + std * torch.randn_like(std)  # std = torch.exp(0.5 * log_var)
```

- Decoder reconstructs input from sampled `z`

---

## 3. Loss Function (Negative ELBO)

We combine:

- **Reconstruction Loss**: Binary Cross Entropy between input and reconstruction
- **KL Divergence**: Distance between approximate posterior `q(z|x)` and prior `p(z) = N(0, I)`




## 4. Evaluation and Visualization

- Compare input images to their reconstructions
- Generate new digits from random latent vectors
- Interpolate between two latent codes

## 📝 Reporting: Analysis and Insights

Include in your report (~2 pages):

- Derivation of the ELBO from the variational objective
- Explanation of the KL and reconstruction loss trade-off
- Visualizations:
- Input vs Reconstruction
- Latent Space Traversals
- Samples from Prior
- Observations about learned latent representations


## ✅ Summary Table

| Component         | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| **Model**         | VAE with encoder + decoder using PyTorch                                    |
| **Loss**          | ELBO = Reconstruction Loss + KL Divergence                                  |
| **Dataset**       | MNIST (grayscale 28×28 digits)                                              |
| **Latent Dim**    | Typically 2–20 (use 2D for visualization)                                   |
| **Output**        | Reconstructions, random generations, latent visualizations