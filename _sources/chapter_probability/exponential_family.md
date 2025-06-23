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
# The Exponential Family and Conjugate Priors

Many common distributions (Gaussian, Bernoulli, Poisson, etc.) belong to the **exponential family**, which has a convenient structure for Bayesian analysis. A probability distribution belongs to the exponential family if it can be written in the form:

$$
p(x \mid \theta) = h(x) \exp\left( \eta(\theta)^\top T(x) - A(\theta) \right)
$$

Where:

* $\theta$: natural (canonical) parameters
* $\eta(\theta)$: natural parameter function
* $T(x)$: sufficient statistics
* $A(\theta)$: log-partition function (normalizer)
* $h(x)$: base measure

---

### Why the Exponential Family Matters for MAP

If we choose a **prior that is conjugate** to the exponential family likelihood, the posterior has the **same functional form** as the prior — this makes both **analysis and computation much easier**.

In particular:

* The posterior is often interpretable as a **prior+data update**.
* It allows for **analytical MAP estimation**.
* The prior can be viewed as **pseudo-observations**, guiding estimation when data is scarce.

---

## Gaussian with Unknown Mean and Variance

Let’s now consider:

$$
x_1, \dots, x_n \sim \mathcal{N}(\mu, \sigma^2)
$$

We now want to estimate both parameters $\mu$ and $\sigma^2$ jointly.

---

### Conjugate Prior: Normal-Inverse-Gamma

The conjugate prior for $(\mu, \sigma^2)$ is the **Normal-Inverse-Gamma distribution**:

$$
\mu \mid \sigma^2 \sim \mathcal{N}(\mu_0, \sigma^2 / \kappa_0), \quad \sigma^2 \sim \text{Inv-Gamma}(\alpha_0, \beta_0)
$$

#### Prior density:

$$
p(\mu, \sigma^2) \propto (\sigma^2)^{-(\alpha_0 + 1)} \exp\left(-\frac{\beta_0}{\sigma^2}\right) \cdot \exp\left( -\frac{\kappa_0}{2\sigma^2} (\mu - \mu_0)^2 \right)
$$

---

### Posterior (up to constant):

Given the likelihood and prior, the posterior is:

$$
\log p(\mu, \sigma^2 \mid x_{1:n}) = \log p(x_{1:n} \mid \mu, \sigma^2) + \log p(\mu, \sigma^2)
$$

$$
\propto -\frac{n}{2} \log \sigma^2 - \frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2
          -\frac{\kappa_0}{2\sigma^2} (\mu - \mu_0)^2
          -(\alpha_0 + 1)\log \sigma^2 - \frac{\beta_0}{\sigma^2}
$$

---

### MAP Estimation

Take partial derivatives of the log-posterior w\.r.t. $\mu$ and $\sigma^2$ and set them to zero.

#### Step 1: MAP for $\mu$ (given $\sigma^2$)

$$
\hat{\mu}_{\text{MAP}} = \frac{\kappa_0 \mu_0 + n \bar{x}}{\kappa_0 + n}
$$

#### Step 2: Plug into posterior, solve for $\sigma^2$

$$
\hat{\sigma}^2_{\text{MAP}} = \frac{1}{\alpha_0 + n + 1}
\left(
\beta_0 + \frac{1}{2} \sum_{i=1}^n (x_i - \bar{x})^2 + \frac{\kappa_0 n}{2(\kappa_0 + n)}(\bar{x} - \mu_0)^2
\right)
$$

This yields the **MAP estimates for both parameters**.

---

## Summary: MAP for Gaussian with Unknown Mean and Variance

Given data $x_1, \dots, x_n \sim \mathcal{N}(\mu, \sigma^2)$, and prior parameters $\mu_0, \kappa_0, \alpha_0, \beta_0$, the **MAP estimates** are:

$$
\hat{\mu}_{\text{MAP}} = \frac{\kappa_0 \mu_0 + n \bar{x}}{\kappa_0 + n}
$$

$$
\hat{\sigma}^2_{\text{MAP}} = \frac{1}{\alpha_0 + n + 1} \left(
\beta_0 + \frac{1}{2} \sum_{i=1}^n (x_i - \bar{x})^2 +
\frac{\kappa_0 n}{2(\kappa_0 + n)}(\bar{x} - \mu_0)^2
\right)
$$


Perfect! Below is a **Python demo and visualization** of **MAP estimation for the Gaussian** with **unknown mean and variance**, using the **Normal-Inverse-Gamma prior**.

---

### ✅ What this code does:

* Simulates $n$ samples from $\mathcal{N}(\mu_{\text{true}}, \sigma^2_{\text{true}})$
* Computes:

  * **MLE** estimates for $\mu$ and $\sigma^2$
  * **MAP** estimates using Normal-Inverse-Gamma prior
* Plots:

  * Histogram of data
  * Fitted MLE and MAP densities
  * True density

---

### ✅ Python Code

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma

def gaussian_map_demo(n=3, mu_true=0.0, sigma_true=1.0,
                      mu0=5.0, kappa0=1.0, alpha0=2.0, beta0=2.0, seed=42):
    np.random.seed(seed)
    x = np.random.normal(mu_true, sigma_true, size=n)
    x_bar = np.mean(x)
    s_squared = np.mean((x - x_bar)**2)

    # === MLE estimates ===
    mu_mle = x_bar
    sigma2_mle = s_squared

    # === MAP estimates ===
    kappa_n = kappa0 + n
    alpha_n = alpha0 + n / 2
    mu_map = (kappa0 * mu0 + n * x_bar) / kappa_n
    beta_n = (
        beta0
        + 0.5 * np.sum((x - x_bar)**2)
        + (kappa0 * n / (2 * kappa_n)) * (x_bar - mu0) ** 2
    )
    sigma2_map = beta_n / (alpha_n + 1)

    # === Print Results ===
    print(f"n = {n} observations")
    print(f"True μ = {mu_true}, σ² = {sigma_true**2:.2f}")
    print(f"MLE: μ̂ = {mu_mle:.3f}, σ̂² = {sigma2_mle:.3f}")
    print(f"MAP: μ̂ = {mu_map:.3f}, σ̂² = {sigma2_map:.3f}")

    # === Plot ===
    x_vals = np.linspace(min(x) - 3, max(x) + 3, 300)
    p_true = norm.pdf(x_vals, mu_true, sigma_true)
    p_mle = norm.pdf(x_vals, mu_mle, np.sqrt(sigma2_mle))
    p_map = norm.pdf(x_vals, mu_map, np.sqrt(sigma2_map))

    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=10, density=True, alpha=0.3, color='gray', label='Data histogram')
    plt.plot(x_vals, p_true, 'g--', lw=2, label='True PDF')
    plt.plot(x_vals, p_mle, 'b-', lw=2, label='MLE fit')
    plt.plot(x_vals, p_map, 'm-', lw=2, label='MAP fit')
    plt.axvline(mu_map, color='purple', linestyle='--')
    plt.axvline(mu_mle, color='blue', linestyle='--')
    plt.axvline(mu0, color='green', linestyle=':', lw=1)
    plt.title(f'Gaussian MAP vs MLE (n = {n})')
    plt.xlabel('$x$')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage with very small n to highlight prior influence
gaussian_map_demo(n=3)
```

---

### 📌 Interpretation:

* When $n$ is small, MAP pulls the estimate toward the prior mean $\mu_0$, and inflates the variance.
* As $n \to \infty$, MAP and MLE converge.
* You can adjust `mu0`, `kappa0`, `alpha0`, and `beta0` to see different prior strengths.
