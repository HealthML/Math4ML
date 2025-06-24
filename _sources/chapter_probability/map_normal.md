---

## MAP Estimation for the Multivariate Gaussian

We now move from **MLE** to **MAP estimation** for the parameters of a multivariate Gaussian. We'll use a **Normal-Wishart prior**, the conjugate prior for a multivariate Gaussian with both unknown **mean** and **covariance**.

### Assumptions

Let $\mathbf{x}_1, \dots, \mathbf{x}_n \in \mathbb{R}^d$ be i.i.d. observations:

$$
\mathbf{x}_i \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})
$$

We place a **Normal-Wishart prior** over $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:

* $\boldsymbol{\Sigma} \sim \mathcal{W}^{-1}(\Psi, \nu)$ (inverse-Wishart)
* $\boldsymbol{\mu} \mid \boldsymbol{\Sigma} \sim \mathcal{N}(\boldsymbol{\mu}_0, \frac{1}{\kappa_0} \boldsymbol{\Sigma})$

This gives a joint prior:

$$
p(\boldsymbol{\mu}, \boldsymbol{\Sigma}) = p(\boldsymbol{\Sigma}) \cdot p(\boldsymbol{\mu} \mid \boldsymbol{\Sigma})
$$

---

### Posterior and MAP Estimates

The posterior is also Normal-Wishart, with updated parameters:

* $\kappa_n = \kappa_0 + n$
* $\nu_n = \nu + n$
* $\boldsymbol{\mu}_n = \frac{\kappa_0 \boldsymbol{\mu}_0 + n \bar{\mathbf{x}}}{\kappa_0 + n}$
* $\Psi_n = \Psi + S + \frac{\kappa_0 n}{\kappa_0 + n} (\bar{\mathbf{x}} - \boldsymbol{\mu}_0)(\bar{\mathbf{x}} - \boldsymbol{\mu}_0)^\top$

where:

* $\bar{\mathbf{x}} = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i$
* $S = \sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^\top$

The **MAP estimates** are the **mode** of the Normal-Wishart posterior:

$$
\hat{\boldsymbol{\mu}}_{\text{MAP}} = \boldsymbol{\mu}_n
\quad \text{and} \quad
\hat{\boldsymbol{\Sigma}}_{\text{MAP}} = \frac{\Psi_n}{\nu_n + d + 1}
$$

> Note: This is the mode of the inverse-Wishart, not the mean (which would use $\nu_n - d - 1$ in the denominator).

---

## ✅ Python Demo: MAP with Normal-Wishart Prior

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

def map_multivariate_normal_demo(n=20, d=2, seed=42):
    np.random.seed(seed)
    
    # === True parameters ===
    mu_true = np.array([1.0, -1.0])
    Sigma_true = np.array([[1.0, 0.8],
                           [0.8, 1.5]])
    
    # === Generate data ===
    X = np.random.multivariate_normal(mu_true, Sigma_true, size=n)
    x_bar = np.mean(X, axis=0)
    S = np.dot((X - x_bar).T, (X - x_bar))  # scatter matrix
    
    # === Prior hyperparameters (Normal-Wishart) ===
    mu0 = np.zeros(d)
    kappa0 = 1.0
    nu = d + 2         # nu > d - 1 required for inverse-Wishart to be proper
    Psi = np.eye(d)    # scale matrix (positive definite)

    # === Posterior parameters ===
    kappa_n = kappa0 + n
    nu_n = nu + n
    mu_n = (kappa0 * mu0 + n * x_bar) / kappa_n
    diff = x_bar - mu0
    Psi_n = Psi + S + (kappa0 * n / kappa_n) * np.outer(diff, diff)

    # === MAP estimates ===
    mu_map = mu_n
    Sigma_map = Psi_n / (nu_n + d + 1)  # mode of inverse-Wishart

    # === MLE for comparison ===
    mu_mle = x_bar
    Sigma_mle = S / n

    # === Output ===
    print("True μ:        ", mu_true)
    print("MLE μ̂:         ", mu_mle)
    print("MAP μ̂:         ", mu_map)
    print("\nTrue Σ:\n", Sigma_true)
    print("MLE Σ̂:\n", Sigma_mle)
    print("MAP Σ̂:\n", Sigma_map)

    # === Visualization ===
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.4, label='Data')
    plt.scatter(*mu_true, c='green', marker='x', s=100, label='True μ')
    plt.scatter(*mu_mle, c='blue', marker='o', s=100, label='MLE μ̂')
    plt.scatter(*mu_map, c='purple', marker='D', s=100, label='MAP μ̂')
    plt.title("MAP vs MLE Estimation (Multivariate Normal)")
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

# Run the demo
map_multivariate_normal_demo()
```

---

### 📌 Interpretation

| Estimate                                    | Formula                                                              | Behavior          |
| ------------------------------------------- | -------------------------------------------------------------------- | ----------------- |
| $\hat{\boldsymbol{\mu}}_{\text{MLE}}$    | $\bar{\mathbf{x}}$                                                 | Data only         |
| $\hat{\boldsymbol{\mu}}_{\text{MAP}}$    | Weighted average of $\boldsymbol{\mu}_0$ and $\bar{\mathbf{x}}$ | Prior + data      |
| $\hat{\boldsymbol{\Sigma}}_{\text{MLE}}$ | $S/n$                                                              | May underestimate |
| $\hat{\boldsymbol{\Sigma}}_{\text{MAP}}$ | Shrinks toward $\Psi / (\nu + d + 1)$                              | Regularized       |

---

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def plot_gaussian_contour(ax, mean, cov, label, color, linestyle='-'):
    rv = multivariate_normal(mean, cov)
    x, y = np.mgrid[-3:4:.01, -4:3:.01]
    pos = np.dstack((x, y))
    z = rv.pdf(pos)
    ax.contour(x, y, z, levels=3, colors=color, linestyles=linestyle, linewidths=1.5, alpha=0.7, label=label)

def map_vs_mle_density_plot(n=20, seed=42):
    np.random.seed(seed)
    
    # True parameters
    mu_true = np.array([1.0, -1.0])
    Sigma_true = np.array([[1.0, 0.8], [0.8, 1.5]])
    
    # Generate data
    X = np.random.multivariate_normal(mu_true, Sigma_true, size=n)
    x_bar = np.mean(X, axis=0)
    S = np.dot((X - x_bar).T, (X - x_bar))

    # Prior parameters
    d = 2
    mu0 = np.zeros(d)
    kappa0 = 1.0
    nu = d + 2
    Psi = np.eye(d)

    # Posterior parameters
    kappa_n = kappa0 + n
    nu_n = nu + n
    mu_n = (kappa0 * mu0 + n * x_bar) / kappa_n
    diff = x_bar - mu0
    Psi_n = Psi + S + (kappa0 * n / kappa_n) * np.outer(diff, diff)

    # MAP
    mu_map = mu_n
    Sigma_map = Psi_n / (nu_n + d + 1)

    # MLE
    mu_mle = x_bar
    Sigma_mle = S / n

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[:, 0], X[:, 1], alpha=0.3, label='Samples')
    ax.scatter(*mu_true, marker='x', color='green', s=100, label='True μ')
    ax.scatter(*mu_mle, marker='o', color='blue', s=100, label='MLE μ̂')
    ax.scatter(*mu_map, marker='D', color='purple', s=100, label='MAP μ̂')

    # Contours
    plot_gaussian_contour(ax, mu_true, Sigma_true, "True", color='green', linestyle='--')
    plot_gaussian_contour(ax, mu_mle, Sigma_mle, "MLE", color='blue', linestyle='-')
    plot_gaussian_contour(ax, mu_map, Sigma_map, "MAP", color='purple', linestyle='-.')

    ax.set_title("2D Density Contours: True vs MLE vs MAP")
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True)
    ax.legend()
    ax.axis('equal')
    plt.tight_layout()
    plt.show()

# Run the visual comparison
map_vs_mle_density_plot()
```

---

### 📌 What You'll See

* The **green dashed ellipse** represents the **true distribution**.
* The **blue solid ellipse** is the **MLE**, which might overfit in small samples.
* The **purple dotted ellipse** is the **MAP**, often smoother and closer to the prior.
* All ellipses are **contours of constant density**, showing shape and orientation.



---

## Limitations of MLE in the Small Sample Case

The maximum likelihood estimator (MLE) works well with large amounts of data, but in the **small sample regime**, it has important limitations:

* **Variance underestimation**: The MLE for variance,

$$
\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2,
$$

is **biased**. In fact, the unbiased estimator uses a denominator of $n - 1$, not $n$.

* **Extreme case $n = 1$**: With only one sample, the MLE of the variance is **not even defined**, since we have

$$
\hat{\sigma}^2_{\text{MLE}} = \frac{1}{1} (x_1 - x_1)^2 = 0,
$$

which clearly **understates our uncertainty**.

These issues motivate a **Bayesian perspective**, where we regularize estimation with a prior.
