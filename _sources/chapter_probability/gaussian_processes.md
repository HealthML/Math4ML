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
# From Bayesian Linear Regression to Gaussian Process Regression

### ✅ Key insight:

In Bayesian linear regression, we place a **prior on the weights** $\mathbf{w}$, which induces a **distribution over functions** $f(\mathbf{x}) = \mathbf{x}^\top \mathbf{w}$.

Gaussian Process Regression generalizes this idea by placing a **prior directly over functions**:

$$
f(\cdot) \sim \mathcal{GP}(m(\cdot), k(\cdot, \cdot))
$$

---

### 1. Bayesian Linear Regression as a GP

In Bayesian linear regression with prior $\mathbf{w} \sim \mathcal{N}(0, \tau^2 \mathbf{I})$, the function

$$
f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}
$$

has a distribution:

$$
\mathbb{E}[f(\mathbf{x})] = 0, \quad
\text{Cov}(f(\mathbf{x}), f(\mathbf{x}')) = \tau^2 \mathbf{x}^\top \mathbf{x}'
$$

✅ This is a **Gaussian Process** with:

* **Mean function**: $m(\mathbf{x}) = 0$
* **Kernel (covariance) function**: $k(\mathbf{x}, \mathbf{x}') = \tau^2 \mathbf{x}^\top \mathbf{x}'$ → **linear kernel**

So:

> **Bayesian linear regression = Gaussian Process regression with a linear kernel**

---

### 2. Generalizing with Kernels

In Gaussian Process Regression (GPR), we don’t explicitly model $f(\mathbf{x})$ with weights — we just specify a kernel $k(\mathbf{x}, \mathbf{x}')$, which encodes smoothness, periodicity, etc.

We then place a **prior**:

$$
f(\cdot) \sim \mathcal{GP}(0, k(\cdot, \cdot))
$$

And for observations $\mathbf{y}$ with noise variance $\sigma^2$, we get:

$$
\mathbf{y} \sim \mathcal{N}(0, \mathbf{K}_{XX} + \sigma^2 \mathbf{I})
$$

where $[\mathbf{K}_{XX}]_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$

---

### 3. Predictive Distribution

For a test input $\mathbf{x}_*$, the predictive distribution is:

$$
p(y_* \mid \mathbf{x}_*, \mathbf{X}, \mathbf{y}) =
\mathcal{N}(\mu_*, \sigma_*^2)
$$

Where:

$$
\mu_* = \mathbf{k}_*^\top (\mathbf{K} + \sigma^2 \mathbf{I})^{-1} \mathbf{y}
\quad , \quad
\sigma_*^2 = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^\top (\mathbf{K} + \sigma^2 \mathbf{I})^{-1} \mathbf{k}_*
$$

with $\mathbf{k}_* = [k(\mathbf{x}_1, \mathbf{x}_*), \dots, k(\mathbf{x}_n, \mathbf{x}_*)]^\top$

---

Here you see the **predictive posterior** plots for two Gaussian Process regressions:

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.spatial.distance import cdist

# === Linear kernel (Bayesian linear regression as GP)
def linear_kernel(x1, x2, tau2=1.0):
    return tau2 * (x1 @ x2.T)

# === RBF kernel (Squared Exponential)
def rbf_kernel(x1, x2, lengthscale=1.0, variance=1.0):
    dists = cdist(x1, x2, 'sqeuclidean')
    return variance * np.exp(-0.5 * dists / lengthscale**2)

# === Gaussian Process Regression
def gp_predict(X_train, y_train, X_test, kernel_func, noise=1e-1, **kernel_params):
    K = kernel_func(X_train, X_train, **kernel_params)
    K_s = kernel_func(X_train, X_test, **kernel_params)
    K_ss = kernel_func(X_test, X_test, **kernel_params)

    K += noise**2 * np.eye(len(X_train))

    K_inv = inv(K)
    mu_post = K_s.T @ K_inv @ y_train
    cov_post = K_ss - K_s.T @ K_inv @ K_s

    return mu_post, cov_post

# === Create data: 1D input, noisy sine wave
np.random.seed(42)
n_train = 100
X_train = np.sort(np.random.uniform(-5, 5, size=n_train)).reshape(-1, 1)
y_train = np.sin(X_train).ravel() + np.random.normal(0, 0.3, size=n_train)

X_test = np.linspace(-6, 6, 300).reshape(-1, 1)

# === GP with Linear Kernel (Bayesian LinReg)
mu_lin, cov_lin = gp_predict(X_train, y_train, X_test, linear_kernel, tau2=1.0, noise=0.3)

# === GP with RBF Kernel
mu_rbf, cov_rbf = gp_predict(X_train, y_train, X_test, rbf_kernel, lengthscale=1.0, variance=1.0, noise=0.3)

# === Plotting
x_plot = X_test.ravel()
std_lin = np.sqrt(np.diag(cov_lin))
std_rbf = np.sqrt(np.diag(cov_rbf))

plt.figure(figsize=(14, 6))

# Linear kernel GP (Bayesian LinReg)
plt.subplot(1, 2, 1)
plt.title("Bayesian Linear Regression as GP (Linear Kernel)")
plt.fill_between(x_plot, mu_lin - 2*std_lin, mu_lin + 2*std_lin, color='lightblue', alpha=0.5, label='Uncertainty (±2σ)')
plt.plot(x_plot, mu_lin, 'b', lw=2, label='Posterior Mean')
plt.scatter(X_train, y_train, c='k', s=10, alpha=0.6, label='Training Data')
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.grid(True)

# RBF kernel GP
plt.subplot(1, 2, 2)
plt.title("Gaussian Process Regression (RBF Kernel)")
plt.fill_between(x_plot, mu_rbf - 2*std_rbf, mu_rbf + 2*std_rbf, color='plum', alpha=0.5, label='Uncertainty (±2σ)')
plt.plot(x_plot, mu_rbf, 'm', lw=2, label='Posterior Mean')
plt.scatter(X_train, y_train, c='k', s=10, alpha=0.6, label='Training Data')
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```


### 🧠 Left: **Bayesian Linear Regression as a GP**

* Uses a **linear kernel**: $k(x, x') = \tau^2 x x'$
* The model can only capture **linear trends** in the data
* The uncertainty grows outside the data range, but the mean function remains a linear fit

---

### 🌈 Right: **Gaussian Process with RBF Kernel**

* Uses the **squared exponential (RBF) kernel**: smooth, flexible, nonparametric
* Captures the sinusoidal structure of the data
* Uncertainty increases far from training points, but fit remains expressive and smooth

---

### 🔍 Comparison: Bayesian Linear Regression vs. Gaussian Process Regression

| Feature                  | **Bayesian Linear Regression (GP w/ Linear Kernel)** | **Gaussian Process Regression (RBF Kernel)**                         |
| ------------------------ | ---------------------------------------------------- | -------------------------------------------------------------------- |
| **Model Class**          | Parametric (finite weights)                          | Nonparametric (function space)                                       |
| **Kernel**               | $k(x, x') = \tau^2 x^\top x'$                        | $k(x, x') = \sigma^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$ |
| **Prior Over Functions** | Linear functions only                                | Infinite-dimensional smooth functions                                |
| **Flexibility**          | Limited to linear or affine trends                   | Very flexible, adapts to complex structure                           |
| **Posterior Mean**       | Linear function of inputs                            | Nonlinear function of inputs                                         |
| **Posterior Variance**   | Grows linearly outside data range                    | Grows smoothly and saturates far from data                           |
| **Computational Cost**   | $\mathcal{O}(d^3)$ (matrix inversion over weights)   | $\mathcal{O}(n^3)$ (matrix inversion over training points)           |
| **Interpretability**     | Coefficients (weights) interpretable                 | Function-level interpretation, less direct                           |


Absolutely — this is one of the most intuitive and powerful ways to motivate **Gaussian Processes (GPs)**:

> As the **limit of Bayesian linear regression in function space**.

Let’s walk through the motivation step by step, building on your idea.

---

## 🧱 Step-by-Step Motivation: From Bayesian Linear Regression to Gaussian Processes

### 1. **Bayesian Linear Regression: Prior Over Functions**

We begin with the Bayesian linear regression model:

$$
f(x) = \mathbf{w}^\top \boldsymbol{\phi}(x)
\quad \text{with} \quad \mathbf{w} \sim \mathcal{N}(0, \tau^2 \mathbf{I})
$$

* Here $\boldsymbol{\phi}(x) \in \mathbb{R}^d$ is a fixed vector of **basis functions**, e.g., polynomials or radial functions.
* The prior on $\mathbf{w}$ induces a **prior on functions**:

  $$
  f(x) \sim \mathcal{GP}(0, k(x, x')) \quad \text{with} \quad k(x, x') = \tau^2 \boldsymbol{\phi}(x)^\top \boldsymbol{\phi}(x')
  $$

This is already a **Gaussian Process**! But the function class is limited by the choice and number of basis functions.

---

### 2. **Evaluate Functions at Finitely Many Inputs**

Now choose $x_1, x_2, \dots, x_n$ along some interval, say $[-1, 1]$, and evaluate:

$$
f(x_1), \dots, f(x_n)
$$

Since $f(x) = \mathbf{w}^\top \boldsymbol{\phi}(x)$, the joint distribution over these values is multivariate Gaussian:

$$
\mathbf{f} = [f(x_1), \dots, f(x_n)]^\top \sim \mathcal{N}(0, \mathbf{K})
\quad \text{with} \quad K_{ij} = k(x_i, x_j)
$$

So: we have a **finite-dimensional GP prior** over functions defined only at those $n$ points.

---

### 3. **Increase the Density of Evaluation Points**

Now imagine increasing $n$ and making the $x_i$ more densely spaced in the input domain.

As $n \to \infty$, the evaluation points become dense in the domain (e.g., $[-1, 1]$), and the prior becomes a distribution over the **entire function**.

➡️ **In the limit**, we obtain a **stochastic process**:
A collection of random variables $\{f(x) : x \in \mathbb{R} \}$, any finite subset of which has a joint Gaussian distribution.

This is precisely the definition of a **Gaussian Process**:

> A **GP** is a distribution over functions such that
>
> $$
> (f(x_1), \dots, f(x_n)) \sim \mathcal{N}(\mathbf{0}, \mathbf{K}) \quad \text{for any } x_1, \dots, x_n
> $$

---

### 4. **Generalizing the Kernel**

In Bayesian linear regression, we get a **linear kernel**:

$$
k(x, x') = \tau^2 x x'
$$

By changing the basis to an **infinite-dimensional** one (e.g. RBF features), we can define much more flexible kernels like:

$$
k_{\text{RBF}}(x, x') = \sigma^2 \exp\left(-\frac{(x - x')^2}{2\ell^2}\right)
$$

Now the function space includes **smooth nonlinear functions** — this gives us the full expressive power of **nonparametric Bayesian regression**.

---

### 🎓 Summary:

| Concept                 | Bayesian Linear Regression                                   | Gaussian Process                   |
| ----------------------- | ------------------------------------------------------------ | ---------------------------------- |
| Function representation | Finite basis: $f(x) = \mathbf{w}^\top \boldsymbol{\phi}(x)$  | Infinite function space            |
| Prior                   | $\mathbf{w} \sim \mathcal{N}(0, \tau^2 I)$                   | $f \sim \mathcal{GP}(0, k(x, x'))$ |
| Kernel                  | $k(x, x') = \boldsymbol{\phi}(x)^\top \boldsymbol{\phi}(x')$ | Any PSD function                   |
| Limit                   | Fixed feature map                                            | Infinite (e.g., RBF, periodic)     |

