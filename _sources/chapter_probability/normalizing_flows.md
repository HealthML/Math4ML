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
# From Change of Variables to Normalizing Flows

We’ve seen in the previous example that if we start with a random vector $\mathbf{X}$ with known density $p_{\mathbf{X}}(\mathbf{x})$, and apply a smooth, invertible function $f: \mathbb{R}^n \to \mathbb{R}^n$, then the resulting random vector $\mathbf{Y} = f(\mathbf{X})$ has a density given by the **change-of-variables formula**:

$$
p_{\mathbf{Y}}(\mathbf{y}) = p_{\mathbf{X}}(f^{-1}(\mathbf{y})) \cdot \left| \det \left( \frac{\partial f^{-1}}{\partial \mathbf{y}} \right) \right|
$$

or equivalently:

$$
p_{\mathbf{Y}}(f(\mathbf{x})) = p_{\mathbf{X}}(\mathbf{x}) \cdot \left| \det \left( \frac{\partial f}{\partial \mathbf{x}} \right) \right|^{-1}
$$

This formula shows how the probability mass "warps" as we pass it through a transformation.

---

### Key Insight

Suppose we want to **model a complex target distribution** $p_{\text{target}}(\mathbf{y})$ for which:

* Sampling directly is hard
* Evaluating the density is expensive or intractable

But if we can find a **smooth, invertible transformation** $f$ such that:

$$
\mathbf{Y} = f(\mathbf{Z}), \quad \text{where } \mathbf{Z} \sim p_{\mathbf{Z}} \text{ (a simple distribution, e.g. standard normal)}
$$

then we can **induce** a complex distribution on $\mathbf{Y}$, and compute the exact density via:

$$
p_{\mathbf{Y}}(\mathbf{y}) = p_{\mathbf{Z}}(f^{-1}(\mathbf{y})) \cdot \left| \det \left( \frac{\partial f^{-1}}{\partial \mathbf{y}} \right) \right|
$$

This is the foundation of **normalizing flows**.

---

## Normalizing Flows: Definition

A **normalizing flow** is a sequence of invertible, differentiable transformations:

$$
\mathbf{z}_0 \sim p_0(\mathbf{z}_0) \quad \xrightarrow{f_1} \quad \mathbf{z}_1 \quad \xrightarrow{f_2} \quad \cdots \quad \xrightarrow{f_K} \quad \mathbf{y} = \mathbf{z}_K
$$

The overall transformation is:

$$
\mathbf{y} = f_K \circ f_{K-1} \circ \cdots \circ f_1(\mathbf{z}_0)
$$

Then, by repeatedly applying the change-of-variables formula, we can compute the density of $\mathbf{y}$:

$$
\log p_{\mathbf{Y}}(\mathbf{y}) = \log p_0(\mathbf{z}_0) - \sum_{k=1}^K \log \left| \det \left( \frac{\partial f_k}{\partial \mathbf{z}_{k-1}} \right) \right|
$$

where $\mathbf{z}_0 = f_1^{-1} \circ \cdots \circ f_K^{-1}(\mathbf{y})$.

---

## Why Normalizing Flows?

* **Flexible modeling**: Flows can represent complex, multi-modal distributions.
* **Exact density evaluation**: Unlike GANs or VAEs, flows provide **exact likelihoods**.
* **Efficient sampling and inference**: If all $f_k$ are computationally efficient, we can sample and compute densities fast.

---

## Visual Analogy

Think of normalizing flows as **sculpting** a simple blob (like a Gaussian) into a complex shape by bending and stretching space — always carefully keeping track of how volumes change, via the Jacobian determinant.

---

## An invertible affine coupling layer

We define a transformation $f: \mathbb{R}^2 \to \mathbb{R}^2$ using an **affine coupling** mechanism:

### Definition

Let $\mathbf{z} = \begin{bmatrix} z_1 \\ z_2 \end{bmatrix} \in \mathbb{R}^2$. Define:

$$
\begin{aligned}
y_1 &= z_1 \\
y_2 &= z_2 \cdot \exp(s(z_1)) + t(z_1)
\end{aligned}
$$

Here:

* $s(\cdot)$ is a **scale function** (e.g., a neural net or a simple function like $s(z_1) = \sin(z_1)$)
* $t(\cdot)$ is a **translation function**

This transformation is **invertible** as long as $\exp(s(z_1)) \ne 0$. The inverse is:

$$
\begin{aligned}
z_1 &= y_1 \\
z_2 &= \left(y_2 - t(y_1)\right) \cdot \exp(-s(y_1))
\end{aligned}
$$

### Jacobian Determinant

The Jacobian matrix $J_f$ of $f$ is:

$$
J_f = \begin{bmatrix}
1 & 0 \\
\frac{\partial y_2}{\partial z_1} & \exp(s(z_1))
\end{bmatrix}
$$

The determinant is easy to compute (lower triangular matrix):

$$
\det J_f = \exp(s(z_1))
$$

This makes log-determinant evaluation efficient:

$$
\log \left| \det J_f \right| = s(z_1)
$$

---

## Why Use This Example?

This layer is:

* **Invertible** and **efficient**
* Has a **triangular Jacobian**, so the determinant is trivial to compute
* Scales to higher dimensions by permuting or splitting coordinates
* Used in real normalizing flows like **RealNVP** and **Glow**

Let's implement the affine coupling layer in Python.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np

def s_function(z1):
    """Scale function s(z1), simple nonlinearity"""
    return np.sin(z1)

def t_function(z1):
    """Translation function t(z1), simple nonlinearity"""
    return np.cos(z1)

def affine_coupling_forward(z):
    """
    Affine coupling forward transformation.
    Input:
        z: np.array of shape (n_samples, 2)
    Output:
        y: transformed output
        log_det_jacobian: log determinant of the Jacobian
    """
    z1 = z[:, 0]
    z2 = z[:, 1]

    s = s_function(z1)
    t = t_function(z1)

    y1 = z1
    y2 = z2 * np.exp(s) + t

    log_det_jacobian = s  # log|det J| = s(z1)

    y = np.stack([y1, y2], axis=1)
    return y, log_det_jacobian

def affine_coupling_inverse(y):
    """
    Affine coupling inverse transformation.
    Input:
        y: np.array of shape (n_samples, 2)
    Output:
        z: inverse transformed output
    """
    y1 = y[:, 0]
    y2 = y[:, 1]

    s = s_function(y1)
    t = t_function(y1)

    z1 = y1
    z2 = (y2 - t) * np.exp(-s)

    z = np.stack([z1, z2], axis=1)
    return z

# Sample inputs
np.random.seed(0)
z_samples = np.random.randn(100, 2)  # Standard normal samples

# Apply forward and inverse transformation
y_samples, log_det = affine_coupling_forward(z_samples)
z_reconstructed = affine_coupling_inverse(y_samples)

# Check reconstruction accuracy
reconstruction_error = np.max(np.abs(z_samples - z_reconstructed))
reconstruction_error

```

```{code-cell} ipython3
:tags: [hide-input]


import numpy as np
import matplotlib.pyplot as plt

# Define the target distribution: banana-shaped
def target_density(x, y):
    # A banana-shaped 2D distribution
    return np.exp(-0.5 * ((x)**2 + (y - 0.1 * x**2 + 1)**2))

# Define the coupling layer transformation (simple affine coupling)
def coupling_forward(z):
    x, y = z[:, 0], z[:, 1]
    s = 0.5 * x  # scale function
    t = 0.2 * x  # translation function
    y_new = y * np.exp(s) + t
    return np.stack([x, y_new], axis=1)

def coupling_inverse(z):
    x, y_new = z[:, 0], z[:, 1]
    s = 0.5 * x
    t = 0.2 * x
    y = (y_new - t) / np.exp(s)
    return np.stack([x, y], axis=1)

# Sample from a simple base distribution (uniform)
n_samples = 10000
z = np.random.uniform(-2, 2, size=(n_samples, 2))

# Transform samples through the coupling layer
x = coupling_forward(z)

# Plot original (uniform) and transformed (approximate target) samples
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# Base distribution
axs[0].scatter(z[:, 0], z[:, 1], alpha=0.3, s=5)
axs[0].set_title("Base Distribution (Uniform)")
axs[0].set_xlim(-3, 3)
axs[0].set_ylim(-3, 3)

# Transformed samples
axs[1].scatter(x[:, 0], x[:, 1], alpha=0.3, s=5, color='orange')
axs[1].set_title("Transformed Samples (Flow Output)")
axs[1].set_xlim(-3, 3)
axs[1].set_ylim(-3, 3)

# True target density (background)
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
zz = target_density(xx, yy)
axs[2].contourf(xx, yy, zz, levels=50, cmap='viridis')
axs[2].set_title("Target Density")
axs[2].set_xlim(-3, 3)
axs[2].set_ylim(-3, 3)

plt.tight_layout()
plt.show()

```

### Optimizing the parameters of the affine coupling layer

We want to **maximize the likelihood** of the transformed samples under the target distribution. 

For a normalizing flow with parameters \$\theta\$ and a base distribution \$p\_Z\$ (e.g., uniform or standard normal), the **log-likelihood of data \$\mathbf{x}\$** is:

$$
\log p_X(\mathbf{x}; \theta) = \log p_Z(f_\theta^{-1}(\mathbf{x})) + \log \left| \det \left( \frac{\partial f_\theta^{-1}}{\partial \mathbf{x}} \right) \right|
$$

If you generate \$n\$ samples from the base distribution, transform them via \$f\_\theta\$, and evaluate them under a known **target density \$p\_{\text{target}}\$**, your objective becomes:

$$
\max_\theta \ \sum_{i=1}^n \log p_{\text{target}}(f_\theta(z^{(i)})) + \log \left| \det \left( \frac{\partial f_\theta}{\partial z^{(i)}} \right) \right|
$$

---

### 🧮 Newton’s Method for Flow Parameter Optimization

Newton’s method requires:

* **Gradient** \$\nabla\_\theta \mathcal{L}\$
* **Hessian** \$H\_\theta = \nabla\_\theta^2 \mathcal{L}\$

The update step is:

$$
\theta_{\text{new}} = \theta_{\text{old}} - H_\theta^{-1} \nabla_\theta \mathcal{L}
$$

This can be **costly**, but in small 2D problems with a small number of parameters (like affine coupling layers with a few affine weights), it’s feasible.

---

Let’s derive the **gradient** and **Hessian** of the log-likelihood with respect to the parameters of a **simple affine coupling layer**, in the context of **normalizing flows**.

---

## 🔧 Setup: Affine Coupling Layer

Assume a 2D input vector:

$$
\mathbf{z} = \begin{bmatrix} z_1 \\ z_2 \end{bmatrix} \sim \text{Uniform}([0,1]^2)
$$

We define a simple **affine coupling layer**:

$$
f_\theta(\mathbf{z}) = \begin{bmatrix} 
z_1 \\
z_2 \cdot \exp(s(z_1)) + t(z_1)
\end{bmatrix}
$$

We use **parametric functions**:

* \$s(z\_1) = a z\_1 + b\$
* \$t(z\_1) = c z\_1 + d\$

The parameters are: \$\theta = \[a, b, c, d]\$

---

## 🎯 Objective: Log-Density of Transformed Sample

Let \$\mathbf{x} = f\_\theta(\mathbf{z})\$, and let \$p\_{\text{target}}(\mathbf{x})\$ be the known density we want to approximate. Then the **loss** is:

$$
\mathcal{L}(\theta) = \sum_{i=1}^n \log p_{\text{target}}(f_\theta(\mathbf{z}^{(i)})) + \log \left| \det \left( \frac{\partial f_\theta}{\partial \mathbf{z}^{(i)}} \right) \right|
$$

We derive \$\nabla\_\theta \mathcal{L}\$ and \$\nabla^2\_\theta \mathcal{L}\$ for a single sample \$\mathbf{z}\$, then extend to a sum.

---

## 📐 Step 1: The Jacobian

For the affine coupling layer:

$$
\mathbf{x} = f_\theta(\mathbf{z}) = 
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
=
\begin{bmatrix}
z_1 \\
z_2 \cdot \exp(a z_1 + b) + c z_1 + d
\end{bmatrix}
$$

Jacobian:

$$
J = \frac{\partial f_\theta}{\partial \mathbf{z}} =
\begin{bmatrix}
1 & 0 \\
z_2 \cdot a \exp(a z_1 + b) + c & \exp(a z_1 + b)
\end{bmatrix}
$$

Determinant of Jacobian:

$$
\det(J) = \exp(a z_1 + b)
$$

So the log-determinant term is:

$$
\log |\det J| = a z_1 + b
$$

---

## 🧮 Step 2: Log-Likelihood

Let \$x = f\_\theta(z)\$, then

$$
\mathcal{L}(\theta; z) = \log p_{\text{target}}(f_\theta(z)) + a z_1 + b
$$

Let’s define:

$$
x_1 = z_1, \quad x_2 = z_2 \cdot \exp(a z_1 + b) + c z_1 + d
$$

---

## 🔁 Step 3: Gradient w\.r.t. \$\theta = \[a, b, c, d]\$

We apply the chain rule:

$$
\nabla_\theta \mathcal{L} = \frac{\partial \log p_{\text{target}}(x)}{\partial x} \cdot \frac{\partial x}{\partial \theta} + \nabla_\theta \log |\det J|
$$

Break down:

### a) \$\frac{\partial x}{\partial \theta}\$

Let’s compute partials of \$x\_2\$:

* \$\frac{\partial x\_2}{\partial a} = z\_2 \cdot z\_1 \cdot \exp(a z\_1 + b)\$
* \$\frac{\partial x\_2}{\partial b} = z\_2 \cdot \exp(a z\_1 + b)\$
* \$\frac{\partial x\_2}{\partial c} = z\_1\$
* \$\frac{\partial x\_2}{\partial d} = 1\$

So,

$$
\frac{\partial x}{\partial \theta} =
\begin{bmatrix}
0 & 0 & 0 & 0 \\
z_2 z_1 e^{a z_1 + b} & z_2 e^{a z_1 + b} & z_1 & 1
\end{bmatrix}
$$

Denote \$\nabla\_x \log p\_{\text{target}} = \[g\_1, g\_2]^\top\$. Then:

$$
\frac{\partial \log p_{\text{target}}(x)}{\partial \theta} =
g_2 \cdot
\begin{bmatrix}
z_2 z_1 e^{a z_1 + b} \\
z_2 e^{a z_1 + b} \\
z_1 \\
1
\end{bmatrix}
$$

### b) Gradient of log-determinant

$$
\nabla_\theta \log |\det J| =
\begin{bmatrix}
z_1 \\
1 \\
0 \\
0
\end{bmatrix}
$$

### 🔚 Final Gradient Expression:

$$
\nabla_\theta \mathcal{L} =
g_2 \cdot
\begin{bmatrix}
z_2 z_1 e^{a z_1 + b} \\
z_2 e^{a z_1 + b} \\
z_1 \\
1
\end{bmatrix}
+
\begin{bmatrix}
z_1 \\
1 \\
0 \\
0
\end{bmatrix}
$$

---

## 🧠 Step 4: Hessian


Let’s now derive the **full Hessian** of the log-likelihood

$$
\mathcal{L}(\theta) = \log p_{\text{target}}(x(\theta)) + \log|\det J(\theta)| = \log p(x_1, x_2) + (a z_1 + b)
$$

where $x_1 = z_1$,
and

$$
x_2 = z_2 \cdot e^{a z_1 + b} + c z_1 + d
$$

and the parameters are $\theta = [a, b, c, d]^\top \in \mathbb{R}^4$.

---

## Step 1: Notation and Precomputations

Let:

* $s = a z_1 + b$
* $e^s = \exp(s)$
* Let $\nabla_x \log p(x) = \begin{bmatrix} g_1 \\ g_2 \end{bmatrix}$
* Let $H_x = \nabla^2_x \log p(x) = \begin{bmatrix} h_{11} & h_{12} \\ h_{21} & h_{22} \end{bmatrix}$

Since only $x_2$ depends on $\theta$, we can focus on that.

We already computed:

$$
\nabla_\theta x_2 = 
\begin{bmatrix}
z_2 z_1 e^s \\
z_2 e^s \\
z_1 \\
1
\end{bmatrix}
$$

And

$$
\nabla_\theta \log |\det J| = 
\begin{bmatrix}
z_1 \\
1 \\
0 \\
0
\end{bmatrix}
$$

So the full gradient is:

$$
\nabla_\theta \mathcal{L} = g_2 \cdot \nabla_\theta x_2 + \nabla_\theta \log |\det J|
$$

---

## Step 2: Hessian Breakdown

The Hessian is:

$$
H_\theta = \nabla^2_\theta \mathcal{L} 
= g_2 \cdot \nabla^2_\theta x_2 + \left( \nabla_\theta x_2 \right) \left( \nabla_\theta g_2 \right)^\top + \nabla^2_\theta \log |\det J|
$$

But since $g_2 = \frac{\partial \log p}{\partial x_2}$, and $\nabla_\theta g_2 = \frac{\partial^2 \log p}{\partial x_2^2} \cdot \nabla_\theta x_2 = h_{22} \cdot \nabla_\theta x_2$, we get:

$$
H_\theta = g_2 \cdot \nabla^2_\theta x_2 + h_{22} \cdot (\nabla_\theta x_2)(\nabla_\theta x_2)^\top + \nabla^2_\theta \log |\det J|
$$

Let’s compute each term separately.

---

## Step 3: Second Derivatives of $x_2$

Recall:

$$
x_2 = z_2 \cdot e^s + c z_1 + d, \quad s = a z_1 + b
$$

So,

$$
\frac{\partial^2 x_2}{\partial a^2} = z_2 z_1^2 e^s, \quad
\frac{\partial^2 x_2}{\partial a \partial b} = z_2 z_1 e^s, \quad
\frac{\partial^2 x_2}{\partial b^2} = z_2 e^s
$$

$$
\frac{\partial^2 x_2}{\partial a \partial c} = 0, \quad \frac{\partial^2 x_2}{\partial a \partial d} = 0, \quad \text{(and all other second mixed terms with c or d are zero)}
$$

So, the Hessian of $x_2$ is:

$$
\nabla^2_\theta x_2 =
\begin{bmatrix}
z_2 z_1^2 e^s & z_2 z_1 e^s & 0 & 0 \\
z_2 z_1 e^s & z_2 e^s & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

---

## Step 4: Hessian of Log-Determinant

Since:

$$
\log |\det J| = a z_1 + b
$$

Then:

$$
\nabla^2_\theta \log |\det J| = 
\begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 
\end{bmatrix}
$$

---

## ✅ Final Expression for the Hessian

$$
H_\theta = g_2 \cdot \nabla^2_\theta x_2 + h_{22} \cdot (\nabla_\theta x_2)(\nabla_\theta x_2)^\top
$$

More explicitly:

```math
H_\theta = 
g_2 \cdot
\begin{bmatrix}
z_2 z_1^2 e^s & z_2 z_1 e^s & 0 & 0 \\
z_2 z_1 e^s & z_2 e^s & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix}
+
h_{22} \cdot
\begin{bmatrix}
z_2 z_1 e^s \\
z_2 e^s \\
z_1 \\
1
\end{bmatrix}
\begin{bmatrix}
z_2 z_1 e^s &
z_2 e^s &
z_1 &
1
\end{bmatrix}
```

This gives a rank-1 update structure in the second term (outer product), useful for efficient Newton optimization.

### implementation

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Target distribution: a banana-shaped distribution
def log_p(x1, x2):
    # Define a simple log-density for a banana distribution
    y1 = x1
    y2 = x2 + 0.1 * x1**2
    return -0.5 * (y1**2 + y2**2)

def grad_log_p(x1, x2):
    y1 = x1
    y2 = x2 + 0.1 * x1**2
    dy1 = -y1
    dy2 = -y2
    dx1 = dy1 + dy2 * 0.2 * x1
    dx2 = dy2
    return np.array([dx1, dx2])

def hess_log_p(x1, x2):
    y1 = x1
    y2 = x2 + 0.1 * x1**2
    d2y1 = -1.0
    d2y2 = -1.0
    dx1x1 = d2y1 + (-1.0) * 0.2 + (-y2) * 0.2
    dx1x2 = 0
    dx2x2 = d2y2
    return np.array([[dx1x1, dx1x2], [dx1x2, dx2x2]])

# Affine coupling layer: parameters theta = [a, b, c, d]
def transform(z1, z2, theta):
    a, b, c, d = theta
    s = a * z1 + b
    x1 = z1
    x2 = z2 * np.exp(s) + c * z1 + d
    return x1, x2

def grad_and_hess(z1, z2, theta):
    a, b, c, d = theta
    s = a * z1 + b
    exp_s = np.exp(s)

    x1 = z1
    x2 = z2 * exp_s + c * z1 + d

    g = grad_log_p(x1, x2)[1]
    h = hess_log_p(x1, x2)[1, 1]

    grad = g * np.array([z2 * z1 * exp_s, z2 * exp_s, z1, 1]) + np.array([z1, 1, 0, 0])
    J = np.array([z2 * z1 * exp_s, z2 * exp_s, z1, 1])
    hess = (
        g * np.array([
            [z2 * z1**2 * exp_s, z2 * z1 * exp_s, 0, 0],
            [z2 * z1 * exp_s, z2 * exp_s, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        + h * np.outer(J, J)
    )
    return grad, hess

def newtons_method(z1, z2, theta0, max_iter=10, tol=1e-6):
    theta = theta0.copy()
    for i in range(max_iter):
        grad, hess = grad_and_hess(z1, z2, theta)
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            print("Hessian is singular.")
            break
        theta -= delta
        if np.linalg.norm(delta) < tol:
            break
    return theta

# Run the method
z1, z2 = 0.5, -0.2
theta0 = np.random.randn(4) * 0.1
theta_opt = newtons_method(z1, z2, theta0)

theta_opt


```


