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
# 📜 Spectral Theorem for Compact Self-Adjoint Operators

:::{prf:theorem} Compact, Self-Adjoint Linear Operator
:label: def-compact-selfadjoint-operator
:nonumber:

Let $ \mathcal{H} $ be a Hilbert space.

A linear operator $ T : \mathcal{H} \to \mathcal{H} $ is called a **compact, self-adjoint linear operator** if it satisfies the following properties:

1. **Linearity**:
   
   $$
   T(\alpha f + \beta g) = \alpha T(f) + \beta T(g)
   \quad \text{for all } f, g \in \mathcal{H}, \ \alpha, \beta \in \mathbb{R} \text{ (or } \mathbb{C} \text{)}
   $$

2. **Self-Adjointness**:
   
   $$
   \langle T f, g \rangle = \langle f, T g \rangle
   \quad \text{for all } f, g \in \mathcal{H}
   $$

3. **Compactness**:
   For every bounded sequence $ \{f_n\} \subset \mathcal{H} $, the sequence $ \{T f_n\} $ has a **convergent subsequence** in $ \mathcal{H} $.
:::

Here is a clear and formal example — written as a MyST proof block — of a **compact, self-adjoint linear operator** on the Hilbert space $L^2([0, 1])$, using an **integral operator with a continuous symmetric kernel**.

---


:::{prf:theorem} Integral Operator on $ L^2([0, 1]) $
:label: ex-integral-operator-compact-selfadjoint
:nonumber:

Let $ \mathcal{H} = L^2([0, 1]) $ and let $ k : [0, 1] \times [0, 1] \to \mathbb{R} $ be a **continuous**, **symmetric** function, i.e.,

$$
k(x, y) = k(y, x) \quad \text{for all } x, y \in [0, 1]
$$

Define the operator $ T : \mathcal{H} \to \mathcal{H} $ by:

$$
(Tf)(x) = \int_0^1 k(x, y) f(y) \, dy
$$

Then $ T $ is a **compact, self-adjoint linear operator**:

- **Linearity**: follows directly from the linearity of the integral.
- **Self-adjointness**: for all $ f, g \in L^2([0, 1]) $,

$$
\langle T f, g \rangle = \int_0^1 \left( \int_0^1 k(x, y) f(y) \, dy \right) g(x) \, dx
= \int_0^1 f(y) \left( \int_0^1 k(x, y) g(x) \, dx \right) dy
= \langle f, T g \rangle
$$

by symmetry of $ k(x, y) $.

- **Compactness**: Since $ k $ is continuous on a compact domain $ [0, 1]^2 $, the operator $ T $ is compact (by the Arzelà–Ascoli theorem or the Hilbert–Schmidt theorem).

Thus, $ T $ satisfies all the conditions of a compact, self-adjoint linear operator.
:::


:::{prf:theorem} RBF Kernel Operator on $ L^2([0, 1]) $
:label: ex-rbf-kernel-operator
:nonumber:

Let $ \mathcal{H} = L^2([0, 1]) $, and let $ \gamma > 0 $. Define the kernel:

$$
k(x, y) = \exp(-\gamma (x - y)^2)
$$

This is the **Radial Basis Function (RBF) kernel**, which is:

- **continuous** on $ [0, 1]^2 $,
- **symmetric**, i.e., $ k(x, y) = k(y, x) $,
- **positive definite**, meaning it induces a positive semi-definite kernel matrix for any finite sample.

Then the integral operator

$$
(Tf)(x) = \int_0^1 \exp(-\gamma (x - y)^2) f(y) \, dy
$$

defines a **compact, self-adjoint linear operator** on $ L^2([0, 1]) $.

:::

:::{prf:theorem} Brownian Motion Kernel Operator on $ L^2([0, 1]) $
:label: ex-min-kernel-operator
:nonumber:

Let $ k(x, y) = \min(x, y) $, defined on $ [0, 1] \times [0, 1] $. This kernel is:

- **continuous** and **symmetric**: $ \min(x, y) = \min(y, x) $
- **positive semi-definite**: it corresponds to the covariance function of standard Brownian motion.

The integral operator:

$$
(Tf)(x) = \int_0^1 \min(x, y) f(y) \, dy
$$

is known as the **Volterra operator** associated with Brownian motion. It is:

- **linear**
- **self-adjoint** (via symmetry of $ \min(x, y) $)
- **compact**, since it is a Hilbert–Schmidt operator with square-integrable kernel.

Thus, it is a **compact, self-adjoint linear operator** on $ L^2([0, 1]) $.

:::




Let $\mathcal{H}$ be a real or complex **Hilbert space**, and let
$T : \mathcal{H} \to \mathcal{H}$ be a **compact, self-adjoint linear operator**.

> Then:
>
> 1. There exists an **orthonormal basis** $\{\phi_i\}_{i \in \mathbb{N}}$ of $\overline{\operatorname{im}(T)} \subseteq \mathcal{H}$ consisting of **eigenvectors of $T$**.
>
> 2. The corresponding eigenvalues $\{\lambda_i\} \subset \mathbb{R}$ are real, with $\lambda_i \to 0$.
>
> 3. $T$ has at most countably many non-zero eigenvalues, and each non-zero eigenvalue has **finite multiplicity**.
>
> 4. For all $f \in \mathcal{H}$, we have:
>
> $$
> T f = \sum_{i=1}^\infty \lambda_i \langle f, \phi_i \rangle \phi_i
> $$
>
> where the sum converges in norm (i.e., in $\mathcal{H}$).

---

## 🧠 Intuition

* Compactness of $T$ is like “finite rank behavior” at infinity.
* Self-adjointness ensures that the eigenvalues are real, and eigenvectors for distinct eigenvalues are orthogonal.
* The spectrum of $T$ consists of **eigenvalues only**, accumulating at 0.
* We can **diagonalize** $T$ in an orthonormal eigenbasis — exactly like symmetric matrices.

---

## ✍️ Sketch of the Proof

We split the proof into a sequence of known results.

---

### 1. **Existence of a Maximum Eigenvalue**

Let $T$ be compact and self-adjoint. Define:

$$
\lambda_1 = \sup_{\|f\| = 1} \langle Tf, f \rangle
$$

This is the **Rayleigh quotient**, and it gives the largest eigenvalue in magnitude. The supremum is **attained** (due to compactness), and the maximizer $f_1$ satisfies:

$$
Tf_1 = \lambda_1 f_1
$$

---

### 2. **Orthogonalization and Iteration (like Gram-Schmidt)**

Define $\mathcal{H}_1 = \{f \in \mathcal{H} : \langle f, f_1 \rangle = 0\}$. Restrict $T$ to $\mathcal{H}_1$, where it remains compact and self-adjoint. Then find the next eigenpair $(\lambda_2, f_2)$, and repeat.

This gives an **orthonormal sequence** of eigenfunctions $\{f_i\}$ with real eigenvalues $\lambda_i \to 0$, due to compactness.

---

### 3. **Convergence of Spectral Expansion**

For any $f \in \mathcal{H}$, let:

$$
f = \sum_{i=1}^\infty \langle f, \phi_i \rangle \phi_i + f_\perp
$$

where $f_\perp \in \ker(T)$. Then:

$$
Tf = \sum_{i=1}^\infty \lambda_i \langle f, \phi_i \rangle \phi_i
$$

The convergence is in $\mathcal{H}$-norm, using Parseval's identity and the fact that $\lambda_i \to 0$.

---

### ✅ Summary Box

**Spectral Theorem (Compact Self-Adjoint Operators)**

Let $ T : \mathcal{H} \to \mathcal{H} $ be compact and self-adjoint.

Then there exists an orthonormal basis $ \{\phi_i\} \subset \mathcal{H} $ consisting of eigenvectors of $ T $, with corresponding real eigenvalues $ \lambda_i \to 0 $, such that:

$$
T f = \sum_{i=1}^\infty \lambda_i \langle f, \phi_i \rangle \phi_i
\quad \text{for all } f \in \mathcal{H}
$$


---

This result is the infinite-dimensional generalization of the fact that a real symmetric matrix has an orthonormal eigenbasis and can be diagonalized.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# Define domain
n = 200
x = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, x)

# Define RBF kernel
gamma = 50
rbf_kernel = np.exp(-gamma * (X - Y) ** 2)

# Define min kernel
min_kernel = np.minimum(X, Y)

# Plotting both kernels side-by-side
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot RBF kernel
im0 = axs[0].imshow(rbf_kernel, extent=[0,1,0,1], origin='lower', cmap='viridis')
axs[0].set_title('RBF Kernel $k(x,y) = \exp(-\\gamma (x-y)^2)$')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
fig.colorbar(im0, ax=axs[0])

# Plot min kernel
im1 = axs[1].imshow(min_kernel, extent=[0,1,0,1], origin='lower', cmap='viridis')
axs[1].set_title('Min Kernel $k(x,y) = \min(x, y)$')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
fig.colorbar(im1, ax=axs[1])

plt.tight_layout()
plt.show()
```
This visualization shows two symmetric, continuous kernels defined on $[0, 1]^2$, each inducing a compact, self-adjoint integral operator on $L^2([0, 1])$:

* **Left panel**: The RBF kernel $k(x, y) = \exp(-\gamma(x - y)^2)$, concentrated along the diagonal where $x \approx y$, modeling local similarity.
* **Right panel**: The Brownian motion kernel $k(x, y) = \min(x, y)$, forming a triangular structure that accumulates information from the origin.

Both kernels generate PSD Gram matrices and operators with eigenfunction decompositions — perfect for illustrating Mercer's theorem in practice.
