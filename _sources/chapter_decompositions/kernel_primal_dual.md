# Kernel Methods: Dual and Primal Perspectives on Finite Datasets
Let’s connect both perspectives and interpret the **finite dataset** in the **primal (feature) view**, just like we did with the integral operator in the dual.

---

## 🔁 Dual View: Kernel as Monte Carlo Inner Product

In the **dual view**, with a kernel $k(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}}$, we define the **integral operator**:

$$
(Tf)(x) = \int k(x, x') f(x') dx' = \int \langle \phi(x), \phi(x') \rangle f(x') dx'
$$

In the **empirical (finite-data) version**:

$$
(T_n f)(x) = \frac{1}{n} \sum_{i=1}^n \langle \phi(x), \phi(x_i) \rangle f(x_i)
= \left\langle \phi(x), \frac{1}{n} \sum_{i=1}^n f(x_i) \phi(x_i) \right\rangle
$$

This reveals the core insight:

> The empirical operator **acts as a projection** of $\phi(x)$ onto the **weighted average of training features**.

---

## 🧠 Primal View: Weighted Sum of Feature Vectors

Define the feature map $\phi: \mathcal{X} \to \mathcal{H}$, where $\mathcal{H}$ is a (possibly infinite-dimensional) Hilbert space.

Given finite training data $x_1, \dots, x_n$, we can now define:

### **Empirical Covariance Operator**

This is the **primal counterpart** of the integral operator:

$$
C_n := \frac{1}{n} \sum_{i=1}^n \phi(x_i) \otimes \phi(x_i)
\quad \text{(maps } \mathcal{H} \to \mathcal{H})
$$

It’s the **covariance operator** of the feature vectors:

* For $v \in \mathcal{H}$, we have:

$$
(C_n v) = \frac{1}{n} \sum_{i=1}^n \langle v, \phi(x_i) \rangle \phi(x_i)
$$

This is exactly like a sample covariance matrix in $\mathbb{R}^d$, but in a Hilbert space.

---

## 🎯 Dual vs. Primal Views: Unified Picture

| Viewpoint             | Object                        | Formula                                                                 | Interpretation                                     |
| --------------------- | ----------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------- |
| **Dual (kernel)**     | Empirical integral operator   | $T_n f(x) = \frac{1}{n} \sum \langle \phi(x), \phi(x_i) \rangle f(x_i)$ | Projects onto span of kernel functions $k(x, x_i)$ |
| **Primal (features)** | Empirical covariance operator | $C_n v = \frac{1}{n} \sum \langle v, \phi(x_i) \rangle \phi(x_i)$       | Projects onto span of feature vectors $\phi(x_i)$  |

Both operators:

* Are rank at most $n$
* Have **equivalent spectra** (nonzero eigenvalues are the same)
* Are approximations to infinite-dimensional population-level operators

---

## 📌 Summary: Training Data in the Primal View

The training data $\{x_i\}_{i=1}^n$, mapped via $\phi$, defines:

1. A **subspace** of the Hilbert space:

   $$
   \mathcal{H}_n = \text{span} \{ \phi(x_1), \dots, \phi(x_n) \}
   $$

2. An **empirical distribution** over that subspace:
   The covariance operator $C_n$ gives an **approximate geometry** of the full feature space using just the samples.

3. The **mean feature vector**:

   $$
   \bar{\phi} = \frac{1}{n} \sum_{i=1}^n \phi(x_i)
   $$

   used in e.g. kernel mean embeddings and Maximum Mean Discrepancy (MMD).

So:

> In the **primal**, the data gives you a subspace and an empirical distribution over features.
> In the **dual**, you get kernel matrices and Monte Carlo approximations to integral operators.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

# Generate synthetic data
np.random.seed(0)
X = np.random.randn(100, 2)

# Normalize data (important for PCA)
X_scaled = StandardScaler().fit_transform(X)

# --- Primal PCA ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- Dual PCA (Kernel PCA with RBF Kernel) ---
gamma = 1.0
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=gamma, fit_inverse_transform=True)
X_kpca = kpca.fit_transform(X_scaled)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c='blue', alpha=0.6)
axes[0].set_title("Primal PCA (Linear Feature Map)")
axes[0].set_xlabel("PC 1")
axes[0].set_ylabel("PC 2")

axes[1].scatter(X_kpca[:, 0], X_kpca[:, 1], c='green', alpha=0.6)
axes[1].set_title("Dual PCA (Kernel PCA with RBF Kernel)")
axes[1].set_xlabel("Kernel PC 1")
axes[1].set_ylabel("Kernel PC 2")

plt.tight_layout()
plt.show()
```

This code generates synthetic data and applies both Primal PCA (using linear feature maps) and Dual PCA (using Kernel PCA with an RBF kernel).