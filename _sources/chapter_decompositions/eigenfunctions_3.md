Here's a dedicated section you can add to your teaching material to explain **eigenfunctions in the context of Mercer's theorem and kernel methods**, geared toward machine learning students:

---

## 🧩 What Are Eigenfunctions? (in the context of kernels)

In linear algebra, eigenvectors are directions that stay the same under a matrix transformation — only their length (the eigenvalue) changes.

In **functional analysis**, **eigenfunctions** play the same role, but for **operators on functions** rather than matrices on vectors.

---

### 🎯 Definition

Let $T$ be an integral operator defined via a **Mercer kernel**:

$$
(Tf)(x) = \int_{\mathcal{X}} k(x, x') f(x') dx'
$$

A function $\phi \in L^2(\mathcal{X})$ is an **eigenfunction** of $T$ with eigenvalue $\lambda \in \mathbb{R}$ if:

$$
(T\phi)(x) = \lambda \phi(x) \quad \text{for all } x \in \mathcal{X}
$$

This is the infinite-dimensional analog of $A\mathbf{v} = \lambda \mathbf{v}$.

---

### 🧠 Intuition

* In finite-dimensional ML, we diagonalize the Gram matrix $\mathbf{K} = V \Lambda V^\top$, where columns of $V$ are eigenvectors.
* In function space, we diagonalize the **integral operator** $T$. Its eigenfunctions $\phi_i$ are like "functional eigenvectors."
* Mercer’s theorem says these eigenfunctions **form an orthonormal basis** of $L^2(\mathcal{X})$.

They are the **building blocks** of the kernel function:

$$
k(x, x') = \sum_{i=1}^\infty \lambda_i \phi_i(x)\phi_i(x')
$$

This is a functional analog of the matrix decomposition $K = \sum \lambda_i v_i v_i^\top$.

---

### 📈 Example: RBF Kernel in 1D

For the **Gaussian kernel** on $\mathcal{X} = [-1, 1]$, the eigenfunctions are **sines and cosines** (Fourier basis), weighted by rapidly decaying eigenvalues.

In fact, in the full $\mathbb{R}$, the RBF kernel is diagonalized by the **Fourier basis**:

$$
\phi_\omega(x) = e^{i \omega x}, \quad \text{with } \lambda(\omega) = e^{-\omega^2 / 4\gamma}
$$

So the Mercer expansion becomes a **Fourier integral**:

$$
k(x, x') = \int_{-\infty}^\infty e^{i\omega(x - x')} \lambda(\omega) d\omega
$$

---

### 🔄 ML Connection

Mercer’s expansion gives a **feature map**:

$$
\phi(x) = (\sqrt{\lambda_1} \phi_1(x), \sqrt{\lambda_2} \phi_2(x), \dots)
$$

So the kernel is an inner product:

$$
k(x, x') = \langle \phi(x), \phi(x') \rangle_{\ell^2}
$$

This justifies kernel methods like:

* Kernel Ridge Regression
* Kernel PCA
* Support Vector Machines
* Gaussian Processes

where **you never need to compute the eigenfunctions explicitly** — the kernel “hides” the complexity of infinite-dimensional space.

---

### 🧩 Summary Table

| Finite-Dimensional Linear Algebra                   | Functional Analysis               |
| --------------------------------------------------- | --------------------------------- |
| Eigenvector $\mathbf{v}$                            | Eigenfunction $\phi$              |
| Matrix $\mathbf{K}$                                 | Integral operator $T$             |
| $\mathbf{K} \mathbf{v} = \lambda \mathbf{v}$        | $T \phi = \lambda \phi$           |
| Gram matrix decomposition                           | Mercer expansion                  |
| Feature vector $\phi(\mathbf{x})$ in $\mathbb{R}^d$ | Feature map $\phi(x)$ in $\ell^2$ |

---

Let me know if you'd like an interactive demo (e.g. with RBF eigenfunctions on $[-1,1]$ visualized numerically), or an illustration of how this basis leads to **kernel PCA** projections.
