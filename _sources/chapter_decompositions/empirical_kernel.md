# From Integral to Sum: Empirical Approximation

Given a kernel $k(x, x')$ and a function $f$, Mercer’s integral operator is:

$$
(Tf)(x) = \int_{\mathcal{X}} k(x, x') f(x') \, dx'
$$

But in ML, we only observe a finite sample $\{ x_1, \dots, x_n \} \subset \mathcal{X}$. So we replace the **integral** with a **sum** — i.e., a **Monte Carlo estimate** of the integral.

$$
(T_n f)(x) := \frac{1}{n} \sum_{i=1}^n k(x, x_i) f(x_i)
$$

This is called the **empirical kernel operator** (or empirical approximation of the integral operator).

---

### 🔍 Interpretation

You can now think of $T_n$ as a discrete matrix:

* Let $f \in \mathbb{R}^n$ be the vector $(f(x_1), \dots, f(x_n))$
* Define the kernel matrix $K \in \mathbb{R}^{n \times n}$ by $K_{ij} = k(x_i, x_j)$
* Then the operation $T_n$ becomes **matrix multiplication**:

$$
(T_n f)(x_j) = \frac{1}{n} \sum_{i=1}^n K_{ji} f(x_i)
\quad \text{or} \quad
\mathbf{T}_n \mathbf{f} = \frac{1}{n} K \mathbf{f}
$$

This lets you **numerically approximate** Mercer’s operator and its spectrum using standard linear algebra.

---

## 🧠 Why This Is So Useful

* We can compute **eigenvectors and eigenvalues** of $\frac{1}{n} K$ to approximate **eigenfunctions and eigenvalues** of $T$.
* **Kernel PCA** and **spectral clustering** use exactly this approximation.
* This is also the foundation of **Nyström approximation** for scaling kernel methods.

---

## 🎯 Summary Table

| Infinite Data (Mercer Operator)         | Finite Data (Empirical Version)                         |
| --------------------------------------- | ------------------------------------------------------- |
| $(Tf)(x) = \int k(x,x') f(x') dx'$      | $(T_n f)(x) = \frac{1}{n} \sum_{i=1}^n k(x,x_i) f(x_i)$ |
| Integral operator on $L^2(\mathcal{X})$ | Matrix operator on $\mathbb{R}^n$                       |
| True eigenfunctions $\phi_i$            | Discrete eigenvectors $u_i \in \mathbb{R}^n$            |
| Mercer expansion                        | Kernel PCA, Nyström, etc.                               |

---

## ✅ Bottom Line

> In practice, **Mercer’s integral operator becomes a matrix** — and you work with eigenvectors of the kernel matrix $K$, which approximate the eigenfunctions of the continuous operator.

So every time you train a model using a **kernel matrix** on a finite dataset, you're implicitly approximating something **infinite-dimensional** — using finite, concrete computations.

