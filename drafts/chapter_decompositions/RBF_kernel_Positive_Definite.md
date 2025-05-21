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
# RBF Kernel Positive Definite

In this chapter, we will state and prove Mercer's theorem, showing that a set of kernel's, so called Mercer kernels exist that represent infinite dimensional reproducing kernel Hilbert spaces. Mercer kernels always produce positive definite kernel matrices.

## Mercer's Theorem

Mercer’s Theorem is a cornerstone in understanding **positive-definite kernels** and their representation in **reproducing kernel Hilbert spaces (RKHS)** — foundational for kernel methods like SVMs and kernel PCA.

Below is a careful statement and proof outline of **Mercer’s Theorem**, suitable for a course that has covered eigenvalues, symmetric matrices, and function spaces.

---

## 📜 Mercer’s Theorem (Simplified Version)

Let $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ be a **symmetric, continuous, positive semi-definite kernel** function on a **compact domain** $\mathcal{X} \subset \mathbb{R}^d$.

> Then there exists an **orthonormal basis** $\{\phi_i\}_{i=1}^\infty$ of $L^2(\mathcal{X})$, and **non-negative eigenvalues** $\{\lambda_i\}_{i=1}^\infty$, such that:
>
> $$
> k(x, x') = \sum_{i=1}^{\infty} \lambda_i \phi_i(x) \phi_i(x') \quad \text{with convergence in } L^2(\mathcal{X} \times \mathcal{X})
> $$
>
> Furthermore, the integral operator:
>
> $$
> (Tf)(x) := \int_{\mathcal{X}} k(x, x') f(x') dx'
> $$
>
> is **compact, self-adjoint**, and **positive semi-definite** on $L^2(\mathcal{X})$.

---

## 🧠 Intuition

Mercer’s Theorem says:

* A symmetric, continuous, PSD kernel defines a **nice integral operator** on functions.
* That operator has a **spectral decomposition**, just like symmetric matrices do.
* The kernel function $k(x, x')$ can be written as a **sum over eigenfunctions** of this operator, just like how a Gram matrix can be decomposed as $K = \sum \lambda_i u_i u_i^\top$.

This justifies using **feature maps** $\phi_i(x) = \sqrt{\lambda_i} \psi_i(x)$ and writing:

$$
k(x, x') = \langle \phi(x), \phi(x') \rangle_{\ell^2}
$$

---

## ✍️ Sketch of the Proof

### Step 1: Define the Integral Operator

Given a kernel $k(x, x')$, define an operator $T$ on $L^2(\mathcal{X})$ by:

$$
(Tf)(x) = \int_{\mathcal{X}} k(x, x') f(x') dx'
$$

* $T$ is **linear**
* $T$ is **self-adjoint** since $k(x, x') = k(x', x)$
* $T$ is **compact**, due to continuity of $k$ on a compact domain

### Step 2: Apply the Spectral Theorem for Compact Self-Adjoint Operators

From functional analysis:

* $T$ has an orthonormal basis of eigenfunctions $\{\phi_i\}_{i=1}^\infty$
* Corresponding eigenvalues $\lambda_i \geq 0$ (since $T$ is PSD)

### Step 3: Represent the Kernel

Show that:

$$
k(x, x') = \sum_{i=1}^\infty \lambda_i \phi_i(x) \phi_i(x')
$$

This expansion converges **absolutely and uniformly** on $\mathcal{X} \times \mathcal{X}$ if $k$ is continuous.

### Step 4: Show PSD and Feature Map Representation

From the expansion, define the map:

$$
\phi(x) := \left( \sqrt{\lambda_1} \phi_1(x), \sqrt{\lambda_2} \phi_2(x), \dots \right)
$$

Then:

$$
k(x, x') = \langle \phi(x), \phi(x') \rangle_{\ell^2}
$$

So the kernel is **an inner product in an infinite-dimensional Hilbert space** — justifying its use in kernel methods.

---

## ✅ Summary Box


**Mercer's Theorem (simplified)**

Let $ k(x, x') $ be a continuous, symmetric, positive semi-definite kernel on a compact domain $ \mathcal{X} \subset \mathbb{R}^d $.

Then there exist orthonormal functions $ \phi_i \in L^2(\mathcal{X}) $ and eigenvalues $ \lambda_i \geq 0 $ such that:

$$
k(x, x') = \sum_{i=1}^{\infty} \lambda_i \phi_i(x) \phi_i(x')
$$
with convergence in $ L^2(\mathcal{X} \times \mathcal{X}) $.

Moreover, $ k $ defines a compact, self-adjoint, PSD operator on $ L^2(\mathcal{X}) $.




## 🧠 Setup: What is the RBF kernel?

Let $\mathbf{x}_1, \dots, \mathbf{x}_n \in \mathbb{R}^d$ be a set of data points.

The **RBF kernel** (also called Gaussian kernel) is defined as:

$$
k(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)
\quad \text{for } \gamma > 0
$$

The **RBF kernel matrix** $\mathbf{K} \in \mathbb{R}^{n \times n}$ has entries:

$$
\mathbf{K}_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)
$$

---

## ✅ Claim

> The RBF kernel matrix $\mathbf{K}$ is **positive semi-definite** for all $\gamma > 0$, i.e., for any $\mathbf{c} \in \mathbb{R}^n$,

$$
\mathbf{c}^\top \mathbf{K} \mathbf{c} \geq 0
$$

Moreover, if all $\mathbf{x}_i$ are distinct, then $\mathbf{K}$ is **positive definite**.

---

## ✍️ Proof (via Mercer's Theorem / Fourier Representation)

The RBF kernel is a special case of a **positive-definite kernel** as characterized by Mercer's theorem, but here’s a more constructive argument:

### Step 1: Express the kernel as an inner product in an infinite-dimensional feature space.

Let’s define the feature map $\phi: \mathbb{R}^d \to \ell^2$ via:

$$
\phi(\mathbf{x}) = \left( \sqrt{a_k} \, \psi_k(\mathbf{x}) \right)_{k=1}^{\infty}
$$

such that:

$$
k(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle
$$

It is known (e.g., via Taylor expansion or Fourier basis) that the RBF kernel corresponds to an **inner product in an infinite-dimensional Hilbert space**, and hence:

$$
k(\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle
\Rightarrow
\mathbf{K}_{ij} = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle
$$

Then, for any $\mathbf{c} \in \mathbb{R}^n$:

$$
\mathbf{c}^\top \mathbf{K} \mathbf{c}
= \sum_{i,j=1}^n c_i c_j \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle
= \left\| \sum_{i=1}^n c_i \phi(\mathbf{x}_i) \right\|^2 \geq 0
$$

✅ Hence, $\mathbf{K}$ is **positive semi-definite**.

---

### 🚀 Positive definiteness

If the $\mathbf{x}_i$ are **pairwise distinct**, then the feature vectors $\phi(\mathbf{x}_i)$ are **linearly independent** in the Hilbert space, and the only way for the sum to vanish is $\mathbf{c} = 0$. Hence:

$$
\mathbf{c}^\top \mathbf{K} \mathbf{c} > 0 \quad \text{for all } \mathbf{c} \ne 0
$$

✅ So $\mathbf{K}$ is **positive definite** if all data points are distinct.

---

### 📦 Summary


**Proposition**: The RBF kernel matrix $ \mathbf{K}_{ij} = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2) $ is positive semi-definite for all $\gamma > 0$, and positive definite if all $\mathbf{x}_i$ are distinct.

**Proof sketch**: The kernel function is an inner product in a Hilbert space, so the Gram matrix $ \mathbf{K} $ has the form $ \mathbf{K} = \Phi \Phi^\top $, which is always PSD.

## **proof by induction** that the **RBF kernel matrix is positive semi-definite**, based on verifying the PSD property for matrices of increasing size. This approach is constructive, concrete, and aligns well with students familiar with induction and Gram matrices.


---

## 🧩 Goal

Let $K \in \mathbb{R}^{n \times n}$, with entries:

$$
K_{ij} = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)
$$

We aim to prove:

> For any $n \in \mathbb{N}$, $K$ is **positive semi-definite**, i.e., for all $\mathbf{c} \in \mathbb{R}^n$:

$$
\mathbf{c}^\top K \mathbf{c} \geq 0
$$

---

## 🧠 Strategy: Induction on Matrix Size $n$

Let’s prove it by **induction on $n$**, the number of input points $\mathbf{x}_1, \dots, \mathbf{x}_n \in \mathbb{R}^d$.

---

### 🧱 Base Case $n = 1$

We have:

$$
K = [1] \quad \text{since } \|\mathbf{x}_1 - \mathbf{x}_1\|^2 = 0 \Rightarrow K_{11} = \exp(0) = 1
$$

Then for any $c \in \mathbb{R}$:

$$
c^\top K c = c^2 \cdot 1 = c^2 \geq 0
$$

✅ Base case holds.

---

### 🔁 Inductive Hypothesis

Assume that for some $n$, the kernel matrix $K_n \in \mathbb{R}^{n \times n}$ formed from $\mathbf{x}_1, \dots, \mathbf{x}_n$ is **positive semi-definite**.

---

### 🔄 Inductive Step: $n+1$

We add a new point $\mathbf{x}_{n+1}$ and form the $(n+1) \times (n+1)$ matrix $K_{n+1}$:

$$
K_{n+1} =
\begin{bmatrix}
K_n & \mathbf{k} \\
\mathbf{k}^\top & 1
\end{bmatrix}
$$

where:

* $K_n \in \mathbb{R}^{n \times n}$ is the existing RBF matrix (assumed PSD)
* $\mathbf{k} \in \mathbb{R}^n$, with entries $k_i = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_{n+1}\|^2)$
* The bottom-right entry is $k(\mathbf{x}_{n+1}, \mathbf{x}_{n+1}) = 1$

Let $\mathbf{c} \in \mathbb{R}^{n+1}$, split as:

$$
\mathbf{c} = \begin{bmatrix} \mathbf{a} \\ b \end{bmatrix}, \quad \mathbf{a} \in \mathbb{R}^n, \ b \in \mathbb{R}
$$

Then:

$$
\mathbf{c}^\top K_{n+1} \mathbf{c} =
\begin{bmatrix} \mathbf{a}^\top & b \end{bmatrix}
\begin{bmatrix}
K_n & \mathbf{k} \\
\mathbf{k}^\top & 1
\end{bmatrix}
\begin{bmatrix} \mathbf{a} \\ b \end{bmatrix}
= \mathbf{a}^\top K_n \mathbf{a} + 2b \mathbf{k}^\top \mathbf{a} + b^2
$$

Let’s define:

$$
f(b) = \mathbf{a}^\top K_n \mathbf{a} + 2b \mathbf{k}^\top \mathbf{a} + b^2
= \left( b + \mathbf{k}^\top \mathbf{a} \right)^2 + \left( \mathbf{a}^\top K_n \mathbf{a} - (\mathbf{k}^\top \mathbf{a})^2 \right)
$$

Note:

* The first term $\left(b + \mathbf{k}^\top \mathbf{a} \right)^2 \geq 0$
* By the **Cauchy-Schwarz inequality**, if $K_n$ is a Gram matrix (as is the case here), then:

  $$
  (\mathbf{k}^\top \mathbf{a})^2 \leq \mathbf{a}^\top K_n \mathbf{a}
  \Rightarrow \mathbf{a}^\top K_n \mathbf{a} - (\mathbf{k}^\top \mathbf{a})^2 \geq 0
  $$

✅ Therefore, $f(b) \geq 0$ for all $\mathbf{a}, b$, i.e., $K_{n+1}$ is PSD.

---

### ✅ Conclusion

By induction, all RBF kernel matrices $K_n \in \mathbb{R}^{n \times n}$ are **positive semi-definite** for all $n$.

---

### 📦 Summary

**Theorem**: RBF kernel matrices are positive semi-definite for all n and all γ > 0.

**Proof**: By induction on the number of data points n, using the structure of the kernel matrix
and properties of quadratic forms and Cauchy-Schwarz inequality.

## EXpanding the Cauchy-Schwarz step in the proof
### 🎯 The Step in Question

In the inductive proof of PSD for the RBF kernel matrix, we reached this expression for any vector $\mathbf{c} = \begin{bmatrix} \mathbf{a} \\ b \end{bmatrix} \in \mathbb{R}^{n+1}$:

$$
\mathbf{c}^\top K_{n+1} \mathbf{c} = \left( b + \mathbf{k}^\top \mathbf{a} \right)^2 + \left( \mathbf{a}^\top K_n \mathbf{a} - (\mathbf{k}^\top \mathbf{a})^2 \right)
$$

We want to argue that:

$$
\mathbf{a}^\top K_n \mathbf{a} - (\mathbf{k}^\top \mathbf{a})^2 \geq 0
$$

This is the **Cauchy-Schwarz step** — and here’s what it means.

---

## 🧠 Setting

* $K_n \in \mathbb{R}^{n \times n}$ is an **RBF kernel matrix**:

  $$
  K_n = \left[ \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2) \right]_{i,j=1}^n
  $$
* The vector $\mathbf{k} \in \mathbb{R}^n$ has entries $k_i = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_{n+1}\|^2)$

We assume (by the inductive hypothesis) that $K_n$ is **positive semi-definite**, which means it is a **Gram matrix**: it can be written as

$$
K_n = \Phi \Phi^\top
$$

for some (possibly infinite-dimensional) feature map $\phi(\mathbf{x})$, where:

$$
\Phi = 
\begin{bmatrix}
\phi(\mathbf{x}_1)^\top \\
\vdots \\
\phi(\mathbf{x}_n)^\top
\end{bmatrix}
\in \mathbb{R}^{n \times d}
$$

and $\phi(\mathbf{x}_i) \in \mathbb{R}^d$ or a Hilbert space.

---

## ✅ Step Explained Using Inner Products

Let’s define:

* $\mathbf{u} = \sum_{i=1}^n a_i \phi(\mathbf{x}_i)$
* $\mathbf{v} = \phi(\mathbf{x}_{n+1})$

Then:

* $\mathbf{a}^\top K_n \mathbf{a} = \|\mathbf{u}\|^2$
* $\mathbf{k}^\top \mathbf{a} = \langle \mathbf{u}, \mathbf{v} \rangle$, since $k_i = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_{n+1}) \rangle$

Now apply the **Cauchy–Schwarz inequality** in the inner product space:

$$
|\langle \mathbf{u}, \mathbf{v} \rangle|^2 \leq \|\mathbf{u}\|^2 \cdot \|\mathbf{v}\|^2
$$

In our case:

* $\mathbf{a}^\top K_n \mathbf{a} = \|\mathbf{u}\|^2$
* $(\mathbf{k}^\top \mathbf{a})^2 = |\langle \mathbf{u}, \mathbf{v} \rangle|^2$
* $\|\mathbf{v}\|^2 = k(\mathbf{x}_{n+1}, \mathbf{x}_{n+1}) = 1$

So:

$$
(\mathbf{k}^\top \mathbf{a})^2 \leq \mathbf{a}^\top K_n \mathbf{a}
\quad \Rightarrow \quad
\mathbf{a}^\top K_n \mathbf{a} - (\mathbf{k}^\top \mathbf{a})^2 \geq 0
$$

✅ This guarantees that the second term in our decomposition is **non-negative**, which is what we needed to conclude PSD.

---

### 📌 Summary

* The RBF kernel matrix $K_n$ is a **Gram matrix**: $K_n = \Phi \Phi^\top$
* So any quadratic form $\mathbf{a}^\top K_n \mathbf{a}$ is a **squared norm**: $\|\sum a_i \phi(\mathbf{x}_i)\|^2$
* The dot product with $\phi(\mathbf{x}_{n+1})$ is **bounded** by Cauchy-Schwarz:

  $$
  (\mathbf{k}^\top \mathbf{a})^2 = |\langle \mathbf{u}, \mathbf{v} \rangle|^2 \leq \|\mathbf{u}\|^2
  $$

