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
# Matrix Norms

Matrix norms provide a way to measure the "size" or "magnitude" of a matrix. They are used throughout machine learning and numerical analysis—for example, to quantify approximation error, assess convergence in optimization algorithms, or bound the spectral properties of linear transformations.

## Definition

A **matrix norm** is a function $ \|\cdot\| : \mathbb{R}^{m \times n} \to \mathbb{R} $ satisfying the following properties for all matrices $ \mathbf{A}, 
\mathbf{B} \in \mathbb{R}^{m \times n} $ and all scalars $ \alpha \in \mathbb{R} $:

1. **Non-negativity**: $ \|\mathbf{A}\| \geq 0 $
2. **Definiteness**: $ \|\mathbf{A}\| = 0 \iff \mathbf{A} = 0 $
3. **Homogeneity**: $ \|\alpha \mathbf{A}\| = |\alpha| \cdot \|\mathbf{A}\| $
4. **Triangle inequality**: $ \|\mathbf{A} + \mathbf{B}\| \leq \|\mathbf{A}\| + \|\mathbf{B}\| $


## Common Matrix Norms

### 1. **Frobenius Norm**

Defined by:

$$
\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\mathrm{tr}(\mathbf{A}^\top \mathbf{A})}
$$

It treats the matrix as a vector in $ \mathbb{R}^{mn} $.

### 2. **Induced (Operator) Norms**

Given a vector norm $ \|\cdot\| $, the **induced matrix norm** is:

$$
\|\mathbf{A}\| = \sup_{\mathbf{x} \neq 0} \frac{\|\mathbf{A} \mathbf{x}\|}{\|\mathbf{x}\|} = \sup_{\|\mathbf{x}\| = 1} \|\mathbf{A} \mathbf{x}\|
$$

Examples:
- **Spectral norm**: Induced by the Euclidean norm $ \|\cdot\|_2 $. 

  Equal to the largest singular value of $ \mathbf{A}. $
- **$ \ell_1 $ norm**: Maximum absolute column sum.
- **$ \ell_\infty $ norm**: Maximum absolute row sum.


## Properties

- Induced norms satisfy the **submultiplicative property**:

$$
\|\mathbf{A}\mathbf{B}\| \leq \|\mathbf{A}\| \cdot \|\mathbf{B}\|
$$

- For the Frobenius norm:

$$
\|\mathbf{A}\mathbf{B}\|_F \leq \|\mathbf{A}\|_F \cdot \|\mathbf{B}\|_F
$$

- All norms on a finite-dimensional vector space are equivalent (they define the same topology), but may differ in scaling.


## Applications in Machine Learning

- In **optimization**, norms define constraints (e.g., Lasso uses $ \ell_1 $-norm penalty).
- In **regularization**, norms quantify complexity of parameter matrices (e.g., weight decay with $ \ell_2 $-norm).
- In **spectral methods**, matrix norms bound approximation error (e.g., spectral norm bounds for generalization).


## Visual Comparison (2D case)

In 2D, vector norms induce different geometries:
- $ \ell_2 $: circular level sets
- $ \ell_1 $: diamond-shaped level sets
- $ \ell_\infty $: square level sets

This influences which directions are favored in optimization and which vectors are "small" under a given norm.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# Define example matrix A
A = np.array([[2, 1],
              [1, 3]])

# Create unit circles in different norms
theta = np.linspace(0, 2 * np.pi, 400)
circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)

# l1 unit ball boundary (diamond)
l1_vectors = []
for v in circle:
    norm = np.sum(np.abs(v))
    l1_vectors.append(v / norm)
l1_vectors = np.array(l1_vectors)

# l2 unit ball (circle)
l2_vectors = circle

# linf unit ball boundary (square)
linf_vectors = []
for v in circle:
    norm = np.max(np.abs(v))
    linf_vectors.append(v / norm)
linf_vectors = np.array(linf_vectors)

# Apply matrix A to each set
l1_transformed = l1_vectors @ A.T
l2_transformed = l2_vectors @ A.T
linf_transformed = linf_vectors @ A.T

# Plot
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# l1 norm effect
ax[0].plot(l1_vectors[:, 0], l1_vectors[:, 1], label='Original')
ax[0].plot(l1_transformed[:, 0], l1_transformed[:, 1], label='Transformed')
ax[0].set_title(r'$\ell_1$ Norm Unit Ball $\rightarrow A$')
ax[0].axis('equal')
ax[0].grid(True)
ax[0].legend()

# l2 norm effect
ax[1].plot(l2_vectors[:, 0], l2_vectors[:, 1], label='Original')
ax[1].plot(l2_transformed[:, 0], l2_transformed[:, 1], label='Transformed')
ax[1].set_title(r'$\ell_2$ Norm Unit Ball $\rightarrow A$')
ax[1].axis('equal')
ax[1].grid(True)
ax[1].legend()

# linf norm effect
ax[2].plot(linf_vectors[:, 0], linf_vectors[:, 1], label='Original')
ax[2].plot(linf_transformed[:, 0], linf_transformed[:, 1], label='Transformed')
ax[2].set_title(r'$\ell_\infty$ Norm Unit Ball $\rightarrow A$')
ax[2].axis('equal')
ax[2].grid(True)
ax[2].legend()

plt.tight_layout()
plt.show()

```


Let’s give formal **definitions and proofs** for several commonly used **induced matrix norms**, also known as **operator norms**, derived from vector norms.

---

## 📚 Setting

Let $\|\cdot\|$ be a **vector norm** on $\mathbb{R}^n$, and define the **induced matrix norm** for $\mathbf{A} \in \mathbb{R}^{m \times n}$ as:

$$
\|\mathbf{A}\| = \sup_{\mathbf{x} \neq 0} \frac{\|\mathbf{A} \mathbf{x}\|}{\|\mathbf{x}\|} = \sup_{\|\mathbf{x}\| = 1} \|\mathbf{A} \mathbf{x}\|
$$

We’ll now state and prove specific formulas for induced norms when the underlying vector norm is:

* $\ell_1$
* $\ell_\infty$
* $\ell_2$ (spectral norm)

---

## 1. Induced $\ell_1$ Norm

**Claim**:
If $\|\mathbf{x}\| = \|\mathbf{x}\|_1$, then:

$$
\|\mathbf{A}\|_1 = \max_{1 \leq j \leq n} \sum_{i=1}^m |A_{ij}|
\quad \text{(maximum absolute column sum)}
$$

You're absolutely right — the proof is missing a key justification: **how to construct a vector $\mathbf{x}$ that attains the bound**, i.e., that the triangle inequality becomes an equality.

Here's the improved proof with that step made explicit.

---

## 1. Induced $\ell_1$ Norm

**Claim**:
If $\|\mathbf{x}\| = \|\mathbf{x}\|_1$, then:

$$
\|\mathbf{A}\|_1 = \max_{1 \leq j \leq n} \sum_{i=1}^m |A_{ij}|
\quad \text{(maximum absolute column sum)}
$$

---

:::{prf:proof}

Let $\mathbf{A} = [a_{ij}] \in \mathbb{R}^{m \times n}$. 

Then by the definition of the induced norm:

$$
\|\mathbf{A}\|_1 = \sup_{\|\mathbf{x}\|_1 = 1} \|\mathbf{A} \mathbf{x}\|_1
= \sup_{\|\mathbf{x}\|_1 = 1} \sum_{i=1}^m \left| \sum_{j=1}^n a_{ij} x_j \right|
$$

Apply the triangle inequality inside the absolute value:

$$
\leq \sup_{\|\mathbf{x}\|_1 = 1} \sum_{i=1}^m \sum_{j=1}^n |a_{ij}| \cdot |x_j|
= \sup_{\|\mathbf{x}\|_1 = 1} \sum_{j=1}^n |x_j| \left( \sum_{i=1}^m |a_{ij}| \right)
$$

Let us define the **column sums**:

$$
c_j := \sum_{i=1}^m |a_{ij}|
$$

Then the expression becomes:

$$
\sum_{j=1}^n |x_j| c_j \leq \max_j c_j \cdot \sum_{j=1}^n |x_j| = \max_j c_j
$$

since $\sum_{j=1}^n |x_j| = \|\mathbf{x}\|_1 = 1$, and this is a convex combination of the $c_j$.

### Attainment of the Maximum

Let $j^* \in \{1, \dots, n\}$ be the index of the column with maximum sum:

$$
c_{j^*} = \max_j \sum_i |a_{ij}|
$$

Now choose the **standard basis vector** $\mathbf{e}_{j^*} \in \mathbb{R}^n$, where:

$$
(\mathbf{e}_{j^*})_j = \begin{cases}
1, & j = j^* \\\\
0, & j \neq j^*
\end{cases}
$$

Then $\|\mathbf{e}_{j^*}\|_1 = 1$, and:

$$
\|\mathbf{A} \mathbf{e}_{j^*}\|_1 = \sum_{i=1}^m \left| a_{i j^*} \right| = c_{j^*}
$$

So the upper bound is **achieved**, and we conclude:

$$
\|\mathbf{A}\|_1 = \max_j \sum_i |a_{ij}|
$$

QED.
:::
---

## 2. Induced $\ell_\infty$ Norm

**Claim**:
If $\|\mathbf{x}\| = \|\mathbf{x}\|_\infty$, then:

$$
\|\mathbf{A}\|_\infty = \max_{1 \leq i \leq m} \sum_{j=1}^n |A_{ij}|
\quad \text{(maximum absolute row sum)}
$$

:::{prf:proof}

Let $\|\mathbf{x}\|_\infty = 1$. 

Then:

$$
\|A \mathbf{x}\|_\infty = \max_{i} \left| \sum_j a_{ij} x_j \right|
\leq \max_i \sum_j |a_{ij}||x_j| \leq \max_i \sum_j |a_{ij}|
$$

Equality is achieved by choosing $x_j = \operatorname{sign}(a_{ij^*})$ at the row $i^*$ with largest sum. So:

$$
\|\mathbf{A}\|_\infty = \max_i \sum_j |a_{ij}|
$$

QED.
:::

## 3. Induced $\ell_2$ Norm (Spectral Norm)

**Claim**:
If $\|\cdot\| = \|\cdot\|_2$, then:

$$
\|\mathbf{A}\|_2 = \sigma_{\max}(\mathbf{A}) = \sqrt{\lambda_{\max}(\mathbf{A}^\top \mathbf{A})}
$$

where $\sigma_{\max}(\mathbf{A})$ is the **largest singular value** of $\mathbf{A}$, and $\lambda_{\max}$ denotes the largest eigenvalue.

:::{prf:proof}

$$
\|\mathbf{A}\|_2 = \sup_{\|\mathbf{x}\|_2 = 1} \|\mathbf{A} \mathbf{x}\|_2
= \sup_{\|\mathbf{x}\|_2 = 1} \sqrt{(\mathbf{A} \mathbf{x})^\top (\mathbf{A} \mathbf{x})}
= \sup_{\|\mathbf{x}\|_2 = 1} \sqrt{\mathbf{x}^\top \mathbf{A}^\top \mathbf{A} \mathbf{x}}
$$

This is the **Rayleigh quotient** of $\mathbf{A}^\top \mathbf{A}$, a symmetric PSD matrix. 

So:

$$
\|\mathbf{A}\|_2 = \sqrt{\lambda_{\max}(\mathbf{A}^\top \mathbf{A})}
$$

QED.
:::
---

## Summary Table

| Vector Norm   | Induced Matrix Norm        |                               |
| ------------- | ----------------------------|----------------------------- |
| $\ell_1$      | Max column sum:  |  $\max_j \sum_i a_{ij}$ |
| $\ell_\infty$ | Max row sum:  | $\max_i \sum_j a_{ij}$ |
| $\ell_2$      | Largest singular value: |$\sqrt{\lambda_{\max}(A^\top A)}$ |    



