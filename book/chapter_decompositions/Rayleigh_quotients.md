# Rayleigh quotients


There turns out to be an interesting connection between the quadratic
form of a symmetric matrix and its eigenvalues. This connection is
provided by the **Rayleigh quotient**

$$R_\mathbf{A}(\mathbf{x}) = \frac{\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}}{\mathbf{x}^{\!\top\!}\mathbf{x}}$$

The Rayleigh quotient has a couple of important properties which the
reader can (and should!) easily verify from the definition:

(i) **Scale invariance**: for any vector $\mathbf{x} \neq \mathbf{0}$
    and any scalar $\alpha \neq 0$,
    $R_\mathbf{A}(\mathbf{x}) = R_\mathbf{A}(\alpha\mathbf{x})$.

(ii) If $\mathbf{x}$ is an eigenvector of $\mathbf{A}$ with eigenvalue
     $\lambda$, then $R_\mathbf{A}(\mathbf{x}) = \lambda$.

We can further show that the Rayleigh quotient is bounded by the largest
and smallest eigenvalues of $\mathbf{A}$. But first we will show a
useful special case of the final result.

*Proposition.* 
For any $\mathbf{x}$ such that $\|\mathbf{x}\|_2 = 1$,

$$\lambda_{\min}(\mathbf{A}) \leq \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} \leq \lambda_{\max}(\mathbf{A})$$

with equality if and only if $\mathbf{x}$ is a corresponding
eigenvector.

*Proof.* We show only the $\max$ case because the argument for the
$\min$ case is entirely analogous.

Since $\mathbf{A}$ is symmetric, we can decompose it as
$\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{\!\top\!}$. Then use
the change of variable $\mathbf{y} = \mathbf{Q}^{\!\top\!}\mathbf{x}$,
noting that the relationship between $\mathbf{x}$ and $\mathbf{y}$ is
one-to-one and that $\|\mathbf{y}\|_2 = 1$ since $\mathbf{Q}$ is
orthogonal. Hence

$$\max_{\|\mathbf{x}\|_2 = 1} \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} = \max_{\|\mathbf{y}\|_2 = 1} \mathbf{y}^{\!\top\!}\mathbf{\Lambda}\mathbf{y} = \max_{y_1^2+\dots+y_n^2=1} \sum_{i=1}^n \lambda_i y_i^2$$

Written this way, it is clear that $\mathbf{y}$ maximizes this
expression exactly if and only if it satisfies
$\sum_{i \in I} y_i^2 = 1$ where
$I = \{i : \lambda_i = \max_{j=1,\dots,n} \lambda_j = \lambda_{\max}(\mathbf{A})\}$
and $y_j = 0$ for $j \not\in I$. That is, $I$ contains the index or
indices of the largest eigenvalue. In this case, the maximal value of
the expression is

$$\sum_{i=1}^n \lambda_i y_i^2 = \sum_{i \in I} \lambda_i y_i^2 = \lambda_{\max}(\mathbf{A}) \sum_{i \in I} y_i^2 = \lambda_{\max}(\mathbf{A})$$

Then writing $\mathbf{q}_1, \dots, \mathbf{q}_n$ for the columns of
$\mathbf{Q}$, we have

$$\mathbf{x} = \mathbf{Q}\mathbf{Q}^{\!\top\!}\mathbf{x} = \mathbf{Q}\mathbf{y} = \sum_{i=1}^n y_i\mathbf{q}_i = \sum_{i \in I} y_i\mathbf{q}_i$$

where we have used the matrix-vector product identity.

Recall that $\mathbf{q}_1, \dots, \mathbf{q}_n$ are eigenvectors of
$\mathbf{A}$ and form an orthonormal basis for $\mathbb{R}^n$. Therefore
by construction, the set $\{\mathbf{q}_i : i \in I\}$ forms an
orthonormal basis for the eigenspace of $\lambda_{\max}(\mathbf{A})$.
Hence $\mathbf{x}$, which is a linear combination of these, lies in that
eigenspace and thus is an eigenvector of $\mathbf{A}$ corresponding to
$\lambda_{\max}(\mathbf{A})$.

We have shown that
$\max_{\|\mathbf{x}\|_2 = 1} \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} = \lambda_{\max}(\mathbf{A})$,
from which we have the general inequality
$\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} \leq \lambda_{\max}(\mathbf{A})$
for all unit-length $\mathbf{x}$. ◻


By the scale invariance of the Rayleigh quotient, we immediately have as
a corollary (since
$\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} = R_{\mathbf{A}}(\mathbf{x})$
for unit $\mathbf{x}$)

*Theorem.* 
(Min-max theorem) For all $\mathbf{x} \neq \mathbf{0}$,

$$\lambda_{\min}(\mathbf{A}) \leq R_\mathbf{A}(\mathbf{x}) \leq \lambda_{\max}(\mathbf{A})$$

with equality if and only if $\mathbf{x}$ is a corresponding
eigenvector.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# Define symmetric matrix
A = np.array([[2, 1],
              [1, 3]])

# Eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eigh(A)
λ_min, λ_max = eigvals

# Generate unit circle points
theta = np.linspace(0, 2*np.pi, 300)
circle = np.stack((np.cos(theta), np.sin(theta)))

# Rayleigh quotient computation
R = np.einsum('ij,ji->i', circle.T @ A, circle)  # x^T A x
R /= np.einsum('ij,ji->i', circle.T, circle)     # x^T x

# Rayleigh extrema
idx_min = np.argmin(R)
idx_max = np.argmax(R)
x_min = circle[:, idx_min]
x_max = circle[:, idx_max]

# Prepare grid for quadratic form level sets
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
XY = np.stack((X, Y), axis=-1)
Z = np.einsum('...i,ij,...j->...', XY, A, XY)
levels = np.linspace(np.min(Z), np.max(Z), 20)

# Create combined figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Rayleigh quotient on unit circle
sc = ax1.scatter(circle[0], circle[1], c=R, cmap='viridis', s=10)
ax1.quiver(0, 0, x_min[0], x_min[1], color='red', scale=1, scale_units='xy', angles='xy', label='argmin R(x)')
ax1.quiver(0, 0, x_max[0], x_max[1], color='orange', scale=1, scale_units='xy', angles='xy', label='argmax R(x)')
for i in range(2):
    eigvec = eigvecs[:, i]
    ax1.quiver(0, 0, eigvec[0], eigvec[1], color='black', alpha=0.5, scale=1, scale_units='xy', angles='xy', width=0.008)
ax1.set_title("Rayleigh Quotient on the Unit Circle")
ax1.set_aspect('equal')
ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(-1.1, 1.1)
ax1.grid(True)
ax1.legend()
plt.colorbar(sc, ax=ax1, label="Rayleigh Quotient $R_A(\\mathbf{x})$")

# Right: Level sets of quadratic form
contour = ax2.contour(X, Y, Z, levels=levels, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8, fmt="%.1f")
ax2.set_title("Level Sets of $\\mathbf{x}^\\top \\mathbf{A} \\mathbf{x}$")
ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")
ax2.axhline(0, color='gray', lw=0.5)
ax2.axvline(0, color='gray', lw=0.5)
for i in range(2):
    vec = eigvecs[:, i] * np.sqrt(eigvals[i])
    ax2.quiver(0, 0, vec[0], vec[1], color='red', scale=1, scale_units='xy', angles='xy', width=0.01, label=f"$\\mathbf{{q}}_{i+1}$")
ax2.set_aspect('equal')
ax2.legend()

plt.suptitle("Rayleigh Quotient and Quadratic Form Level Sets", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
```

This combined visualization brings together the **Rayleigh quotient** and the **level sets of the quadratic form** $\mathbf{x}^\top \mathbf{A} \mathbf{x}$:

* **Left panel**: Rayleigh quotient $R_\mathbf{A}(\mathbf{x})$ on the unit circle

  * Color shows how the value varies with direction.
  * Extremes occur at eigenvector directions (marked with arrows).

* **Right panel**: Level sets (contours) of the quadratic form

  * Elliptical shapes aligned with eigenvectors.
  * Red vectors indicate principal axes (scaled eigenvectors).

Together, these panels illustrate how the **direction of a vector determines how strongly it is scaled** by the symmetric matrix, and how this scaling relates to the matrix's **eigenstructure**.

✅ As guaranteed by the **Min–Max Theorem**, the maximum and minimum of the Rayleigh quotient occur precisely at the **eigenvectors corresponding to the largest and smallest eigenvalues**.



---

## ✅ Theorem: Real symmetric matrices cannot produce rotation

### 🧾 Statement

Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ be a **real symmetric matrix**. Then:

> The linear transformation $\mathbf{x} \mapsto \mathbf{A}\mathbf{x}$ **does not rotate** vectors — i.e., it cannot produce a transformation that changes the direction of a vector **without preserving its span**.

In particular:

* The transformation **does not rotate angles**
* The transformation has a basis of **orthogonal eigenvectors**
* Therefore, all action is **stretching/compressing along fixed directions**, not rotation

---

## 🧠 Intuition

Rotation mixes directions. But symmetric matrices:

* Have **real eigenvalues**
* Are **orthogonally diagonalizable**
* Have **mutually orthogonal eigenvectors**

So the matrix acts by **scaling along fixed orthogonal axes**, without changing the direction between basis vectors — i.e., no twisting, hence no rotation.

---

## ✏️ Proof (2D case, generalizes easily)

Let $\mathbf{A} \in \mathbb{R}^{2 \times 2}$ be symmetric:

$$
\mathbf{A} = \begin{pmatrix} a & b \\ b & d \end{pmatrix}
$$

We’ll show that $\mathbf{A}$ cannot produce a true rotation.

### Step 1: Diagonalize $\mathbf{A}$

Because $\mathbf{A}$ is real symmetric, there exists an orthogonal matrix $\mathbf{Q}$ and diagonal $\mathbf{\Lambda}$ such that:

$$
\mathbf{A} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^\top
$$

That is, $\mathbf{A}$ acts as:

* A rotation (or reflection) $\mathbf{Q}^\top$
* A stretch along axes $\mathbf{\Lambda}$
* A second rotation (or reflection) $\mathbf{Q}$

But since $\mathbf{Q}$ and $\mathbf{Q}^\top$ cancel out geometrically (they are transposes of each other), this results in:

> A transformation that **scales but does not rotate** relative to the basis of eigenvectors.

### Step 2: Show $\mathbf{A}$ preserves alignment

Let $\mathbf{v}$ be any eigenvector of $\mathbf{A}$. Then:

$$
\mathbf{A} \mathbf{v} = \lambda \mathbf{v}
$$

So $\mathbf{v}$ is **mapped to a scalar multiple of itself** — its **direction doesn’t change**.

Because $\mathbb{R}^2$ has two linearly independent eigenvectors (since symmetric matrices are always diagonalizable), **no vector is rotated out of its original span** — just scaled.

Hence, the transformation only **stretches**, **compresses**, or **reflects**, but never rotates.

---

## 🚫 Counterexample: Rotation matrix is not symmetric

The rotation matrix:

$$
\mathbf{R}_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
$$

is **not symmetric** unless $\theta = 0$ or $\pi$, where it reduces to identity or negation.

It **does not** have real eigenvectors (except at those degenerate angles), and it **rotates** all directions.

---

## ✅ Conclusion

**Rotation requires asymmetry.**

If a linear transformation rotates vectors (changes direction without preserving alignment), the matrix must be **non-symmetric**.

---

## ✅ Corollary

A matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ can perform rotation **only if**:

* It is **not symmetric**, and
* It has **complex eigenvalues** (at least in 2D rotation)

Excellent and important question. The answer is:

> ❗️**Not all non-symmetric matrices have an eigen-decomposition over $\mathbb{R}$ or even $\mathbb{C}$.**

Let’s unpack what this means.

---

## ✅ What is an Eigen-Decomposition?

An **eigen-decomposition** of a matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ means:

$$
\mathbf{A} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^{-1}
$$

Where:

* $\mathbf{\Lambda}$ is a diagonal matrix of eigenvalues
* $\mathbf{V}$ contains eigenvectors as columns
* $\mathbf{V}^{-1}$ exists (i.e., $\mathbf{V}$ is invertible)

This decomposition is **only possible when $\mathbf{A}$ is diagonalizable**.

---

## ❌ Not All Matrices Are Diagonalizable

A matrix is **not diagonalizable** if:

* It **does not have enough linearly independent eigenvectors** (i.e., the geometric multiplicity < algebraic multiplicity)

This can happen even if all the eigenvalues are real!

### 🔴 Example (Defective Matrix):

$$
\mathbf{A} = \begin{pmatrix}
1 & 1 \\
0 & 1
\end{pmatrix}
$$

* Eigenvalue: $\lambda = 1$
* But only **one** linearly independent eigenvector
* So it **cannot be diagonalized**

---

## ✅ When Does a Matrix Have an Eigen-Decomposition?

| Matrix Type                    | Diagonalizable? | Notes                                            |
| ------------------------------ | --------------- | ------------------------------------------------ |
| Symmetric (real)               | ✅ Always        | Eigen-decomposition with orthogonal eigenvectors |
| Diagonalizable (in general)    | ✅ Yes           | Can write $A = V \Lambda V^{-1}$                 |
| Defective (non-diagonalizable) | ❌ No            | Needs Jordan form instead                        |

---

## 🔁 Jordan Decomposition: The General Replacement

If a matrix is **not diagonalizable**, it still has a **Jordan decomposition**:

$$
\mathbf{A} = \mathbf{P} \mathbf{J} \mathbf{P}^{-1}
$$

Where:

* $\mathbf{J}$ is **block diagonal**: eigenvalues + possible **Jordan blocks**
* This captures **generalized eigenvectors**

So **every square matrix** has a **Jordan decomposition**, but **not every one has an eigen-decomposition**.

---

## ✅ Summary

* **Symmetric matrices**: always have an eigen-decomposition (with real, orthogonal eigenvectors)
* **Non-symmetric matrices**:

  * May have a complete eigen-decomposition (if diagonalizable)
  * May **not**, if they are **defective**
* In the general case, you must use **Jordan form**

A matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ has **complex eigenvalues or eigenvectors** when:

### ✅ 1. The matrix is **not symmetric** (i.e., $\mathbf{A} \ne \mathbf{A}^\top$)

* Real symmetric matrices **always** have **real** eigenvalues and orthogonal eigenvectors.
* Non-symmetric real matrices can have complex eigenvalues and eigenvectors.

### ✅ 2. The **characteristic polynomial** has **complex roots**

For example, consider:

$$
\mathbf{A} = \begin{pmatrix} 0 & -2 \\ 1 & 0 \end{pmatrix}
$$

Its characteristic polynomial is:

$$
\det(\mathbf{A} - \lambda \mathbf{I}) = \lambda^2 + 1
$$

The roots are:

$$
\lambda = \pm i
$$

So it has **pure imaginary eigenvalues**, and its eigenvectors are also **complex**.
## ✅ Quick Answer:

The eigenvectors and their transformed versions $\mathbf{A} \mathbf{v} = \lambda \mathbf{v}$ **are** parallel — **but only in complex vector space** $\mathbb{C}^n$.

In **real space**, we usually visualize:

* The **real part** of a complex vector: $\mathrm{Re}(\mathbf{v})$
* The **imaginary part** of a complex vector: $\mathrm{Im}(\mathbf{v})$

But neither of these alone is invariant under multiplication by $\lambda \in \mathbb{C}$. So when you look at:

$$
\mathbf{v} = \mathrm{Re}(\mathbf{v}) + i \cdot \mathrm{Im}(\mathbf{v})
$$

and apply $\mathbf{A}$, what you see in the real plane is:

$$
\mathrm{Re}(\mathbf{A} \mathbf{v}) \quad \text{vs.} \quad \mathrm{Re}(\lambda \mathbf{v})
$$

These are **not scalar multiples** of $\mathrm{Re}(\mathbf{v})$ or $\mathrm{Im}(\mathbf{v})$, because complex scaling **mixes real and imaginary components** — unless $\lambda$ is real.

---

## 🔍 Example

Say:

$$
\lambda = a + ib, \quad \mathbf{v} = \begin{pmatrix} x + iy \\ z + iw \end{pmatrix}
$$

Then:

$$
\lambda \mathbf{v} = (a + ib)(\text{real} + i \cdot \text{imag}) = \text{mix of real and imaginary}
$$

So $\mathbf{A} \mathbf{v} = \lambda \mathbf{v}$, but $\mathrm{Re}(\mathbf{A} \mathbf{v})$ will **not be parallel** to $\mathrm{Re}(\mathbf{v})$ alone — it's a rotated and scaled mixture.

---

## 🧠 Bottom Line

> **Eigenvectors and their transformations are parallel in $\mathbb{C}^n$, but not necessarily in $\mathbb{R}^n$.**


> Note: The eigenvectors and their transformations are parallel in complex space, but their real and imaginary parts generally point in different directions due to complex scaling (rotation + stretch).

---

## 🧠 Intuition

* Complex eigenvalues often indicate **rotational behavior** in linear dynamical systems.
* The matrix above rotates vectors by 90° and has no real direction that stays on its span after transformation — hence no real eigenvectors.

---

## 🔄 Summary

| Matrix Type        | Eigenvalues       | Eigenvectors      |
| ------------------ | ----------------- | ----------------- |
| Symmetric real     | Real              | Real & orthogonal |
| Non-symmetric real | Real or complex   | Real or complex   |
| Complex (any)      | Complex (general) | Complex (general) |

