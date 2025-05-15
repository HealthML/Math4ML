# Real symmetric matrices cannot produce rotation

### 🧾 Statement
:::{prf:theorem}
:label: trm-symmetric-no-rotation
:nonumber:

Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ be a **real symmetric matrix**.

Then:

The linear transformation $\mathbf{x} \mapsto \mathbf{A}\mathbf{x}$ **does not rotate** vectors — i.e., it cannot produce a transformation that changes the direction of a vector **without preserving its span**.

In particular:

* The transformation **does not rotate angles**
* The transformation has a basis of **orthogonal eigenvectors**
* Therefore, all action is **stretching/compressing along fixed directions**, not rotation
:::

## 🧠 Intuition

Rotation mixes directions. But symmetric matrices:

* Have **real eigenvalues**
* Are **orthogonally diagonalizable**
* Have **mutually orthogonal eigenvectors**

So the matrix acts by **scaling along fixed orthogonal axes**, without changing the direction between basis vectors — i.e., no twisting, hence no rotation.

---
:::{prf:proof} (2D case, generalizes easily)

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

Let $\mathbf{v}$ be any eigenvector of $\mathbf{A}$.

Then:

$$
\mathbf{A} \mathbf{v} = \lambda \mathbf{v}
$$

So $\mathbf{v}$ is **mapped to a scalar multiple of itself** — its **direction doesn’t change**.

Because $\mathbb{R}^2$ has two linearly independent eigenvectors (since symmetric matrices are always diagonalizable), **no vector is rotated out of its original span** — just scaled.

Hence, the transformation only **stretches**, **compresses**, or **reflects**, but never rotates.
:::

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

