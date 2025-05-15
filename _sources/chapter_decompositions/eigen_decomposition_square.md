## The Eigen-Decomposition
Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ be an invertible square matrix.

Denote the basis of eigenvectors
$\mathbf{q}_1, \dots, \mathbf{q}_n$ and their eigenvalues
$\lambda_1, \dots, \lambda_n$.

Let $\mathbf{Q}$ be a matrix with $\mathbf{q}_1, \dots, \mathbf{q}_n$ as its columns, and

$$\mathbf{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_n).$$

The **eigen-decomposition** of $\mathbf{A}$ means:

$$
\mathbf{A} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^{-1}
$$

This decomposition is **only possible when $\mathbf{A}$ is invertible**.

---

## ❌ Not All Matrices Are Diagonalizable

A matrix is **not diagonalizable** if:

* It **does not have enough linearly independent eigenvectors** (i.e., the geometric multiplicity < algebraic multiplicity)

This can happen even if $\mathbf{A}$ is invertible!

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

