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
# Symmetric matrices

A matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is said to be
**symmetric** if it is equal to its own transpose
($\mathbf{A} = \mathbf{A}^{\!\top\!}$), meaning that $A_{ij} = A_{ji}$
for all $(i,j)$.

This definition seems harmless but turns out to
have some strong implications. 

## Spectral Decopmosition

:::{prf:theorem} Spectral Theorem
:label: trm-spectral-decomposition
:nonumber:

If $\mathbf{A} \in \mathbb{R}^{n \times n}$ is
symmetric, then there exists an orthonormal basis for $\mathbb{R}^n$
consisting of eigenvectors of $\mathbf{A}$.
:::

The practical application of this theorem is a particular factorization
of symmetric matrices, referred to as the **eigendecomposition** or
**spectral decomposition**.

Denote the orthonormal basis of eigenvectors
$\mathbf{q}_1, \dots, \mathbf{q}_n$ and their eigenvalues
$\lambda_1, \dots, \lambda_n$.

Let $\mathbf{Q}$ be an orthogonal matrix with $\mathbf{q}_1, \dots, \mathbf{q}_n$ as its columns, and

$$\mathbf{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_n).$$

Since by definition $\mathbf{A}\mathbf{q}_i = \lambda_i\mathbf{q}_i$ for every
$i$, the following relationship holds:

$$\mathbf{A}\mathbf{Q} = \mathbf{Q}\mathbf{\Lambda}$$

Right-multiplying
by $\mathbf{Q}^{\!\top\!}$, we arrive at the decomposition

$$\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{\!\top\!}$$

### Quadratic forms

Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ be a symmetric matrix.

The expression $\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}$ is called a
**quadratic form**.


Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ be a symmetric matrix, and
recall that the expression $\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}$
is called a quadratic form of $\mathbf{A}$. It is in some cases helpful
to rewrite the quadratic form in terms of the individual elements that
make up $\mathbf{A}$ and $\mathbf{x}$:

$$\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} = \sum_{i=1}^n\sum_{j=1}^n A_{ij}x_ix_j$$

This identity is valid for any square matrix (need not be symmetric),
although quadratic forms are usually only discussed in the context of
symmetric matrices.

