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
# The fundamental subspaces of a matrix

The fundamental subspaces of a matrix $\mathbf{A}$ are the four subspaces associated with the matrix and its transpose. 
These subspaces are important in linear algebra and numerical analysis, particularly in the context of solving linear systems and eigenvalue problems.
We also provide the projections onto these subspaces, which are useful for various applications such as least squares problems and dimensionality reduction. The proof of these projection formulas relies on the properties of the Moore-Penrose pseudoinverse and the orthogonal projections onto subspaces.

We denote the matrix $\mathbf{A}$ as an $m \times n$ matrix, where $m$ is the number of rows and $n$ is the number of columns. 

The four fundamental subspaces are:

## 1. **Column Space (Range) of $\mathbf{A}$**: 
The column space of a matrix $\mathbf{A}$ is the set of all possible linear combinations of its columns. It represents the span of the columns of $\mathbf{A}$ and is denoted as $\text{Col}(\mathbf{A})$ or $\text{Range}(\mathbf{A})$.

:::{prf:lemma} Projection onto the Column Space
:label: trm-projection-column-space
:nonumber:

The projection of a vector $\mathbf{b}$ onto the column space of a matrix $\mathbf{A}$ is given by:

$$
\mathbf{P}_{\text{Col}(\mathbf{A})}(\mathbf{b}) = \mathbf{A}\mathbf{A}^+ \mathbf{b}
$$
:::

## 2. **Null Space (Kernel) of $\mathbf{A}$**: 
The null space of a matrix $\mathbf{A}$ is the set of all vectors $\mathbf{x}$ such that $\mathbf{A}\mathbf{x} = \mathbf{0}$. It represents the solutions to the homogeneous equation associated with $\mathbf{A}$ and is denoted as $\text{Null}(\mathbf{A})$ or $\text{Ker}(\mathbf{A})$.

:::{prf:lemma} Projection onto the Null Space
:label: trm-projection-null-space
:nonumber:

The projection of a vector $\mathbf{b}$ onto the null space of a matrix $\mathbf{A}$ is given by:

$$
\mathbf{P}_{\text{Null}(\mathbf{A})}(\mathbf{b}) = \left(\mathbf{I} - \mathbf{P}_{\text{Col}(\mathbf{A})}\right)(\mathbf{b}) = \mathbf{b} - \mathbf{A}\mathbf{A}^+ \mathbf{b}
$$
:::

## 3. **Row Space of $\mathbf{A}$**: 
The row space of a matrix $\mathbf{A}$ is the set of all possible linear combinations of its rows. It is equivalent to the column space of its transpose, $\mathbf{A}^\top$, and is denoted as $\text{Row}(\mathbf{A})$ or $\text{Col}(\mathbf{A}^\top)$.

:::{prf:lemma} Projection onto the Row Space
:label: trm-projection-row-space
:nonumber:

The projection of a vector $\mathbf{b}$ onto the row space of a matrix $\mathbf{A}$ is given by:

$$
\mathbf{P}_{\text{Row}(\mathbf{A})}(\mathbf{b}) = \mathbf{A}^+\mathbf{A}\mathbf{b}
$$
:::


## 4. **Left Null Space (Kernel) of $\mathbf{A}$**:
The left null space of a matrix $\mathbf{A}$ is the set of all vectors $\mathbf{y}$ such that $\mathbf{A}^\top\mathbf{y} = \mathbf{0}$. It represents the solutions to the homogeneous equation associated with $\mathbf{A}^\top$ and is denoted as $\text{Null}(\mathbf{A}^\top)$ or $\text{Ker}(\mathbf{A}^\top)$.

:::{prf:lemma} Projection onto the Left Null Space
:label: trm-projection-left-null-space
:nonumber:

The projection of a vector $\mathbf{b}$ onto the left null space of a matrix $\mathbf{A}$ is given by:

$$
\mathbf{P}_{\text{Null}(\mathbf{A}^\top)}(\mathbf{b}) = \left(\mathbf{I} - \mathbf{P}_{\text{Row}(\mathbf{A})}\right)(\mathbf{b}) = \mathbf{b} - \mathbf{A}^+\mathbf{A}\mathbf{b}
$$
:::


## Singular Value Decomposition and the four fundamental subspaces
The SVD provides a powerful way to understand the four fundamental
subspaces of a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$.

The SVD of $\mathbf{A}$ is given by:

$$
\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\!\top\!}
$$

where $\mathbf{U} \in \mathbb{R}^{m \times m}$ and $\mathbf{V} \in \mathbb{R}^{n \times n}$ are orthogonal matrices, and $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$ is a diagonal matrix with the singular values of $\mathbf{A}$ on its diagonal.

:::{prf:lemma} SVD and the Four Fundamental Subspaces
:label: trm-svd-four-subspaces
:nonumber:

The SVD of a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ can be used to identify the four fundamental subspaces as follows:
1. **Column Space**: $\text{Col}(\mathbf{A}) = \text{span}(\mathbf{U}_r)$, where $\mathbf{U}_r$ consists of the first $r$ columns of $\mathbf{U}$ corresponding to non-zero singular values.
2. **Row Space**: $\text{Row}(\mathbf{A}) = \text{span}(\mathbf{V}_r)$, where $\mathbf{V}_r$ consists of the first $r$ columns of $\mathbf{V}$ corresponding to non-zero singular values.
3. **Null Space**: $\text{Null}(\mathbf{A}) = \text{span}(\mathbf{V}_{n-r})$, where $\mathbf{V}_{n-r}$ consists of the last $n-r$ columns of $\mathbf{V}$ corresponding to zero singular values.
4. **Left Null Space**: $\text{Null}(\mathbf{A}^\top) = \text{span}(\mathbf{U}_{m-r})$, where $\mathbf{U}_{m-r}$ consists of the last $m-r$ columns of $\mathbf{U}$ corresponding to zero singular values.
:::

## Summary
The four fundamental subspaces of a matrix $\mathbf{A}$ are essential in understanding the structure of the matrix and its properties. 
The projections onto these subspaces can be computed using the Moore-Penrose pseudoinverse, which provides a powerful tool for solving linear systems and performing dimensionality reduction. 
The SVD further enhances our understanding by revealing the relationships between these subspaces through the orthogonal matrices and singular values.