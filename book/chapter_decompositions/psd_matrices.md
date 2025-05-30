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
# Positive (semi-)definite matrices

>A symmetric matrix $\mathbf{A}$ is **positive semi-definite** if for all $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} \geq 0$. 
>
>Sometimes people write $\mathbf{A} \succeq 0$ to indicate that $\mathbf{A}$ is positive
semi-definite.

> A symmetric matrix $\mathbf{A}$ is **positive definite** if for all nonzero $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} > 0$.
>
>Sometimes people write $\mathbf{A} \succ 0$ to indicate that $\mathbf{A}$ is positive definite.

Note that positive definiteness is a strictly stronger property than
positive semi-definiteness, in the sense that every positive definite
matrix is positive semi-definite but not vice-versa.

These properties are related to eigenvalues in the following way.

:::{prf:proposition} Eigenvalues of Positive Definite Matrices 
:label: trm-psd-eigenvalues
:nonumber:
A symmetric matrix is positive semi-definite if and only if all of its
eigenvalues are nonnegative, and positive definite if and only if all of
its eigenvalues are positive.
:::

:::{prf:proof}
Suppose $A$ is positive semi-definite, and let $\mathbf{x}$ be
an eigenvector of $\mathbf{A}$ with eigenvalue $\lambda$.
Then

$$0 \leq \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} = \mathbf{x}^{\!\top\!}(\lambda\mathbf{x}) = \lambda\mathbf{x}^{\!\top\!}\mathbf{x} = \lambda\|\mathbf{x}\|_2^2$$

Since $\mathbf{x} \neq \mathbf{0}$ (by the assumption that it is an
eigenvector), we have $\|\mathbf{x}\|_2^2 > 0$, so we can divide both
sides by $\|\mathbf{x}\|_2^2$ to arrive at $\lambda \geq 0$.

If $\mathbf{A}$ is positive definite, the inequality above holds strictly,
so $\lambda > 0$.

This proves one direction.

To simplify the proof of the other direction, we will use the machinery
of Rayleigh quotients.

Suppose that $\mathbf{A}$ is symmetric and all
its eigenvalues are nonnegative.

Then for all
$\mathbf{x} \neq \mathbf{0}$,

$$0 \leq \lambda_{\min}(\mathbf{A}) \leq R_\mathbf{A}(\mathbf{x})$$

Since $\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}$ matches
$R_\mathbf{A}(\mathbf{x})$ in sign, we conclude that $\mathbf{A}$ is
positive semi-definite. 

If the eigenvalues of $\mathbf{A}$ are all
strictly positive, then $0 < \lambda_{\min}(\mathbf{A})$, whence it
follows that $\mathbf{A}$ is positive definite. ◻
:::

## Gram matrices
In many machine learning algorithms, especially those involving regression, classification, or kernel methods, we frequently work with **data matrices** $\mathbf{A} \in \mathbb{R}^{m \times n}$, where each **row** represents a sample and each **column** a feature. From such matrices, we often compute **matrices of inner products** like $\mathbf{A}^\top \mathbf{A}$. These matrices — called **Gram matrices** — encode the pairwise **similarity between features** (or, in kernelized settings, between samples), and play a central role in optimization problems such as least squares, ridge regression, and principal component analysis.

:::{prf:proposition} Gram Matrices
:label: trm-gram-matrices
:nonumber:

Suppose $\mathbf{A} \in \mathbb{R}^{m \times n}$.

Then $\mathbf{A}^{\!\top\!}\mathbf{A}$ is positive semi-definite.

If $\operatorname{null}(\mathbf{A}) = \{\mathbf{0}\}$, then
$\mathbf{A}^{\!\top\!}\mathbf{A}$ is positive definite.
:::

:::{prf:proof}

For any $\mathbf{x} \in \mathbb{R}^n$,

$$\mathbf{x}^{\!\top\!} (\mathbf{A}^{\!\top\!}\mathbf{A})\mathbf{x} = (\mathbf{A}\mathbf{x})^{\!\top\!}(\mathbf{A}\mathbf{x}) = \|\mathbf{A}\mathbf{x}\|_2^2 \geq 0$$

so $\mathbf{A}^{\!\top\!}\mathbf{A}$ is positive semi-definite.

Note that $\|\mathbf{A}\mathbf{x}\|_2^2 = 0$ implies
$\|\mathbf{A}\mathbf{x}\|_2 = 0$, which in turn implies
$\mathbf{A}\mathbf{x} = \mathbf{0}$ (recall that this is a property of
norms).

If $\operatorname{null}(\mathbf{A}) = \{\mathbf{0}\}$,
$\mathbf{A}\mathbf{x} = \mathbf{0}$ implies $\mathbf{x} = \mathbf{0}$,
so
$\mathbf{x}^{\!\top\!} (\mathbf{A}^{\!\top\!}\mathbf{A})\mathbf{x} = 0$
if and only if $\mathbf{x} = \mathbf{0}$, and thus
$\mathbf{A}^{\!\top\!}\mathbf{A}$ is positive definite. ◻
:::

We observe that kernel matrices computed for all pairs of instances in a data set are positive semi definite. In fact, many kernel functions, like for example the RBF kernel, guarantee positive definiteness of the kernel matrix as long as all data points are pairwise distinct.


## Invertibility

Positive definite matrices are invertible (since their eigenvalues are
nonzero), whereas positive semi-definite matrices might not be. 

However, if you already have a positive semi-definite matrix, it is possible to
perturb its diagonal slightly to produce a positive definite matrix.

:::{prf:proposition}
:label: trm-A-plus-eps
:nonumber:

If $\mathbf{A}$ is positive semi-definite and $\epsilon > 0$, then
$\mathbf{A} + \epsilon\mathbf{I}$ is positive definite.
:::

:::{prf:proof}
Assuming $\mathbf{A}$ is positive semi-definite and
$\epsilon > 0$, we have for any $\mathbf{x} \neq \mathbf{0}$ that

$$\mathbf{x}^{\!\top\!}(\mathbf{A}+\epsilon\mathbf{I})\mathbf{x} = \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} + \epsilon\mathbf{x}^{\!\top\!}\mathbf{I}\mathbf{x} = \underbrace{\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}}_{\geq 0} + \underbrace{\epsilon\|\mathbf{x}\|_2^2}_{> 0} > 0$$

as claimed. ◻
:::

An obvious but frequently useful consequence of the two propositions we
have just shown is that
$\mathbf{A}^{\!\top\!}\mathbf{A} + \epsilon\mathbf{I}$ is positive
definite (and in particular, invertible) for *any* matrix $\mathbf{A}$
and any $\epsilon > 0$.

## The geometry of positive definite quadratic forms

A useful way to understand quadratic forms is by the geometry of their
level sets. 
A **level set** or **isocontour** of a function is the set
of all inputs such that the function applied to those inputs yields a
given output.

Mathematically, the $c$-isocontour of $f$ is
$\{\mathbf{x} \in \operatorname{dom} f : f(\mathbf{x}) = c\}$.

Let us consider the special case
$f(\mathbf{x}) = \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}$ where
$\mathbf{A}$ is a positive definite matrix.

Since $\mathbf{A}$ is
positive definite, it has a unique matrix square root
$\mathbf{A}^{\frac{1}{2}} = \mathbf{Q}\mathbf{\Lambda}^{\frac{1}{2}}\mathbf{Q}^{\!\top\!}$,
where $\mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{\!\top\!}$ is the
eigendecomposition of $\mathbf{A}$ and
$\mathbf{\Lambda}^{\frac{1}{2}} = \operatorname{diag}(\sqrt{\lambda_1}, \dots \sqrt{\lambda_n})$.

It is easy to see that this matrix $\mathbf{A}^{\frac{1}{2}}$ is
positive definite (consider its eigenvalues) and satisfies
$\mathbf{A}^{\frac{1}{2}}\mathbf{A}^{\frac{1}{2}} = \mathbf{A}$. Fixing
a value $c \geq 0$, the $c$-isocontour of $f$ is the set of
$\mathbf{x} \in \mathbb{R}^n$ such that

$$c = \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} = \mathbf{x}^{\!\top\!}\mathbf{A}^{\frac{1}{2}}\mathbf{A}^{\frac{1}{2}}\mathbf{x} = \|\mathbf{A}^{\frac{1}{2}}\mathbf{x}\|_2^2$$

where we have used the symmetry of $\mathbf{A}^{\frac{1}{2}}$. Making
the change of variable
$\mathbf{z} = \mathbf{A}^{\frac{1}{2}}\mathbf{x}$, we have the condition
$\|\mathbf{z}\|_2 = \sqrt{c}$. 

That is, the values $\mathbf{z}$ lie on a
sphere of radius $\sqrt{c}$. 

These can be parameterized as
$\mathbf{z} = \sqrt{c}\hat{\mathbf{z}}$ where $\hat{\mathbf{z}}$ has
$\|\hat{\mathbf{z}}\|_2 = 1$. 

Then since
$\mathbf{A}^{-\frac{1}{2}} = \mathbf{Q}\mathbf{\Lambda}^{-\frac{1}{2}}\mathbf{Q}^{\!\top\!}$,
we have

$$\mathbf{x} = \mathbf{A}^{-\frac{1}{2}}\mathbf{z} = \mathbf{Q}\mathbf{\Lambda}^{-\frac{1}{2}}\mathbf{Q}^{\!\top\!}\sqrt{c}\hat{\mathbf{z}} = \sqrt{c}\mathbf{Q}\mathbf{\Lambda}^{-\frac{1}{2}}\tilde{\mathbf{z}}$$

where $\tilde{\mathbf{z}} = \mathbf{Q}^{\!\top\!}\hat{\mathbf{z}}$ also
satisfies $\|\tilde{\mathbf{z}}\|_2 = 1$ since $\mathbf{Q}$ is
orthogonal. 

Using this parameterization, we see that the solution set
$\{\mathbf{x} \in \mathbb{R}^n : f(\mathbf{x}) = c\}$ is the image of
the unit sphere
$\{\tilde{\mathbf{z}} \in \mathbb{R}^n : \|\tilde{\mathbf{z}}\|_2 = 1\}$
under the invertible linear map
$\mathbf{x} = \sqrt{c}\mathbf{Q}\mathbf{\Lambda}^{-\frac{1}{2}}\tilde{\mathbf{z}}$.

What we have gained with all these manipulations is a clear algebraic
understanding of the $c$-isocontour of $f$ in terms of a sequence of
linear transformations applied to a well-understood set. We begin with
the unit sphere, then scale every axis $i$ by
$\lambda_i^{-\frac{1}{2}}$, resulting in an axis-aligned ellipsoid.
Observe that the axis lengths of the ellipsoid are proportional to the
inverse square roots of the eigenvalues of $\mathbf{A}$. Hence larger
eigenvalues correspond to shorter axis lengths, and vice-versa.

Then this axis-aligned ellipsoid undergoes a rigid transformation (i.e.
one that preserves length and angles, such as a rotation/reflection)
given by $\mathbf{Q}$. The result of this transformation is that the
axes of the ellipse are no longer along the coordinate axes in general,
but rather along the directions given by the corresponding eigenvectors.
To see this, consider the unit vector $\mathbf{e}_i \in \mathbb{R}^n$
that has $[\mathbf{e}_i]_j = \delta_{ij}$. In the pre-transformed space,
this vector points along the axis with length proportional to
$\lambda_i^{-\frac{1}{2}}$. But after applying the rigid transformation
$\mathbf{Q}$, the resulting vector points in the direction of the
corresponding eigenvector $\mathbf{q}_i$, since

$$\mathbf{Q}\mathbf{e}_i = \sum_{j=1}^n [\mathbf{e}_i]_j\mathbf{q}_j = \mathbf{q}_i$$

where we have used the matrix-vector product identity from earlier.

**In summary:** the isocontours of
$f(\mathbf{x}) = \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}$ are
ellipsoids such that the axes point in the directions of the
eigenvectors of $\mathbf{A}$, and the radii of these axes are
proportional to the inverse square roots of the corresponding
eigenvalues.


To demonstrate the eigenvalue decomposition of a positive semi-definite matrix, we will be looking at Principal Component Analysis (PCA) algorithm in the next section. The algorithm is a technique used for applications like dimensionality reduction, lossy data compression, feature extraction and data visualization. 

