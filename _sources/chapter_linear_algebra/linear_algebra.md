# Linear Algebra

In this section we present important classes of spaces in which our data
will live and our operations will take place: vector spaces, metric
spaces, normed spaces, and inner product spaces. Generally speaking,
these are defined in such a way as to capture one or more important
properties of Euclidean space but in a more general way.

## Vector spaces

**Vector spaces** are the basic setting in which linear algebra happens.
A vector space $V$ is a set (the elements of which are called
**vectors**) on which two operations are defined: vectors can be added
together, and vectors can be multiplied by real numbers[^1] called
**scalars**. $V$ must satisfy

(i) There exists an additive identity (written $\mathbf{0}$) in $V$ such
    that $\mathbf{x}+\mathbf{0} = \mathbf{x}$ for all $\mathbf{x} \in V$

(ii) For each $\mathbf{x} \in V$, there exists an additive inverse
     (written $\mathbf{-x}$) such that
     $\mathbf{x}+(\mathbf{-x}) = \mathbf{0}$

(iii) There exists a multiplicative identity (written $1$) in
      $\mathbb{R}$ such that $1\mathbf{x} = \mathbf{x}$ for all
      $\mathbf{x} \in V$

(iv) Commutativity: $\mathbf{x}+\mathbf{y} = \mathbf{y}+\mathbf{x}$ for
     all $\mathbf{x}, \mathbf{y} \in V$

(v) Associativity:
    $(\mathbf{x}+\mathbf{y})+\mathbf{z} = \mathbf{x}+(\mathbf{y}+\mathbf{z})$
    and $\alpha(\beta\mathbf{x}) = (\alpha\beta)\mathbf{x}$ for all
    $\mathbf{x}, \mathbf{y}, \mathbf{z} \in V$ and
    $\alpha, \beta \in \mathbb{R}$

(vi) Distributivity:
     $\alpha(\mathbf{x}+\mathbf{y}) = \alpha\mathbf{x} + \alpha\mathbf{y}$
     and $(\alpha+\beta)\mathbf{x} = \alpha\mathbf{x} + \beta\mathbf{x}$
     for all $\mathbf{x}, \mathbf{y} \in V$ and
     $\alpha, \beta \in \mathbb{R}$

### Euclidean space

The quintessential vector space is **Euclidean space**, which we denote
$\mathbb{R}^n$. The vectors in this space consist of $n$-tuples of real
numbers: $$\mathbf{x} = (x_1, x_2, \dots, x_n)$$ For our purposes, it
will be useful to think of them as $n \times 1$ matrices, or **column
vectors**:
$$\mathbf{x} = \begin{bmatrix}x_1 \\ x_2 \\ \vdots \\ x_n\end{bmatrix}$$
Addition and scalar multiplication are defined component-wise on vectors
in $\mathbb{R}^n$:
$$\mathbf{x} + \mathbf{y} = \begin{bmatrix}x_1 + y_1 \\ \vdots \\ x_n + y_n\end{bmatrix}, \hspace{0.5cm} \alpha\mathbf{x} = \begin{bmatrix}\alpha x_1 \\ \vdots \\ \alpha x_n\end{bmatrix}$$
Euclidean space is used to mathematically represent physical space, with
notions such as distance, length, and angles. Although it becomes hard
to visualize for $n > 3$, these concepts generalize mathematically in
obvious ways. Even when you're working in more general settings than
$\mathbb{R}^n$, it is often useful to visualize vector addition and
scalar multiplication in terms of 2D vectors in the plane or 3D vectors
in space.

### Subspaces

Vector spaces can contain other vector spaces. If $V$ is a vector space,
then $S \subseteq V$ is said to be a **subspace** of $V$ if

(i) $\mathbf{0} \in S$

(ii) $S$ is closed under addition: $\mathbf{x}, \mathbf{y} \in S$
     implies $\mathbf{x}+\mathbf{y} \in S$

(iii) $S$ is closed under scalar multiplication:
      $\mathbf{x} \in S, \alpha \in \mathbb{R}$ implies
      $\alpha\mathbf{x} \in S$

Note that $V$ is always a subspace of $V$, as is the trivial vector
space which contains only $\mathbf{0}$.

As a concrete example, a line passing through the origin is a subspace
of Euclidean space.

Some of the most important subspaces are those induced by linear maps.
If $T : V \to W$ is a linear map, we define the **nullspace**[^2] of $T$
as $$\Null(T) = \{\mathbf{x} \in V \mid T\mathbf{x} = \mathbf{0}\}$$ and
the **range** (or the **columnspace** if we are considering the matrix
form) of $T$ as
$$\range(T) = \{\mathbf{y} \in W \mid \text{$\exists \mathbf{x} \in V$ such that $T\mathbf{x} = \mathbf{y}$}\}$$
It is a good exercise to verify that the nullspace and range of a linear
map are always subspaces of its domain and codomain, respectively.

## Metric spaces

Metrics generalize the notion of distance from Euclidean space (although
metric spaces need not be vector spaces).

A **metric** on a set $S$ is a function $d : S \times S \to \mathbb{R}$
that satisfies

(i) $d(x,y) \geq 0$, with equality if and only if $x = y$

(ii) $d(x,y) = d(y,x)$

(iii) $d(x,z) \leq d(x,y) + d(y,z)$ (the so-called **triangle
      inequality**)

for all $x, y, z \in S$.

A key motivation for metrics is that they allow limits to be defined for
mathematical objects other than real numbers. We say that a sequence
$\{x_n\} \subseteq S$ converges to the limit $x$ if for any
$\epsilon > 0$, there exists $N \in \mathbb{N}$ such that
$d(x_n, x) < \epsilon$ for all $n \geq N$. Note that the definition for
limits of sequences of real numbers, which you have likely seen in a
calculus class, is a special case of this definition when using the
metric $d(x, y) = |x-y|$.

## Normed spaces

Norms generalize the notion of length from Euclidean space.

A **norm** on a real vector space $V$ is a function
$\|\cdot\| : V \to \mathbb{R}$ that satisfies

(i) $\|\mathbf{x}\| \geq 0$, with equality if and only if
    $\mathbf{x} = \mathbf{0}$

(ii) $\|\alpha\mathbf{x}\| = |\alpha|\|\mathbf{x}\|$

(iii) $\|\mathbf{x}+\mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$
      (the **triangle inequality** again)

for all $\mathbf{x}, \mathbf{y} \in V$ and all $\alpha \in \mathbb{R}$.
A vector space endowed with a norm is called a **normed vector space**,
or simply a **normed space**.

Note that any norm on $V$ induces a distance metric on $V$:
$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x}-\mathbf{y}\|$$ One can verify
that the axioms for metrics are satisfied under this definition and
follow directly from the axioms for norms. Therefore any normed space is
also a metric space.[^3]

We will typically only be concerned with a few specific norms on
$\mathbb{R}^n$: $$\begin{aligned}
\|\mathbf{x}\|_1 &= \sum_{i=1}^n |x_i| \\
\|\mathbf{x}\|_2 &= \sqrt{\sum_{i=1}^n x_i^2} \\
\|\mathbf{x}\|_p &= \left(\sum_{i=1}^n |x_i|^p\right)^\frac{1}{p} \hspace{0.5cm}\hspace{0.5cm} (p \geq 1) \\
\|\mathbf{x}\|_\infty &= \max_{1 \leq i \leq n} |x_i|
\end{aligned}$$ Note that the 1- and 2-norms are special cases of the
$p$-norm, and the $\infty$-norm is the limit of the $p$-norm as $p$
tends to infinity. We require $p \geq 1$ for the general definition of
the $p$-norm because the triangle inequality fails to hold if $p < 1$.
(Try to find a counterexample!)

Here's a fun fact: for any given finite-dimensional vector space $V$,
all norms on $V$ are equivalent in the sense that for two norms
$\|\cdot\|_A, \|\cdot\|_B$, there exist constants $\alpha, \beta > 0$
such that
$$\alpha\|\mathbf{x}\|_A \leq \|\mathbf{x}\|_B \leq \beta\|\mathbf{x}\|_A$$
for all $\mathbf{x} \in V$. Therefore convergence in one norm implies
convergence in any other norm. This rule may not apply in
infinite-dimensional vector spaces such as function spaces, though.

## Inner product spaces

An **inner product** on a real vector space $V$ is a function
$\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$ satisfying

(i) $\langle \mathbf{x}, \mathbf{x} \rangle \geq 0$, with equality if
    and only if $\mathbf{x} = \mathbf{0}$

(ii) $\langle \alpha\mathbf{x} + \beta\mathbf{y}, \mathbf{z} \rangle = \alpha\langle \mathbf{x}, \mathbf{z} \rangle + \beta\langle \mathbf{y}, \mathbf{z} \rangle$

(iii) $\langle \mathbf{x}, \mathbf{y} \rangle = \langle \mathbf{y}, \mathbf{x} \rangle$

for all $\mathbf{x}, \mathbf{y}, \mathbf{z} \in V$ and all
$\alpha,\beta \in \mathbb{R}$. A vector space endowed with an inner
product is called an **inner product space**.

Note that any inner product on $V$ induces a norm on $V$:
$$\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$$ One
can verify that the axioms for norms are satisfied under this definition
and follow (almost) directly from the axioms for inner products.
Therefore any inner product space is also a normed space (and hence also
a metric space).[^4]

Two vectors $\mathbf{x}$ and $\mathbf{y}$ are said to be **orthogonal**
if $\langle \mathbf{x}, \mathbf{y} \rangle = 0$; we write
$\mathbf{x} \perp \mathbf{y}$ for shorthand. Orthogonality generalizes
the notion of perpendicularity from Euclidean space. If two orthogonal
vectors $\mathbf{x}$ and $\mathbf{y}$ additionally have unit length
(i.e. $\|\mathbf{x}\| = \|\mathbf{y}\| = 1$), then they are described as
**orthonormal**.

The standard inner product on $\mathbb{R}^n$ is given by
$$\langle \mathbf{x}, \mathbf{y} \rangle = \sum_{i=1}^n x_iy_i = \mathbf{x}^{\!\top\!}\mathbf{y}$$
The matrix notation on the righthand side (see the Transposition section
if it's unfamiliar) arises because this inner product is a special case
of matrix multiplication where we regard the resulting $1 \times 1$
matrix as a scalar. The inner product on $\mathbb{R}^n$ is also often
written $\mathbf{x}\cdot\mathbf{y}$ (hence the alternate name **dot
product**). The reader can verify that the two-norm $\|\cdot\|_2$ on
$\mathbb{R}^n$ is induced by this inner product.

### Pythagorean Theorem

The well-known Pythagorean theorem generalizes naturally to arbitrary
inner product spaces.

::: theorem
If $\mathbf{x} \perp \mathbf{y}$, then
$$\|\mathbf{x}+\mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2$$
:::

::: proof
*Proof.* Suppose $\mathbf{x} \perp \mathbf{y}$, i.e.
$\langle \mathbf{x}, \mathbf{y} \rangle = 0$. Then
$$\|\mathbf{x}+\mathbf{y}\|^2 = \langle \mathbf{x}+\mathbf{y}, \mathbf{x}+\mathbf{y} \rangle = \langle \mathbf{x}, \mathbf{x} \rangle + \langle \mathbf{y}, \mathbf{x} \rangle + \langle \mathbf{x}, \mathbf{y} \rangle + \langle \mathbf{y}, \mathbf{y} \rangle = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2$$
as claimed. ◻
:::

### Cauchy-Schwarz inequality

This inequality is sometimes useful in proving bounds:
$$|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \|\mathbf{x}\| \cdot \|\mathbf{y}\|$$
for all $\mathbf{x}, \mathbf{y} \in V$. Equality holds exactly when
$\mathbf{x}$ and $\mathbf{y}$ are scalar multiples of each other (or
equivalently, when they are linearly dependent).

## Transposition

If $\mathbf{A} \in \mathbb{R}^{m \times n}$, its **transpose**
$\mathbf{A}^{\!\top\!} \in \mathbb{R}^{n \times m}$ is given by
$(\mathbf{A}^{\!\top\!})_{ij} = A_{ji}$ for each $(i, j)$. In other
words, the columns of $\mathbf{A}$ become the rows of
$\mathbf{A}^{\!\top\!}$, and the rows of $\mathbf{A}$ become the columns
of $\mathbf{A}^{\!\top\!}$.

The transpose has several nice algebraic properties that can be easily
verified from the definition:

(i) $(\mathbf{A}^{\!\top\!})^{\!\top\!} = \mathbf{A}$

(ii) $(\mathbf{A}+\mathbf{B})^{\!\top\!} = \mathbf{A}^{\!\top\!} + \mathbf{B}^{\!\top\!}$

(iii) $(\alpha \mathbf{A})^{\!\top\!} = \alpha \mathbf{A}^{\!\top\!}$

(iv) $(\mathbf{A}\mathbf{B})^{\!\top\!} = \mathbf{B}^{\!\top\!} \mathbf{A}^{\!\top\!}$

## Eigenthings

For a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, there may
be vectors which, when $\mathbf{A}$ is applied to them, are simply
scaled by some constant. We say that a nonzero vector
$\mathbf{x} \in \mathbb{R}^n$ is an **eigenvector** of $\mathbf{A}$
corresponding to **eigenvalue** $\lambda$ if
$$\mathbf{A}\mathbf{x} = \lambda\mathbf{x}$$ The zero vector is excluded
from this definition because
$\mathbf{A}\mathbf{0} = \mathbf{0} = \lambda\mathbf{0}$ for every
$\lambda$.

We now give some useful results about how eigenvalues change after
various manipulations.

::: proposition
Let $\mathbf{x}$ be an eigenvector of $\mathbf{A}$ with corresponding
eigenvalue $\lambda$. Then

(i) For any $\gamma \in \mathbb{R}$, $\mathbf{x}$ is an eigenvector of
    $\mathbf{A} + \gamma\mathbf{I}$ with eigenvalue $\lambda + \gamma$.

(ii) If $\mathbf{A}$ is invertible, then $\mathbf{x}$ is an eigenvector
     of $\mathbf{A}^{-1}$ with eigenvalue $\lambda^{-1}$.

(iii) $\mathbf{A}^k\mathbf{x} = \lambda^k\mathbf{x}$ for any
      $k \in \mathbb{Z}$ (where $\mathbf{A}^0 = \mathbf{I}$ by
      definition).
:::

::: proof
*Proof.* (i) follows readily:
$$(\mathbf{A} + \gamma\mathbf{I})\mathbf{x} = \mathbf{A}\mathbf{x} + \gamma\mathbf{I}\mathbf{x} = \lambda\mathbf{x} + \gamma\mathbf{x} = (\lambda + \gamma)\mathbf{x}$$

\(ii\) Suppose $\mathbf{A}$ is invertible. Then
$$\mathbf{x} = \mathbf{A}^{-1}\mathbf{A}\mathbf{x} = \mathbf{A}^{-1}(\lambda\mathbf{x}) = \lambda\mathbf{A}^{-1}\mathbf{x}$$
Dividing by $\lambda$, which is valid because the invertibility of
$\mathbf{A}$ implies $\lambda \neq 0$, gives
$\lambda^{-1}\mathbf{x} = \mathbf{A}^{-1}\mathbf{x}$.

\(iii\) The case $k \geq 0$ follows immediately by induction on $k$.
Then the general case $k \in \mathbb{Z}$ follows by combining the
$k \geq 0$ case with (ii). ◻
:::

## Trace

The **trace** of a square matrix is the sum of its diagonal entries:
$$\tr(\mathbf{A}) = \sum_{i=1}^n A_{ii}$$ The trace has several nice
algebraic properties:

(i) $\tr(\mathbf{A}+\mathbf{B}) = \tr(\mathbf{A}) + \tr(\mathbf{B})$

(ii) $\tr(\alpha\mathbf{A}) = \alpha\tr(\mathbf{A})$

(iii) $\tr(\mathbf{A}^{\!\top\!}) = \tr(\mathbf{A})$

(iv) $\tr(\mathbf{A}\mathbf{B}\mathbf{C}\mathbf{D}) = \tr(\mathbf{B}\mathbf{C}\mathbf{D}\mathbf{A}) = \tr(\mathbf{C}\mathbf{D}\mathbf{A}\mathbf{B}) = \tr(\mathbf{D}\mathbf{A}\mathbf{B}\mathbf{C})$

The first three properties follow readily from the definition. The last
is known as **invariance under cyclic permutations**. Note that the
matrices cannot be reordered arbitrarily, for example
$\tr(\mathbf{A}\mathbf{B}\mathbf{C}\mathbf{D}) \neq \tr(\mathbf{B}\mathbf{A}\mathbf{C}\mathbf{D})$
in general. Also, there is nothing special about the product of four
matrices -- analogous rules hold for more or fewer matrices.

Interestingly, the trace of a matrix is equal to the sum of its
eigenvalues (repeated according to multiplicity):
$$\tr(\mathbf{A}) = \sum_i \lambda_i(\mathbf{A})$$

## Determinant

The **determinant** of a square matrix can be defined in several
different confusing ways, none of which are particularly important for
our purposes; go look at an introductory linear algebra text (or
Wikipedia) if you need a definition. But it's good to know the
properties:

(i) $\det(\mathbf{I}) = 1$

(ii) $\det(\mathbf{A}^{\!\top\!}) = \det(\mathbf{A})$

(iii) $\det(\mathbf{A}\mathbf{B}) = \det(\mathbf{A})\det(\mathbf{B})$

(iv) $\det(\mathbf{A}^{-1}) = \det(\mathbf{A})^{-1}$

(v) $\det(\alpha\mathbf{A}) = \alpha^n \det(\mathbf{A})$

Interestingly, the determinant of a matrix is equal to the product of
its eigenvalues (repeated according to multiplicity):
$$\det(\mathbf{A}) = \prod_i \lambda_i(\mathbf{A})$$

## Orthogonal matrices

A matrix $\mathbf{Q} \in \mathbb{R}^{n \times n}$ is said to be
**orthogonal** if its columns are pairwise orthonormal. This definition
implies that
$$\mathbf{Q}^{\!\top\!} \mathbf{Q} = \mathbf{Q}\mathbf{Q}^{\!\top\!} = \mathbf{I}$$
or equivalently, $\mathbf{Q}^{\!\top\!} = \mathbf{Q}^{-1}$. A nice thing
about orthogonal matrices is that they preserve inner products:
$$(\mathbf{Q}\mathbf{x})^{\!\top\!}(\mathbf{Q}\mathbf{y}) = \mathbf{x}^{\!\top\!} \mathbf{Q}^{\!\top\!} \mathbf{Q}\mathbf{y} = \mathbf{x}^{\!\top\!} \mathbf{I}\mathbf{y} = \mathbf{x}^{\!\top\!}\mathbf{y}$$
A direct result of this fact is that they also preserve 2-norms:
$$\|\mathbf{Q}\mathbf{x}\|_2 = \sqrt{(\mathbf{Q}\mathbf{x})^{\!\top\!}(\mathbf{Q}\mathbf{x})} = \sqrt{\mathbf{x}^{\!\top\!}\mathbf{x}} = \|\mathbf{x}\|_2$$
Therefore multiplication by an orthogonal matrix can be considered as a
transformation that preserves length, but may rotate or reflect the
vector about the origin.

## Symmetric matrices

A matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is said to be
**symmetric** if it is equal to its own transpose
($\mathbf{A} = \mathbf{A}^{\!\top\!}$), meaning that $A_{ij} = A_{ji}$
for all $(i,j)$. This definition seems harmless enough but turns out to
have some strong implications. We summarize the most important of these
as

::: theorem
(Spectral Theorem) If $\mathbf{A} \in \mathbb{R}^{n \times n}$ is
symmetric, then there exists an orthonormal basis for $\mathbb{R}^n$
consisting of eigenvectors of $\mathbf{A}$.
:::

The practical application of this theorem is a particular factorization
of symmetric matrices, referred to as the **eigendecomposition** or
**spectral decomposition**. Denote the orthonormal basis of eigenvectors
$\mathbf{q}_1, \dots, \mathbf{q}_n$ and their eigenvalues
$\lambda_1, \dots, \lambda_n$. Let $\mathbf{Q}$ be an orthogonal matrix
with $\mathbf{q}_1, \dots, \mathbf{q}_n$ as its columns, and
$\mathbf{\Lambda} = \diag(\lambda_1, \dots, \lambda_n)$. Since by
definition $\mathbf{A}\mathbf{q}_i = \lambda_i\mathbf{q}_i$ for every
$i$, the following relationship holds:
$$\mathbf{A}\mathbf{Q} = \mathbf{Q}\mathbf{\Lambda}$$ Right-multiplying
by $\mathbf{Q}^{\!\top\!}$, we arrive at the decomposition
$$\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{\!\top\!}$$

### Rayleigh quotients

Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ be a symmetric matrix. The
expression $\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}$ is called a
**quadratic form**.

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

::: proposition
For any $\mathbf{x}$ such that $\|\mathbf{x}\|_2 = 1$,
$$\lambda_{\min}(\mathbf{A}) \leq \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} \leq \lambda_{\max}(\mathbf{A})$$
with equality if and only if $\mathbf{x}$ is a corresponding
eigenvector.
:::

::: proof
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
:::

By the scale invariance of the Rayleigh quotient, we immediately have as
a corollary (since
$\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} = R_{\mathbf{A}}(\mathbf{x})$
for unit $\mathbf{x}$)

::: theorem
(Min-max theorem) For all $\mathbf{x} \neq \mathbf{0}$,
$$\lambda_{\min}(\mathbf{A}) \leq R_\mathbf{A}(\mathbf{x}) \leq \lambda_{\max}(\mathbf{A})$$
with equality if and only if $\mathbf{x}$ is a corresponding
eigenvector.
:::

## Positive (semi-)definite matrices

A symmetric matrix $\mathbf{A}$ is **positive semi-definite** if for all
$\mathbf{x} \in \mathbb{R}^n$,
$\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} \geq 0$. Sometimes people
write $\mathbf{A} \succeq 0$ to indicate that $\mathbf{A}$ is positive
semi-definite.

A symmetric matrix $\mathbf{A}$ is **positive definite** if for all
nonzero $\mathbf{x} \in \mathbb{R}^n$,
$\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} > 0$. Sometimes people write
$\mathbf{A} \succ 0$ to indicate that $\mathbf{A}$ is positive definite.
Note that positive definiteness is a strictly stronger property than
positive semi-definiteness, in the sense that every positive definite
matrix is positive semi-definite but not vice-versa.

These properties are related to eigenvalues in the following way.

::: proposition
A symmetric matrix is positive semi-definite if and only if all of its
eigenvalues are nonnegative, and positive definite if and only if all of
its eigenvalues are positive.
:::

::: proof
*Proof.* Suppose $A$ is positive semi-definite, and let $\mathbf{x}$ be
an eigenvector of $\mathbf{A}$ with eigenvalue $\lambda$. Then
$$0 \leq \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} = \mathbf{x}^{\!\top\!}(\lambda\mathbf{x}) = \lambda\mathbf{x}^{\!\top\!}\mathbf{x} = \lambda\|\mathbf{x}\|_2^2$$
Since $\mathbf{x} \neq \mathbf{0}$ (by the assumption that it is an
eigenvector), we have $\|\mathbf{x}\|_2^2 > 0$, so we can divide both
sides by $\|\mathbf{x}\|_2^2$ to arrive at $\lambda \geq 0$. If
$\mathbf{A}$ is positive definite, the inequality above holds strictly,
so $\lambda > 0$. This proves one direction.

To simplify the proof of the other direction, we will use the machinery
of Rayleigh quotients. Suppose that $\mathbf{A}$ is symmetric and all
its eigenvalues are nonnegative. Then for all
$\mathbf{x} \neq \mathbf{0}$,
$$0 \leq \lambda_{\min}(\mathbf{A}) \leq R_\mathbf{A}(\mathbf{x})$$
Since $\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}$ matches
$R_\mathbf{A}(\mathbf{x})$ in sign, we conclude that $\mathbf{A}$ is
positive semi-definite. If the eigenvalues of $\mathbf{A}$ are all
strictly positive, then $0 < \lambda_{\min}(\mathbf{A})$, whence it
follows that $\mathbf{A}$ is positive definite. ◻
:::

As an example of how these matrices arise, consider

::: proposition
Suppose $\mathbf{A} \in \mathbb{R}^{m \times n}$. Then
$\mathbf{A}^{\!\top\!}\mathbf{A}$ is positive semi-definite. If
$\Null(\mathbf{A}) = \{\mathbf{0}\}$, then
$\mathbf{A}^{\!\top\!}\mathbf{A}$ is positive definite.
:::

::: proof
*Proof.* For any $\mathbf{x} \in \mathbb{R}^n$,
$$\mathbf{x}^{\!\top\!} (\mathbf{A}^{\!\top\!}\mathbf{A})\mathbf{x} = (\mathbf{A}\mathbf{x})^{\!\top\!}(\mathbf{A}\mathbf{x}) = \|\mathbf{A}\mathbf{x}\|_2^2 \geq 0$$
so $\mathbf{A}^{\!\top\!}\mathbf{A}$ is positive semi-definite.

Note that $\|\mathbf{A}\mathbf{x}\|_2^2 = 0$ implies
$\|\mathbf{A}\mathbf{x}\|_2 = 0$, which in turn implies
$\mathbf{A}\mathbf{x} = \mathbf{0}$ (recall that this is a property of
norms). If $\Null(\mathbf{A}) = \{\mathbf{0}\}$,
$\mathbf{A}\mathbf{x} = \mathbf{0}$ implies $\mathbf{x} = \mathbf{0}$,
so
$\mathbf{x}^{\!\top\!} (\mathbf{A}^{\!\top\!}\mathbf{A})\mathbf{x} = 0$
if and only if $\mathbf{x} = \mathbf{0}$, and thus
$\mathbf{A}^{\!\top\!}\mathbf{A}$ is positive definite. ◻
:::

Positive definite matrices are invertible (since their eigenvalues are
nonzero), whereas positive semi-definite matrices might not be. However,
if you already have a positive semi-definite matrix, it is possible to
perturb its diagonal slightly to produce a positive definite matrix.

::: proposition
If $\mathbf{A}$ is positive semi-definite and $\epsilon > 0$, then
$\mathbf{A} + \epsilon\mathbf{I}$ is positive definite.
:::

::: proof
*Proof.* Assuming $\mathbf{A}$ is positive semi-definite and
$\epsilon > 0$, we have for any $\mathbf{x} \neq \mathbf{0}$ that
$$\mathbf{x}^{\!\top\!}(\mathbf{A}+\epsilon\mathbf{I})\mathbf{x} = \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} + \epsilon\mathbf{x}^{\!\top\!}\mathbf{I}\mathbf{x} = \underbrace{\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}}_{\geq 0} + \underbrace{\epsilon\|\mathbf{x}\|_2^2}_{> 0} > 0$$
as claimed. ◻
:::

An obvious but frequently useful consequence of the two propositions we
have just shown is that
$\mathbf{A}^{\!\top\!}\mathbf{A} + \epsilon\mathbf{I}$ is positive
definite (and in particular, invertible) for *any* matrix $\mathbf{A}$
and any $\epsilon > 0$.

### The geometry of positive definite quadratic forms

A useful way to understand quadratic forms is by the geometry of their
level sets. A **level set** or **isocontour** of a function is the set
of all inputs such that the function applied to those inputs yields a
given output. Mathematically, the $c$-isocontour of $f$ is
$\{\mathbf{x} \in \dom f : f(\mathbf{x}) = c\}$.

Let us consider the special case
$f(\mathbf{x}) = \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}$ where
$\mathbf{A}$ is a positive definite matrix. Since $\mathbf{A}$ is
positive definite, it has a unique matrix square root
$\mathbf{A}^{\frac{1}{2}} = \mathbf{Q}\mathbf{\Lambda}^{\frac{1}{2}}\mathbf{Q}^{\!\top\!}$,
where $\mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{\!\top\!}$ is the
eigendecomposition of $\mathbf{A}$ and
$\mathbf{\Lambda}^{\frac{1}{2}} = \diag(\sqrt{\lambda_1}, \dots \sqrt{\lambda_n})$.
It is easy to see that this matrix $\mathbf{A}^{\frac{1}{2}}$ is
positive definite (consider its eigenvalues) and satisfies
$\mathbf{A}^{\frac{1}{2}}\mathbf{A}^{\frac{1}{2}} = \mathbf{A}$. Fixing
a value $c \geq 0$, the $c$-isocontour of $f$ is the set of
$\mathbf{x} \in \mathbb{R}^n$ such that
$$c = \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} = \mathbf{x}^{\!\top\!}\mathbf{A}^{\frac{1}{2}}\mathbf{A}^{\frac{1}{2}}\mathbf{x} = \|\mathbf{A}^{\frac{1}{2}}\mathbf{x}\|_2^2$$
where we have used the symmetry of $\mathbf{A}^{\frac{1}{2}}$. Making
the change of variable
$\mathbf{z} = \mathbf{A}^{\frac{1}{2}}\mathbf{x}$, we have the condition
$\|\mathbf{z}\|_2 = \sqrt{c}$. That is, the values $\mathbf{z}$ lie on a
sphere of radius $\sqrt{c}$. These can be parameterized as
$\mathbf{z} = \sqrt{c}\hat{\mathbf{z}}$ where $\hat{\mathbf{z}}$ has
$\|\hat{\mathbf{z}}\|_2 = 1$. Then since
$\mathbf{A}^{-\frac{1}{2}} = \mathbf{Q}\mathbf{\Lambda}^{-\frac{1}{2}}\mathbf{Q}^{\!\top\!}$,
we have
$$\mathbf{x} = \mathbf{A}^{-\frac{1}{2}}\mathbf{z} = \mathbf{Q}\mathbf{\Lambda}^{-\frac{1}{2}}\mathbf{Q}^{\!\top\!}\sqrt{c}\hat{\mathbf{z}} = \sqrt{c}\mathbf{Q}\mathbf{\Lambda}^{-\frac{1}{2}}\tilde{\mathbf{z}}$$
where $\tilde{\mathbf{z}} = \mathbf{Q}^{\!\top\!}\hat{\mathbf{z}}$ also
satisfies $\|\tilde{\mathbf{z}}\|_2 = 1$ since $\mathbf{Q}$ is
orthogonal. Using this parameterization, we see that the solution set
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

In summary: the isocontours of
$f(\mathbf{x}) = \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}$ are
ellipsoids such that the axes point in the directions of the
eigenvectors of $\mathbf{A}$, and the radii of these axes are
proportional to the inverse square roots of the corresponding
eigenvalues.

## Singular value decomposition

Singular value decomposition (SVD) is a widely applicable tool in linear
algebra. Its strength stems partially from the fact that *every matrix*
$\mathbf{A} \in \mathbb{R}^{m \times n}$ has an SVD (even non-square
matrices)! The decomposition goes as follows:
$$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\!\top\!}$$ where
$\mathbf{U} \in \mathbb{R}^{m \times m}$ and
$\mathbf{V} \in \mathbb{R}^{n \times n}$ are orthogonal matrices and
$\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$ is a diagonal matrix with
the **singular values** of $\mathbf{A}$ (denoted $\sigma_i$) on its
diagonal.

By convention, the singular values are given in non-increasing order,
i.e.
$$\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_{\min(m,n)} \geq 0$$
Only the first $r$ singular values are nonzero, where $r$ is the rank of
$\mathbf{A}$.

Observe that the SVD factors provide eigendecompositions for
$\mathbf{A}^{\!\top\!}\mathbf{A}$ and $\mathbf{A}\mathbf{A}^{\!\top\!}$:
$$\begin{aligned}
\mathbf{A}^{\!\top\!}\mathbf{A} &= (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\!\top\!})^{\!\top\!}\mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\!\top\!} = \mathbf{V}\mathbf{\Sigma}^{\!\top\!}\mathbf{U}^{\!\top\!}\mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\!\top\!} = \mathbf{V}\mathbf{\Sigma}^{\!\top\!}\mathbf{\Sigma}\mathbf{V}^{\!\top\!} \\
\mathbf{A}\mathbf{A}^{\!\top\!} &= \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\!\top\!}(\mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\!\top\!})^{\!\top\!} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\!\top\!}\mathbf{V}\mathbf{\Sigma}^{\!\top\!}\mathbf{U}^{\!\top\!} = \mathbf{U}\mathbf{\Sigma}\mathbf{\Sigma}^{\!\top\!}\mathbf{U}^{\!\top\!}
\end{aligned}$$ It follows immediately that the columns of $\mathbf{V}$
(the **right-singular vectors** of $\mathbf{A}$) are eigenvectors of
$\mathbf{A}^{\!\top\!}\mathbf{A}$, and the columns of $\mathbf{U}$ (the
**left-singular vectors** of $\mathbf{A}$) are eigenvectors of
$\mathbf{A}\mathbf{A}^{\!\top\!}$.

The matrices $\mathbf{\Sigma}^{\!\top\!}\mathbf{\Sigma}$ and
$\mathbf{\Sigma}\mathbf{\Sigma}^{\!\top\!}$ are not necessarily the same
size, but both are diagonal with the squared singular values
$\sigma_i^2$ on the diagonal (plus possibly some zeros). Thus the
singular values of $\mathbf{A}$ are the square roots of the eigenvalues
of $\mathbf{A}^{\!\top\!}\mathbf{A}$ (or equivalently, of
$\mathbf{A}\mathbf{A}^{\!\top\!}$)[^5].

## Some useful matrix identities

### Matrix-vector product as linear combination of matrix columns

::: proposition
Let $\mathbf{x} \in \mathbb{R}^n$ be a vector and
$\mathbf{A} \in \mathbb{R}^{m \times n}$ a matrix with columns
$\mathbf{a}_1, \dots, \mathbf{a}_n$. Then
$$\mathbf{A}\mathbf{x} = \sum_{i=1}^n x_i\mathbf{a}_i$$
:::

This identity is extremely useful in understanding linear operators in
terms of their matrices' columns. The proof is very simple (consider
each element of $\mathbf{A}\mathbf{x}$ individually and expand by
definitions) but it is a good exercise to convince yourself.

### Sum of outer products as matrix-matrix product

An **outer product** is an expression of the form
$\mathbf{a}\mathbf{b}^{\!\top\!}$, where $\mathbf{a} \in \mathbb{R}^m$
and $\mathbf{b} \in \mathbb{R}^n$. By inspection it is not hard to see
that such an expression yields an $m \times n$ matrix such that
$$[\mathbf{a}\mathbf{b}^{\!\top\!}]_{ij} = a_ib_j$$ It is not
immediately obvious, but the sum of outer products is actually
equivalent to an appropriate matrix-matrix product! We formalize this
statement as

::: proposition
Let $\mathbf{a}_1, \dots, \mathbf{a}_k \in \mathbb{R}^m$ and
$\mathbf{b}_1, \dots, \mathbf{b}_k \in \mathbb{R}^n$. Then
$$\sum_{\ell=1}^k \mathbf{a}_\ell\mathbf{b}_\ell^{\!\top\!} = \mathbf{A}\mathbf{B}^{\!\top\!}$$
where
$$\mathbf{A} = \begin{bmatrix}\mathbf{a}_1 & \cdots & \mathbf{a}_k\end{bmatrix}, \hspace{0.5cm} \mathbf{B} = \begin{bmatrix}\mathbf{b}_1 & \cdots & \mathbf{b}_k\end{bmatrix}$$
:::

::: proof
*Proof.* For each $(i,j)$, we have
$$\left[\sum_{\ell=1}^k \mathbf{a}_\ell\mathbf{b}_\ell^{\!\top\!}\right]_{ij} = \sum_{\ell=1}^k [\mathbf{a}_\ell\mathbf{b}_\ell^{\!\top\!}]_{ij} = \sum_{\ell=1}^k [\mathbf{a}_\ell]_i[\mathbf{b}_\ell]_j = \sum_{\ell=1}^k A_{i\ell}B_{j\ell}$$
This last expression should be recognized as an inner product between
the $i$th row of $\mathbf{A}$ and the $j$th row of $\mathbf{B}$, or
equivalently the $j$th column of $\mathbf{B}^{\!\top\!}$. Hence by the
definition of matrix multiplication, it is equal to
$[\mathbf{A}\mathbf{B}^{\!\top\!}]_{ij}$. ◻
:::

### Quadratic forms

Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ be a symmetric matrix, and
recall that the expression $\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}$
is called a quadratic form of $\mathbf{A}$. It is in some cases helpful
to rewrite the quadratic form in terms of the individual elements that
make up $\mathbf{A}$ and $\mathbf{x}$:
$$\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} = \sum_{i=1}^n\sum_{j=1}^n A_{ij}x_ix_j$$
This identity is valid for any square matrix (need not be symmetric),
although quadratic forms are usually only discussed in the context of
symmetric matrices.

# Calculus and Optimization

Much of machine learning is about minimizing a **cost function** (also
called an **objective function** in the optimization community), which
is a scalar function of several variables that typically measures how
poorly our model fits the data we have.

## Extrema

Optimization is about finding **extrema**, which depending on the
application could be minima or maxima. When defining extrema, it is
necessary to consider the set of inputs over which we're optimizing.
This set $\mathcal{X} \subseteq \mathbb{R}^d$ is called the **feasible
set**. If $\mathcal{X}$ is the entire domain of the function being
optimized (as it often will be for our purposes), we say that the
problem is **unconstrained**. Otherwise the problem is **constrained**
and may be much harder to solve, depending on the nature of the feasible
set.

Suppose $f : \mathbb{R}^d \to \mathbb{R}$. A point $\mathbf{x}$ is said
to be a **local minimum** (resp. **local maximum**) of $f$ in
$\mathcal{X}$ if $f(\mathbf{x}) \leq f(\mathbf{y})$ (resp.
$f(\mathbf{x}) \geq f(\mathbf{y})$) for all $\mathbf{y}$ in some
neighborhood $N \subseteq \mathcal{X}$ about $\mathbf{x}$.[^6]
Furthermore, if $f(\mathbf{x}) \leq f(\mathbf{y})$ for all
$\mathbf{y} \in \mathcal{X}$, then $\mathbf{x}$ is a **global minimum**
of $f$ in $\mathcal{X}$ (similarly for global maximum). If the phrase
"in $\mathcal{X}$" is unclear from context, assume we are optimizing
over the whole domain of the function.

The qualifier **strict** (as in e.g. a strict local minimum) means that
the inequality sign in the definition is actually a $>$ or $<$, with
equality not allowed. This indicates that the extremum is unique within
some neighborhood.

Observe that maximizing a function $f$ is equivalent to minimizing $-f$,
so optimization problems are typically phrased in terms of minimization
without loss of generality. This convention (which we follow here)
eliminates the need to discuss minimization and maximization separately.

## Gradients

The single most important concept from calculus in the context of
machine learning is the **gradient**. Gradients generalize derivatives
to scalar functions of several variables. The gradient of
$f : \mathbb{R}^d \to \mathbb{R}$, denoted $\nabla f$, is given by
$$\nabla f = \begin{bmatrix}\pdv{f}{x_1} \\ \vdots \\ \pdv{f}{x_n}\end{bmatrix}
\hspace{0.5cm}\text{i.e.}\hspace{0.5cm}
[\nabla f]_i = \pdv{f}{x_i}$$ Gradients have the following very
important property: $\nabla f(\mathbf{x})$ points in the direction of
**steepest ascent** from $\mathbf{x}$. Similarly,
$-\nabla f(\mathbf{x})$ points in the direction of **steepest descent**
from $\mathbf{x}$. We will use this fact frequently when iteratively
minimizing a function via **gradient descent**.

## The Jacobian

The **Jacobian** of $f : \mathbb{R}^n \to \mathbb{R}^m$ is a matrix of
first-order partial derivatives: $$\mathbf{J}_f = \begin{bmatrix}
    \pdv{f_1}{x_1} & \hdots & \pdv{f_1}{x_n} \\
    \vdots & \ddots & \vdots \\
    \pdv{f_m}{x_1} & \hdots & \pdv{f_m}{x_n}\end{bmatrix}
\hspace{0.5cm}\text{i.e.}\hspace{0.5cm}
[\mathbf{J}_f]_{ij} = \pdv{f_i}{x_j}$$ Note the special case $m = 1$,
where $\nabla f = \mathbf{J}_f^{\!\top\!}$.

## The Hessian

The **Hessian** matrix of $f : \mathbb{R}^d \to \mathbb{R}$ is a matrix
of second-order partial derivatives: $$\nabla^2 f = \begin{bmatrix}
    \pdv[2]{f}{x_1} & \hdots & \pdv{f}{x_1}{x_d} \\
    \vdots & \ddots & \vdots \\
    \pdv{f}{x_d}{x_1} & \hdots & \pdv[2]{f}{x_d}\end{bmatrix}
\hspace{0.5cm}\text{i.e.}\hspace{0.5cm}
[\nabla^2 f]_{ij} = {\pdv{f}{x_i}{x_j}}$$ Recall that if the partial
derivatives are continuous, the order of differentiation can be
interchanged (Clairaut's theorem), so the Hessian matrix will be
symmetric. This will typically be the case for differentiable functions
that we work with.

The Hessian is used in some optimization algorithms such as Newton's
method. It is expensive to calculate but can drastically reduce the
number of iterations needed to converge to a local minimum by providing
information about the curvature of $f$.

## Matrix calculus

Since a lot of optimization reduces to finding points where the gradient
vanishes, it is useful to have differentiation rules for matrix and
vector expressions. We give some common rules here. Probably the two
most important for our purposes are $$\begin{aligned}
\nabla_\mathbf{x} &(\mathbf{a}^{\!\top\!}\mathbf{x}) = \mathbf{a} \\
\nabla_\mathbf{x} &(\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}) = (\mathbf{A} + \mathbf{A}^{\!\top\!})\mathbf{x}
\end{aligned}$$ Note that this second rule is defined only if
$\mathbf{A}$ is square. Furthermore, if $\mathbf{A}$ is symmetric, we
can simplify the result to $2\mathbf{A}\mathbf{x}$.

### The chain rule

Most functions that we wish to optimize are not completely arbitrary
functions, but rather are composed of simpler functions which we know
how to handle. The chain rule gives us a way to calculate derivatives
for a composite function in terms of the derivatives of the simpler
functions that make it up.

The chain rule from single-variable calculus should be familiar:
$$(f \circ g)'(x) = f'(g(x))g'(x)$$ where $\circ$ denotes function
composition. There is a natural generalization of this rule to
multivariate functions.

::: proposition
Suppose $f : \mathbb{R}^m \to \mathbb{R}^k$ and
$g : \mathbb{R}^n \to \mathbb{R}^m$. Then
$f \circ g : \mathbb{R}^n \to \mathbb{R}^k$ and
$$\mathbf{J}_{f \circ g}(\mathbf{x}) = \mathbf{J}_f(g(\mathbf{x}))\mathbf{J}_g(\mathbf{x})$$
:::

In the special case $k = 1$ we have the following corollary since
$\nabla f = \mathbf{J}_f^{\!\top\!}$.

::: corollary
Suppose $f : \mathbb{R}^m \to \mathbb{R}$ and
$g : \mathbb{R}^n \to \mathbb{R}^m$. Then
$f \circ g : \mathbb{R}^n \to \mathbb{R}$ and
$$\nabla (f \circ g)(\mathbf{x}) = \mathbf{J}_g(\mathbf{x})^{\!\top\!} \nabla f(g(\mathbf{x}))$$
:::

## Taylor's theorem

Taylor's theorem has natural generalizations to functions of more than
one variable. We give the version presented in [@numopt].

::: theorem
(Taylor's theorem) Suppose $f : \mathbb{R}^d \to \mathbb{R}$ is
continuously differentiable, and let $\mathbf{h} \in \mathbb{R}^d$. Then
there exists $t \in (0,1)$ such that
$$f(\mathbf{x} + \mathbf{h}) = f(\mathbf{x}) + \nabla f(\mathbf{x} + t\mathbf{h})^{\!\top\!}\mathbf{h}$$
Furthermore, if $f$ is twice continuously differentiable, then
$$\nabla f(\mathbf{x} + \mathbf{h}) = \nabla f(\mathbf{x}) + \int_0^1 \nabla^2 f(\mathbf{x} + t\mathbf{h})\mathbf{h} \dd{t}$$
and there exists $t \in (0,1)$ such that
$$f(\mathbf{x} + \mathbf{h}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^{\!\top\!}\mathbf{h} + \frac{1}{2}\mathbf{h}^{\!\top\!}\nabla^2f(\mathbf{x}+t\mathbf{h})\mathbf{h}$$
:::

This theorem is used in proofs about conditions for local minima of
unconstrained optimization problems. Some of the most important results
are given in the next section.

## Conditions for local minima

::: proposition
If $\mathbf{x}^*$ is a local minimum of $f$ and $f$ is continuously
differentiable in a neighborhood of $\mathbf{x}^*$, then
$\nabla f(\mathbf{x}^*) = \mathbf{0}$.
:::

::: proof
*Proof.* Let $\mathbf{x}^*$ be a local minimum of $f$, and suppose
towards a contradiction that $\nabla f(\mathbf{x}^*) \neq \mathbf{0}$.
Let $\mathbf{h} = -\nabla f(\mathbf{x}^*)$, noting that by the
continuity of $\nabla f$ we have
$$\lim_{t \to 0} -\nabla f(\mathbf{x}^* + t\mathbf{h}) = -\nabla f(\mathbf{x}^*) = \mathbf{h}$$
Hence
$$\lim_{t \to 0} \mathbf{h}^{\!\top\!}\nabla f(\mathbf{x}^* + t\mathbf{h}) = \mathbf{h}^{\!\top\!}\nabla f(\mathbf{x}^*) = -\|\mathbf{h}\|_2^2 < 0$$
Thus there exists $T > 0$ such that
$\mathbf{h}^{\!\top\!}\nabla f(\mathbf{x}^* + t\mathbf{h}) < 0$ for all
$t \in [0,T]$. Now we apply Taylor's theorem: for any $t \in (0,T]$,
there exists $t' \in (0,t)$ such that
$$f(\mathbf{x}^* + t\mathbf{h}) = f(\mathbf{x}^*) + t\mathbf{h}^{\!\top\!} \nabla f(\mathbf{x}^* + t'\mathbf{h}) < f(\mathbf{x}^*)$$
whence it follows that $\mathbf{x}^*$ is not a local minimum, a
contradiction. Hence $\nabla f(\mathbf{x}^*) = \mathbf{0}$. ◻
:::

The proof shows us why the vanishing gradient is necessary for an
extremum: if $\nabla f(\mathbf{x})$ is nonzero, there always exists a
sufficiently small step $\alpha > 0$ such that
$f(\mathbf{x} - \alpha\nabla f(\mathbf{x}))) < f(\mathbf{x})$. For this
reason, $-\nabla f(\mathbf{x})$ is called a **descent direction**.

Points where the gradient vanishes are called **stationary points**.
Note that not all stationary points are extrema. Consider
$f : \mathbb{R}^2 \to \mathbb{R}$ given by $f(x,y) = x^2 - y^2$. We have
$\nabla f(\mathbf{0}) = \mathbf{0}$, but the point $\mathbf{0}$ is the
minimum along the line $y = 0$ and the maximum along the line $x = 0$.
Thus it is neither a local minimum nor a local maximum of $f$. Points
such as these, where the gradient vanishes but there is no local
extremum, are called **saddle points**.

We have seen that first-order information (i.e. the gradient) is
insufficient to characterize local minima. But we can say more with
second-order information (i.e. the Hessian). First we prove a necessary
second-order condition for local minima.

::: proposition
If $\mathbf{x}^*$ is a local minimum of $f$ and $f$ is twice
continuously differentiable in a neighborhood of $\mathbf{x}^*$, then
$\nabla^2 f(\mathbf{x}^*)$ is positive semi-definite.
:::

::: proof
*Proof.* Let $\mathbf{x}^*$ be a local minimum of $f$, and suppose
towards a contradiction that $\nabla^2 f(\mathbf{x}^*)$ is not positive
semi-definite. Let $\mathbf{h}$ be such that
$\mathbf{h}^{\!\top\!}\nabla^2 f(\mathbf{x}^*)\mathbf{h} < 0$, noting
that by the continuity of $\nabla^2 f$ we have
$$\lim_{t \to 0} \nabla^2 f(\mathbf{x}^* + t\mathbf{h}) = \nabla^2 f(\mathbf{x}^*)$$
Hence
$$\lim_{t \to 0} \mathbf{h}^{\!\top\!}\nabla^2 f(\mathbf{x}^* + t\mathbf{h})\mathbf{h} = \mathbf{h}^{\!\top\!}\nabla^2 f(\mathbf{x}^*)\mathbf{h} < 0$$
Thus there exists $T > 0$ such that
$\mathbf{h}^{\!\top\!}\nabla^2 f(\mathbf{x}^* + t\mathbf{h})\mathbf{h} < 0$
for all $t \in [0,T]$. Now we apply Taylor's theorem: for any
$t \in (0,T]$, there exists $t' \in (0,t)$ such that
$$f(\mathbf{x}^* + t\mathbf{h}) = f(\mathbf{x}^*) + \underbrace{t\mathbf{h}^{\!\top\!}\nabla f(\mathbf{x}^*)}_0 + \frac{1}{2}t^2\mathbf{h}^{\!\top\!}\nabla^2 f(\mathbf{x}^* + t'\mathbf{h})\mathbf{h} < f(\mathbf{x}^*)$$
where the middle term vanishes because
$\nabla f(\mathbf{x}^*) = \mathbf{0}$ by the previous result. It follows
that $\mathbf{x}^*$ is not a local minimum, a contradiction. Hence
$\nabla^2 f(\mathbf{x}^*)$ is positive semi-definite. ◻
:::

Now we give sufficient conditions for local minima.

::: proposition
Suppose $f$ is twice continuously differentiable with $\nabla^2 f$
positive semi-definite in a neighborhood of $\mathbf{x}^*$, and that
$\nabla f(\mathbf{x}^*) = \mathbf{0}$. Then $\mathbf{x}^*$ is a local
minimum of $f$. Furthermore if $\nabla^2 f(\mathbf{x}^*)$ is positive
definite, then $\mathbf{x}^*$ is a strict local minimum.
:::

::: proof
*Proof.* Let $B$ be an open ball of radius $r > 0$ centered at
$\mathbf{x}^*$ which is contained in the neighborhood. Applying Taylor's
theorem, we have that for any $\mathbf{h}$ with $\|\mathbf{h}\|_2 < r$,
there exists $t \in (0,1)$ such that
$$f(\mathbf{x}^* + \mathbf{h}) = f(\mathbf{x}^*) + \underbrace{\mathbf{h}^{\!\top\!}\nabla f(\mathbf{x}^*)}_0 + \frac{1}{2}\mathbf{h}^{\!\top\!}\nabla^2 f(\mathbf{x}^* + t\mathbf{h})\mathbf{h} \geq f(\mathbf{x}^*)$$
The last inequality holds because
$\nabla^2 f(\mathbf{x}^* + t\mathbf{h})$ is positive semi-definite
(since $\|t\mathbf{h}\|_2 = t\|\mathbf{h}\|_2 < \|\mathbf{h}\|_2 < r$),
so
$\mathbf{h}^{\!\top\!}\nabla^2 f(\mathbf{x}^* + t\mathbf{h})\mathbf{h} \geq 0$.
Since $f(\mathbf{x}^*) \leq f(\mathbf{x}^* + \mathbf{h})$ for all
directions $\mathbf{h}$ with $\|\mathbf{h}\|_2 < r$, we conclude that
$\mathbf{x}^*$ is a local minimum.

Now further suppose that $\nabla^2 f(\mathbf{x}^*)$ is strictly positive
definite. Since the Hessian is continuous we can choose another ball
$B'$ with radius $r' > 0$ centered at $\mathbf{x}^*$ such that
$\nabla^2 f(\mathbf{x})$ is positive definite for all
$\mathbf{x} \in B'$. Then following the same argument as above (except
with a strict inequality now since the Hessian is positive definite) we
have $f(\mathbf{x}^* + \mathbf{h}) > f(\mathbf{x}^*)$ for all
$\mathbf{h}$ with $0 < \|\mathbf{h}\|_2 < r'$. Hence $\mathbf{x}^*$ is a
strict local minimum. ◻
:::

Note that, perhaps counterintuitively, the conditions
$\nabla f(\mathbf{x}^*) = \mathbf{0}$ and $\nabla^2 f(\mathbf{x}^*)$
positive semi-definite are not enough to guarantee a local minimum at
$\mathbf{x}^*$! Consider the function $f(x) = x^3$. We have $f'(0) = 0$
and $f''(0) = 0$ (so the Hessian, which in this case is the $1 \times 1$
matrix $\begin{bmatrix}0\end{bmatrix}$, is positive semi-definite). But
$f$ has a saddle point at $x = 0$. The function $f(x) = -x^4$ is an even
worse offender -- it has the same gradient and Hessian at $x = 0$, but
$x = 0$ is a strict local maximum for this function!

For these reasons we require that the Hessian remains positive
semi-definite as long as we are close to $\mathbf{x}^*$. Unfortunately,
this condition is not practical to check computationally, but in some
cases we can verify it analytically (usually by showing that
$\nabla^2 f(\mathbf{x})$ is p.s.d. for all
$\mathbf{x} \in \mathbb{R}^d$). Also, if $\nabla^2 f(\mathbf{x}^*)$ is
strictly positive definite, the continuity assumption on $f$ implies
this condition, so we don't have to worry.

## Convexity

**Convexity** is a term that pertains to both sets and functions. For
functions, there are different degrees of convexity, and how convex a
function is tells us a lot about its minima: do they exist, are they
unique, how quickly can we find them using optimization algorithms, etc.
In this section, we present basic results regarding convexity, strict
convexity, and strong convexity.

### Convex sets

<figure id="fig:convexset">
<figure>
<img src="convex-set.png" />
<figcaption>A convex set</figcaption>
</figure>
<figure>
<img src="nonconvex-set.png" />
<figcaption>A non-convex set</figcaption>
</figure>
<figcaption>What convex sets look like</figcaption>
</figure>

A set $\mathcal{X} \subseteq \mathbb{R}^d$ is **convex** if
$$t\mathbf{x} + (1-t)\mathbf{y} \in \mathcal{X}$$ for all
$\mathbf{x}, \mathbf{y} \in \mathcal{X}$ and all $t \in [0,1]$.

Geometrically, this means that all the points on the line segment
between any two points in $\mathcal{X}$ are also in $\mathcal{X}$. See
Figure [1](#fig:convexset){reference-type="ref"
reference="fig:convexset"} for a visual.

Why do we care whether or not a set is convex? We will see later that
the nature of minima can depend greatly on whether or not the feasible
set is convex. Undesirable pathological results can occur when we allow
the feasible set to be arbitrary, so for proofs we will need to assume
that it is convex. Fortunately, we often want to minimize over all of
$\mathbb{R}^d$, which is easily seen to be a convex set.

### Basics of convex functions

In the remainder of this section, assume
$f : \mathbb{R}^d \to \mathbb{R}$ unless otherwise noted. We'll start
with the definitions and then give some results.

A function $f$ is **convex** if
$$f(t\mathbf{x} + (1-t)\mathbf{y}) \leq t f(\mathbf{x}) + (1-t)f(\mathbf{y})$$
for all $\mathbf{x}, \mathbf{y} \in \dom f$ and all $t \in [0,1]$.

If the inequality holds strictly (i.e. $<$ rather than $\leq$) for all
$t \in (0,1)$ and $\mathbf{x} \neq \mathbf{y}$, then we say that $f$ is
**strictly convex**.

A function $f$ is **strongly convex with parameter $m$** (or
**$m$-strongly convex**) if the function
$$\mathbf{x} \mapsto f(\mathbf{x}) - \frac{m}{2}\|\mathbf{x}\|_2^2$$ is
convex.

These conditions are given in increasing order of strength; strong
convexity implies strict convexity which implies convexity.

![What convex functions look
like](convex-function.png){#fig:convexfunction width="\\linewidth"}

Geometrically, convexity means that the line segment between two points
on the graph of $f$ lies on or above the graph itself. See Figure
[2](#fig:convexfunction){reference-type="ref"
reference="fig:convexfunction"} for a visual.

Strict convexity means that the graph of $f$ lies strictly above the
line segment, except at the segment endpoints. (So actually the function
in the figure appears to be strictly convex.)

### Consequences of convexity

Why do we care if a function is (strictly/strongly) convex?

Basically, our various notions of convexity have implications about the
nature of minima. It should not be surprising that the stronger
conditions tell us more about the minima.

::: proposition
Let $\mathcal{X}$ be a convex set. If $f$ is convex, then any local
minimum of $f$ in $\mathcal{X}$ is also a global minimum.
:::

::: proof
*Proof.* Suppose $f$ is convex, and let $\mathbf{x}^*$ be a local
minimum of $f$ in $\mathcal{X}$. Then for some neighborhood
$N \subseteq \mathcal{X}$ about $\mathbf{x}^*$, we have
$f(\mathbf{x}) \geq f(\mathbf{x}^*)$ for all $\mathbf{x} \in N$. Suppose
towards a contradiction that there exists
$\tilde{\mathbf{x}} \in \mathcal{X}$ such that
$f(\tilde{\mathbf{x}}) < f(\mathbf{x}^*)$.

Consider the line segment
$\mathbf{x}(t) = t\mathbf{x}^* + (1-t)\tilde{\mathbf{x}}, ~ t \in [0,1]$,
noting that $\mathbf{x}(t) \in \mathcal{X}$ by the convexity of
$\mathcal{X}$. Then by the convexity of $f$,
$$f(\mathbf{x}(t)) \leq tf(\mathbf{x}^*) + (1-t)f(\tilde{\mathbf{x}}) < tf(\mathbf{x}^*) + (1-t)f(\mathbf{x}^*) = f(\mathbf{x}^*)$$
for all $t \in (0,1)$.

We can pick $t$ to be sufficiently close to $1$ that
$\mathbf{x}(t) \in N$; then $f(\mathbf{x}(t)) \geq f(\mathbf{x}^*)$ by
the definition of $N$, but $f(\mathbf{x}(t)) < f(\mathbf{x}^*)$ by the
above inequality, a contradiction.

It follows that $f(\mathbf{x}^*) \leq f(\mathbf{x})$ for all
$\mathbf{x} \in \mathcal{X}$, so $\mathbf{x}^*$ is a global minimum of
$f$ in $\mathcal{X}$. ◻
:::

::: proposition
Let $\mathcal{X}$ be a convex set. If $f$ is strictly convex, then there
exists at most one local minimum of $f$ in $\mathcal{X}$. Consequently,
if it exists it is the unique global minimum of $f$ in $\mathcal{X}$.
:::

::: proof
*Proof.* The second sentence follows from the first, so all we must show
is that if a local minimum exists in $\mathcal{X}$ then it is unique.

Suppose $\mathbf{x}^*$ is a local minimum of $f$ in $\mathcal{X}$, and
suppose towards a contradiction that there exists a local minimum
$\tilde{\mathbf{x}} \in \mathcal{X}$ such that
$\tilde{\mathbf{x}} \neq \mathbf{x}^*$.

Since $f$ is strictly convex, it is convex, so $\mathbf{x}^*$ and
$\tilde{\mathbf{x}}$ are both global minima of $f$ in $\mathcal{X}$ by
the previous result. Hence $f(\mathbf{x}^*) = f(\tilde{\mathbf{x}})$.
Consider the line segment
$\mathbf{x}(t) = t\mathbf{x}^* + (1-t)\tilde{\mathbf{x}}, ~ t \in [0,1]$,
which again must lie entirely in $\mathcal{X}$. By the strict convexity
of $f$,
$$f(\mathbf{x}(t)) < tf(\mathbf{x}^*) + (1-t)f(\tilde{\mathbf{x}}) = tf(\mathbf{x}^*) + (1-t)f(\mathbf{x}^*) = f(\mathbf{x}^*)$$
for all $t \in (0,1)$. But this contradicts the fact that $\mathbf{x}^*$
is a global minimum. Therefore if $\tilde{\mathbf{x}}$ is a local
minimum of $f$ in $\mathcal{X}$, then
$\tilde{\mathbf{x}} = \mathbf{x}^*$, so $\mathbf{x}^*$ is the unique
minimum in $\mathcal{X}$. ◻
:::

It is worthwhile to examine how the feasible set affects the
optimization problem. We will see why the assumption that $\mathcal{X}$
is convex is needed in the results above.

Consider the function $f(x) = x^2$, which is a strictly convex function.
The unique global minimum of this function in $\mathbb{R}$ is $x = 0$.
But let's see what happens when we change the feasible set
$\mathcal{X}$.

(i) $\mathcal{X} = \{1\}$: This set is actually convex, so we still have
    a unique global minimum. But it is not the same as the unconstrained
    minimum!

(ii) $\mathcal{X} = \mathbb{R} \setminus \{0\}$: This set is non-convex,
     and we can see that $f$ has no minima in $\mathcal{X}$. For any
     point $x \in \mathcal{X}$, one can find another point
     $y \in \mathcal{X}$ such that $f(y) < f(x)$.

(iii) $\mathcal{X} = (-\infty,-1] \cup [0,\infty)$: This set is
      non-convex, and we can see that there is a local minimum
      ($x = -1$) which is distinct from the global minimum ($x = 0$).

(iv) $\mathcal{X} = (-\infty,-1] \cup [1,\infty)$: This set is
     non-convex, and we can see that there are two global minima
     ($x = \pm 1$).

### Showing that a function is convex

Hopefully the previous section has convinced the reader that convexity
is an important property. Next we turn to the issue of showing that a
function is (strictly/strongly) convex. It is of course possible (in
principle) to directly show that the condition in the definition holds,
but this is usually not the easiest way.

::: proposition
Norms are convex.
:::

::: proof
*Proof.* Let $\|\cdot\|$ be a norm on a vector space $V$. Then for all
$\mathbf{x}, \mathbf{y} \in V$ and $t \in [0,1]$,
$$\|t\mathbf{x} + (1-t)\mathbf{y}\| \leq \|t\mathbf{x}\| + \|(1-t)\mathbf{y}\| = |t|\|\mathbf{x}\| + |1-t|\|\mathbf{y}\| = t\|\mathbf{x}\| + (1-t)\|\mathbf{y}\|$$
where we have used respectively the triangle inequality, the homogeneity
of norms, and the fact that $t$ and $1-t$ are nonnegative. Hence
$\|\cdot\|$ is convex. ◻
:::

::: proposition
Suppose $f$ is differentiable. Then $f$ is convex if and only if
$$f(\mathbf{y}) \geq f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$$
for all $\mathbf{x}, \mathbf{y} \in \dom f$.
:::

::: proof
*Proof.* To-do. ◻
:::

::: proposition
Suppose $f$ is twice differentiable. Then

(i) $f$ is convex if and only if $\nabla^2 f(\mathbf{x}) \succeq 0$ for
    all $\mathbf{x} \in \dom f$.

(ii) If $\nabla^2 f(\mathbf{x}) \succ 0$ for all
     $\mathbf{x} \in \dom f$, then $f$ is strictly convex.

(iii) $f$ is $m$-strongly convex if and only if
      $\nabla^2 f(\mathbf{x}) \succeq mI$ for all
      $\mathbf{x} \in \dom f$.
:::

::: proof
*Proof.* Omitted. ◻
:::

::: proposition
If $f$ is convex and $\alpha \geq 0$, then $\alpha f$ is convex.
:::

::: proof
*Proof.* Suppose $f$ is convex and $\alpha \geq 0$. Then for all
$\mathbf{x}, \mathbf{y} \in \dom(\alpha f) = \dom f$, $$\begin{aligned}
(\alpha f)(t\mathbf{x} + (1-t)\mathbf{y}) &= \alpha f(t\mathbf{x} + (1-t)\mathbf{y}) \\
&\leq \alpha\left(tf(\mathbf{x}) + (1-t)f(\mathbf{y})\right) \\
&= t(\alpha f(\mathbf{x})) + (1-t)(\alpha f(\mathbf{y})) \\
&= t(\alpha f)(\mathbf{x}) + (1-t)(\alpha f)(\mathbf{y})
\end{aligned}$$ so $\alpha f$ is convex. ◻
:::

::: proposition
If $f$ and $g$ are convex, then $f+g$ is convex. Furthermore, if $g$ is
strictly convex, then $f+g$ is strictly convex, and if $g$ is
$m$-strongly convex, then $f+g$ is $m$-strongly convex.
:::

::: proof
*Proof.* Suppose $f$ and $g$ are convex. Then for all
$\mathbf{x}, \mathbf{y} \in \dom (f+g) = \dom f \cap \dom g$,
$$\begin{aligned}
(f+g)(t\mathbf{x} + (1-t)\mathbf{y}) &= f(t\mathbf{x} + (1-t)\mathbf{y}) + g(t\mathbf{x} + (1-t)\mathbf{y}) \\
&\leq tf(\mathbf{x}) + (1-t)f(\mathbf{y}) + g(t\mathbf{x} + (1-t)\mathbf{y}) & \text{convexity of $f$} \\
&\leq tf(\mathbf{x}) + (1-t)f(\mathbf{y}) + tg(\mathbf{x}) + (1-t)g(\mathbf{y}) & \text{convexity of $g$} \\
&= t(f(\mathbf{x}) + g(\mathbf{x})) + (1-t)(f(\mathbf{y}) + g(\mathbf{y})) \\
&= t(f+g)(\mathbf{x}) + (1-t)(f+g)(\mathbf{y})
\end{aligned}$$ so $f + g$ is convex.

If $g$ is strictly convex, the second inequality above holds strictly
for $\mathbf{x} \neq \mathbf{y}$ and $t \in (0,1)$, so $f+g$ is strictly
convex.

If $g$ is $m$-strongly convex, then the function
$h(\mathbf{x}) \equiv g(\mathbf{x}) - \frac{m}{2}\|\mathbf{x}\|_2^2$ is
convex, so $f+h$ is convex. But
$$(f+h)(\mathbf{x}) \equiv f(\mathbf{x}) + h(\mathbf{x}) \equiv f(\mathbf{x}) + g(\mathbf{x}) - \frac{m}{2}\|\mathbf{x}\|_2^2 \equiv (f+g)(\mathbf{x}) - \frac{m}{2}\|\mathbf{x}\|_2^2$$
so $f+g$ is $m$-strongly convex. ◻
:::

::: proposition
If $f_1, \dots, f_n$ are convex and $\alpha_1, \dots, \alpha_n \geq 0$,
then $$\sum_{i=1}^n \alpha_i f_i$$ is convex.
:::

::: proof
*Proof.* Follows from the previous two propositions by induction. ◻
:::

::: proposition
If $f$ is convex, then
$g(\mathbf{x}) \equiv f(\mathbf{A}\mathbf{x} + \mathbf{b})$ is convex
for any appropriately-sized $\mathbf{A}$ and $\mathbf{b}$.
:::

::: proof
*Proof.* Suppose $f$ is convex and $g$ is defined like so. Then for all
$\mathbf{x}, \mathbf{y} \in \dom g$, $$\begin{aligned}
g(t\mathbf{x} + (1-t)\mathbf{y}) &= f(\mathbf{A}(t\mathbf{x} + (1-t)\mathbf{y}) + \mathbf{b}) \\
&= f(t\mathbf{A}\mathbf{x} + (1-t)\mathbf{A}\mathbf{y} + \mathbf{b}) \\
&= f(t\mathbf{A}\mathbf{x} + (1-t)\mathbf{A}\mathbf{y} + t\mathbf{b} + (1-t)\mathbf{b}) \\
&= f(t(\mathbf{A}\mathbf{x} + \mathbf{b}) + (1-t)(\mathbf{A}\mathbf{y} + \mathbf{b})) \\
&\leq tf(\mathbf{A}\mathbf{x} + \mathbf{b}) + (1-t)f(\mathbf{A}\mathbf{y} + \mathbf{b}) & \text{convexity of $f$} \\
&= tg(\mathbf{x}) + (1-t)g(\mathbf{y})
\end{aligned}$$ Thus $g$ is convex. ◻
:::

::: proposition
If $f$ and $g$ are convex, then
$h(\mathbf{x}) \equiv \max\{f(\mathbf{x}), g(\mathbf{x})\}$ is convex.
:::

::: proof
*Proof.* Suppose $f$ and $g$ are convex and $h$ is defined like so. Then
for all $\mathbf{x}, \mathbf{y} \in \dom h$, $$\begin{aligned}
h(t\mathbf{x} + (1-t)\mathbf{y}) &= \max\{f(t\mathbf{x} + (1-t)\mathbf{y}), g(t\mathbf{x} + (1-t)\mathbf{y})\} \\
&\leq \max\{tf(\mathbf{x}) + (1-t)f(\mathbf{y}), tg(\mathbf{x}) + (1-t)g(\mathbf{y})\} \\
&\leq \max\{tf(\mathbf{x}), tg(\mathbf{x})\} + \max\{(1-t)f(\mathbf{y}), (1-t)g(\mathbf{y})\} \\
&= t\max\{f(\mathbf{x}), g(\mathbf{x})\} + (1-t)\max\{f(\mathbf{y}), g(\mathbf{y})\} \\
&= th(\mathbf{x}) + (1-t)h(\mathbf{y})
\end{aligned}$$ Note that in the first inequality we have used convexity
of $f$ and $g$ plus the fact that $a \leq c, b \leq d$ implies
$\max\{a,b\} \leq \max\{c,d\}$. In the second inequality we have used
the fact that $\max\{a+b, c+d\} \leq \max\{a,c\} + \max\{b,d\}$.

Thus $h$ is convex. ◻
:::

### Examples

A good way to gain intuition about the distinction between convex,
strictly convex, and strongly convex functions is to consider examples
where the stronger property fails to hold.

Functions that are convex but not strictly convex:

(i) $f(\mathbf{x}) = \mathbf{w}^{\!\top\!}\mathbf{x} + \alpha$ for any
    $\mathbf{w} \in \mathbb{R}^d, \alpha \in \mathbb{R}$. Such a
    function is called an **affine function**, and it is both convex and
    concave. (In fact, a function is affine if and only if it is both
    convex and concave.) Note that linear functions and constant
    functions are special cases of affine functions.

(ii) $f(\mathbf{x}) = \|\mathbf{x}\|_1$

Functions that are strictly but not strongly convex:

(i) $f(x) = x^4$. This example is interesting because it is strictly
    convex but you cannot show this fact via a second-order argument
    (since $f''(0) = 0$).

(ii) $f(x) = \exp(x)$. This example is interesting because it's bounded
     below but has no local minimum.

(iii) $f(x) = -\log x$. This example is interesting because it's
      strictly convex but not bounded below.

Functions that are strongly convex:

(i) $f(\mathbf{x}) = \|\mathbf{x}\|_2^2$

## Orthogonal projections

We now consider a particular kind of optimization problem that is
particularly well-understood and can often be solved in closed form:
given some point $\mathbf{x}$ in an inner product space $V$, find the
closest point to $\mathbf{x}$ in a subspace $S$ of $V$. This process is
referred to as **projection onto a subspace**.

The following diagram should make it geometrically clear that, at least
in Euclidean space, the solution is intimately related to orthogonality
and the Pythagorean theorem:

::: center
![image](orthogonal-projection.png){width="50%"}
:::

Here $\mathbf{y}$ is an arbitrary element of the subspace $S$, and
$\mathbf{y}^*$ is the point in $S$ such that $\mathbf{x}-\mathbf{y}^*$
is perpendicular to $S$. The hypotenuse of a right triangle (in this
case $\|\mathbf{x}-\mathbf{y}\|$) is always longer than either of the
legs (in this case $\|\mathbf{x}-\mathbf{y}^*\|$ and
$\|\mathbf{y}^*-\mathbf{y}\|$), and when $\mathbf{y} \neq \mathbf{y}^*$
there always exists such a triangle between $\mathbf{x}$, $\mathbf{y}$,
and $\mathbf{y}^*$.

Our intuition from Euclidean space suggests that the closest point to
$\mathbf{x}$ in $S$ has the perpendicularity property described above,
and we now show that this is indeed the case.

::: proposition
Suppose $\mathbf{x} \in V$ and $\mathbf{y} \in S$. Then $\mathbf{y}^*$
is the unique minimizer of $\|\mathbf{x}-\mathbf{y}\|$ over
$\mathbf{y} \in S$ if and only if $\mathbf{x}-\mathbf{y}^* \perp S$.
:::

::: proof
*Proof.* $(\implies)$ Suppose $\mathbf{y}^*$ is the unique minimizer of
$\|\mathbf{x}-\mathbf{y}\|$ over $\mathbf{y} \in S$. That is,
$\|\mathbf{x}-\mathbf{y}^*\| \leq \|\mathbf{x}-\mathbf{y}\|$ for all
$\mathbf{y} \in S$, with equality only if $\mathbf{y} = \mathbf{y}^*$.
Fix $\mathbf{v} \in S$ and observe that $$\begin{aligned}
g(t) &:= \|\mathbf{x}-\mathbf{y}^*+t\mathbf{v}\|^2 \\
&= \langle \mathbf{x}-\mathbf{y}^*+t\mathbf{v}, \mathbf{x}-\mathbf{y}^*+t\mathbf{v} \rangle \\
&= \langle \mathbf{x}-\mathbf{y}^*, \mathbf{x}-\mathbf{y}^* \rangle - 2t\langle \mathbf{x}-\mathbf{y}^*, \mathbf{v} \rangle + t^2\langle \mathbf{v}, \mathbf{v} \rangle \\
&= \|\mathbf{x}-\mathbf{y}^*\|^2 - 2t\langle \mathbf{x}-\mathbf{y}^*, \mathbf{v} \rangle + t^2\|\mathbf{v}\|^2
\end{aligned}$$ must have a minimum at $t = 0$ as a consequence of this
assumption. Thus
$$0 = g'(0) = \left.-2\langle \mathbf{x}-\mathbf{y}^*, \mathbf{v} \rangle + 2t\|\mathbf{v}\|^2\right|_{t=0} = -2\langle \mathbf{x}-\mathbf{y}^*, \mathbf{v} \rangle$$
giving $\mathbf{x}-\mathbf{y}^* \perp \mathbf{v}$. Since $\mathbf{v}$
was arbitrary in $S$, we have $\mathbf{x}-\mathbf{y}^* \perp S$ as
claimed.

$(\impliedby)$ Suppose $\mathbf{x}-\mathbf{y}^* \perp S$. Observe that
for any $\mathbf{y} \in S$, $\mathbf{y}^*-\mathbf{y} \in S$ because
$\mathbf{y}^* \in S$ and $S$ is closed under subtraction. Under the
hypothesis, $\mathbf{x}-\mathbf{y}^* \perp \mathbf{y}^*-\mathbf{y}$, so
by the Pythagorean theorem,
$$\|\mathbf{x}-\mathbf{y}\| = \|\mathbf{x}-\mathbf{y}^*+\mathbf{y}^*-\mathbf{y}\| = \|\mathbf{x}-\mathbf{y}^*\| + \|\mathbf{y}^*-\mathbf{y}\| \geq \|\mathbf{x} - \mathbf{y}^*\|$$
and in fact the inequality is strict when $\mathbf{y} \neq \mathbf{y}^*$
since this implies $\|\mathbf{y}^*-\mathbf{y}\| > 0$. Thus
$\mathbf{y}^*$ is the unique minimizer of $\|\mathbf{x}-\mathbf{y}\|$
over $\mathbf{y} \in S$. ◻
:::

Since a unique minimizer in $S$ can be found for any $\mathbf{x} \in V$,
we can define an operator
$$P\mathbf{x} = \argmin_{\mathbf{y} \in S} \|\mathbf{x}-\mathbf{y}\|$$
Observe that $P\mathbf{y} = \mathbf{y}$ for any $\mathbf{y} \in S$,
since $\mathbf{y}$ has distance zero from itself and every other point
in $S$ has positive distance from $\mathbf{y}$. Thus
$P(P\mathbf{x}) = P\mathbf{x}$ for any $\mathbf{x}$ (i.e., $P^2 = P$)
because $P\mathbf{x} \in S$. The identity $P^2 = P$ is actually one of
the defining properties of a **projection**, the other being linearity.

An immediate consequence of the previous result is that
$\mathbf{x} - P\mathbf{x} \perp S$ for any $\mathbf{x} \in V$, and
conversely that $P$ is the unique operator that satisfies this property
for all $\mathbf{x} \in V$. For this reason, $P$ is known as an
**orthogonal projection**.

If we choose an orthonormal basis for the target subspace $S$, it is
possible to write down a more specific expression for $P$.

::: proposition
If $\mathbf{e}_1, \dots, \mathbf{e}_m$ is an orthonormal basis for $S$,
then
$$P\mathbf{x} = \sum_{i=1}^m \langle \mathbf{x}, \mathbf{e}_i \rangle\mathbf{e}_i$$
:::

::: proof
*Proof.* Let $\mathbf{e}_1, \dots, \mathbf{e}_m$ be an orthonormal basis
for $S$, and suppose $\mathbf{x} \in V$. Then for all $j = 1, \dots, m$,
$$\begin{aligned}
\left\langle \mathbf{x}-\sum_{i=1}^m \langle \mathbf{x}, \mathbf{e}_i \rangle\mathbf{e}_i, \mathbf{e}_j \right\rangle &= \langle \mathbf{x}, \mathbf{e}_j \rangle - \sum_{i=1}^m \langle \mathbf{x}, \mathbf{e}_i \rangle\underbrace{\langle \mathbf{e}_i, \mathbf{e}_j \rangle}_{\delta_{ij}} \\
&= \langle \mathbf{x}, \mathbf{e}_j \rangle - \langle \mathbf{x}, \mathbf{e}_j \rangle \\
&= 0
\end{aligned}$$ We have shown that the claimed expression, call it
$\tilde{P}\mathbf{x}$, satisfies
$\mathbf{x} - \tilde{P}\mathbf{x} \perp \mathbf{e}_j$ for every element
$\mathbf{e}_j$ of the orthonormal basis for $S$. It follows (by
linearity of the inner product) that
$\mathbf{x} - \tilde{P}\mathbf{x} \perp S$, so the previous result
implies $P = \tilde{P}$. ◻
:::

The fact that $P$ is a linear operator (and thus a proper projection, as
earlier we showed $P^2 = P$) follows readily from this result.

# Probability

Probability theory provides powerful tools for modeling and dealing with
uncertainty.

## Basics

Suppose we have some sort of randomized experiment (e.g. a coin toss,
die roll) that has a fixed set of possible outcomes. This set is called
the **sample space** and denoted $\Omega$.

We would like to define probabilities for some **events**, which are
subsets of $\Omega$. The set of events is denoted $\mathcal{F}$.[^7] The
**complement** of the event $A$ is another event,
$A^\text{c} = \Omega \setminus A$.

Then we can define a **probability measure**
$\mathbb{P} : \mathcal{F} \to [0,1]$ which must satisfy

(i) $\mathbb{P}(\Omega) = 1$

(ii) **Countable additivity**: for any countable collection of disjoint
     sets $\{A_i\} \subseteq \mathcal{F}$,
     $$\mathbb{P}\bigg(\bigcup_i A_i\bigg) = \sum_i \mathbb{P}(A_i)$$

The triple $(\Omega, \mathcal{F}, \mathbb{P})$ is called a **probability
space**.[^8]

If $\mathbb{P}(A) = 1$, we say that $A$ occurs **almost surely** (often
abbreviated a.s.).[^9], and conversely $A$ occurs **almost never** if
$\mathbb{P}(A) = 0$.

From these axioms, a number of useful rules can be derived.

::: proposition
Let $A$ be an event. Then

(i) $\mathbb{P}(A^\text{c}) = 1 - \mathbb{P}(A)$.

(ii) If $B$ is an event and $B \subseteq A$, then
     $\mathbb{P}(B) \leq \mathbb{P}(A)$.

(iii) $0 = \mathbb{P}(\varnothing) \leq \mathbb{P}(A) \leq \mathbb{P}(\Omega) = 1$
:::

::: proof
*Proof.* (i) Using the countable additivity of $\mathbb{P}$, we have
$$\mathbb{P}(A) + \mathbb{P}(A^\text{c}) = \mathbb{P}(A \mathbin{\dot{\cup}} A^\text{c}) = \mathbb{P}(\Omega) = 1$$

To show (ii), suppose $B \in \mathcal{F}$ and $B \subseteq A$. Then
$$\mathbb{P}(A) = \mathbb{P}(B \mathbin{\dot{\cup}} (A \setminus B)) = \mathbb{P}(B) + \mathbb{P}(A \setminus B) \geq \mathbb{P}(B)$$
as claimed.

For (iii): the middle inequality follows from (ii) since
$\varnothing \subseteq A \subseteq \Omega$. We also have
$$\mathbb{P}(\varnothing) = \mathbb{P}(\varnothing \mathbin{\dot{\cup}} \varnothing) = \mathbb{P}(\varnothing) + \mathbb{P}(\varnothing)$$
by countable additivity, which shows $\mathbb{P}(\varnothing) = 0$. ◻
:::

::: proposition
If $A$ and $B$ are events, then
$\mathbb{P}(A \cup B) = \mathbb{P}(A) + \mathbb{P}(B) - \mathbb{P}(A \cap B)$.
:::

::: proof
*Proof.* The key is to break the events up into their various
overlapping and non-overlapping parts. $$\begin{aligned}
\mathbb{P}(A \cup B) &= \mathbb{P}((A \cap B) \mathbin{\dot{\cup}} (A \setminus B) \mathbin{\dot{\cup}} (B \setminus A)) \\
&= \mathbb{P}(A \cap B) + \mathbb{P}(A \setminus B) + \mathbb{P}(B \setminus A) \\
&= \mathbb{P}(A \cap B) + \mathbb{P}(A) - \mathbb{P}(A \cap B) + \mathbb{P}(B) - \mathbb{P}(A \cap B) \\
&= \mathbb{P}(A) + \mathbb{P}(B) - \mathbb{P}(A \cap B)
\end{aligned}$$ ◻
:::

::: proposition
If $\{A_i\} \subseteq \mathcal{F}$ is a countable set of events,
disjoint or not, then
$$\mathbb{P}\bigg(\bigcup_i A_i\bigg) \leq \sum_i \mathbb{P}(A_i)$$
:::

This inequality is sometimes referred to as **Boole's inequality** or
the **union bound**.

::: proof
*Proof.* Define $B_1 = A_1$ and
$B_i = A_i \setminus (\bigcup_{j < i} A_j)$ for $i > 1$, noting that
$\bigcup_{j \leq i} B_j = \bigcup_{j \leq i} A_j$ for all $i$ and the
$B_i$ are disjoint. Then
$$\mathbb{P}\bigg(\bigcup_i A_i\bigg) = \mathbb{P}\bigg(\bigcup_i B_i\bigg) = \sum_i \mathbb{P}(B_i) \leq \sum_i \mathbb{P}(A_i)$$
where the last inequality follows by monotonicity since
$B_i \subseteq A_i$ for all $i$. ◻
:::

### Conditional probability

The **conditional probability** of event $A$ given that event $B$ has
occurred is written $\mathbb{P}(A | B)$ and defined as
$$\mathbb{P}(A | B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)}$$
assuming $\mathbb{P}(B) > 0$.[^10]

### Chain rule

Another very useful tool, the **chain rule**, follows immediately from
this definition:
$$\mathbb{P}(A \cap B) = \mathbb{P}(A | B)\mathbb{P}(B) = \mathbb{P}(B | A)\mathbb{P}(A)$$

### Bayes' rule

Taking the equality from above one step further, we arrive at the simple
but crucial **Bayes' rule**:
$$\mathbb{P}(A | B) = \frac{\mathbb{P}(B | A)\mathbb{P}(A)}{\mathbb{P}(B)}$$
It is sometimes beneficial to omit the normalizing constant and write
$$\mathbb{P}(A | B) \propto \mathbb{P}(A)\mathbb{P}(B | A)$$ Under this
formulation, $\mathbb{P}(A)$ is often referred to as the **prior**,
$\mathbb{P}(A | B)$ as the **posterior**, and $\mathbb{P}(B | A)$ as the
**likelihood**.

In the context of machine learning, we can use Bayes' rule to update our
"beliefs" (e.g. values of our model parameters) given some data that
we've observed.

## Random variables

A **random variable** is some uncertain quantity with an associated
probability distribution over the values it can assume.

Formally, a random variable on a probability space
$(\Omega, \mathcal{F}, \mathbb{P})$ is a function[^11]
$X: \Omega \to \mathbb{R}$.[^12]

We denote the range of $X$ by
$X(\Omega) = \{X(\omega) : \omega \in \Omega\}$. To give a concrete
example (taken from [@pitman]), suppose $X$ is the number of heads in
two tosses of a fair coin. The sample space is
$$\Omega = \{hh, tt, ht, th\}$$ and $X$ is determined completely by the
outcome $\omega$, i.e. $X = X(\omega)$. For example, the event $X = 1$
is the set of outcomes $\{ht, th\}$.

It is common to talk about the values of a random variable without
directly referencing its sample space. The two are related by the
following definition: the event that the value of $X$ lies in some set
$S \subseteq \mathbb{R}$ is
$$X \in S = \{\omega \in \Omega : X(\omega) \in S\}$$ Note that special
cases of this definition include $X$ being equal to, less than, or
greater than some specified value. For example
$$\mathbb{P}(X = x) = \mathbb{P}(\{\omega \in \Omega : X(\omega) = x\})$$

A word on notation: we write $p(X)$ to denote the entire probability
distribution of $X$ and $p(x)$ for the evaluation of the function $p$ at
a particular value $x \in X(\Omega)$. Hopefully this (reasonably
standard) abuse of notation is not too distracting. If $p$ is
parameterized by some parameters $\theta$, we write
$p(X; \mathbf{\theta})$ or $p(x; \mathbf{\theta})$, unless we are in a
Bayesian setting where the parameters are considered a random variable,
in which case we condition on the parameters.

### The cumulative distribution function

The **cumulative distribution function** (c.d.f.) gives the probability
that a random variable is at most a certain value:
$$F(x) = \mathbb{P}(X \leq x)$$ The c.d.f. can be used to give the
probability that a variable lies within a certain range:
$$\mathbb{P}(a < X \leq b) = F(b) - F(a)$$

### Discrete random variables

A **discrete random variable** is a random variable that has a countable
range and assumes each value in this range with positive probability.
Discrete random variables are completely specified by their
**probability mass function** (p.m.f.) $p : X(\Omega) \to [0,1]$ which
satisfies $$\sum_{x \in X(\Omega)} p(x) = 1$$ For a discrete $X$, the
probability of a particular value is given exactly by its p.m.f.:
$$\mathbb{P}(X = x) = p(x)$$

### Continuous random variables

A **continuous random variable** is a random variable that has an
uncountable range and assumes each value in this range with probability
zero. Most of the continuous random variables that one would encounter
in practice are **absolutely continuous random variables**[^13], which
means that there exists a function $p : \mathbb{R} \to [0,\infty)$ that
satisfies $$F(x) \equiv \int_{-\infty}^x p(z)\dd{z}$$ The function $p$
is called a **probability density function** (abbreviated p.d.f.) and
must satisfy $$\int_{-\infty}^\infty p(x)\dd{x} = 1$$ The values of this
function are not themselves probabilities, since they could exceed 1.
However, they do have a couple of reasonable interpretations. One is as
relative probabilities; even though the probability of each particular
value being picked is technically zero, some points are still in a sense
more likely than others.

One can also think of the density as determining the probability that
the variable will lie in a small range about a given value. This is
because, for small $\epsilon > 0$,
$$\mathbb{P}(x-\epsilon \leq X \leq x+\epsilon) = \int_{x-\epsilon}^{x+\epsilon} p(z)\dd{z} \approx 2\epsilon p(x)$$
using a midpoint approximation to the integral.

Here are some useful identities that follow from the definitions above:
$$\begin{aligned}
\mathbb{P}(a \leq X \leq b) &= \int_a^b p(x)\dd{x} \\
p(x) &= F'(x)
\end{aligned}$$

### Other kinds of random variables

There are random variables that are neither discrete nor continuous. For
example, consider a random variable determined as follows: flip a fair
coin, then the value is zero if it comes up heads, otherwise draw a
number uniformly at random from $[1,2]$. Such a random variable can take
on uncountably many values, but only finitely many of these with
positive probability. We will not discuss such random variables because
they are rather pathological and require measure theory to analyze.

## Joint distributions

Often we have several random variables and we would like to get a
distribution over some combination of them. A **joint distribution** is
exactly this. For some random variables $X_1, \dots, X_n$, the joint
distribution is written $p(X_1, \dots, X_n)$ and gives probabilities
over entire assignments to all the $X_i$ simultaneously.

### Independence of random variables

We say that two variables $X$ and $Y$ are **independent** if their joint
distribution factors into their respective distributions, i.e.
$$p(X, Y) = p(X)p(Y)$$ We can also define independence for more than two
random variables, although it is more complicated. Let
$\{X_i\}_{i \in I}$ be a collection of random variables indexed by $I$,
which may be infinite. Then $\{X_i\}$ are independent if for every
finite subset of indices $i_1, \dots, i_k \in I$ we have
$$p(X_{i_1}, \dots, X_{i_k}) = \prod_{j=1}^k p(X_{i_j})$$ For example,
in the case of three random variables, $X, Y, Z$, we require that
$p(X,Y,Z) = p(X)p(Y)p(Z)$ as well as $p(X,Y) = p(X)p(Y)$,
$p(X,Z) = p(X)p(Z)$, and $p(Y,Z) = p(Y)p(Z)$.

It is often convenient (though perhaps questionable) to assume that a
bunch of random variables are **independent and identically
distributed** (i.i.d.) so that their joint distribution can be factored
entirely: $$p(X_1, \dots, X_n) = \prod_{i=1}^n p(X_i)$$ where
$X_1, \dots, X_n$ all share the same p.m.f./p.d.f.

### Marginal distributions

If we have a joint distribution over some set of random variables, it is
possible to obtain a distribution for a subset of them by "summing out"
(or "integrating out" in the continuous case) the variables we don't
care about: $$p(X) = \sum_{y} p(X, y)$$

## Great Expectations

If we have some random variable $X$, we might be interested in knowing
what is the "average" value of $X$. This concept is captured by the
**expected value** (or **mean**) $\mathbb{E}[X]$, which is defined as
$$\mathbb{E}[X] = \sum_{x \in X(\Omega)} xp(x)$$ for discrete $X$ and as
$$\mathbb{E}[X] = \int_{-\infty}^\infty xp(x)\dd{x}$$ for continuous
$X$.

In words, we are taking a weighted sum of the values that $X$ can take
on, where the weights are the probabilities of those respective values.
The expected value has a physical interpretation as the "center of mass"
of the distribution.

### Properties of expected value

A very useful property of expectation is that of linearity:
$$\mathbb{E}\left[\sum_{i=1}^n \alpha_i X_i + \beta\right] = \sum_{i=1}^n \alpha_i \mathbb{E}[X_i] + \beta$$
Note that this holds even if the $X_i$ are not independent!

But if they are independent, the product rule also holds:
$$\mathbb{E}\left[\prod_{i=1}^n X_i\right] = \prod_{i=1}^n \mathbb{E}[X_i]$$

## Variance

Expectation provides a measure of the "center" of a distribution, but
frequently we are also interested in what the "spread" is about that
center. We define the variance $\operatorname{Var}(X)$ of a random
variable $X$ by
$$\operatorname{Var}(X) = \mathbb{E}\left[\left(X - \mathbb{E}[X]\right)^2\right]$$
In words, this is the average squared deviation of the values of $X$
from the mean of $X$. Using a little algebra and the linearity of
expectation, it is straightforward to show that
$$\operatorname{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$

### Properties of variance

Variance is not linear (because of the squaring in the definition), but
one can show the following:
$$\operatorname{Var}(\alpha X + \beta) = \alpha^2 \operatorname{Var}(X)$$
Basically, multiplicative constants become squared when they are pulled
out, and additive constants disappear (since the variance contributed by
a constant is zero).

Furthermore, if $X_1, \dots, X_n$ are uncorrelated[^14], then
$$\operatorname{Var}(X_1 + \dots + X_n) = \operatorname{Var}(X_1) + \dots + \operatorname{Var}(X_n)$$

### Standard deviation

Variance is a useful notion, but it suffers from that fact the units of
variance are not the same as the units of the random variable (again
because of the squaring). To overcome this problem we can use **standard
deviation**, which is defined as $\sqrt{\operatorname{Var}(X)}$. The
standard deviation of $X$ has the same units as $X$.

## Covariance

Covariance is a measure of the linear relationship between two random
variables. We denote the covariance between $X$ and $Y$ as
$\operatorname{Cov}(X, Y)$, and it is defined to be
$$\operatorname{Cov}(X, Y) = \mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]$$
Note that the outer expectation must be taken over the joint
distribution of $X$ and $Y$.

Again, the linearity of expectation allows us to rewrite this as
$$\operatorname{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$
Comparing these formulas to the ones for variance, it is not hard to see
that $\operatorname{Var}(X) = \operatorname{Cov}(X, X)$.

A useful property of covariance is that of **bilinearity**:
$$\begin{aligned}
\operatorname{Cov}(\alpha X + \beta Y, Z) &= \alpha\operatorname{Cov}(X, Z) + \beta\operatorname{Cov}(Y, Z) \\
\operatorname{Cov}(X, \alpha Y + \beta Z) &= \alpha\operatorname{Cov}(X, Y) + \beta\operatorname{Cov}(X, Z)
\end{aligned}$$

### Correlation

Normalizing the covariance gives the **correlation**:
$$\rho(X, Y) = \frac{\operatorname{Cov}(X, Y)}{\sqrt{\operatorname{Var}(X)\operatorname{Var}(Y)}}$$
Correlation also measures the linear relationship between two variables,
but unlike covariance always lies between $-1$ and $1$.

Two variables are said to be **uncorrelated** if
$\operatorname{Cov}(X, Y) = 0$ because $\operatorname{Cov}(X, Y) = 0$
implies that $\rho(X, Y) = 0$. If two variables are independent, then
they are uncorrelated, but the converse does not hold in general.

## Random vectors

So far we have been talking about **univariate distributions**, that is,
distributions of single variables. But we can also talk about
**multivariate distributions** which give distributions of **random
vectors**:
$$\mathbf{X} = \begin{bmatrix}X_1 \\ \vdots \\ X_n\end{bmatrix}$$ The
summarizing quantities we have discussed for single variables have
natural generalizations to the multivariate case.

Expectation of a random vector is simply the expectation applied to each
component:
$$\mathbb{E}[\mathbf{X}] = \begin{bmatrix}\mathbb{E}[X_1] \\ \vdots \\ \mathbb{E}[X_n]\end{bmatrix}$$

The variance is generalized by the **covariance matrix**:
$$\mathbf{\Sigma} = \mathbb{E}[(\mathbf{X} - \mathbb{E}[\mathbf{X}])(\mathbf{X} - \mathbb{E}[\mathbf{X}])^{\!\top\!}] = \begin{bmatrix}
\operatorname{Var}(X_1) & \operatorname{Cov}(X_1, X_2) & \hdots & \operatorname{Cov}(X_1, X_n) \\
\operatorname{Cov}(X_2, X_1) & \operatorname{Var}(X_2) & \hdots & \operatorname{Cov}(X_2, X_n) \\
\vdots & \vdots & \ddots & \vdots \\
\operatorname{Cov}(X_n, X_1) & \operatorname{Cov}(X_n, X_2) & \hdots & \operatorname{Var}(X_n)
\end{bmatrix}$$ That is, $\Sigma_{ij} = \operatorname{Cov}(X_i, X_j)$.
Since covariance is symmetric in its arguments, the covariance matrix is
also symmetric. It's also positive semi-definite: for any $\mathbf{x}$,
$$\mathbf{x}^{\!\top\!}\mathbf{\Sigma}\mathbf{x} = \mathbf{x}^{\!\top\!}\mathbb{E}[(\mathbf{X} - \mathbb{E}[\mathbf{X}])(\mathbf{X} - \mathbb{E}[\mathbf{X}])^{\!\top\!}]\mathbf{x} = \mathbb{E}[\mathbf{x}^{\!\top\!}(\mathbf{X} - \mathbb{E}[\mathbf{X}])(\mathbf{X} - \mathbb{E}[\mathbf{X}])^{\!\top\!}\mathbf{x}] = \mathbb{E}[((\mathbf{X} - \mathbb{E}[\mathbf{X}])^{\!\top\!}\mathbf{x})^2] \geq 0$$
The inverse of the covariance matrix, $\mathbf{\Sigma}^{-1}$, is
sometimes called the **precision matrix**.

## Estimation of Parameters

Now we get into some basic topics from statistics. We make some
assumptions about our problem by prescribing a **parametric** model
(e.g. a distribution that describes how the data were generated), then
we fit the parameters of the model to the data. How do we choose the
values of the parameters?

### Maximum likelihood estimation

A common way to fit parameters is **maximum likelihood estimation**
(MLE). The basic principle of MLE is to choose values that "explain" the
data best by maximizing the probability/density of the data we've seen
as a function of the parameters. Suppose we have random variables
$X_1, \dots, X_n$ and corresponding observations $x_1, \dots, x_n$. Then
$$\hat{\mathbf{\theta}}_\textsc{mle} = \argmax_\mathbf{\theta} \mathcal{L}(\mathbf{\theta})$$
where $\mathcal{L}$ is the **likelihood function**
$$\mathcal{L}(\mathbf{\theta}) = p(x_1, \dots, x_n; \mathbf{\theta})$$
Often, we assume that $X_1, \dots, X_n$ are i.i.d. Then we can write
$$p(x_1, \dots, x_n; \theta) = \prod_{i=1}^n p(x_i; \mathbf{\theta})$$
At this point, it is usually convenient to take logs, giving rise to the
**log-likelihood**
$$\log\mathcal{L}(\mathbf{\theta}) = \sum_{i=1}^n \log p(x_i; \mathbf{\theta})$$
This is a valid operation because the probabilities/densities are
assumed to be positive, and since log is a monotonically increasing
function, it preserves ordering. In other words, any maximizer of
$\log\mathcal{L}$ will also maximize $\mathcal{L}$.

For some distributions, it is possible to analytically solve for the
maximum likelihood estimator. If $\log\mathcal{L}$ is differentiable,
setting the derivatives to zero and trying to solve for
$\mathbf{\theta}$ is a good place to start.

### Maximum a posteriori estimation

A more Bayesian way to fit parameters is through **maximum a posteriori
estimation** (MAP). In this technique we assume that the parameters are
a random variable, and we specify a prior distribution
$p(\mathbf{\theta})$. Then we can employ Bayes' rule to compute the
posterior distribution of the parameters given the observed data:
$$p(\mathbf{\theta} | x_1, \dots, x_n) \propto p(\mathbf{\theta})p(x_1, \dots, x_n | \mathbf{\theta})$$
Computing the normalizing constant is often intractable, because it
involves integrating over the parameter space, which may be very
high-dimensional. Fortunately, if we just want the MAP estimate, we
don't care about the normalizing constant! It does not affect which
values of $\mathbf{\theta}$ maximize the posterior. So we have
$$\hat{\mathbf{\theta}}_\textsc{map} = \argmax_\mathbf{\theta} p(\mathbf{\theta})p(x_1, \dots, x_n | \mathbf{\theta})$$
Again, if we assume the observations are i.i.d., then we can express
this in the equivalent, and possibly friendlier, form
$$\hat{\mathbf{\theta}}_\textsc{map} = \argmax_\mathbf{\theta} \left(\log p(\mathbf{\theta}) + \sum_{i=1}^n \log p(x_i | \mathbf{\theta})\right)$$
A particularly nice case is when the prior is chosen carefully such that
the posterior comes from the same family as the prior. In this case the
prior is called a **conjugate prior**. For example, if the likelihood is
binomial and the prior is beta, the posterior is also beta. There are
many conjugate priors; the reader may find this [table of conjugate
priors](https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions)
useful.

## The Gaussian distribution

There are many distributions, but one of particular importance is the
**Gaussian distribution**, also known as the **normal distribution**. It
is a continuous distribution, parameterized by its mean
$\bm\mu \in \mathbb{R}^d$ and positive-definite covariance matrix
$\mathbf{\Sigma} \in \mathbb{R}^{d \times d}$, with density
$$p(\mathbf{x}; \bm\mu, \mathbf{\Sigma}) = \frac{1}{\sqrt{(2\pi)^d \det(\mathbf{\Sigma})}}\exp\left(-\frac{1}{2}(\mathbf{x} - \bm\mu)^{\!\top\!}\mathbf{\Sigma}^{-1}(\mathbf{x} - \bm\mu)\right)$$
Note that in the special case $d = 1$, the density is written in the
more recognizable form
$$p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
We write $\mathbf{X} \sim \mathcal{N}(\bm\mu, \mathbf{\Sigma})$ to
denote that $\mathbf{X}$ is normally distributed with mean $\bm\mu$ and
variance $\mathbf{\Sigma}$.

### The geometry of multivariate Gaussians

The geometry of the multivariate Gaussian density is intimately related
to the geometry of positive definite quadratic forms, so make sure the
material in that section is well-understood before tackling this
section.

First observe that the p.d.f. of the multivariate Gaussian can be
rewritten as
$$p(\mathbf{x}; \bm\mu, \mathbf{\Sigma}) = g(\tilde{\mathbf{x}}^{\!\top\!}\mathbf{\Sigma}^{-1}\tilde{\mathbf{x}})$$
where $\tilde{\mathbf{x}} = \mathbf{x} - \bm\mu$ and
$g(z) = [(2\pi)^d \det(\mathbf{\Sigma})]^{-\frac{1}{2}}\exp\left(-\frac{z}{2}\right)$.
Writing the density in this way, we see that after shifting by the mean
$\bm\mu$, the density is really just a simple function of its precision
matrix's quadratic form.

Here is a key observation: this function $g$ is **strictly monotonically
decreasing** in its argument. That is, $g(a) > g(b)$ whenever $a < b$.
Therefore, small values of
$\tilde{\mathbf{x}}^{\!\top\!}\mathbf{\Sigma}^{-1}\tilde{\mathbf{x}}$
(which generally correspond to points where $\tilde{\mathbf{x}}$ is
closer to $\mathbf{0}$, i.e. $\mathbf{x} \approx \bm\mu$) have
relatively high probability densities, and vice-versa. Furthermore,
because $g$ is *strictly* monotonic, it is injective, so the
$c$-isocontours of $p(\mathbf{x}; \bm\mu, \mathbf{\Sigma})$ are the
$g^{-1}(c)$-isocontours of the function
$\mathbf{x} \mapsto \tilde{\mathbf{x}}^{\!\top\!}\mathbf{\Sigma}^{-1}\tilde{\mathbf{x}}$.
That is, for any $c$,
$$\{\mathbf{x} \in \mathbb{R}^d : p(\mathbf{x}; \bm\mu, \mathbf{\Sigma}) = c\} = \{\mathbf{x} \in \mathbb{R}^d : \tilde{\mathbf{x}}^{\!\top\!}\mathbf{\Sigma}^{-1}\tilde{\mathbf{x}} = g^{-1}(c)\}$$
In words, these functions have the same isocontours but different
isovalues.

Recall the executive summary of the geometry of positive definite
quadratic forms: the isocontours of
$f(\mathbf{x}) = \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}$ are
ellipsoids such that the axes point in the directions of the
eigenvectors of $\mathbf{A}$, and the lengths of these axes are
proportional to the inverse square roots of the corresponding
eigenvalues. Therefore in this case, the isocontours of the density are
ellipsoids (centered at $\bm\mu$) with axis lengths proportional to the
inverse square roots of the eigenvalues of $\mathbf{\Sigma}^{-1}$, or
equivalently, the square roots of the eigenvalues of $\mathbf{\Sigma}$.

[^1]: More generally, vector spaces can be defined over any **field**
    $\mathbb{F}$. We take $\mathbb{F} = \mathbb{R}$ in this document to
    avoid an unnecessary diversion into abstract algebra.

[^2]: It is sometimes called the **kernel** by algebraists, but we
    eschew this terminology because the word "kernel" has another
    meaning in machine learning.

[^3]: If a normed space is complete with respect to the distance metric
    induced by its norm, we say that it is a **Banach space**.

[^4]: If an inner product space is complete with respect to the distance
    metric induced by its inner product, we say that it is a **Hilbert
    space**.

[^5]: Recall that $\mathbf{A}^{\!\top\!}\mathbf{A}$ and
    $\mathbf{A}\mathbf{A}^{\!\top\!}$ are positive semi-definite, so
    their eigenvalues are nonnegative, and thus taking square roots is
    always well-defined.

[^6]: A **neighborhood** about $\mathbf{x}$ is an open set which
    contains $\mathbf{x}$.

[^7]: $\mathcal{F}$ is required to be a $\sigma$-algebra for technical
    reasons; see [@rigorousprob].

[^8]: Note that a probability space is simply a measure space in which
    the measure of the whole space equals 1.

[^9]: This is a probabilist's version of the measure-theoretic term
    *almost everywhere*.

[^10]: In some cases it is possible to define conditional probability on
    events of probability zero, but this is significantly more technical
    so we omit it.

[^11]: The function must be measurable.

[^12]: More generally, the codomain can be any measurable space, but
    $\mathbb{R}$ is the most common case by far and sufficient for our
    purposes.

[^13]: Random variables that are continuous but not absolutely
    continuous are called **singular random variables**. We will not
    discuss them, assuming rather that all continuous random variables
    admit a density function.

[^14]: We haven't defined this yet; see the Correlation section below
