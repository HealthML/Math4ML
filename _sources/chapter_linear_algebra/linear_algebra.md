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
numbers:
$$\mathbf{x} = (x_1, x_2, \dots, x_n)$$

For our purposes, it
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
If $T : V \to W$ is a linear map, we define the **nullspace** (or **kernel**) of $T$
as

$\operatorname{null}(T) = \{\mathbf{x} \in V \mid T\mathbf{x} = \mathbf{0}\}.$

and
the **range** (or the **columnspace** if we are considering the matrix
form) of $T$ as

$\operatorname{range}(T) = \{\mathbf{y} \in W \mid \exists \mathbf{x} \in V : T\mathbf{x} = \mathbf{y}\}.$

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

$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x}-\mathbf{y}\|$

One can verify that the axioms for metrics are satisfied under this definition and
follow directly from the axioms for norms. Therefore any normed space is
also a metric space.[^3]

We will typically only be concerned with a few specific norms on
$\mathbb{R}^n$:
$$\begin{aligned}
\|\mathbf{x}\|_1 &= \sum_{i=1}^n |x_i| \\
\|\mathbf{x}\|_2 &= \sqrt{\sum_{i=1}^n x_i^2} \\
\|\mathbf{x}\|_p &= \left(\sum_{i=1}^n |x_i|^p\right)^\frac{1}{p} \hspace{0.5cm}\hspace{0.5cm} (p \geq 1) \\
\|\mathbf{x}\|_\infty &= \max_{1 \leq i \leq n} |x_i|
\end{aligned}$$

Note that the 1- and 2-norms are special cases of the
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
$$\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$$

One can verify that the axioms for norms are satisfied under this definition
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

*Theorem.* 
If $\mathbf{x} \perp \mathbf{y}$, then
$$\|\mathbf{x}+\mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2.$$


*Proof.* Suppose $\mathbf{x} \perp \mathbf{y}$, i.e.
$\langle \mathbf{x}, \mathbf{y} \rangle = 0$. Then
$$\|\mathbf{x}+\mathbf{y}\|^2 = \langle \mathbf{x}+\mathbf{y}, \mathbf{x}+\mathbf{y} \rangle = \langle \mathbf{x}, \mathbf{x} \rangle + \langle \mathbf{y}, \mathbf{x} \rangle + \langle \mathbf{x}, \mathbf{y} \rangle + \langle \mathbf{y}, \mathbf{y} \rangle = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2$$

as claimed. ◻

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
$$\mathbf{A}\mathbf{x} = \lambda\mathbf{x}$$

The zero vector is excluded
from this definition because
$\mathbf{A}\mathbf{0} = \mathbf{0} = \lambda\mathbf{0}$ for every
$\lambda$.

We now give some useful results about how eigenvalues change after
various manipulations.

*Proposition.*
Let $\mathbf{x}$ be an eigenvector of $\mathbf{A}$ with corresponding
eigenvalue $\lambda$. Then

(i) For any $\gamma \in \mathbb{R}$, $\mathbf{x}$ is an eigenvector of
    $\mathbf{A} + \gamma\mathbf{I}$ with eigenvalue $\lambda + \gamma$.

(ii) If $\mathbf{A}$ is invertible, then $\mathbf{x}$ is an eigenvector
     of $\mathbf{A}^{-1}$ with eigenvalue $\lambda^{-1}$.

(iii) $\mathbf{A}^k\mathbf{x} = \lambda^k\mathbf{x}$ for any
      $k \in \mathbb{Z}$ (where $\mathbf{A}^0 = \mathbf{I}$ by
      definition).


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


## Trace

The **trace** of a square matrix is the sum of its diagonal entries:
$$\operatorname{tr}(\mathbf{A}) = \sum_{i=1}^n A_{ii}$$

The trace has several nice
algebraic properties:

(i) $\operatorname{tr}(\mathbf{A}+\mathbf{B}) = \operatorname{tr}(\mathbf{A}) + \operatorname{tr}(\mathbf{B})$

(ii) $\operatorname{tr}(\alpha\mathbf{A}) = \alpha\operatorname{tr}(\mathbf{A})$

(iii) $\operatorname{tr}(\mathbf{A}^{\!\top\!}) = \operatorname{tr}(\mathbf{A})$

(iv) $\operatorname{tr}(\mathbf{A}\mathbf{B}\mathbf{C}\mathbf{D}) = \operatorname{tr}(\mathbf{B}\mathbf{C}\mathbf{D}\mathbf{A}) = \operatorname{tr}(\mathbf{C}\mathbf{D}\mathbf{A}\mathbf{B}) = \operatorname{tr}(\mathbf{D}\mathbf{A}\mathbf{B}\mathbf{C})$

The first three properties follow readily from the definition. The last
is known as **invariance under cyclic permutations**. Note that the
matrices cannot be reordered arbitrarily, for example
$\operatorname{tr}(\mathbf{A}\mathbf{B}\mathbf{C}\mathbf{D}) \neq \operatorname{tr}(\mathbf{B}\mathbf{A}\mathbf{C}\mathbf{D})$
in general. Also, there is nothing special about the product of four
matrices -- analogous rules hold for more or fewer matrices.

Interestingly, the trace of a matrix is equal to the sum of its
eigenvalues (repeated according to multiplicity):
$$\operatorname{tr}(\mathbf{A}) = \sum_i \lambda_i(\mathbf{A})$$

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

*Theorem.* 
(Spectral Theorem) If $\mathbf{A} \in \mathbb{R}^{n \times n}$ is
symmetric, then there exists an orthonormal basis for $\mathbb{R}^n$
consisting of eigenvectors of $\mathbf{A}$.

The practical application of this theorem is a particular factorization
of symmetric matrices, referred to as the **eigendecomposition** or
**spectral decomposition**. Denote the orthonormal basis of eigenvectors
$\mathbf{q}_1, \dots, \mathbf{q}_n$ and their eigenvalues
$\lambda_1, \dots, \lambda_n$. Let $\mathbf{Q}$ be an orthogonal matrix
with $\mathbf{q}_1, \dots, \mathbf{q}_n$ as its columns, and
$\mathbf{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$. Since by
definition $\mathbf{A}\mathbf{q}_i = \lambda_i\mathbf{q}_i$ for every
$i$, the following relationship holds:
$$\mathbf{A}\mathbf{Q} = \mathbf{Q}\mathbf{\Lambda}$$

Right-multiplying
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

*Proposition.* 
A symmetric matrix is positive semi-definite if and only if all of its
eigenvalues are nonnegative, and positive definite if and only if all of
its eigenvalues are positive.


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


As an example of how these matrices arise, consider

*Proposition.*
Suppose $\mathbf{A} \in \mathbb{R}^{m \times n}$. Then
$\mathbf{A}^{\!\top\!}\mathbf{A}$ is positive semi-definite. If
$\operatorname{null}(\mathbf{A}) = \{\mathbf{0}\}$, then
$\mathbf{A}^{\!\top\!}\mathbf{A}$ is positive definite.


*Proof.* For any $\mathbf{x} \in \mathbb{R}^n$,
$$\mathbf{x}^{\!\top\!} (\mathbf{A}^{\!\top\!}\mathbf{A})\mathbf{x} = (\mathbf{A}\mathbf{x})^{\!\top\!}(\mathbf{A}\mathbf{x}) = \|\mathbf{A}\mathbf{x}\|_2^2 \geq 0$$

so $\mathbf{A}^{\!\top\!}\mathbf{A}$ is positive semi-definite.

Note that $\|\mathbf{A}\mathbf{x}\|_2^2 = 0$ implies
$\|\mathbf{A}\mathbf{x}\|_2 = 0$, which in turn implies
$\mathbf{A}\mathbf{x} = \mathbf{0}$ (recall that this is a property of
norms). If $\operatorname{null}(\mathbf{A}) = \{\mathbf{0}\}$,
$\mathbf{A}\mathbf{x} = \mathbf{0}$ implies $\mathbf{x} = \mathbf{0}$,
so
$\mathbf{x}^{\!\top\!} (\mathbf{A}^{\!\top\!}\mathbf{A})\mathbf{x} = 0$
if and only if $\mathbf{x} = \mathbf{0}$, and thus
$\mathbf{A}^{\!\top\!}\mathbf{A}$ is positive definite. ◻

Positive definite matrices are invertible (since their eigenvalues are
nonzero), whereas positive semi-definite matrices might not be. However,
if you already have a positive semi-definite matrix, it is possible to
perturb its diagonal slightly to produce a positive definite matrix.

*Proposition.* 
If $\mathbf{A}$ is positive semi-definite and $\epsilon > 0$, then
$\mathbf{A} + \epsilon\mathbf{I}$ is positive definite.

*Proof.* Assuming $\mathbf{A}$ is positive semi-definite and
$\epsilon > 0$, we have for any $\mathbf{x} \neq \mathbf{0}$ that
$$\mathbf{x}^{\!\top\!}(\mathbf{A}+\epsilon\mathbf{I})\mathbf{x} = \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x} + \epsilon\mathbf{x}^{\!\top\!}\mathbf{I}\mathbf{x} = \underbrace{\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}}_{\geq 0} + \underbrace{\epsilon\|\mathbf{x}\|_2^2}_{> 0} > 0$$

as claimed. ◻

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
$\{\mathbf{x} \in \operatorname{dom} f : f(\mathbf{x}) = c\}$.

Let us consider the special case
$f(\mathbf{x}) = \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}$ where
$\mathbf{A}$ is a positive definite matrix. Since $\mathbf{A}$ is
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
$$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\!\top\!}$$

where
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
\end{aligned}$$

It follows immediately that the columns of $\mathbf{V}$
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

*Proposition.* 
Let $\mathbf{x} \in \mathbb{R}^n$ be a vector and
$\mathbf{A} \in \mathbb{R}^{m \times n}$ a matrix with columns
$\mathbf{a}_1, \dots, \mathbf{a}_n$. Then
$$\mathbf{A}\mathbf{x} = \sum_{i=1}^n x_i\mathbf{a}_i$$

This identity is extremely useful in understanding linear operators in
terms of their matrices' columns. The proof is very simple (consider
each element of $\mathbf{A}\mathbf{x}$ individually and expand by
definitions) but it is a good exercise to convince yourself.

### Sum of outer products as matrix-matrix product

An **outer product** is an expression of the form
$\mathbf{a}\mathbf{b}^{\!\top\!}$, where $\mathbf{a} \in \mathbb{R}^m$
and $\mathbf{b} \in \mathbb{R}^n$. By inspection it is not hard to see
that such an expression yields an $m \times n$ matrix such that
$$[\mathbf{a}\mathbf{b}^{\!\top\!}]_{ij} = a_ib_j$$

It is not
immediately obvious, but the sum of outer products is actually
equivalent to an appropriate matrix-matrix product! We formalize this
statement as

*Proposition.* 
Let $\mathbf{a}_1, \dots, \mathbf{a}_k \in \mathbb{R}^m$ and
$\mathbf{b}_1, \dots, \mathbf{b}_k \in \mathbb{R}^n$. Then
$$\sum_{\ell=1}^k \mathbf{a}_\ell\mathbf{b}_\ell^{\!\top\!} = \mathbf{A}\mathbf{B}^{\!\top\!}$$

where
$$\mathbf{A} = \begin{bmatrix}\mathbf{a}_1 & \cdots & \mathbf{a}_k\end{bmatrix}, \hspace{0.5cm} \mathbf{B} = \begin{bmatrix}\mathbf{b}_1 & \cdots & \mathbf{b}_k\end{bmatrix}$$

*Proof.* For each $(i,j)$, we have
$$\left[\sum_{\ell=1}^k \mathbf{a}_\ell\mathbf{b}_\ell^{\!\top\!}\right]_{ij} = \sum_{\ell=1}^k [\mathbf{a}_\ell\mathbf{b}_\ell^{\!\top\!}]_{ij} = \sum_{\ell=1}^k [\mathbf{a}_\ell]_i[\mathbf{b}_\ell]_j = \sum_{\ell=1}^k A_{i\ell}B_{j\ell}$$

This last expression should be recognized as an inner product between
the $i$th row of $\mathbf{A}$ and the $j$th row of $\mathbf{B}$, or
equivalently the $j$th column of $\mathbf{B}^{\!\top\!}$. Hence by the
definition of matrix multiplication, it is equal to
$[\mathbf{A}\mathbf{B}^{\!\top\!}]_{ij}$. ◻

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

