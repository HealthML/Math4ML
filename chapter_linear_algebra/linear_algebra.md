# Linear Algebra

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
