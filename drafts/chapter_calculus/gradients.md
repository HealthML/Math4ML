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

$$\nabla f = \begin{bmatrix}\frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n}\end{bmatrix}
\hspace{0.5cm}\text{i.e.}\hspace{0.5cm}
[\nabla f]_i = \frac{\partial f}{\partial x_i}$$

Gradients have the following very
important property: $\nabla f(\mathbf{x})$ points in the direction of
**steepest ascent** from $\mathbf{x}$. Similarly,
$-\nabla f(\mathbf{x})$ points in the direction of **steepest descent**
from $\mathbf{x}$. We will use this fact frequently when iteratively
minimizing a function via **gradient descent**.

## The Jacobian

The **Jacobian** of $f : \mathbb{R}^n \to \mathbb{R}^m$ is a matrix of
first-order partial derivatives: 

$$\mathbf{J}_f = \begin{bmatrix}
    \frac{\partial f_1}{\partial x_1} & \dots & \frac{\partial f_1}{\partial x_n} \\
    \vdots & \ddots & \vdots \\
    \frac{\partial f_m}{\partial x_1} & \dots & \frac{\partial f_m}{\partial x_n}\end{bmatrix}
\hspace{0.5cm}\text{i.e.}\hspace{0.5cm}
[\mathbf{J}_f]_{ij} = \frac{\partial f_i}{\partial x_j}$$

Note the special case $m = 1$,
where $\nabla f = \mathbf{J}_f^{\!\top\!}$.

