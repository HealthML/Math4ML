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

## The Hessian

The **Hessian** matrix of $f : \mathbb{R}^d \to \mathbb{R}$ is a matrix
of second-order partial derivatives: 

$$\nabla^2 f = \begin{bmatrix}
    \frac{\partial^2 f}{\partial x_1^2} & \dots & \frac{\partial^2 f}{\partial x_1 \partial x_d} \\
    \vdots & \ddots & \vdots \\
    \frac{\partial^2 f}{\partial x_d \partial x_1} & \dots & \frac{\partial^2 f}{\partial x_d^2}\end{bmatrix}
\hspace{0.5cm}\text{i.e.}\hspace{0.5cm}
[\nabla^2 f]_{ij} = {\frac{\partial^2 f}{\partial x_i \partial x_j}}$$ 

Recall that if the partial
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
most important for our purposes are 

$$\begin{aligned}
\nabla_\mathbf{x} &(\mathbf{a}^{\!\top\!}\mathbf{x}) = \mathbf{a} \\
\nabla_\mathbf{x} &(\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}) = (\mathbf{A} + \mathbf{A}^{\!\top\!})\mathbf{x}
\end{aligned}$$ 

Note that this second rule is defined only if
$\mathbf{A}$ is square. Furthermore, if $\mathbf{A}$ is symmetric, we
can simplify the result to $2\mathbf{A}\mathbf{x}$.

### The chain rule

Most functions that we wish to optimize are not completely arbitrary
functions, but rather are composed of simpler functions which we know
how to handle. The chain rule gives us a way to calculate derivatives
for a composite function in terms of the derivatives of the simpler
functions that make it up.

The chain rule from single-variable calculus should be familiar:

$$(f \circ g)'(x) = f'(g(x))g'(x)$$ 

where $\circ$ denotes function
composition. There is a natural generalization of this rule to
multivariate functions.

*Proposition.*
Suppose $f : \mathbb{R}^m \to \mathbb{R}^k$ and
$g : \mathbb{R}^n \to \mathbb{R}^m$. Then
$f \circ g : \mathbb{R}^n \to \mathbb{R}^k$ and

$$\mathbf{J}_{f \circ g}(\mathbf{x}) = \mathbf{J}_f(g(\mathbf{x}))\mathbf{J}_g(\mathbf{x})$$


In the special case $k = 1$ we have the following corollary since
$\nabla f = \mathbf{J}_f^{\!\top\!}$.

 corollary
Suppose $f : \mathbb{R}^m \to \mathbb{R}$ and
$g : \mathbb{R}^n \to \mathbb{R}^m$. Then
$f \circ g : \mathbb{R}^n \to \mathbb{R}$ and

$$\nabla (f \circ g)(\mathbf{x}) = \mathbf{J}_g(\mathbf{x})^{\!\top\!} \nabla f(g(\mathbf{x}))$$
