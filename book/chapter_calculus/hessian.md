# The Hessian

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



