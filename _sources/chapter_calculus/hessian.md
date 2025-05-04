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
# The Hessian

The **Hessian** matrix of a scalar-valued function $ f : \mathbb{R}^d \to \mathbb{R} $ is a square matrix
of second-order partial derivatives:

$$\nabla^2 f(\mathbf{x}) = \begin{bmatrix}
    \frac{\partial^2 f}{\partial x_1^2} & \dots & \frac{\partial^2 f}{\partial x_1 \partial x_d} \\
    \vdots & \ddots & \vdots \\
    \frac{\partial^2 f}{\partial x_d \partial x_1} & \dots & \frac{\partial^2 f}{\partial x_d^2}
\end{bmatrix}, \quad\text{i.e.,}\quad
[\nabla^2 f]_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j} $$

If the second partial derivatives are continuous (as they often are in optimization), then by **Clairaut's theorem**,
the order of differentiation can be interchanged: $ \frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i} $,
which implies that the Hessian matrix is symmetric.

---

## First-Order Taylor Expansion

Recall, that we can create a locally linear approximation to a function at a point $\mathbf{x}_0 \in \mathbb{R}^d $ using the gradient at $\nabla f(\mathbf{x}_0)$.

$$ f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top (\mathbf{x} - \mathbf{x}_0) . $$

This affine approximation is also known as the **first-order Taylor approximation**.

It agrees with the original function in value and gradient at the point $ \mathbf{x}_0 $.
## Second-Order Taylor Expansion

The Hessian appears naturally in the **second-order Taylor approximation** of a function around a point $ \mathbf{x}_0 \in \mathbb{R}^d $.
For a sufficiently smooth function $ f : \mathbb{R}^d \to \mathbb{R} $, we can approximate its values near $ \mathbf{x}_0 $ as:

$$ f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top (\mathbf{x} - \mathbf{x}_0) + \frac{1}{2}(\mathbf{x} - \mathbf{x}_0)^\top \nabla^2 f(\mathbf{x}_0)(\mathbf{x} - \mathbf{x}_0). $$

This is a **local quadratic approximation** to the function. It agrees with the original function in value, gradient, and Hessian at the point $ \mathbf{x}_0 $.

### Interpretation:
- The **gradient** term captures the linear behavior (slope) of the function near $ \mathbf{x}_0 $.
- The **Hessian** term captures the curvature — how the gradient changes in different directions.

---

We illustrate both the first-order and the second-order Taylor expansion using the following function.

$$ f(x, y) = \log(1 + x^2 + y^2) $$

We compute the first-order and second-order Taylor approximations at the point $ (x_0, y_0) = (0.3, 0.3) $.

The true function and the linear approximation match in value and gradient at the point $ (x_0, y_0)$ but differ elsewhere. Similarly, the quadratic approximation match in value, gradient, and Hessian at this point but differ elsewhere.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function and its gradient and Hessian
f = lambda x, y: np.log(1 + x**2 + y**2)
x0, y0 = 0.3, 0.3

# Compute value, gradient, and Hessian at (x0, y0)
r2 = x0**2 + y0**2
f0 = np.log(1 + r2)
grad = np.array([2*x0, 2*y0]) / (1 + r2)
H = (2 / (1 + r2)) * np.eye(2) - (4 * np.outer([x0, y0], [x0, y0])) / (1 + r2)**2

# Taylor expansion up to second order
def f_taylor_first_order(x, y):
    dx = x - x0
    dy = y - y0
    delta = np.array([dx, dy])
    return f0 + (grad @ delta).sum()

# Taylor expansion up to second order
def f_taylor_second_order(x, y):
    dx = x - x0
    dy = y - y0
    delta = np.array([dx, dy])
    return f0 + (grad @ delta).sum() + 0.5 * (delta @ H @ delta).sum()

# Create grid for plotting
x_vals = np.linspace(x0-1, x0+1, 100)
y_vals = np.linspace(y0-1, y0+1, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z_true = f(X,Y)
Z_first = np.zeros(X.shape)
Z_second = np.zeros(X.shape)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z_first[i,j] = f_taylor_first_order(X[i,j],Y[i,j])
        Z_second[i,j] = f_taylor_second_order(X[i,j],Y[i,j])

# Plot both Taylor approximations
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

true_surface1 = ax1.plot_surface(X, Y, Z_true, cmap='viridis', alpha=0.6)
approx_surface1 = ax1.plot_surface(X, Y, Z_first, cmap='coolwarm', alpha=0.7)
ax1.scatter(x0, y0, f0, color='red', s=50, label=r'$\mathbf{x}_0$')
ax1.set_title("First-Order Taylor Approximation")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend()
ax1.set_zlim([-0.5,2])

true_surface2 = ax2.plot_surface(X, Y, Z_true, cmap='viridis', alpha=0.6)
approx_surface2 = ax2.plot_surface(X, Y, Z_second, cmap='coolwarm', alpha=0.7)
ax2.scatter(x0, y0, f0, color='red', s=50, label=r'$\mathbf{x}_0$')
ax2.set_title("Second-Order Taylor Approximation")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.legend()
ax2.set_zlim([-0.5,2])

plt.tight_layout()
plt.show()
```
This visualization shows how the first-order (left) and second-order (right) Taylor expansions approximate the original function locally around the point $ (0.3,0.3) $, but deviates farther away. Both approximations are shown in blue to red and the original function in yellow to green colors.

## Summary and Outlook

An advantage of a local quadratic approximation is that we can find its minimum analytically.
This idea lies at the heart of **Newton's method**.
The Hessian matrix also allows us also to better understand the properties of stationary points of a function and derive **second-order conditions of minima**.

Before we will explore these two topics further, we first have to better understand the **properties of matrices** such as the Hessian. So let's turn to the topic of **matrix algebra**.
