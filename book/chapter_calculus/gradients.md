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
# Gradients

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

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# Define a vector-valued function f: R2 -> R2
def f(x, y):
    # Example nonlinear function
    return np.array([x**2 - y, y**2 + x])

# Define the Jacobian of f
def jacobian(x, y):
    # df1/dx = 2x, df1/dy = -1
    # df2/dx = 1,  df2/dy = 2y
    return np.array([[2*x, -1],
                     [1,   2*y]])

# Set up grid for visualization
x_vals = np.linspace(-2, 2, 25)
y_vals = np.linspace(-2, 2, 25)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute Jacobian norm (Frobenius) at each grid point
J_norm = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        J = jacobian(X[i,j], Y[i,j])
        J_norm[i,j] = np.linalg.norm(J, ord='fro')

# Plot 1: Heatmap of Jacobian norm
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, J_norm, levels=20, cmap='plasma')
plt.colorbar(label='||J_f(x,y)||_F')
plt.title('Jacobian Norm of $f(x,y)$')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, linestyle=':')

# Plot 2: Local linear approximation at selected points
points = [(-1, -1), (0, 0), (1, 1)]
scale = 0.5  # scaling for visualization
plt.subplot(1, 2, 2)
plt.title('Local Linear Approximation via Jacobian')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, linestyle=':')

# Plot original basis vectors at each point and their images under J
for (x0, y0) in points:
    J = jacobian(x0, y0)
    # Basis vectors
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    # Mapped vectors
    Je1 = J @ e1
    Je2 = J @ e2
    # Plot the point
    plt.scatter(x0, y0, color='black')
    # Plot original basis arrows
    plt.quiver(x0, y0, e1[0]*scale, e1[1]*scale, 
               angles='xy', scale_units='xy', scale=1, color='blue', width=0.003)
    plt.quiver(x0, y0, e2[0]*scale, e2[1]*scale, 
               angles='xy', scale_units='xy', scale=1, color='green', width=0.003)
    # Plot transformed basis arrows
    plt.quiver(x0, y0, Je1[0]*scale, Je1[1]*scale, 
               angles='xy', scale_units='xy', scale=1, color='red', width=0.003)
    plt.quiver(x0, y0, Je2[0]*scale, Je2[1]*scale, 
               angles='xy', scale_units='xy', scale=1, color='orange', width=0.003)
    # Annotate
    plt.text(x0+0.1, y0+0.1, f"({x0},{y0})")
    
plt.legend(['point','e1','e2','J*e1','J*e2'], loc='upper left')
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()
```

1. **Plots a heatmap** of the Frobenius norm of the Jacobian $\mathbf{J}_f(x,y)$ for 
   $$
     f(x,y) = \begin{bmatrix} x^2 - y \\ y^2 + x \end{bmatrix},
   $$
   revealing regions where the mapping changes most rapidly.

2. **Displays the local linear approximation** at three sample points $(-1,-1)$, $(0,0)$, and $(1,1)$. At each point:
   - The **blue and green arrows** are the standard basis vectors $\mathbf{e}_1$ and $\mathbf{e}_2$.
   - The **red and orange arrows** are their images under the Jacobian, $\mathbf{J}_f \mathbf{e}_1$ and $\mathbf{J}_f \mathbf{e}_2$.
   - These illustrate how the Jacobian linearly approximates the action of $f$ in a neighborhood: small steps in input space are mapped to (approximately) linear steps in output space.