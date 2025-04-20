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
# Extrema

Optimization is about finding **extrema**, which depending on the
application could be minima or maxima. When defining extrema, it is
necessary to consider the set of inputs over which we're optimizing.
This set $\mathcal{X} \subseteq \mathbb{R}^d$ is called the **feasible
set**. If $\mathcal{X}$ is the entire domain of the function being
optimized (as it often will be for our purposes), we say that the
problem is **unconstrained**. Otherwise the problem is **constrained**
and may be much harder to solve, depending on the nature of the feasible
set.

Suppose $f : \mathbb{R}^d \to \mathbb{R}$.
A point $\mathbf{x}$ is said to be a **local minimum** (resp. **local maximum**) of $f$ in
$\mathcal{X}$ if $f(\mathbf{x}) \leq f(\mathbf{y})$ (resp. $f(\mathbf{x}) \geq f(\mathbf{y})$) for all $\mathbf{y}$ in some neighborhood $N \subseteq \mathcal{X}$ about $\mathbf{x}$.

Furthermore, if $f(\mathbf{x}) \leq f(\mathbf{y})$ for all $\mathbf{y} \in \mathcal{X}$, then $\mathbf{x}$ is a **global minimum** of $f$ in $\mathcal{X}$ (similarly for global maximum). If the phrase "in $\mathcal{X}$" is unclear from context, assume we are optimizing
over the whole domain of the function.

The qualifier **strict** (as in e.g. a strict local minimum) means that the inequality sign in the definition is actually a $>$ or $<$, with equality not allowed. This indicates that the extremum is unique within some neighborhood.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# Define the objective function and its gradient
def f(x, y):
    # A slightly non-convex function: tilted quadratic + sinusoidal ripples
    return (x - 1)**2 + 2*(y - 2)**2 + 0.5 * np.sin(3*x) * np.sin(3*y)

def grad_f(x, y):
    # Partial derivatives
    dfdx = 2*(x - 1) + 1.5 * np.cos(3*x) * np.sin(3*y)
    dfdy = 4*(y - 2) + 1.5 * np.sin(3*x) * np.cos(3*y)
    return np.array([dfdx, dfdy])

# Grid setup
x_vals = np.linspace(-1, 3, 200)
y_vals = np.linspace(-1, 5, 200)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# Plot contours
plt.figure(figsize=(8, 6))
contours = plt.contour(X, Y, Z, levels=30, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)

# Plot gradient descent vector field (negative gradient)
skip = 15
GX, GY = grad_f(X, Y)
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip],
           -GX[::skip, ::skip], -GY[::skip, ::skip],
           color='gray', alpha=0.6, headwidth=3)

# Simulate gradient descent from several starting points
starts = [(-0.5, 4), (3, -0.5), (2.5, 4.5)]
lr = 0.05
num_steps = 50

for x0, y0 in starts:
    path = np.zeros((num_steps+1, 2))
    path[0] = np.array([x0, y0])
    for i in range(num_steps):
        grad = grad_f(path[i, 0], path[i, 1])
        path[i+1] = path[i] - lr * grad
    plt.plot(path[:, 0], path[:, 1], marker='o', markersize=3,
             label=f"start=({x0:.1f},{y0:.1f})")

# Final touches
plt.title("Contour of $f(x,y)$ with Gradient Descent Paths")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
```
The above figure shows:

- **Contours of**  

  $$
    f(x,y) = (x - 1)^2 + 2(y - 2)^2 + \tfrac12\sin(3x)\sin(3y),
  $$
  which has a unique global minimum near $(1,2)$ but small ripples that create nonconvexity.

- **Gray arrows**, indicating the negative gradient direction at each point.

- **Colored paths**, showing gradient‑descent trajectories started from three different initial points. You can see how each path “flows” downhill along the contours, eventually converging to the basin of the global minimum.

---

**Why this helps build intuition:**

1. **Contours** are level sets of the cost function. Regions where contours are close together indicate steep slopes, while widely spaced contours are flat.
2. **Gradient arrows** point in the direction of greatest decrease. Following them leads you toward a local minimum.
3. **Paths from different starts** illustrate how gradient descent navigates toward minima and how nonconvex ripples can slow it down or trap it in local basins.
4. Adjusting the **learning rate** or **initialization** changes these trajectories dramatically—showing why step‑size tuning and good initialization matter in practice.

This visualization combines level sets, gradient fields, and descent paths to give a clear, geometric picture of unconstrained optimization in two dimensions.
