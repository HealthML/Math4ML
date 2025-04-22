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
# The Derivative (1D)

Let $f:\mathbb{R}\to\mathbb{R}$ be a 1D real-valued function.

The **derivative** $f'(x)$ is

$$
  f'(x) \;=\; \lim_{h\to0} \frac{f(x+h)-f(x)}{h},
$$
if this limit exists.

- If $f'(a)$ exists, we say **$f$ is differentiable at $a$.**  
- If differentiable for all $x$ in an interval, we say **$f$ is differentiable on that interval**.  
- $f'(x)$ is the **instantaneous rate of change** of $f$ at $x$.  
- $f'(x)$ is the **slope of the tangent line** to the graph of $f$ at $x$.

To illustrate the definition of the derivative $\displaystyle f'(x_0)=\lim_{h\to0}\frac{f(x_0+h)-f(x_0)}{h}$, we can visualize the tangent line of a function at a point $x_0$ as the limit of secant lines for decreasing $h$.
The secant line between two points $(x_0, f(x_0))$ and $(x_0+h, f(x_0+h))$ has slope

$$
  \frac{f(x_0+h)-f(x_0)}{h}.
$$
As $h$ approaches $0$, the secant line approaches the tangent line at $(x_0, f(x_0))$.
The slope of the tangent line is given by the derivative $f'(x_0)$.
To visualize this, we can plot the function $f(x)=\sin(x)$ and its tangent line at a point $x_0=0.5$, along with several secant lines for different values of $h$.
```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# Define the function and its analytical derivative
def f(x):
    return np.sin(x)

def df(x):
    return np.cos(x)

# Choose a point to visualize the derivative
x0 = 0.5
f0 = f(x0)
df0 = df(x0)

# Domain for plotting
x = np.linspace(-1, 2, 400)

# Compute the tangent line at x0
y_tan = f0 + df0 * (x - x0)

# Define a set of h values to illustrate secant lines
h_values = [1.0, 0.5, 0.2, 0.1]

# Set up the plot
plt.figure(figsize=(8, 6))

# Plot the function
plt.plot(x, f(x), label="f(x) = sin(x)")

# Plot the tangent line
plt.plot(x, y_tan, label="Tangent at x₀", linewidth=2)

# Plot secant lines for each h
for h in h_values:
    slope = (f(x0 + h) - f0) / h
    y_sec = f0 + slope * (x - x0)
    plt.plot(x, y_sec, label=f"Secant h={h:.1f}", linestyle='--')

# Mark the point of tangency
plt.scatter([x0], [f0], color='black', zorder=5)
plt.text(x0, f0 + 0.1, "x₀", ha='center')

# Final decorations
plt.title("Visualization of Derivative as Limit of Secant Lines")
plt.xlabel("x")
plt.ylabel("f(x) and Secant/Tangent Lines")
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()
```


## Basic Differentiation Rules

1. **Constant rule**  
   $\displaystyle D(C)=0$.  
2. **Power rule**  
   $\displaystyle D(x^n)=n\,x^{n-1}$, $n\in\mathbb{R}$.  
3. **Exponential**  
   $\displaystyle D(e^x)=e^x$.  
4. **Logarithm**  
   $\displaystyle D(\ln x)=\tfrac1x$.  

## Sum, Product, Quotient, Constant Multiple

Let $f$ and $g$ be differentiable, $C$ a constant:

- **Constant multiple**  
  $\displaystyle D\bigl[C\,f(x)\bigr] = C\,f'(x)$.
- **Sum rule**  
  $\displaystyle D\bigl[f(x)+g(x)\bigr] = f'(x)+g'(x)$.
- **Product rule**  
  $\displaystyle D\bigl[f(x)\,g(x)\bigr] = f(x)\,g'(x) + g(x)\,f'(x)$.
- **Quotient rule**  

  $$
    D\!\Bigl[\tfrac{f(x)}{g(x)}\Bigr]
    = \frac{g(x)\,f'(x)\;-\;f(x)\,g'(x)}{[g(x)]^2}.
  $$

## Partial Derivatives

For $y=f(x_1,\dots,x_n)$, the **partial derivative** wrt $x_i$ is

$$
  \frac{\partial y}{\partial x_i}
  = \lim_{h\to0}
    \frac{f(x_1,\dots,x_i+h,\dots,x_n)
         -f(x_1,\dots,x_i,\dots,x_n)}{h}.
$$
This is the derivative of $f$ with respect to $x_i$, treating all other variables as constants.

# Gradients

Gradients generalize derivatives to scalar functions of several variables.

The gradient of
$f : \mathbb{R}^d \to \mathbb{R}$, denoted $\nabla f$, is given by

$$\nabla f = \begin{bmatrix}\frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n}\end{bmatrix}
\hspace{0.5cm}\text{i.e.}\hspace{0.5cm}
[\nabla f]_i = \frac{\partial f}{\partial x_i}$$

Gradients have the following very important property:

:::{prf:theorem} Gradient descent
:label: thm-gradient-descent
:nonumber:
$\nabla f(\mathbf{x})$ points in the direction of **steepest ascent** from $\mathbf{x}$. 
Similarly, $-\nabla f(\mathbf{x})$ points in the direction of **steepest descent** from $\mathbf{x}$. 
:::


:::{prf:proof} using the directional derivative and Cauchy–Schwarz

Let $f:\mathbb{R}^n\to\mathbb{R}$ be differentiable at a point $\mathbf{x}$, and let $\nabla f(\mathbf{x})$ denote its gradient there.  For any unit vector $\mathbf{u}\in\mathbb{R}^n$ ($\|\mathbf{u}\|=1$), the **directional derivative** of $f$ at $\mathbf{x}$ in the direction $\mathbf{u}$ is

$$
  D_{\mathbf{u}}f(\mathbf{x})
  \;=\;\lim_{h\to0^+}\frac{f(\mathbf{x}+h\mathbf{u})-f(\mathbf{x})}{h}
  \;=\;\nabla f(\mathbf{x})^\top \mathbf{u}.
$$
We wish to find the direction $\mathbf{u}$ that **minimizes** this rate of change (i.e.\ yields the steepest instantaneous decrease).  Since $\mathbf{u}$ is constrained to have unit length, we solve

$$
  \min_{\|\mathbf{u}\|=1}\;\nabla f(\mathbf{x})^\top \mathbf{u}.
$$
By the Cauchy–Schwarz inequality,

$$
  \nabla f(\mathbf{x})^\top \mathbf{u}
  \;\ge\; -\|\nabla f(\mathbf{x})\|\;\|\mathbf{u}\|
  \;=\;-\,\|\nabla f(\mathbf{x})\|.
$$

Moreover, equality holds exactly when $\mathbf{u}$ is a negative multiple of $\nabla f(\mathbf{x})$.  Imposing $\|\mathbf{u}\|=1$ gives the unique minimizer

$$
  \mathbf{u}^*
  =-\,\frac{\nabla f(\mathbf{x})}{\|\nabla f(\mathbf{x})\|}.
$$

Hence

$$
  D_{\mathbf{u}^*}f(\mathbf{x})
  =-\|\nabla f(\mathbf{x})\|,
$$
which is the smallest possible directional derivative over all unit directions.  In other words, moving (infinitesimally) in the direction of $-\nabla f(\mathbf{x})$ decreases $f$ **most rapidly**.  
:::

We will use this fact frequently when iteratively minimizing a function via **gradient descent**.

In the following example we show gradient descent for the function $f(x,y)=(x-1)^2+2(y-2)^2+\frac{1}{2}\sin(3x)\sin(3y)$, which has a unique global minimum near $(1,2)$ but small ripples that cause gradient descent to get stuck in local minima.

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
