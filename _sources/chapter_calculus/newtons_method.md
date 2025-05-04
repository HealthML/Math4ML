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
# Newton's Method

An advantage of the **second-order Taylor expansion** is that it provides a **local quadratic approximation** of a function. Quadratic functions are simple to minimize analytically — and this idea is the foundation of **Newton's method**.

---

## **Motivation from Second-Order Taylor Expansion**

Recall the second-order Taylor expansion of a function $f: \mathbb{R}^d \to \mathbb{R}$ around a point $\mathbf{x}_0$:

$$
f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top (\mathbf{x} - \mathbf{x}_0) + \frac{1}{2}(\mathbf{x} - \mathbf{x}_0)^\top \nabla^2 f(\mathbf{x}_0)(\mathbf{x} - \mathbf{x}_0).
$$

We now minimize this quadratic approximation with respect to $\mathbf{x}$. The minimum occurs where the gradient vanishes:

$$
\nabla f(\mathbf{x}_0) + \nabla^2 f(\mathbf{x}_0)(\mathbf{x} - \mathbf{x}_0) = 0
$$

Solving for $\mathbf{x}$, we get the **Newton update rule**:

$$
\boxed{
\mathbf{x}_{\text{new}} = \mathbf{x}_0 - \left[\nabla^2 f(\mathbf{x}_0)\right]^{-1} \nabla f(\mathbf{x}_0)
}
$$

---

## Newton's Method: Iterative Algorithm

Given an initial guess $\mathbf{x}_0$, we repeat:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \left[\nabla^2 f(\mathbf{x}_k)\right]^{-1} \nabla f(\mathbf{x}_k)
$$

until convergence. Unlike gradient descent, which moves in the direction of steepest descent, Newton’s method uses **second-order curvature information** to take a more direct path toward the minimum.

---

## ⚠️ Caveat

* If the Hessian is not **positive definite**, the update may not be a descent direction.

Excellent point! 👏

You're absolutely right — **adding a line search** to Newton's method is a **very common and practical enhancement**. Here's why it helps:

---

## ⚠️ Why Newton’s Method Alone Can Fail

The Newton update:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - [\nabla^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)
$$

assumes that the **Hessian is positive definite**, which ensures a descent direction and local convexity. But in practice:

* The **Hessian may be indefinite** (especially far from the minimum).
* The step $\Delta \mathbf{x}$ can be **too large**, causing divergence.

---

## ✅ Line Search to the Rescue

To improve robustness, we combine Newton's direction with a **line search**:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \eta_k \mathbf{d}_k,
\quad \text{where} \quad \mathbf{d}_k = -[\nabla^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)
$$

Here, $\eta_k$ is chosen using a **backtracking line search**, typically satisfying the **Armijo condition** (just like in gradient descent).

---

### 🔄 Newton's Method with Line Search: Pseudocode

1. Compute Newton direction:
   $\mathbf{d}_k = -[\nabla^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)$
2. Perform line search:
   Find $\eta_k \in (0, 1]$ such that

   $$
   f(\mathbf{x}_k + \eta_k \mathbf{d}_k) \leq f(\mathbf{x}_k) + c_1 \eta_k \nabla f(\mathbf{x}_k)^\top \mathbf{d}_k
   $$
3. Update:
   $\mathbf{x}_{k+1} = \mathbf{x}_k + \eta_k \mathbf{d}_k$

---

### ✅ Advantages

* Handles **indefinite or ill-conditioned Hessians**
* Prevents overshooting
* Preserves **fast convergence** near the minimum
* Makes Newton’s method practical for real optimization problems

---

### 👉 Would You Like:

* A **modified Newton implementation** that includes Armijo backtracking?
* A **visualization** of the line search steps along the Newton direction?

I’d be happy to generate that for you next!


---

Let's visualize Newton's method on the same function as before:

$$
f(x, y) = \log(1 + x^2 + y^2)
$$

We'll start at a point $\mathbf{x}_0$, compute the Newton step, and plot both the function and the path taken by Newton's method.

---
Here's a modified version of your previous **Newton's method implementation**, now with **Armijo backtracking line search** added. This ensures that the method is robust even when the Hessian is not positive definite or when steps are too aggressive.

---

## ✅ Newton's Method with Armijo Backtracking

### 🔧 Code

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function, gradient, and Hessian
def f(x):
    return np.log(1 + np.sum(x**2))

def grad_f(x):
    return 2 * x / (1 + np.sum(x**2))

def hess_f(x):
    r2 = np.sum(x**2)
    I = np.eye(len(x))
    outer = np.outer(x, x)
    return (2 / (1 + r2)) * I - (4 / (1 + r2)**2) * outer

# Backtracking line search (Armijo)
def armijo_backtracking(f, grad_f, x, d, alpha=1.0, beta=0.5, c1=1e-4):
    f_x = f(x)
    grad_dot_d = grad_f(x).dot(d)
    while f(x + alpha * d) > f_x + c1 * alpha * grad_dot_d:
        alpha *= beta
    return alpha

# Newton's method with line search
def newton_with_backtracking(f, grad_f, hess_f, x0, tol=1e-6, max_iter=20):
    x = x0.copy()
    path = [x.copy()]
    etas = []

    for _ in range(max_iter):
        grad = grad_f(x)
        H = hess_f(x)
        try:
            d = -np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            print("Hessian is singular or ill-conditioned. Stopping.")
            break

        if np.linalg.norm(d) < tol:
            break

        eta = armijo_backtracking(f, grad_f, x, d)
        x = x + eta * d
        path.append(x.copy())
        etas.append(eta)

    return np.array(path), etas

# Run method
x0 = np.array([1.5, 1.5])
path, etas = newton_with_backtracking(f, grad_f, hess_f, x0)

# Create grid for visualization
x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.log(1 + X**2 + Y**2)

# Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Overlay path
ax.plot(path[:, 0], path[:, 1], [f(p) for p in path], color='red', marker='o', label="Newton steps")
ax.scatter(*x0, f(x0), color='blue', label=r'Initial point $\mathbf{x}_0$', s=60)
ax.scatter(*path[-1], f(path[-1]), color='green', label='Final point', s=60)

ax.set_title("Newton's Method with Line Search on $f(x, y) = \log(1 + x^2 + y^2)$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.tight_layout()
plt.show()
```

---

### 🧠 Notes

* Backtracking ensures each step **decreases the objective**.
* Useful when the **Hessian is ill-conditioned** or **far from the minimum**.
* Keeps Newton’s method fast **and stable**.

Would you like to visualize the step sizes (etas) or apply this to a non-convex function next?
