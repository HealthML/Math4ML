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



```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the scalar-valued function f: R^2 -> R
def f(x, y):
    return np.log(1 + x**2 + y**2)

# Point around which we expand
x0, y0 = 1.0, 1.0
r2 = x0**2 + y0**2

# Compute value, gradient, and Hessian at (x0, y0)
f0 = np.log(1 + r2)
grad = np.array([2*x0, 2*y0]) / (1 + r2)
H = (2 / (1 + r2)) * np.eye(2) - (4 * np.outer([x0, y0], [x0, y0])) / (1 + r2)**2

# Vectorized second-order Taylor approximation
def f_taylor(X, Y):
    dx = X - x0
    dy = Y - y0
    delta = np.stack([dx, dy], axis=-1)  # shape (N, M, 2)
    linear = np.tensordot(delta, grad, axes=([2], [0]))  # shape (N, M)
    quad = np.einsum('...i,ij,...j->...', delta, H, delta)  # shape (N, M)
    return f0 + linear + 0.5 * quad

# Create grid
x_vals = np.linspace(-1, 3, 100)
y_vals = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z_true = f(X, Y)
Z_approx = f_taylor(X, Y)
Z_error = np.abs(Z_true - Z_approx)

# Plot: true function, Taylor approx, and error
fig = plt.figure(figsize=(18, 5))

# True function
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z_true, cmap='viridis', alpha=0.9)
ax1.set_title("True Function: $f(x, y) = \log(1 + x^2 + y^2)$")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("f(x, y)")
ax1.scatter(x0, y0, f0, color='red', s=50, label=r'$\mathbf{x}_0$')
ax1.legend()

# Taylor approximation
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_approx, cmap='plasma', alpha=0.9)
ax2.set_title("Second-Order Taylor Approximation")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("T2(x, y)")

# Error surface
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
surf3 = ax3.plot_surface(X, Y, Z_error, cmap='inferno', alpha=0.9)
ax3.set_title("Approximation Error: $|f(x, y) - T_2(x, y)|$")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("Error")

plt.tight_layout()
plt.show()
```

Perfect! You're now looking at **Taylor expansions for scalar-valued functions** $f : \mathbb{R}^n \to \mathbb{R}$. This generalization is foundational for:

* Multivariate approximation
* Optimization (e.g. Newton's method)
* Machine learning models (e.g. loss function landscapes)

---

## ✅ Second-Order Taylor Expansion in $\mathbb{R}^n$

Let $f : \mathbb{R}^n \to \mathbb{R}$ be twice continuously differentiable. The **second-order Taylor expansion** of $f$ around a point $\mathbf{x}_0$ is:

$$
f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top (\mathbf{x} - \mathbf{x}_0) + \frac{1}{2} (\mathbf{x} - \mathbf{x}_0)^\top \nabla^2 f(\mathbf{x}_0) (\mathbf{x} - \mathbf{x}_0)
$$

This approximation is:

* Exact up to first and second derivatives at $\mathbf{x}_0$
* A **quadratic form** in $\mathbf{x} - \mathbf{x}_0$

---

## 🎯 Remainder Term in $\mathbb{R}^n$

The **remainder** or **error** term (in Lagrange form) is:

$$
R(\mathbf{x}) = \frac{1}{6} \sum_{i,j,k} \frac{\partial^3 f}{\partial x_i \partial x_j \partial x_k}(\boldsymbol{\xi}) (x_i - x_{0i})(x_j - x_{0j})(x_k - x_{0k})
$$

or more compactly, in terms of the third-order tensor of derivatives:

$$
R(\mathbf{x}) = \mathcal{O}(\|\mathbf{x} - \mathbf{x}_0\|^3)
$$

So the second-order approximation becomes **very accurate** when $\|\mathbf{x} - \mathbf{x}_0\|$ is small.

---

## 📈 Visualization Idea (2D case)

We can visualize the true function and its second-order Taylor approximation for a function like:

$$
f(x, y) = \log(1 + x^2 + y^2)
$$

Which is:

* Scalar-valued
* Twice differentiable
* Non-quadratic (so approximation is informative)

We'll plot:

* The true surface $f(x, y)$
* The quadratic approximation centered at $(x_0, y_0)$
* The absolute error $|f(x, y) - T_2(x, y)|$

---

## ✅ Shall I generate that visualization next?

Let me know:

* The center point $\mathbf{x}_0$ (e.g. $(1, 1)$)
* Any preferred function (or we use $\log(1 + x^2 + y^2)$)
* Whether you want the **error surface** plotted separately or as a heatmap over the function



Yes — and this is a great question!

The **Taylor series** is not *always* exact for general functions, but under certain regularity conditions, it can be **provably exact** in the limit (i.e., as the degree $k \to \infty$), or even **exact after a finite number of terms** (e.g., for polynomials).

Let’s break this down step by step.

---

## 🎓 What We Mean by “Exact”

Let $f : \mathbb{R}^n \to \mathbb{R}$ be a smooth function and $\mathbf{x}_0 \in \mathbb{R}^n$. We say the Taylor series is **exact** at $\mathbf{x}$ if:

$$
f(\mathbf{x}) = \sum_{|\alpha| \ge 0} \frac{D^\alpha f(\mathbf{x}_0)}{\alpha!} (\mathbf{x} - \mathbf{x}_0)^\alpha
$$

That is, the infinite Taylor series **converges** to the actual value of the function.

---

## ✅ Case 1: When the Taylor Series is Exact — The Analytic Case

### **Theorem (Multivariate Taylor Series — Analytic Functions)**

Let $f : \mathbb{R}^n \to \mathbb{R}$ be **analytic** at $\mathbf{x}_0$, meaning that it equals its Taylor series in a neighborhood of $\mathbf{x}_0$:

$$
f(\mathbf{x}) = \sum_{k=0}^\infty \sum_{|\alpha| = k} \frac{D^\alpha f(\mathbf{x}_0)}{\alpha!} (\mathbf{x} - \mathbf{x}_0)^\alpha
\quad \text{for all } \mathbf{x} \text{ near } \mathbf{x}_0.
$$

✅ Then the Taylor series is exact (it converges to $f(\mathbf{x})$).

**Proof idea**:

* Use bounds on all derivatives $D^\alpha f$
* Show the remainder $R_k(\mathbf{x}) \to 0$ as $k \to \infty$
* This follows from standard results in analysis (e.g., ratio test + uniform convergence)

---

## 🚫 Case 2: When the Taylor Series Is Not Exact

There exist **infinitely differentiable functions** $f \in C^\infty$ for which the Taylor series **does not converge to $f$**.

### Counterexample (bump function):

$$
f(x) = 
\begin{cases}
e^{-1/x^2} & \text{if } x \neq 0 \\
0 & \text{if } x = 0
\end{cases}
$$

Then $f \in C^\infty(\mathbb{R})$, and **all** derivatives at 0 vanish:

$$
f^{(k)}(0) = 0 \quad \text{for all } k
$$

So its Taylor series at 0 is identically zero:

$$
T(x) = 0
$$

But $f(x) \neq 0$ for $x \neq 0$ — the Taylor series **fails** to approximate $f$ at all!

---

## ✅ Case 3: When the Series is Exactly Finite (Polynomial Case)

If $f$ is a **polynomial of degree $k$**, then:

$$
f(\mathbf{x}) = \sum_{|\alpha| \le k} \frac{D^\alpha f(\mathbf{x}_0)}{\alpha!} (\mathbf{x} - \mathbf{x}_0)^\alpha
$$

This is exact because all derivatives of order $> k$ vanish.

---

## 🧠 Summary

| Function Type           | Taylor Series         | Error Term Behavior              |
| ----------------------- | --------------------- | -------------------------------- |
| Polynomial (degree $k$) | Exact after $k$ terms | Zero                             |
| Analytic                | Converges to $f$      | $R_k \to 0$ as $k \to \infty$    |
| Smooth but non-analytic | Not exact             | May fail even if $f^{(k)}$ exist |

---

## ✍️ Conclusion

* The Taylor series is **exact** for **analytic functions**
* It is **finite and exact** for **polynomials**
* It is **not always exact** for **infinitely differentiable but non-analytic functions**
* The remainder term gives a way to **bound the error**, and shows convergence if the derivatives are well-behaved

Would you like a symbolic or numerical example in code (e.g., computing and comparing remainder errors)?
