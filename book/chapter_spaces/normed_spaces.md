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
# Normed spaces

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

$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x}-\mathbf{y}\|$

One can verify that the axioms for metrics are satisfied under this definition and
follow directly from the axioms for norms. Therefore any normed space is
also a metric space.

We will typically only be concerned with a few specific norms on
$\mathbb{R}^n$:

$$\begin{aligned}
\|\mathbf{x}\|_1 &= \sum_{i=1}^n |x_i| \\
\|\mathbf{x}\|_2 &= \sqrt{\sum_{i=1}^n x_i^2} \\
\|\mathbf{x}\|_p &= \left(\sum_{i=1}^n |x_i|^p\right)^\frac{1}{p} \hspace{0.5cm}\hspace{0.5cm} (p \geq 1) \\
\|\mathbf{x}\|_\infty &= \max_{1 \leq i \leq n} |x_i|
\end{aligned}$$

Here's a visualization of the **unit norm balls** in $\mathbb{R}^2$ for the most common norms:

- $\ell_p$ norms for different values of $p$ $ p = 1, 2, 3, \infty $
- A **counterexample** for $ p = 0.5 $, shown as a dashed line, labeled clearly as “not a norm”

These “balls” show the set of all points $\mathbf{x} \in \mathbb{R}^2$ such that $\|\mathbf{x}\| = 1$ under each norm.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

def unit_norm_ball(p, num_points=300):
    """
    Generate points on the unit ball of the Lp norm in R^2.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = np.cos(theta)
    y = np.sin(theta)

    if p == np.inf:
        # Infinity norm: max(|x|, |y|) = 1 (a square)
        return np.array([
            [-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]  # square
        ]).T
    else:
        norm = (np.abs(x)**p + np.abs(y)**p)**(1/p)
        return x / norm, y / norm

# Norm values to plot
norms = [0.5, 1, 2, 3, np.inf]
colors = ['gray', 'red', 'blue', 'green', 'orange']
styles = ['--', '-', '-', '-', '-']
labels = [
    r"$\|\mathbf{x}\|_{0.5}$ (not a norm)",
    r"$\|\mathbf{x}\|_1$",
    r"$\|\mathbf{x}\|_2$",
    r"$\|\mathbf{x}\|_3$",
    r"$\|\mathbf{x}\|_\infty$"
]

# Set up plot
plt.figure(figsize=(8, 8))

for p, color, style, label in zip(norms, colors, styles, labels):
    x, y = unit_norm_ball(p)
    plt.plot(x, y, linestyle=style, color=color, label=label)

# Decorations
plt.gca().set_aspect('equal')
plt.title("Unit Norm Balls in $\\mathbb{R}^2$")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.show()
```
- The **solid curves** for $ p = 1, 2, 3, \infty $ all enclose **convex shapes**—valid norm balls.
- The **dashed gray curve** for $ p = 0.5 $ appears **star-shaped and non-convex**, violating the triangle inequality and thus **not forming a norm**.
- It’s a powerful visual cue for why the condition $ p \geq 1 $ is essential for valid norms.


Note that the 1- and 2-norms are special cases of the
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



## Normed Spaces in Machine Learning

Normed spaces generalize the idea of **length** and thus naturally appear whenever machine learning algorithms quantify vector magnitude or enforce regularization.

### Nearest Centroid Classifier under different norms

Let's visualize how different norms affect the decision boundary of a nearest centroid classifier.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
# Generate simple 2D synthetic data
class1 = np.random.randn(20, 2) + np.array([1, 2])
class2 = np.random.randn(20, 2) + np.array([-1, -2])
X = np.vstack((class1, class2))
y = np.array([0]*20 + [1]*20)

# Compute centroids
centroids = np.array([X[y==0].mean(axis=0), X[y==1].mean(axis=0)])

# Define metrics
def lp_distance(x, c, p):
    if p == np.inf:
        return np.max(np.abs(x - c), axis=-1)
    else:
        return np.sum(np.abs(x - c) ** p, axis=-1) ** (1/p)

# Grid for plotting decision boundaries
grid_x, grid_y = np.meshgrid(np.linspace(-6, 6, 400), np.linspace(-6, 6, 400))
grid = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)

# Plot boundaries for each norm
norms = [1, 2, np.inf]
titles = [r"$\ell_1$ (Manhattan)", r"$\ell_2$ (Euclidean)", r"$\ell_\infty$ (Max norm)"]

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

for ax, p, title in zip(axs, norms, titles):
    # Compute distances to each centroid
    d0 = lp_distance(grid, centroids[0], p)
    d1 = lp_distance(grid, centroids[1], p)
    
    # Predict by choosing the closer centroid
    pred = np.where(d0 < d1, 0, 1)
    pred = pred.reshape(grid_x.shape)
    
    # Plot decision boundary
    ax.contourf(grid_x, grid_y, pred, levels=1, alpha=0.3, colors=["red", "blue"])
    
    # Plot data points and centroids
    ax.scatter(class1[:, 0], class1[:, 1], color="red", label="Class 0")
    ax.scatter(class2[:, 0], class2[:, 1], color="blue", label="Class 1")
    ax.scatter(*centroids[0], color="darkred", s=100, marker="x")
    ax.scatter(*centroids[1], color="darkblue", s=100, marker="x")

    
    ax.set_title(f"Nearest Centroid Classifier\nwith {title} Distance")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":")
    ax.legend()

plt.tight_layout()
plt.show()
```


### Further Examples in Machine Learning
In machine learning, norms and metrics are not just theoretical constructs; they play a crucial role in various algorithms and applications. Here are some examples:

1. **Distance Metrics in Classification and Clustering**
    - **k-Nearest Neighbors (k-NN):** The k-NN algorithm uses distance metrics (like L1 or L2 norms) to find the nearest neighbors of a point in feature space. The choice of metric can significantly affect the performance of the classifier.
    - **Clustering Algorithms:** Algorithms like k-means clustering use distance metrics to assign points to clusters based on their proximity to cluster centroids.


2. **Regularization (Ridge and Lasso)**  
Regularization methods in machine learning, such as **ridge regression** (L2 regularization) and **lasso regression** (L1 regularization), explicitly use norms on parameter vectors:

- **L2 Regularization (Ridge):**

$$\text{Loss}_{ridge}(\mathbf{w}) = \text{MSE}(\mathbf{w}) + \lambda \|\mathbf{w}\|_2^2$$

(penalizes the squared Euclidean norm, encouraging small parameter values.)

- **L1 Regularization (Lasso):**
     
$$\text{Loss}_{lasso}(\mathbf{w}) = \text{MSE}(\mathbf{w}) + \lambda \|\mathbf{w}\|_1$$

(penalizes the sum of absolute parameter values, promoting sparsity in the solution.)

3. **Measuring Errors and Convergence (Gradient Descent)**  
When running optimization algorithms such as **gradient descent**, one commonly uses norms to measure how far parameter updates move between iterations:
     
$$\|\mathbf{w}_{t+1} - \mathbf{w}_{t}\|_2 \quad \text{or} \quad \|\nabla f(\mathbf{w}_t)\|_2$$

The algorithm stops when the magnitude (norm) of parameter updates or gradients becomes sufficiently small.

