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
## Inner product spaces

An **inner product** on a real vector space $V$ is a function
$\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$ satisfying

(i) $\langle \mathbf{x}, \mathbf{x} \rangle \geq 0$, with equality if
    and only if $\mathbf{x} = \mathbf{0}$

(ii) $\langle \alpha\mathbf{x} + \beta\mathbf{y}, \mathbf{z} \rangle = \alpha\langle \mathbf{x}, \mathbf{z} \rangle + \beta\langle \mathbf{y}, \mathbf{z} \rangle$

(iii) $\langle \mathbf{x}, \mathbf{y} \rangle = \langle \mathbf{y}, \mathbf{x} \rangle$

for all $\mathbf{x}, \mathbf{y}, \mathbf{z} \in V$ and all
$\alpha,\beta \in \mathbb{R}$. A vector space endowed with an inner
product is called an **inner product space**.

Note that any inner product on $V$ induces a norm on $V$:

$$\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$$

One can verify that the axioms for norms are satisfied under this definition
and follow (almost) directly from the axioms for inner products.
Therefore any inner product space is also a normed space (and hence also
a metric space).

Two vectors $\mathbf{x}$ and $\mathbf{y}$ are said to be **orthogonal**
if $\langle \mathbf{x}, \mathbf{y} \rangle = 0$; we write
$\mathbf{x} \perp \mathbf{y}$ for shorthand. Orthogonality generalizes
the notion of perpendicularity from Euclidean space. If two orthogonal
vectors $\mathbf{x}$ and $\mathbf{y}$ additionally have unit length
(i.e. $\|\mathbf{x}\| = \|\mathbf{y}\| = 1$), then they are described as
**orthonormal**.

The standard inner product on $\mathbb{R}^n$ is given by

$$\langle \mathbf{x}, \mathbf{y} \rangle = \sum_{i=1}^n x_iy_i = \mathbf{x}^{\!\top\!}\mathbf{y}$$

The matrix notation on the righthand side (see the Transposition section
if it's unfamiliar) arises because this inner product is a special case
of matrix multiplication where we regard the resulting $1 \times 1$
matrix as a scalar. The inner product on $\mathbb{R}^n$ is also often
written $\mathbf{x}\cdot\mathbf{y}$ (hence the alternate name **dot
product**). The two-norm $\|\cdot\|_2$ on
$\mathbb{R}^n$ is induced by this inner product.

The inner product on $\mathbb{R}^n$ induces the **length** (or **two-norm**) on $\mathbb{R}^n$:

$$\|\mathbf{x}\|_2 = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle} = \sqrt{\sum_{i=1}^n x_i^2}$$
This is the familiar Euclidean length of a vector in $\mathbb{R}^n$.

The inner product on $\mathbb{R}^n$ induces the following
**angle** between two vectors $\mathbf{x}$ and $\mathbf{y}$:
$$\theta = \arccos\left(\frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\|\mathbf{x}\| \|\mathbf{y}\|}\right)$$
This angle is well-defined as long as $\mathbf{x}$ and $\mathbf{y}$ are
not both the zero vector. The angle is $0$ if $\mathbf{x}$ and
$\mathbf{y}$ are parallel (i.e. $\mathbf{x} = t\mathbf{y}$ for some
$t \in \mathbb{R}$), and $\pi/2$ if they are orthogonal. The cosine of the
angle is given by the **cosine similarity** between $\mathbf{x}$ and
$\mathbf{y}$:
$$\cos(\theta) = \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\|\mathbf{x}\| \|\mathbf{y}\|}$$
This is a common measure of similarity between two vectors, and is
often used in machine learning applications.

### Pythagorean Theorem

The well-known Pythagorean theorem generalizes naturally to arbitrary
inner product spaces.

:::{prf:theorem} Pythagorean theorem

If $\mathbf{x} \perp \mathbf{y}$, then

$$\|\mathbf{x}+\mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2.$$
:::

:::{prf:proof}
Suppose $\mathbf{x} \perp \mathbf{y}$, i.e.
$\langle \mathbf{x}, \mathbf{y} \rangle = 0$. Then

$$\|\mathbf{x}+\mathbf{y}\|^2 = \langle \mathbf{x}+\mathbf{y}, \mathbf{x}+\mathbf{y} \rangle = \langle \mathbf{x}, \mathbf{x} \rangle + \langle \mathbf{y}, \mathbf{x} \rangle + \langle \mathbf{x}, \mathbf{y} \rangle + \langle \mathbf{y}, \mathbf{y} \rangle = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2$$

as claimed. ◻
:::

Below is a Python script that creates a visual demonstration of the Pythagorean theorem in an inner product space. In this example, we choose two perpendicular vectors $\mathbf{x}$ and $\mathbf{y}$ (for instance, $\mathbf{x}=(3, 0)$ and $\mathbf{y}=(0, 4)$) and plot these vectors along with their sum $\mathbf{x}+\mathbf{y}$. The script then annotates the computed norms and verifies that

$$
\|\mathbf{x}+\mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2.
$$

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# Define two perpendicular vectors x and y.
x = np.array([3, 0])
y = np.array([0, 4])
sum_xy = x + y

# Compute the norms (Euclidean norm)
norm_x = np.linalg.norm(x)
norm_y = np.linalg.norm(y)
norm_sum = np.linalg.norm(sum_xy)

# Verify the Pythagorean theorem numerically
lhs = norm_sum**2
rhs = norm_x**2 + norm_y**2

# Set up the plot
plt.figure(figsize=(8, 8))
origin = np.array([0, 0])

# Plot vector x, y, and x+y
plt.quiver(*origin, *x, angles='xy', scale_units='xy', scale=1, color='blue', label=r"$\mathbf{x}$")
plt.quiver(*origin, *y, angles='xy', scale_units='xy', scale=1, color='green', label=r"$\mathbf{y}$")
plt.quiver(*origin, *sum_xy, angles='xy', scale_units='xy', scale=1, color='red', label=r"$\mathbf{x}+\mathbf{y}$")

# Mark the right angle at the origin
plt.plot([x[0], x[0]], [0, y[1]], 'k--', linewidth=1)

# Annotate the plot with norm values
plt.text(x[0]/2, x[1]-0.3, f"{norm_x:.1f}", color='blue', fontsize=12)
plt.text(y[0]-1.0, y[1]/2, f"{norm_y:.1f}", color='green', fontsize=12)
plt.text(sum_xy[0]/2+0.2, sum_xy[1]/2+0.2, f"{norm_sum:.1f}", color='red', fontsize=12)

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Pythagorean Theorem in an Inner Product Space")
plt.xlim(-1, 7)
plt.ylim(-1, 7)
plt.grid(True, linestyle=':')
plt.legend()

# Print the numerical check in the console
print(f"||x+y||^2 = {lhs:.2f}")
print(f"||x||^2 + ||y||^2 = {rhs:.2f}")

plt.tight_layout()
plt.show()
```

---

### Explanation:

- **Vector Definitions:**  
  We define $\mathbf{x} = (3, 0)$ and $\mathbf{y} = (0, 4)$, which are perpendicular, and compute their sum $\mathbf{x} + \mathbf{y} = (3, 4)$.

- **Norm Calculations:**  
  The Euclidean norms are computed using `np.linalg.norm`, yielding $\|\mathbf{x}\| = 3$, $\|\mathbf{y}\| = 4$, and $\|\mathbf{x}+\mathbf{y}\| = 5$. The Pythagorean theorem is numerically verified since $5^2 = 3^2 + 4^2$.

- **Visualization:**  
  - The **blue arrow** represents $\mathbf{x}$.  
  - The **green arrow** represents $\mathbf{y}$.  
  - The **red arrow** represents the resultant vector $\mathbf{x}+\mathbf{y}$.  
  A dashed line indicates the right angle between $\mathbf{x}$ and $\mathbf{y}$.

- **Annotations:**  
  Norm values are annotated near the middle of each vector, and the computed squared norm values are printed to the console.


### Cauchy-Schwarz inequality
The Cauchy-Schwarz inequality is a fundamental result in linear algebra and functional analysis. It states that the absolute value of the inner product of two vectors is less than or equal to the product of their norms.
This inequality is a powerful tool in various fields, including machine learning, statistics, and optimization.
::: {prf:theorem} Cauchy–Schwarz Inequality
For all $\mathbf{x}, \mathbf{y} \in V$,

$$
|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \|\mathbf{x}\| \cdot \|\mathbf{y}\|,
$$
with equality if and only if $\mathbf{x}$ and $\mathbf{y}$ are linearly dependent.
:::

For a proof see the Appendix of the book.

Below is a Python script that provides a visual explanation of the Cauchy–Schwarz inequality by relating the dot product to the cosine of the angle between vectors and illustrating its geometric implications for a linear classifier’s decision boundary. In this visualization, we fix a vector $\mathbf{x}$ and draw several vectors $\mathbf{y}$ on a circle (so that $\|\mathbf{y}\|$ is fixed). For each such $\mathbf{y}$, we annotate the computed dot product (which equals $\|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$). In a separate subplot, we also plot the function $f(\theta)=\|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$ versus $\theta$ to show that the dot product is maximized when $\mathbf{y}$ is aligned with $\mathbf{x}$ and minimized when it is opposite.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# Define a fixed vector x.
x = np.array([2, 1])
norm_x = np.linalg.norm(x)

# Define a fixed norm for y.
norm_y = 2.0

# Create a figure with two subplots.
plt.figure(figsize=(12, 5))

# Subplot 1: Visualizing x and several y vectors.
plt.subplot(1, 2, 1)
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.gca().set_aspect('equal')

# Plot vector x.
plt.quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy', scale=1, color='blue', label=r"$\mathbf{x}$")
plt.text(x[0]*1.05, x[1]*1.05, r"$\mathbf{x}$", color='blue', fontsize=12)

# Define several angles for y (in radians).
angles_deg = [0, 45, 90, 135, 180]
thetas = np.deg2rad(angles_deg)
colors = ['red', 'orange', 'green', 'purple', 'brown']

# Plot a circle for reference (all y with fixed norm).
theta_circle = np.linspace(0, 2*np.pi, 300)
circle_x = norm_y * np.cos(theta_circle)
circle_y = norm_y * np.sin(theta_circle)
plt.plot(circle_x, circle_y, 'k--', alpha=0.3, label="Circle (fixed norm)")

# Plot vectors y at the specified angles and annotate their dot product.
for theta, c, deg in zip(thetas, colors, angles_deg):
    y = norm_y * np.array([np.cos(theta), np.sin(theta)])
    plt.quiver(0, 0, y[0], y[1], angles='xy', scale_units='xy', scale=1, color=c, alpha=0.8)
    dp = np.dot(x, y)
    # Place the annotation near the end of y.
    plt.text(y[0]*1.1, y[1]*1.1, f"{dp:.2f}", color=c, fontsize=10)
    plt.text(y[0]*0.8, y[1]*0.8, f"{deg}°", color=c, fontsize=10)

plt.title("Dot Product and Cosine Geometry")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.legend(loc="upper right")

# Subplot 2: Plot the dot product as a function of the angle between x and y.
plt.subplot(1, 2, 2)
theta_vals = np.linspace(0, np.pi, 300)
dp_vals = norm_x * norm_y * np.cos(theta_vals)
plt.plot(theta_vals, dp_vals, label=r"$\|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$", color='magenta')
plt.axhline(norm_x * norm_y, color='gray', linestyle='--', label="Max")
plt.axhline(-norm_x * norm_y, color='gray', linestyle='--', label="Min")
plt.xlabel(r"$\theta$ (radians)")
plt.ylabel("Dot product")
plt.title("Dot Product vs. Angle")
plt.legend()

plt.tight_layout()
plt.show()
```

### Explanation

- **Subplot 1 (Geometry in $\mathbb{R}^2$)**:  
  The blue arrow represents the fixed vector $\mathbf{x}$. The red, orange, green, purple, and brown arrows are various vectors $\mathbf{y}$ on the circle of radius $2$ (i.e. with fixed norm $\|\mathbf{y}\| = 2$) at angles $0^\circ$, $45^\circ$, $90^\circ$, $135^\circ$, and $180^\circ$, respectively. For each $\mathbf{y}$, the dot product $\langle \mathbf{x}, \mathbf{y} \rangle$ is calculated and annotated. This illustrates that the dot product depends on the cosine of the angle between $\mathbf{x}$ and $\mathbf{y}$.

- **Subplot 2 (Function Plot)**:  
  This subplot shows the function $f(\theta) = \|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$ as a function of the angle $\theta$. The maximum and minimum possible dot products occur when $\theta=0$ or $\theta=\pi$, corresponding to perfect alignment and opposite direction, respectively. This visualization reinforces the concept that the dot product is essentially $\|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$.


## Inner Product Spaces in Machine Learning

Inner product spaces allow us to generalize ideas of angles, lengths, and orthogonality beyond traditional Euclidean geometry. They are foundational in machine learning algorithms involving geometric intuition, similarity measurement, and projection methods.

### Examples of Inner Products in ML:

#### 1. **Linear Classifiers (Dot product similarity)**

Many linear classifiers (like perceptrons, logistic regression, and linear SVMs) rely directly on the standard inner product (dot product):

- **Decision functions** for linear classifiers often take the form:
  
$$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b = \langle \mathbf{w}, \mathbf{x} \rangle + b$$

This explicitly uses the inner product to measure similarity between the feature vector $\mathbf{x}$ and the learned weight vector $\mathbf{w}$.

Below is an added paragraph followed by a Python script that visually illustrates the relationship between dot product similarity and the geometry of the hyperplane in a linear classifier.

---

**Additional Paragraph:**

The dot product not only measures the similarity between two vectors but also has a natural geometric interpretation that directly relates to the decision boundary of linear classifiers. In a linear classifier, the decision function

$$
f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b
$$

can be viewed as measuring how much the input vector $\mathbf{x}$ aligns with the weight vector $\mathbf{w}$. Geometrically, $\mathbf{w}$ points in a direction of maximum increase of the function, so that the hyperplane defined by $\mathbf{w}^\top \mathbf{x} + b = 0$ is perpendicular to $\mathbf{w}$. The signed distance of any point from this hyperplane is proportional to the dot product $\langle \mathbf{w}, \mathbf{x} \rangle$ (after accounting for the bias $b$). Thus, a larger (more positive) dot product indicates that $\mathbf{x}$ lies further on one side of the hyperplane (and is therefore classified in one class), while a more negative dot product indicates placement on the opposite side. This elegant connection between inner products and geometry underpins many classification algorithms.

---

**Python Visualization Script:**

The script below creates a two-dimensional visualization. It plots:
- A hyperplane defined by $ \mathbf{w}^\top \mathbf{x} + b = 0 $.
- The weight vector $\mathbf{w}$.
- A sample input vector $\mathbf{x}$ and its projection onto $\mathbf{w}$, which shows how the dot product measures the similarity.
- Annotations demonstrating the geometric interpretation.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# Define weight vector and bias for a linear classifier
w = np.array([2, -1])
b = 1.0

# Define the decision boundary: w^T x + b = 0
# For 2D: 2*x - y + 1 = 0 => y = 2*x + 1
x_vals = np.linspace(-5, 5, 200)
y_boundary = 2 * x_vals + 1

# Choose a sample point x (for which we want to illustrate the projection)
x_sample = np.array([3, 2])

# Compute the projection of x_sample onto w
# The projection is given by: proj_w(x) = (x.w/||w||^2)*w
w_norm_sq = np.dot(w, w)
projection_coef = np.dot(x_sample, w) / w_norm_sq
x_proj = projection_coef * w

# Set up the plot
plt.figure(figsize=(8, 8))

# Plot the decision boundary
plt.plot(x_vals, y_boundary, 'k--', label=r"Decision boundary: $w^\top x + b = 0$")

# Plot the weight vector originating from the origin
origin = np.array([0, 0])
plt.quiver(*origin, *w, color='blue', angles='xy', scale_units='xy', scale=1, width=0.005)
plt.text(w[0]*1.1, w[1]*1.1, r"$w$", color='blue', fontsize=14)

# Plot the sample point x_sample
plt.scatter(x_sample[0], x_sample[1], color='red', s=100, label=r"Sample $\mathbf{x}$")
plt.text(x_sample[0] + 0.2, x_sample[1] + 0.2, r"$\mathbf{x}$", color='red', fontsize=14)

# Plot the projection of x_sample onto w
plt.scatter(x_proj[0], x_proj[1], color='green', s=100, label=r"Projection $\mathrm{proj}_w(\mathbf{x})$")
plt.plot([x_sample[0], x_proj[0]], [x_sample[1], x_proj[1]], 'r--', label=r"Residual")
plt.quiver(*origin, *x_proj, color='green', angles='xy', scale_units='xy', scale=1, width=0.005)

# Annotate the distance (dot product similarity)
dot_val = np.dot(w, x_sample)
plt.text((x_sample[0] + x_proj[0]) / 2, (x_sample[1] + x_proj[1]) / 2, 
         f"{dot_val:.2f}", color='purple', fontsize=12)

# Set plot limits and labels
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Dot Product Similarity and Hyperplane Geometry")
plt.legend(loc='upper left')
plt.grid(True, linestyle=':')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()
```

---

**Explanation of the Script:**

- The **decision boundary** is plotted as a dashed line, representing $ w^\top x + b = 0 $.
- The **blue arrow** represents the weight vector $ w $, indicating the direction in which the classifier’s decision function increases.
- The **red point** is a sample input vector $ \mathbf{x} $, and its **red dashed line** shows the error between $ \mathbf{x} $ and its projection onto $ w $.
- The **green point** is the projection of $ \mathbf{x} $ onto $ w $ (i.e., $ \mathrm{proj}_w(\mathbf{x}) $), and the length of this projection is closely related to the dot product $ \langle \mathbf{w}, \mathbf{x} \rangle $.
- An annotation displays the numerical value of the dot product, reinforcing how the inner product measures the similarity between $ \mathbf{x} $ and $ \mathbf{w} $ relative to the hyperplane.


---

#### 2. **Kernels as Generalized Inner Products**

The notion of inner product spaces provides a powerful generalization called **kernels**, leading to the idea of **kernel methods** in machine learning. Kernels allow us to implicitly map data into high-dimensional spaces where classification or regression tasks become simpler.

A kernel function $k(\mathbf{x}, \mathbf{y})$ is defined as:

$$k(\mathbf{x}, \mathbf{y}) = \langle \phi(\mathbf{x}), \phi(\mathbf{y}) \rangle$$

where:
- $\phi(\mathbf{x})$ is a feature mapping from the original feature space into a possibly high-dimensional (or even infinite-dimensional) inner product space.
- This new space is called a **Reproducing Kernel Hilbert Space (RKHS)**.

Common kernels include:

- **Polynomial kernel**: $k(\mathbf{x}, \mathbf{y}) = (\mathbf{x}^\top \mathbf{y} + c)^d$
- **Gaussian (RBF) kernel**: $k(\mathbf{x}, \mathbf{y}) = \exp(-\gamma\|\mathbf{x}-\mathbf{y}\|^2)$

---

### Inner Product Space Example with Nearest Centroid Classifier:

The nearest centroid classifier assigns points to the nearest centroid based on distance. We've used Euclidean distance so far, which implicitly relies on the standard inner product. But by generalizing this inner product through kernels, we can obtain interesting classifiers that measure similarity in richer ways.

**Kernelized Nearest Centroid Classifier**:

Instead of explicitly computing the centroid in the original space, we can implicitly compute distances in a high-dimensional space using kernels.

Given data points $\mathbf{x}_i$, the centroid $\mathbf{c}_k$ for class $k$ in the mapped space is given by:

$$\mathbf{c}_k = \frac{1}{N_k}\sum_{i:y_i=k}\phi(\mathbf{x}_i)$$

Then the distance to the centroid becomes:

$$\|\phi(\mathbf{x}) - \mathbf{c}_k\|^2 
= \langle \phi(\mathbf{x}) - \mathbf{c}_k, \phi(\mathbf{x}) - \mathbf{c}_k \rangle
= k(\mathbf{x},\mathbf{x}) - \frac{2}{N_k}\sum_{i:y_i=k}k(\mathbf{x}, \mathbf{x}_i) 
+ \frac{1}{N_k^2}\sum_{i,j:y_i,y_j=k} k(\mathbf{x}_i, \mathbf{x}_j)$$

Thus, a kernelized nearest centroid classifier can classify points using arbitrary inner product spaces defined by kernels, allowing the classifier to handle complex, nonlinear patterns in the data.

Consider a training set $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$ with class labels $y_i \in \{1, \ldots, K\}$, and let $\phi: \mathbb{R}^n \to \mathcal{H}$ be a feature mapping into an (often high-dimensional) Hilbert space. In a standard nearest centroid classifier, the centroid for class $k$ is computed as

$$
\mathbf{c}_k = \frac{1}{N_k}\sum_{i:y_i=k}\phi(\mathbf{x}_i),
$$

where $N_k$ is the number of training examples in class $k$. When classifying a new point $\mathbf{x}$, one typically assigns it to the class with the closest centroid in $\mathcal{H}$, measured by the squared distance

$$
\|\phi(\mathbf{x}) - \mathbf{c}_k\|^2.
$$

Expanding this distance using the properties of an inner product yields

$$
\|\phi(\mathbf{x}) - \mathbf{c}_k\|^2 = \langle\phi(\mathbf{x}),\phi(\mathbf{x})\rangle - 2\langle\phi(\mathbf{x}),\mathbf{c}_k\rangle + \langle \mathbf{c}_k, \mathbf{c}_k\rangle.
$$

By the kernel trick, we define a kernel function $k(\mathbf{x},\mathbf{x}') = \langle\phi(\mathbf{x}),\phi(\mathbf{x}')\rangle$, which allows us to express all inner products in terms of $k$. In particular,

- $\langle\phi(\mathbf{x}),\phi(\mathbf{x})\rangle = k(\mathbf{x},\mathbf{x})$,
- $\langle\phi(\mathbf{x}),\mathbf{c}_k\rangle = \frac{1}{N_k}\sum_{i:y_i=k} k(\mathbf{x},\mathbf{x}_i)$, and
- $\langle \mathbf{c}_k, \mathbf{c}_k \rangle = \frac{1}{N_k^2}\sum_{i,j:y_i,y_j=k} k(\mathbf{x}_i,\mathbf{x}_j)$.

Thus, the squared distance between $\phi(\mathbf{x})$ and the centroid $\mathbf{c}_k$ can be written entirely in terms of the kernel as

$$
\|\phi(\mathbf{x}) - \mathbf{c}_k\|^2 = k(\mathbf{x},\mathbf{x}) - \frac{2}{N_k}\sum_{i:y_i=k} k(\mathbf{x},\mathbf{x}_i) + \frac{1}{N_k^2}\sum_{i,j:y_i,y_j=k} k(\mathbf{x}_i,\mathbf{x}_j).
$$

Using this kernelized distance, the classifier assigns $\mathbf{x}$ to the class $k$ for which $\|\phi(\mathbf{x}) - \mathbf{c}_k\|^2$ is minimal. In this way, the kernelized nearest centroid classifier operates solely via inner products—thus allowing the algorithm to implicitly work in high-dimensional feature spaces without ever computing the mapping $\phi$ explicitly.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

class KernelNearestCentroid:
    def __init__(self, kernel=None):
        """
        Initialize the Kernelized Nearest Centroid Classifier.
        
        Parameters:
        -----------
        kernel : function or None
            A function that takes two vectors and returns a scalar,
            representing the inner product in the feature space.
            If None, a default RBF kernel with sigma=1.0 is used.
        """
        if kernel is None:
            # Default: RBF kernel with sigma=1.0 (gamma=1/(2*sigma^2)=0.5)
            self.kernel = lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / 2.0)
        else:
            self.kernel = kernel
        self.classes_ = None
        self.class_indices_ = {}  # mapping from class to indices in training set
        self.N_k_ = {}            # number of training examples per class
        self.K_train_ = None      # kernel matrix for training data
        self.K_centroid_sqr_ = {} # precomputed term: (1/N_k^2)*sum_{i,j in class k} k(x_i, x_j)
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y):
        """
        Fit the kernelized nearest centroid classifier.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Class labels.
        """
        self.X_train_ = np.asarray(X)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(self.y_train_)
        n = self.X_train_.shape[0]
        # Precompute kernel matrix on training data
        self.K_train_ = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                k_val = self.kernel(self.X_train_[i], self.X_train_[j])
                self.K_train_[i, j] = k_val
                self.K_train_[j, i] = k_val
        
        # For each class, store indices and precompute centroid norm squared in feature space
        for cls in self.classes_:
            indices = np.where(self.y_train_ == cls)[0]
            self.class_indices_[cls] = indices
            N_k = len(indices)
            self.N_k_[cls] = N_k
            # Compute the double-sum term for class centroid: (1/N_k^2)*sum_{i,j in class} k(x_i, x_j)
            K_cls = self.K_train_[np.ix_(indices, indices)]
            self.K_centroid_sqr_[cls] = np.sum(K_cls) / (N_k**2)
    
    def decision_function(self, x):
        """
        Compute the squared distance in feature space from x to each class centroid.
        The kernelized squared distance for class k is given by:
        
            d^2(x, c_k) = k(x, x) 
                          - (2/N_k)*sum_{i:y_i=k} k(x, x_i)
                          + (1/N_k^2)*sum_{i,j:y_i,y_j=k} k(x_i,x_j)
        
        Returns:
        --------
        distances : dict
            Dictionary mapping class label to the computed squared distance.
        """
        distances = {}
        k_xx = self.kernel(x, x)
        for cls in self.classes_:
            indices = self.class_indices_[cls]
            N_k = self.N_k_[cls]
            # Compute sum_{i in class} k(x, x_i)
            k_x_class = np.array([self.kernel(x, self.X_train_[i]) for i in indices])
            term2 = (2.0 / N_k) * np.sum(k_x_class)
            term3 = self.K_centroid_sqr_[cls]
            distances[cls] = k_xx - term2 + term3
        return distances
    
    def predict(self, X):
        """
        Predict the class labels for the given set of data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        
        Returns:
        --------
        y_pred : ndarray, shape (n_samples,)
            Predicted class labels.
        """
        X = np.asarray(X)
        preds = []
        for x in X:
            distances = self.decision_function(x)
            pred = min(distances, key=distances.get)
            preds.append(pred)
        return np.array(preds)

# -------------------------------
# Demonstration on Synthetic Data
# -------------------------------
if __name__ == "__main__":
    # Generate synthetic 2D data for two classes
    np.random.seed(42)
    X_class0 = np.random.randn(20, 2) + np.array([1, 1])
    X_class1 = np.random.randn(20, 2) + np.array([-1, -1])
    X_train = np.vstack((X_class0, X_class1))
    y_train = np.array([0]*20 + [1]*20)
    
    # Instantiate and fit the classifier with an RBF kernel
    clf = KernelNearestCentroid(kernel=lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / 2.0))
    clf.fit(X_train, y_train)
    
    # Create a grid for visualizing the decision boundary
    xx, yy = np.meshgrid(np.linspace(-4, 4, 300), np.linspace(-4, 4, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)
    
    # Plot decision regions and training data
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X_train[y_train==0][:, 0], X_train[y_train==0][:, 1], 
                color='blue', label='Class 0', edgecolor='k')
    plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], 
                color='red', label='Class 1', edgecolor='k')
    plt.title("Kernelized Nearest Centroid Classifier")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()
```

---

## Representer Theorem for Linear Classifiers

In many machine learning problems, especially those involving regularized risk minimization, the **representer theorem** guarantees that the solution to the optimization problem can be written as a finite linear combination of the training samples. For linear classifiers, consider the following regularized formulation:

$$
\min_{\mathbf{w}, b} \; \sum_{i=1}^n L(y_i, \mathbf{w}^\top \mathbf{x}_i + b) + \lambda\, \|\mathbf{w}\|^2,
$$

where:
- $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ are the training examples with $\mathbf{x}_i \in \mathbb{R}^d$ and corresponding labels $y_i$,
- $L$ is a loss function (such as the hinge loss for support vector machines or the logistic loss for logistic regression),
- $\lambda > 0$ is a regularization parameter,
- $\|\mathbf{w}\|^2$ represents the squared Euclidean norm, which penalizes the complexity of the classifier.

The representer theorem asserts that there exists a solution $(\mathbf{w}^*, b^*)$ where the optimal weight vector $\mathbf{w}^*$ lies in the span of the training data. That is, one can express

$$
\mathbf{w}^* = \sum_{i=1}^n \alpha_i \mathbf{x}_i,
$$

for some coefficients $\alpha_1, \alpha_2, \dots, \alpha_n \in \mathbb{R}$. Consequently, the decision function becomes

$$
f(\mathbf{x}) = \mathbf{w}^{*\top} \mathbf{x} + b^* 
= \left(\sum_{i=1}^n \alpha_i \mathbf{x}_i^\top\right) \mathbf{x} + b^*
= \sum_{i=1}^n \alpha_i\, \langle \mathbf{x}_i, \mathbf{x} \rangle + b^*.
$$

This representation has profound implications:

- **Finite Representation:** Even if the underlying hypothesis space (e.g., an RKHS in the kernelized setting) is infinite-dimensional, regularization forces the solution to lie in the finite-dimensional span of the training examples.
- **Computational Efficiency:** The optimization problem is effectively reduced to finding the finite set of coefficients $\alpha_i$ (and $b$), which can be computed efficiently using kernel methods or convex optimization techniques.
- **Interpretability:** In linear classifiers, this representation reveals that the learned decision boundary is entirely determined by the training data. In the case of support vector machines, for instance, only a subset of the training points (the support vectors) will have nonzero coefficients $\alpha_i$, directly indicating which examples are critical for classification.

Thus, for linear classifiers with decision boundaries of the form $\mathbf{w}^\top \mathbf{x} + b$, the representer theorem not only ensures that the problem has a solution in the span of the data but also provides practical and theoretical benefits by reducing the complexity of the learning task.

---

## Representer Theorem for Linear Regression

In linear regression, particularly in its regularized form (such as ridge regression), the learning problem can be formulated as

$$
\min_{\mathbf{w}, b} \; \sum_{i=1}^n \left(y_i - \mathbf{w}^\top \mathbf{x}_i - b\right)^2 + \lambda\, \|\mathbf{w}\|^2,
$$

where:
- $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ are the training examples, with $\mathbf{x}_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$,
- $\lambda > 0$ is a regularization parameter,
- $\|\mathbf{w}\|^2$ is the squared Euclidean norm used to prevent overfitting by penalizing large weights.

The representer theorem tells us that despite the possibly infinite-dimensional nature of the hypothesis space in other settings, in regularized linear regression the optimal weight vector $\mathbf{w}^*$ can always be expressed as a linear combination of the training input vectors. That is, there exist coefficients $\alpha_1, \alpha_2, \dots, \alpha_n \in \mathbb{R}$ such that

$$
\mathbf{w}^* = \sum_{i=1}^n \alpha_i \mathbf{x}_i.
$$

Using this representation, the predicted output for a new input $\mathbf{x}$ becomes

$$
f(\mathbf{x}) = \mathbf{w}^{*\top} \mathbf{x} + b^* 
= \left(\sum_{i=1}^n \alpha_i \mathbf{x}_i^\top\right) \mathbf{x} + b^*
= \sum_{i=1}^n \alpha_i\, \langle \mathbf{x}_i, \mathbf{x} \rangle + b^*.
$$

This formulation has several important implications:

- **Finite Representation:** Even though there might be an infinite number of directions in the function space, the regularized problem constrains the solution to lie within the finite-dimensional span of the training examples.
- **Computational Efficiency:** The solution is characterized by the finite set of coefficients $\alpha_i$ (along with the bias $b^*$). In practice, this makes methods such as kernel ridge regression computationally tractable because the solution can be expressed in terms of kernel evaluations between training data points.
- **Interpretability:** The representation clearly shows that the learned regression function is entirely determined by the training data. For example, in ridge regression, the contribution of each training example is moderated by its corresponding coefficient $\alpha_i$.

Thus, for linear regression with regularization, the representer theorem guarantees that the optimal solution is in the span of the training inputs and can be written entirely in terms of inner products between these inputs, providing both theoretical insight and practical advantages in computation and model interpretation.

--- 
## Representer Theorem for Linear Classifiers and Linear Regression

Below is a standard proof that establishes the representer theorem for linear classifiers and linear regression, i.e. that there exists an optimal solution $(\mathbf{w}^*, b^*)$ with

$$
\mathbf{w}^* = \sum_{i=1}^n \alpha_i \mathbf{x}_i.
$$

---

### Proof:

Assume that we are solving a regularized risk minimization problem over $\mathbf{w} \in \mathbb{R}^d$ and $b \in \mathbb{R}$ of the form

$$
\min_{\mathbf{w}, b} \; \sum_{i=1}^n L\bigl(y_i, \mathbf{w}^\top \mathbf{x}_i + b\bigr) + \lambda\, \|\mathbf{w}\|^2,
$$

where $L$ is a loss function, and $\lambda>0$ is the regularization parameter. Let 
$$
V = \operatorname{span}\{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\}
$$
be the subspace spanned by the training data.

For any vector $\mathbf{w} \in \mathbb{R}^d$, we can decompose it uniquely as

$$
\mathbf{w} = \mathbf{w}_0 + \mathbf{w}_1,
$$

where $\mathbf{w}_0 \in V$ and $\mathbf{w}_1 \in V^\perp$ (the orthogonal complement of $V$). Notice that by construction, for every training example $\mathbf{x}_i$ (which lies in $V$), we have

$$
\langle \mathbf{w}_1, \mathbf{x}_i \rangle = 0.
$$

Thus, the prediction for each training example is

$$
\mathbf{w}^\top \mathbf{x}_i + b = (\mathbf{w}_0 + \mathbf{w}_1)^\top \mathbf{x}_i + b = \mathbf{w}_0^\top \mathbf{x}_i + \mathbf{w}_1^\top \mathbf{x}_i + b = \mathbf{w}_0^\top \mathbf{x}_i + b.
$$

This shows that the component $\mathbf{w}_1$ (lying in $V^\perp$) does not affect the predictions on the training data.

Now, consider the regularization term $\|\mathbf{w}\|^2$. Because $\mathbf{w}_0$ and $\mathbf{w}_1$ are orthogonal, we have

$$
\|\mathbf{w}\|^2 = \|\mathbf{w}_0\|^2 + \|\mathbf{w}_1\|^2.
$$

If $\mathbf{w}_1 \neq \mathbf{0}$, then $\|\mathbf{w}\|^2 > \|\mathbf{w}_0\|^2$ but the predictions remain the same. Since the objective includes the regularization term $\lambda\,\|\mathbf{w}\|^2$, any optimal solution can be improved (or at least not worsened) by setting $\mathbf{w}_1 = \mathbf{0}$. In other words, there is no benefit to having a component in $V^\perp$.

Thus, we can always find an optimal weight vector $\mathbf{w}^*$ that lies entirely in $V$; that is,

$$
\mathbf{w}^* = \mathbf{w}_0 \quad \text{with} \quad \mathbf{w}^* \in V.
$$

Because $V$ is the span of $\{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\}$, there exist scalars $\alpha_1, \alpha_2, \dots, \alpha_n$ such that

$$
\mathbf{w}^* = \sum_{i=1}^n \alpha_i \mathbf{x}_i.
$$

This completes the proof that the optimal solution can be expressed as a linear combination of the training data.

---

### Explanation:

- **Key Observation:** The loss term depends only on $\mathbf{w}^\top \mathbf{x}_i$. Since any component of $\mathbf{w}$ perpendicular to all training examples (i.e. in $V^\perp$) does not affect the loss, including it only increases the regularization penalty.
- **Conclusion:** We can always set the $V^\perp$ component to zero without changing the predictions, thereby arriving at an optimal solution that lies in the span of the training data. Consequently, the optimal weight vector can be written as

$$
\mathbf{w}^* = \sum_{i=1}^n \alpha_i \mathbf{x}_i.
$$

This representer theorem is central in kernel methods and many linear models because it reduces an infinite-dimensional search (if one considers a feature space mapping) to a finite-dimensional problem based solely on the training samples.




### Insights for Students:

- **Inner products** provide a flexible geometric tool for measuring angles and similarity.
- **Kernels** use inner products implicitly to map data into spaces where classification is simplified.
- The kernelized nearest centroid classifier offers a straightforward way to appreciate how inner product spaces generalize standard linear classifiers, enhancing their expressive power in practical ML scenarios. 

This approach naturally motivates the importance of inner product spaces and kernels in machine learning.