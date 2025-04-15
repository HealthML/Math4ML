## Inner Product Spaces in Machine Learning

Inner product spaces allow us to generalize ideas of angles, lengths, and orthogonality beyond traditional Euclidean geometry. They are foundational in machine learning algorithms involving geometric intuition, similarity measurement, and projection methods.

### Examples of Inner Products in ML:

#### 1. **Linear Classifiers (Dot product similarity)**

Many linear classifiers (like perceptrons, logistic regression, and linear SVMs) rely directly on the standard inner product (dot product):

- **Decision functions** for linear classifiers often take the form:
  
$$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b = \langle \mathbf{w}, \mathbf{x} \rangle + b$$

This explicitly uses the inner product to measure similarity between the feature vector $\mathbf{x}$ and the learned weight vector $\mathbf{w}$.

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
plt.xlim(-2, 4)
plt.ylim(-2, 4)
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
This formulation allows us to use the kernel trick to compute distances in the feature space without explicitly mapping the data into that space, making it computationally efficient and enabling the use of high-dimensional feature spaces.