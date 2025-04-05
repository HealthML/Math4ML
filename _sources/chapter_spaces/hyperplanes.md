## Hyperplanes and Linear Classification

The Nearest Centroid Classifier belongs to a broader family of linear classifiers, meaning the decision boundary is a linear function separating the classes. For binary classification, such as distinguishing benign from malignant biopsies, the decision boundary separates the space into two **halfspaces**.

### A Hyperplane in Euclidean Space

In an $n$-dimensional Euclidean space $\mathbb{R}^n$, a **hyperplane** is a flat, $(n-1)$-dimensional subset that divides the space into two distinct halfspaces.

Formally, a hyperplane can be expressed as:

$$
\mathbf{w}^T \mathbf{x} + b = 0
$$

where:
- $\mathbf{w} \in \mathbb{R}^n$ is a normal vector perpendicular to the hyperplane.
- $b \in \mathbb{R}$ is a scalar offset determining the hyperplane's position relative to the origin.


A halfspace in $\mathbb{R}^n$ is defined by:

$$
\{\mathbf{x} \in \mathbb{R}^n \mid \mathbf{w}^T \mathbf{x} + b \geq 0\}
$$

Where:
- $\mathbf{w}$ is a vector perpendicular to the decision boundary
- $b$ is a scalar offset


## Decision Boundary for the Nearest Centroid Classifier
For the nearest centroid classifier, the boundary is a hyperplane precisely halfway between the centroids, perpendicular to the line connecting them.

### Deriving the Separating Hyperplane for a Two-Class Problem

Consider two classes with centroids $\mathbf{c}_1$ and $\mathbf{c}_2$.
The decision boundary (hyperplane) is the set of points equidistant from both centroids. 
Mathematically, we have:

$$
\|\mathbf{x}-\mathbf{c}_1\|^2 = \|\mathbf{x}-\mathbf{c}_2\|^2
$$

Expanding both sides and simplifying:

$$
\mathbf{x}^T\mathbf{x} - 2\mathbf{c}_1^T\mathbf{x} + \mathbf{c}_1^T\mathbf{c}_1 = \mathbf{x}^T\mathbf{x} - 2\mathbf{c}_2^T\mathbf{x} + \mathbf{c}_2^T\mathbf{c}_2
$$

Subtracting $\mathbf{x}^T\mathbf{x}$ from both sides:

$$
-2\mathbf{c}_1^T\mathbf{x} + \mathbf{c}_1^T\mathbf{c}_1 = -2\mathbf{c}_2^T\mathbf{x} + \mathbf{c}_2^T\mathbf{c}_2
$$

Rearranging to isolate $\mathbf{x}$:

$$
2(\mathbf{c}_2 - \mathbf{c}_1)^T \mathbf{x} = \mathbf{c}_2^T\mathbf{c}_2 - \mathbf{c}_1^T\mathbf{c}_1
$$

Thus, the separating hyperplane equation is:

$$
(\mathbf{c}_2 - \mathbf{c}_1)^T \mathbf{x} - \frac{1}{2}(\mathbf{c}_2^T\mathbf{c}_2 - \mathbf{c}_1^T\mathbf{c}_1) = 0
$$

Here, the normal vector $\mathbf{w}$ is given by $(\mathbf{c}_2 - \mathbf{c}_1)$, and the offset $b$ is $-\frac{1}{2}(\mathbf{c}_2^T\mathbf{c}_2 - \mathbf{c}_1^T\mathbf{c}_1)$.
