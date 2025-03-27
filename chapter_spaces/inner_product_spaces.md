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
a metric space).[^4]

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
product**). The reader can verify that the two-norm $\|\cdot\|_2$ on
$\mathbb{R}^n$ is induced by this inner product.

### Pythagorean Theorem

The well-known Pythagorean theorem generalizes naturally to arbitrary
inner product spaces.

*Theorem.* 
If $\mathbf{x} \perp \mathbf{y}$, then

$$\|\mathbf{x}+\mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2.$$


*Proof.* Suppose $\mathbf{x} \perp \mathbf{y}$, i.e.
$\langle \mathbf{x}, \mathbf{y} \rangle = 0$. Then

$$\|\mathbf{x}+\mathbf{y}\|^2 = \langle \mathbf{x}+\mathbf{y}, \mathbf{x}+\mathbf{y} \rangle = \langle \mathbf{x}, \mathbf{x} \rangle + \langle \mathbf{y}, \mathbf{x} \rangle + \langle \mathbf{x}, \mathbf{y} \rangle + \langle \mathbf{y}, \mathbf{y} \rangle = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2$$

as claimed. ◻

### Cauchy-Schwarz inequality

This inequality is sometimes useful in proving bounds:

$$|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \|\mathbf{x}\| \cdot \|\mathbf{y}\|$$

for all $\mathbf{x}, \mathbf{y} \in V$. Equality holds exactly when
$\mathbf{x}$ and $\mathbf{y}$ are scalar multiples of each other (or
equivalently, when they are linearly dependent).



## Inner Product Spaces in Machine Learning

Inner product spaces allow us to generalize ideas of angles, lengths, and orthogonality beyond traditional Euclidean geometry. They are foundational in machine learning algorithms involving geometric intuition, similarity measurement, and projection methods.

### Examples of Inner Products in ML:

#### 1. **Linear Classifiers (Dot product similarity)**

Many linear classifiers (like perceptrons, logistic regression, and linear SVMs) rely directly on the standard inner product (dot product):

- **Decision functions** for linear classifiers often take the form:
  \[
  f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b = \langle \mathbf{w}, \mathbf{x} \rangle + b
  \]

This explicitly uses the inner product to measure similarity between the feature vector \(\mathbf{x}\) and the learned weight vector \(\mathbf{w}\).

---

#### 2. **Kernels as Generalized Inner Products**

The notion of inner product spaces provides a powerful generalization called **kernels**, leading to the idea of **kernel methods** in machine learning. Kernels allow us to implicitly map data into high-dimensional spaces where classification or regression tasks become simpler.

A kernel function \( k(\mathbf{x}, \mathbf{y}) \) is defined as:
\[
k(\mathbf{x}, \mathbf{y}) = \langle \phi(\mathbf{x}), \phi(\mathbf{y}) \rangle
\]

where:
- \(\phi(\mathbf{x})\) is a feature mapping from the original feature space into a possibly high-dimensional (or even infinite-dimensional) inner product space.
- This new space is called a **Reproducing Kernel Hilbert Space (RKHS)**.

Common kernels include:

- **Polynomial kernel**: \( k(\mathbf{x}, \mathbf{y}) = (\mathbf{x}^\top \mathbf{y} + c)^d \)
- **Gaussian (RBF) kernel**: \( k(\mathbf{x}, \mathbf{y}) = \exp(-\gamma\|\mathbf{x}-\mathbf{y}\|^2) \)

---

### Inner Product Space Example with Nearest Centroid Classifier:

The nearest centroid classifier assigns points to the nearest centroid based on distance. We've used Euclidean distance so far, which implicitly relies on the standard inner product. But by generalizing this inner product through kernels, we can obtain interesting classifiers that measure similarity in richer ways.

**Kernelized Nearest Centroid Classifier**:

Instead of explicitly computing the centroid in the original space, we can implicitly compute distances in a high-dimensional space using kernels.

Given data points \( \mathbf{x}_i \), the centroid \(\mathbf{c}_k\) for class \(k\) in the mapped space is given by:
\[
\mathbf{c}_k = \frac{1}{N_k}\sum_{i:y_i=k}\phi(\mathbf{x}_i)
\]

Then the distance to the centroid becomes:
\[
\|\phi(\mathbf{x}) - \mathbf{c}_k\|^2 
= \langle \phi(\mathbf{x}) - \mathbf{c}_k, \phi(\mathbf{x}) - \mathbf{c}_k \rangle
= k(\mathbf{x},\mathbf{x}) - \frac{2}{N_k}\sum_{i:y_i=k}k(\mathbf{x}, \mathbf{x}_i) 
+ \frac{1}{N_k^2}\sum_{i,j:y_i,y_j=k} k(\mathbf{x}_i, \mathbf{x}_j)
\]

Thus, a kernelized nearest centroid classifier can classify points using arbitrary inner product spaces defined by kernels, allowing the classifier to handle complex, nonlinear patterns in the data.

---

### Insights for Students:

- **Inner products** provide a flexible geometric tool for measuring angles and similarity.
- **Kernels** use inner products implicitly to map data into spaces where classification is simplified.
- The kernelized nearest centroid classifier offers a straightforward way to appreciate how inner product spaces generalize standard linear classifiers, enhancing their expressive power in practical ML scenarios. 

This approach naturally motivates the importance of inner product spaces and kernels in machine learning.