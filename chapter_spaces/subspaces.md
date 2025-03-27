### Subspaces

Vector spaces can contain other vector spaces.
If $V$ is a vector space, then $S \subseteq V$ is said to be a **subspace** of $V$ if

(i) $\mathbf{0} \in S$

(ii) $S$ is closed under vector addition: $\mathbf{x}, \mathbf{y} \in S$
     implies $\mathbf{x}+\mathbf{y} \in S$

(iii) $S$ is closed under scalar multiplication:
      $\mathbf{x} \in S, \alpha \in \mathbb{R}$ implies
      $\alpha\mathbf{x} \in S$

Note that $V$ is always a subspace of $V$, as is the trivial vector
space which contains only $\mathbf{0}$.

As a concrete example, a line passing through the origin is a subspace
of Euclidean space.

### Examples of Subspaces in Machine Learning

#### Example 1: Principal Component Analysis (PCA)

PCA is a dimensionality reduction method frequently used to simplify high-dimensional data by projecting it onto a lower-dimensional **subspace**. PCA finds directions (principal components) in the original vector space along which the variance of data is maximized. 

- **Subspace property:**  
  The set of all linear combinations of the first \( k \) principal components forms a subspace of the original feature space. This subspace naturally includes the zero vector, and any addition or scalar multiplication within this reduced-dimensional space remains within it.

#### Example 2: Linear Regression

In linear regression, the predicted values \(\hat{\mathbf{y}}\) of a linear model form a **column space** of the data matrix \(\mathbf{X}\):

- Given:
\[
\hat{\mathbf{y}} = \mathbf{X}\mathbf{\beta}, \quad \mathbf{X} \in \mathbb{R}^{n\times d}, \quad \mathbf{\beta} \in \mathbb{R}^{d\times 1}
\]

- **Subspace property:**  
  The set of all possible predictions \(\hat{\mathbf{y}}\) for different coefficients \(\mathbf{\beta}\) is the column space of \(\mathbf{X}\), a subspace of \(\mathbb{R}^n\). It contains the zero vector (achieved by setting all \(\mathbf{\beta}\) to zero), and is closed under vector addition and scalar multiplication.

---

### Counterexamples of Subspaces in Machine Learning

It's also instructive to highlight what does **not** constitute a subspace.

#### Counterexample 1: Nearest Centroid Classifier's Decision Boundary

Consider the separating hyperplane obtained from the Nearest Centroid Classifier for two classes:

- As derived previously, this hyperplane is defined as:
\[
(\mathbf{c}_2 - \mathbf{c}_1)^T \mathbf{x} - \frac{1}{2}(\mathbf{c}_2^T\mathbf{c}_2 - \mathbf{c}_1^T\mathbf{c}_1) = 0
\]

- **Not a subspace:**  
  This hyperplane is a linear manifold but not a subspace unless it passes exactly through the origin. Generally, this hyperplane does **not contain the zero vector**, and hence it violates condition (i) of the subspace definition.

#### Counterexample 2: Affine Spaces in General (e.g., Linear classifiers with bias term)

Linear classifiers such as SVM, Logistic Regression, or Perceptron typically have a bias term \( b \):

- Decision boundary:
\[
\mathbf{w}^T \mathbf{x} + b = 0
\]

- **Not a subspace:**  
  Unless \( b = 0 \), this decision boundary (an affine hyperplane) does not pass through the origin and thus does not satisfy the zero-vector condition.

---

### Visualizing the difference:

To visualize intuitively:

- **Subspace:** Imagine a plane passing exactly through the origin in 3D space (a subspace).
- **Not a subspace:** Imagine shifting this plane away from the origin; it still looks "flat," but since it doesn't include the origin, it is no longer a subspace—just an affine manifold.

