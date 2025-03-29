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

