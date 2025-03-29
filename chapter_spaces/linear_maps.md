### Linear Maps

Some of the most important subspaces are those induced by linear maps.
If $T : V \to W$ is a linear map, we define the **nullspace** (or **kernel**) of $T$
as

$\operatorname{null}(T) = \{\mathbf{x} \in V \mid T\mathbf{x} = \mathbf{0}\}.$

and
the **range** (or the **columnspace** if we are considering the matrix
form) of $T$ as

$\operatorname{range}(T) = \{\mathbf{y} \in W \mid \exists \mathbf{x} \in V : T\mathbf{x} = \mathbf{y}\}.$

It is a good exercise to verify that the nullspace and range of a linear
map are always subspaces of its domain and codomain, respectively.

The **rank** of a linear map $T$ is the dimension of its range, and the
**nullity** of $T$ is the dimension of its nullspace.


## Affine Spaces

An **affine space** is a geometric structure closely related to vector spaces, but without a distinguished origin. Formally, an affine space can be thought of as a vector space that's been "shifted" away from the origin.

Given a vector space $V$, an affine space is defined as:

$$A = \{\mathbf{v} + \mathbf{a} \mid \mathbf{v} \in V\}$$

for some fixed vector $\mathbf{a}$.
Unlike a subspace, an affine space:

- **Does not necessarily contain the zero vector**.
- Is **not closed under addition or scalar multiplication** unless specifically shifted back to the origin.
- Can be visualized as a subspace that's been translated (shifted) by some offset.

**Simple intuition:**
- A line passing through the origin is a subspace.
- A line parallel to that one but not passing through the origin is an affine space.

**Example (in ML):**  
The separating hyperplane for classifiers like SVM or logistic regression (with a non-zero intercept) is typically an affine space, since it doesn't usually pass through the origin. Similarly, if PCA were done without mean-centering, its resulting space would also be an affine space rather than a subspace.


### Affine Map
Some of the most important affine spaces are those induced by affine maps.
Given vector spaces $V$ and $W$, a function $f: V \to W$ is called an **affine map** (or affine transformation) if it can be expressed in the form:

$$f(\mathbf{x}) = L(\mathbf{x}) + \mathbf{b}$$

where:

- $L : V \to W$ is a linear map (i.e., a transformation satisfying $L(\alpha \mathbf{x} + \beta \mathbf{y}) = \alpha L(\mathbf{x}) + \beta L(\mathbf{y})$),
- $\mathbf{b} \in W$ is a fixed vector representing a translation (shift).

An **affine space** induced by this affine map is the set of all possible outputs of $f$:

$$A = \{ L(\mathbf{x}) + \mathbf{b} \mid \mathbf{x} \in V \}.$$

#### Properties:

- If $\mathbf{b} = \mathbf{0}$, the affine map reduces to a linear map, and $A$ is a vector subspace.
- If $\mathbf{b} \neq \mathbf{0}$, the affine space $A$ is essentially a vector subspace (the image of $L$) shifted away from the origin by $\mathbf{b}$.

#### Intuition:

Affine spaces generalize linear subspaces by allowing them to be translated away from the origin. Thus, they retain the geometric shape and structure of subspaces but lose the special role of the zero vector.

**Example (Machine Learning context):**  
- The separating hyperplane of linear classifiers like logistic regression, perceptron, or SVM with a non-zero bias term is precisely an affine space induced by an affine map.



#### Example 1: Principal Component Analysis (PCA)


In machine learning, we often encounter datasets with many features—sometimes hundreds or thousands of dimensions. Visualizing or analyzing data in such high-dimensional spaces can be challenging. To simplify this problem, we often try to find a smaller number of new features that capture most of the important information in the data.

Principal Component Analysis (PCA) is a common method to achieve this simplification by projecting data onto a lower-dimensional **subspace**.

Here's an intuitive idea of how PCA works, without any advanced linear algebra:

- Suppose you have a dataset represented as points scattered in a high-dimensional space, like a cloud of points.

- PCA finds a few ($k$) "directions" in space along which the data varies most strongly. These so-called principal components form new axes or coordinates for representing your data. Typically, these new axes summarize your original data quite effectively.

- The set of all possible points you can represent using these new axes (directions) is exactly a **subspace**:
    - It includes the zero vector (imagine your new axes intersecting exactly at the origin).
    - If you add any two points in this simplified representation, you remain within the same set.
    - Scaling any point by a scalar (stretching or shrinking) also keeps the result in the subspace.

- **Subspace property:**  
  The set of all linear combinations of the first $k$ principal components forms a subspace of the original feature space. This subspace naturally includes the zero vector, and any addition or scalar multiplication within this reduced-dimensional space remains within it.

Thus, PCA essentially identifies a smaller, simpler subspace within the larger, complicated original feature space, providing a practical and computationally useful example of subspaces in machine learning.

**Visual analogy:**  
Imagine projecting a three-dimensional cloud of points onto a two-dimensional plane that captures most of the shape and distribution. The plane through the origin onto which you project the data is exactly a subspace.

*(Later, when students are comfortable, you can revisit PCA formally, explaining precisely how these directions are determined using eigenvalues, eigenvectors, and the SVD.)*

#### Example 2: Linear Regression

In linear regression, the predicted values $\hat{\mathbf{y}}$ of a linear model form a **column space** of the data matrix $\mathbf{X}$:

- Given:

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{\beta}, \quad \mathbf{X} \in \mathbb{R}^{n\times d}, \quad \mathbf{\beta} \in \mathbb{R}^{d\times 1}$$

- **Subspace property:**  
  The set of all possible predictions $\hat{\mathbf{y}}$ for different coefficients $\mathbf{\beta}$ is the column space of $\mathbf{X}$, a subspace of $\mathbb{R}^n$.
  It contains the zero vector (achieved by setting all $\mathbf{\beta}$ to zero), and is closed under vector addition and scalar multiplication.
