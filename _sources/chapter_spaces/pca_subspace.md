### Examples of Subspaces in Machine Learning

#### Example 1: Principal Component Analysis (PCA)

PCA is a dimensionality reduction method frequently used to simplify high-dimensional data by projecting it onto a lower-dimensional **subspace**. PCA finds directions (principal components) in the original vector space along which the variance of data is maximized. 

- **Subspace property:**  
  The set of all linear combinations of the first $k$ principal components forms a subspace of the original feature space. This subspace naturally includes the zero vector, and any addition or scalar multiplication within this reduced-dimensional space remains within it.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic 2D data with correlation
mean = [0, 0]
cov = [[3, 1.5], [1.5, 1]]  # covariance matrix
n_samples = 200
X = np.random.multivariate_normal(mean, cov, n_samples)

# Center the data (subtract the mean)
X_centered = X - np.mean(X, axis=0)

# Perform PCA using SVD
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
PC1 = Vt[0]  # first principal component
PC2 = Vt[1]  # second principal component

# Project each data point onto the first principal component (PC1)
proj_PC1 = (X_centered @ PC1.reshape(-1, 1)) * PC1.reshape(1, -1)

# Plotting
plt.figure(figsize=(8,8))

# Plot the centered data points
plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.3, label='Centered Data')

# Plot the principal component directions as arrows (starting at the origin)
origin = np.array([0, 0])
plt.quiver(*origin, PC1[0]*S[0], PC1[1]*S[0], color='r', scale=8, label='PC1', width=0.005)
plt.quiver(*origin, PC2[0]*S[1], PC2[1]*S[1], color='b', scale=8, label='PC2', width=0.005)

# Draw the line representing the one-dimensional subspace spanned by PC1:
pc1_line = np.linspace(-5, 5, 100)
pc1_line_coords = np.outer(pc1_line, PC1)
plt.plot(pc1_line_coords[:, 0], pc1_line_coords[:, 1], color='r', linestyle='--', label='Subspace (PC1)')

# Plot the projection of the data onto PC1 (these points lie on the PC1 line)
plt.scatter(proj_PC1[:, 0], proj_PC1[:, 1], color='green', alpha=0.5, label='Projections onto PC1')

# Explicitly mark the zero vector, which must lie on any subspace including PC1
plt.scatter([0], [0], color='black', s=100, marker='x', label='Zero Vector')

# Add labels and legend
plt.xlabel("X₁")
plt.ylabel("X₂")
plt.title("PCA Subspace Visualization (Including Zero Vector)")
plt.legend()
plt.axis('equal')
plt.show()
```

- **Data Generation and Centering:**  
  The script creates a 2D data set with correlated features, then centers it.

- **PCA via SVD:**  
  Singular Value Decomposition (SVD) is applied to the centered data. The first row of `Vt` (i.e. `PC1`) is the direction of maximum variance (principal subspace) and the second row (`PC2`) is orthogonal to it.

- **Projection:**  
  Each data point is projected onto `PC1` by computing the inner product with `PC1` and then scaling the direction vector accordingly. This yields the set of points that exactly lie in the one-dimensional subspace spanned by `PC1`.

- **Visualization:**  
  The scatter plot shows (i) the original centered data, (ii) the principal component directions as arrows, (iii) the line representing the subspace spanned by `PC1`, and (iv) the projected data points. This effectively demonstrates that the set of projections forms a subspace, and that the subspace is closed under linear operations.
- **Subspace Property:**
  The set of all linear combinations of the first $k$ principal components forms a subspace of the original feature space. This subspace naturally includes the zero vector, and any addition or scalar multiplication within this reduced-dimensional space remains within it.


#### Relationship with Linear Maps

Recall that the nullspace and range of a linear map are subspaces of the domain and codomain, respectively. Affine subspaces naturally arise when we consider translations of these ranges. For instance, when performing PCA on original (uncentered) data, the transformation is typically given by

$$
\hat{\mathbf{x}} = \mathbf{\mu} + P(\mathbf{x} - \mathbf{\mu}),
$$
where $\mathbf{\mu}$ is the mean of the data and $P$ is the projection onto the subspace spanned by the principal components. The set of all reconstructed points $\hat{\mathbf{x}}$ forms an affine subspace of the original space, which is a translation of the linear subspace $ \operatorname{range}(P) $ by the mean $\mathbf{\mu}$.

In summary, while a subspace must contain the zero vector and be closed under all linear operations, an affine subspace is simply a translated version of a subspace. Affine subspaces are prevalent in applications such as PCA (when data is not centered) and are a key concept for understanding the geometry induced by linear maps in practical settings.
