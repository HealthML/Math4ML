---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: math4ml
  language: python
  name: python3
---
# Basis Functions Form a Vector Space

We can generalize the polynomial-feature-based vector space to a broader class of functions expressed as linear combinations of basis functions, not just monomials.
In machine learning, many models can be understood as learning a function $ f(\mathbf{x}) $ that maps input features $ \mathbf{x} \in \mathbb{R}^d $ to an output — such as a predicted value in regression or a class score in classification. A powerful and general approach is to express this function as a **linear combination of basis functions**,  

$$
f(\mathbf{x}) = \sum_{j=1}^m a_j \cdot \phi_j(\mathbf{x}),
$$

where $\mathbf{x}$ and $ f(\mathbf{x}) $ are the model input and output, e.g., in regression or classifiction, $ \Phi = \{ \phi_1, \phi_2, \dots, \phi_m \} $ is a set of fixed **basis functions**, and the coefficients $ a_j \in \mathbb{R} $ are **learnable weights** optimized during training. The basis functions $ \phi_j(\mathbf{x}) $ transform the raw input vector $ \mathbf{x} $ into a new **feature space** where the target function can be represented more easily. These functions can take many forms — from simple monomials to radial basis functions (RBFs), Fourier components, or neural network activations — and the choice of $ \Phi $ plays a central role in the model's expressiveness and generalization. Understanding the structure of these functions and the space they span lays the foundation for many classical and modern learning algorithms.

:::{prf:theorem} Linear function spaces over general basis functions
:label: thm-feature-map-vector-space
:nonumber:

Let $\mathbf{x} \in \mathbb{R}^d$ be an input vector, and let $ \Phi = \{ \phi_1, \phi_2, \dots, \phi_m \} $ be a fixed set of real-valued functions, each mapping $ \mathbb{R}^d \to \mathbb{R} $.

Define the set:

$$
V_\Phi = \left\{ f(\mathbf{x}) = \sum_{j=1}^m a_j \cdot \phi_j(\mathbf{x}) \;\middle|\; a_j \in \mathbb{R} \right\}.
$$

Then $ V_\Phi $ is a vector space over $ \mathbb{R} $.
:::

This holds whether the functions $ \phi_j $ are monomials, radial basis functions (RBFs), sigmoids, Fourier basis functions, or any other fixed set of functions. We will show some examples of these functions later. First, we verify the vector space axioms.

:::{prf:proof}
Let $ f(\mathbf{x}) = \sum_{j=1}^m a_j \phi_j(\mathbf{x}) $ and $ g(\mathbf{x}) = \sum_{j=1}^m b_j \phi_j(\mathbf{x}) $ be two elements in $ V_\Phi $.

- **Closure under addition:**

  $$
  (f + g)(\mathbf{x}) = \sum_{j=1}^m (a_j + b_j) \cdot \phi_j(\mathbf{x}) \in V_\Phi.
  $$

- **Closure under scalar multiplication:**

  $$
  (\lambda f)(\mathbf{x}) = \sum_{j=1}^m (\lambda a_j) \cdot \phi_j(\mathbf{x}) \in V_\Phi.
  $$

- **Zero function:**

  $$
  0(\mathbf{x}) = \sum_{j=1}^m 0 \cdot \phi_j(\mathbf{x})
  $$

  is in $ V_\Phi $ and acts as the additive identity.

- **Additive inverse:**

  $$
  (-f)(\mathbf{x}) = \sum_{j=1}^m (-a_j) \cdot \phi_j(\mathbf{x}),
  $$

  with $ f + (-f) = 0 $.

- **Remaining axioms** (associativity, commutativity, distributivity) follow from the properties of real numbers.

Hence, $ V_\Phi $ is a vector space over $ \mathbb{R} $.
:::

---

## Common Basis Functions in $ V_\Phi $ in Machine Learning

$ \phi_j(\mathbf{x}) $: Feature functions — could be:

### 🔢 1. **Monomial Feature**  

$$
\phi(\mathbf{x}) = x_1^2 \cdot x_2
$$

- Represents interaction and nonlinearity.
- Common in polynomial regression and kernel methods.
- Curved and asymmetric — useful for modeling feature interactions.

### 🌐 2. **Radial Basis Function (RBF)**  

$$
\phi(\mathbf{x}) = \exp(-\gamma \|\mathbf{x}\|^2)
$$

- Peaks at the origin and decays radially.
- Encodes **locality** — only nearby inputs have large activation.
- Basis of RBF networks and RBF kernel SVMs.

### 🎵 3. **Fourier Feature**  

$$
\phi(\mathbf{x}) = \sin(\omega^\top \mathbf{x}), \quad \omega = [1, 1]
$$

- Encodes periodicity or oscillations in space.
- Used in signal processing and random Fourier features for kernel approximation.
- Smooth but non-monotonic.

### 🧠 4. **Neural Net Activation (Tanh)**  

$$
\phi(\mathbf{x}) = \tanh(w^\top \mathbf{x}), \quad w = [1, 1]
$$

- S-shaped nonlinearity common in neural networks.
- Compresses input into [−1, 1] range.
- Nonlinear but continuous and differentiable.

### Visualization of Basis Functions

Here’s a visualization of the four **feature functions** $ \phi_j(\mathbf{x}) $, showing different $ \phi_j(x) $ behave over a scalar input $ x \in \mathbb{R} $.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# 1D input range
x = np.linspace(-4, 4, 500).reshape(-1, 1)

# --- Feature Functions ---

# 1. Monomial: x^3
phi_monomial = x[:, 0]**3

# 2. RBF centered at 0
gamma = 0.5
phi_rbf = np.exp(-gamma * (x[:, 0]**2))

# 3. Fourier: sin(2x)
phi_fourier = np.sin(2 * x[:, 0])

# 4. Neural Net Activation: tanh(2x)
phi_nn = np.tanh(2 * x[:, 0])

# --- Plotting ---
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

titles = [
    r"Monomial: $\phi(x) = x^3$",
    r"RBF: $\phi(x) = \exp(-\gamma x^2)$",
    r"Fourier: $\phi(x) = \sin(2x)$",
    r"NN Activation: $\phi(x) = \tanh(2x)$"
]

functions = [phi_monomial, phi_rbf, phi_fourier, phi_nn]

for ax, y_vals, title in zip(axes, functions, titles):
    ax.plot(x, y_vals, color='royalblue')
    ax.set_title(title)
    ax.grid(True)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\phi(x)$")

plt.tight_layout()
plt.show()
```

What This Shows:

| Feature     | Behavior |
|-------------|----------|
| $ x^3 $   | Nonlinear, unbounded, odd symmetry |
| $ \exp(-\gamma x^2) $ | Peak at center, decays rapidly — localized |
| $ \sin(2x) $ | Periodic — captures repeating structure |
| $ \tanh(2x) $ | S-shaped, saturates for large |x| — smooth thresholding |

Here is a each type of function plotted over a 2D input space $ \mathbf{x} = [x_1, x_2] $:

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# Define a 2D input grid
x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)
X_grid = np.c_[X1.ravel(), X2.ravel()]  # shape: (10000, 2)

# --- Basis Functions ---

# 1. Monomial: x1^2 * x2
monomial = (X_grid[:, 0] ** 2) * X_grid[:, 1]

# 2. RBF centered at origin
gamma = 2.0
rbf = np.exp(-gamma * (X_grid[:, 0] ** 2 + X_grid[:, 1] ** 2))

# 3. Fourier: sin(w^T x), with w = [1, 1]
fourier = np.sin(X_grid @ np.array([1, 1]))

# 4. Neural Net Activation: tanh(w^T x + b), w = [1, 1], b = 0
nn_activation = np.tanh(X_grid @ np.array([1, 1]))

# --- Plotting ---
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

functions = [monomial, rbf, fourier, nn_activation]
titles = [
    r"Monomial: $\phi(\mathbf{x}) = x_1^2 x_2$",
    r"RBF: $\phi(\mathbf{x}) = \exp(-\gamma \|\mathbf{x}\|^2)$",
    r"Fourier: $\phi(\mathbf{x}) = \sin(\omega^\top \mathbf{x})$",
    r"NN Activation: $\phi(\mathbf{x}) = \tanh(w^\top \mathbf{x})$"
]

for ax, Z, title in zip(axes, functions, titles):
    contour = ax.contourf(X1, X2, Z.reshape(X1.shape), levels=20, cmap='coolwarm')
    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    fig.colorbar(contour, ax=ax)

plt.tight_layout()
plt.show()
```

| Basis Functions $ \phi_j $ | Space $ V_\Phi $ Represents            | Used In                             |
|------------------------------|------------------------------------------|-------------------------------------|
| Monomials                    | Polynomial function space $ P_n^d $    | Polynomial regression, SVMs        |
| RBFs                         | Radial function space                    | RBF kernel SVM, RBF networks        |
| Sinusoids                    | Fourier basis (periodic functions)       | Fourier analysis, signal processing |
| Neural activations           | Neural network hypothesis space          | MLPs, deep learning                 |


## Example: Multivariate Polynomials in $ \mathbb{R}^2$

We start with an example in $ \mathbb{R}^2$ that makes the theoretical concept more tangible.

Let’s consider the space $ P_2^2$, i.e., all real-valued polynomials in two variables $ x_1, x_2$ of total degree **at most 2**.

We write $ \mathbf{x} = [x_1, x_2] \in \mathbb{R}^2$, and we're interested in all polynomials of the form:

$$
p(\mathbf{x}) = a_{00} + a_{10}x_1 + a_{01}x_2 + a_{20}x_1^2 + a_{11}x_1x_2 + a_{02}x_2^2
$$

Where each $ a_{ij} \in \mathbb{R}$, and the **total degree** of each monomial (i.e., $ i + j$) is at most 2.

---

### Basis for $ P_2^2$

A natural basis for this space is the set of monomials:

$$
\{1,\ x_1,\ x_2,\ x_1^2,\ x_1x_2,\ x_2^2\}
$$

So, $ \dim P_2^2 = 6$, and every element of $ P_2^2$ is a **linear combination** of these basis monomials.

---

### Example Polynomial

Let’s define:

$$
p(\mathbf{x}) = 3 + 2x_1 - x_2 + 5x_1x_2 - x_2^2
$$

This polynomial is clearly in $ P_2^2$ because:
- All monomials are of total degree ≤ 2.
- The coefficients $ [3, 2, -1, 0, 5, -1]$ match the basis ordering.

---

### Vector Space Properties (concretely)

- **Addition:** If we add two such polynomials, we just add their coefficients.
- **Scalar multiplication:** Scaling multiplies all coefficients.
- **Zero element:** The polynomial $ 0 = 0 + 0x_1 + 0x_2 + 0x_1^2 + 0x_1x_2 + 0x_2^2 $
- **Inverse:** Negate each coefficient.

## Polynomial Features in Machine Learning

Using polynomial vector spaces, we can enhance simple machine learning algorithms by explicitly representing complex, nonlinear relationships. We will use the **Nearest Centroid Classifier**, a linear classifier that classifies samples based on their distance to the centroid of each class. This classifier is simple yet effective, especially when combined with basis functions.

Let's look at a challenging synthetic sata set, the "concentric circles" dataset that contains two classes arranged as points along concentric circles of different radius that are not linearly separable in the original space.

```{code-cell} ipython3
:tags: [hide-input]
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Generate circles (small number to keep it clear)
X, y = make_circles(n_samples=80, factor=0.5, noise=0.05, random_state=42)
# Plot the dataset
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=60)
plt.title("Concentric Circles Dataset")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()
```

The space $ P_n^2 $ gives you a natural **feature map** to transform the data into a higher-dimensional space where the classes may become linearly separable. The transformation is defined as:

$$
\phi: \mathbb{R}^2 \to \mathbb{R}^m, \quad \mathbf{x} \mapsto [1, x_1, x_2, x_1^2, x_1x_2, x_2^2, \dots]
$$

So let's implement this in Python.

```{code-cell} ipython3
def polynomial_features_deg2(X):
    """
    Computes degree-2 polynomial features for 2D input X.
    Input shape: (n_samples, 2)
    Output shape: (n_samples, 5) with features:
    [x1, x2, x1^2, x1*x2, x2^2]
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    return np.column_stack([x1, x2, x1**2, x1 * x2, x2**2])
```
Let's visualize the transformation of the concentric circles dataset using polynomial features.

```{code-cell} ipython3
:tags: [hide-input]
# Polynomial transform (degree 2, no bias)
X_poly = polynomial_features_deg2(X)

# Extract original and new features
x1, x2 = X[:, 0], X[:, 1]
x1_sq = X_poly[:, 2]
x2_sq = X_poly[:, 4]

# Plot 3D view of transformed space
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 5))

# Original 2D space
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(x1, x2, c=y, cmap='coolwarm', s=60)
for i in range(len(X)):
    ax1.text(x1[i]+0.02, x2[i]+0.02, str(i), fontsize=10)
ax1.set_xlabel(r"$x_1$")
ax1.set_ylabel(r"$x_2$")
ax1.set_title(r"Original Space ($\mathbb{R}^2$)")
ax1.axis('equal')

# Transformed 3D space (x1^2 vs x2^2)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(x1_sq, x2_sq, X_poly[:, 3], c=y, cmap='coolwarm', s=60)
for i in range(len(X)):
    ax2.text(x1_sq[i], x2_sq[i], X_poly[i, 3], str(i), fontsize=9)
ax2.set_xlabel(r"$x_1^2$")
ax2.set_ylabel(r"$x_2^2$")
ax2.set_zlabel(r"$x_1 x_2$")
ax2.set_title(r"Transformed Feature Space ($P_2^2$)")

plt.tight_layout()
plt.show()
```
We see that in the transformed space $ P_2^2 $ (right), the data points are more spread out and the clusters are linearly separable.
So, let's apply the Nearest Centroid Classifier to the original and the transformed data.

```{code-cell} ipython3
:tags: [hide-input]
from sklearn.neighbors import NearestCentroid

# Model A: Nearest Centroid on raw features
model_raw = NearestCentroid()
model_raw.fit(X, y)

# Model B: Nearest Centroid on polynomially transformed features
X_poly = polynomial_features_deg2(X)
model_poly = NearestCentroid()
model_poly.fit(X_poly, y)

# Create meshgrid for plotting decision boundaries
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 300),
                     np.linspace(-1.5, 1.5, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

# Predict with raw model
Z_raw = model_raw.predict(grid).reshape(xx.shape)

# Predict with polynomial model
grid_poly = polynomial_features_deg2(grid)
Z_poly = model_poly.predict(grid_poly).reshape(xx.shape)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Raw features plot
axes[0].contourf(xx, yy, Z_raw, alpha=0.4, cmap='coolwarm')
axes[0].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
axes[0].set_title("Centroid Classifier (Raw Features)")
axes[0].axis('equal')

# Polynomial features plot
axes[1].contourf(xx, yy, Z_poly, alpha=0.4, cmap='coolwarm')
axes[1].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
axes[1].set_title("Centroid Classifier ($P_2^2$ Features)")
axes[1].axis('equal')

plt.tight_layout()
plt.show()
```

The left plot shows the decision boundary of the Nearest Centroid Classifier using the original features, while the right plot shows the decision boundary after applying polynomial feature expansion. The polynomial transformation allows the classifier to separate the classes effectively. Note that the Nearest Centroid classifier is a linear classifier. While the decision boundary forms a circle in the original space $\mathbb{R}^2$, it is a flat hyperplane in the transformed space. This is a key point when using non-linear basis functions in the context of linear machine learning methods: Learned functions are linear in the transformed space, but can be nonlinear in the original space.

## Further Examples (Moons + XOR)

Let's also compare the polynomial $ P_2^2 $ transformation to the Radial Basis Function (RBF) transformation and add two more challenging datasets: the **Two Moons** and **XOR** datasets.

The Two Moons dataset consists of two interleaving half circles, while the XOR dataset consists of four points arranged in a square pattern. Both datasets are not linearly separable in their original feature space.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

# --- Polynomial feature function (degree 2) ---
def polynomial_features_deg2(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    return np.column_stack([x1, x2, x1**2, x1 * x2, x2**2])

# --- RBF feature function ---
def rbf_features(X, centers, gamma=1.0):
    return np.exp(-gamma * np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))

# --- RBF centers: grid over input space
grid_centers = np.array([[x, y] for x in np.linspace(-2, 2, 5) for y in np.linspace(-2, 2, 5)])

# --- Dataset 1: Concentric Circles ---
X0, y0 = make_circles(n_samples=80, factor=0.5, noise=0.1, random_state=42)
X0 = StandardScaler().fit_transform(X0)

# --- Dataset 2: Two Moons ---
X1, y1 = make_moons(n_samples=80, noise=0.05, random_state=1)
X1 = StandardScaler().fit_transform(X1)

# --- Dataset 3: XOR pattern ---
X2_base = np.array([[0, 0], [1, 1], [0, 1], [1, 0]]) * 2 - 1
y2_base = np.array([0, 0, 1, 1])
X2 = np.tile(X2_base, (20, 1)) + np.random.normal(scale=0.2, size=(80, 2))
y2 = np.tile(y2_base, 20)
X2 = StandardScaler().fit_transform(X2)

datasets = [
    ("Circles", X0, y0),
    ("Two Moons", X1, y1),
    ("XOR", X2, y2)
]

# --- Plotting ---
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i, (name, X, y) in enumerate(datasets):
    # Raw model
    model_raw = NearestCentroid()
    model_raw.fit(X, y)

    # Polynomial model
    X_poly = polynomial_features_deg2(X)
    model_poly = NearestCentroid()
    model_poly.fit(X_poly, y)

    # RBF model
    X_rbf = rbf_features(X, grid_centers, gamma=2.0)
    model_rbf = NearestCentroid()
    model_rbf.fit(X_rbf, y)

    # Grid for visualization
    xx, yy = np.meshgrid(np.linspace(-2, 2, 300),
                         np.linspace(-2, 2, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_poly = polynomial_features_deg2(grid)
    grid_rbf = rbf_features(grid, grid_centers, gamma=2.0)

    # Predictions
    Z_raw = model_raw.predict(grid).reshape(xx.shape)
    Z_poly = model_poly.predict(grid_poly).reshape(xx.shape)
    Z_rbf = model_rbf.predict(grid_rbf).reshape(xx.shape)

    # --- Plot Raw ---
    axes[i, 0].contourf(xx, yy, Z_raw, alpha=0.4, cmap='coolwarm')
    axes[i, 0].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')
    axes[i, 0].set_title(f"{name} - Raw Features")
    axes[i, 0].axis('equal')

    # --- Plot Polynomial ---
    axes[i, 1].contourf(xx, yy, Z_poly, alpha=0.4, cmap='coolwarm')
    axes[i, 1].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')
    axes[i, 1].set_title(f"{name} - $P_2^2$ Features")
    axes[i, 1].axis('equal')

    # --- Plot RBF ---
    axes[i, 2].contourf(xx, yy, Z_rbf, alpha=0.4, cmap='coolwarm')
    axes[i, 2].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')
    axes[i, 2].set_title(f"{name} - RBF Features")
    axes[i, 2].axis('equal')

plt.tight_layout()
plt.show()
```

We see that both feature transformations perform well on the **concentric circles** dataset and the **XOR** dataset, but the **polynomial features** are not as effective on the **two moons** dataset as the RBF features, which can capture the local structure of the data.

