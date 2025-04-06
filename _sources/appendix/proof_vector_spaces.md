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
# Multivariate Basis Functions Form a Vector Space
We can generalize the polynomial-feature-based vector space to a broader class of **functions expressed as linear combinations of basis functions** — where the basis functions $ \phi_j(\mathbf{x}) $ can be *anything*, not just monomials.

Let’s formalize this idea that is  **at the core of kernel methods and feature mappings** in machine learning.


:::{prf:theorem} Linear function spaces over general basis functions
:label: thm-feature-map-vector-space
:nonumber:

Let $\mathbf{x} \in \mathbb{R}^d$ be an input vector, and let $ \Phi = \{ \phi_1, \phi_2, \dots, \phi_m \} $ be a fixed set of real-valued functions, each mapping $ \mathbb{R}^d \to \mathbb{R} $.

Define the set:

$$
V_\Phi = \left\{ f(\mathbf{x}) = \sum_{j=1}^m a_j \cdot \phi_j(\mathbf{x}) \;\middle|\; a_j \in \mathbb{R} \right\}.
$$

Then $ V_\Phi $ is a vector space over $ \mathbb{R} $.

This holds whether the functions $ \phi_j $ are monomials, radial basis functions (RBFs), sigmoids, Fourier basis functions, or any other fixed set of functions.
:::

We verify the vector space axioms.

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

## 🧠 Interpretation in ML

- $ \phi_j(\mathbf{x}) $: Feature functions — could be:


Here’s a visualization of four common **feature functions** $ \phi_j(\mathbf{x}) $, each plotted over a 2D input space $ \mathbf{x} = [x_1, x_2] $:

---

### 🔢 1. **Monomial Feature**  

$$
\phi(\mathbf{x}) = x_1^2 \cdot x_2
$$

- Represents interaction and nonlinearity.
- Common in polynomial regression and kernel methods.
- Curved and asymmetric — useful for modeling feature interactions.

---

### 🌐 2. **Radial Basis Function (RBF)**  

$$
\phi(\mathbf{x}) = \exp(-\gamma \|\mathbf{x}\|^2)
$$

- Peaks at the origin and decays radially.
- Encodes **locality** — only nearby inputs have large activation.
- Basis of RBF networks and RBF kernel SVMs.

---

### 🎵 3. **Fourier Feature**  

$$
\phi(\mathbf{x}) = \sin(\omega^\top \mathbf{x}), \quad \omega = [1, 1]
$$

- Encodes periodicity or oscillations in space.
- Used in signal processing and random Fourier features for kernel approximation.
- Smooth but non-monotonic.

---

### 🧠 4. **Neural Net Activation (Tanh)**  

$$
\phi(\mathbf{x}) = \tanh(w^\top \mathbf{x}), \quad w = [1, 1]
$$

- S-shaped nonlinearity common in neural networks.
- Compresses input into [−1, 1] range.
- Nonlinear but continuous and differentiable.

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
    ax.set_xlabel("$x$")
    ax.set_ylabel("$\phi(x)$")

plt.tight_layout()
plt.show()
```

This shows how different feature functions $ \phi_j(x) $ behave over a scalar input $ x \in \mathbb{R} $.

## ✅ 1D Visualization Code

What This Shows:

| Feature     | Behavior |
|-------------|----------|
| $ x^3 $   | Nonlinear, unbounded, odd symmetry |
| $ \exp(-\gamma x^2) $ | Peak at center, decays rapidly — localized |
| $ \sin(2x) $ | Periodic — captures repeating structure |
| $ \tanh(2x) $ | S-shaped, saturates for large |x| — smooth thresholding |


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


- $ f(\mathbf{x}) $: Model output — e.g., in regression or classification

- $ a_j $: Learnable weights — fitted during training

---

## ✅ Summary Table

| Basis Functions $ \phi_j $ | Space $ V_\Phi $ Represents            | Used In                             |
|------------------------------|------------------------------------------|-------------------------------------|
| Monomials                    | Polynomial function space $ P_n^d $    | Polynomial regression, SVMs        |
| RBFs                         | Radial function space                    | RBF kernel SVM, RBF networks        |
| Sinusoids                    | Fourier basis (periodic functions)       | Fourier analysis, signal processing |
| Neural activations           | Neural network hypothesis space          | MLPs, deep learning                 |

