---
In the figure below, we define three radial‐basis functions (centers at $-2,0,2$, width $\sigma=1$), compute their Jacobian (derivative with respect to $x$), and plot both the basis functions and their derivatives.  

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# 1) Define RBF basis functions and their derivatives (Jacobian)
centers = np.array([-2.0, 0.0, 2.0])
sigma = 1.0

def phi(x):
    """Compute RBF features for array x."""
    return np.exp(- (x[:, None] - centers[None, :])**2 / (2 * sigma**2))

def dphi(x):
    """Compute derivative d phi / dx for each RBF feature."""
    P = phi(x)
    return P * (-(x[:, None] - centers[None, :]) / (sigma**2))

# 2) Domain and evaluation
x = np.linspace(-5, 5, 400)
Phi = phi(x)    # shape (400,3)
DP  = dphi(x)   # shape (400,3)

# 3) Highlight point x0
x0 = 0.0
phi0  = phi(np.array([x0]))[0]
dphi0 = dphi(np.array([x0]))[0]

# 4) Plot basis and Jacobian
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Top: RBF features
for i, c in enumerate(centers):
    axes[0].plot(x, Phi[:, i], label=f'$\phi_{i}(x)$ (center={c})')
axes[0].scatter([x0]*3, phi0, color=['C0','C1','C2'], s=60)
axes[0].set_ylabel('Basis $\phi_i(x)$')
axes[0].legend()
axes[0].set_title('RBF Basis Functions and Their Derivatives')

# Bottom: Jacobian entries
for i in range(len(centers)):
    axes[1].plot(x, DP[:, i], '--', label=f'$\partial_x\phi_{i}(x)$')
axes[1].scatter([x0]*3, dphi0, color=['C0','C1','C2'], s=60)
axes[1].set_ylabel(r'$\partial \phi_i/\partial x$')
axes[1].set_xlabel('$x$')
axes[1].legend()

plt.tight_layout()
plt.show()
```

**Explanation:**

1. **Basis mapping $\phi:\mathbb{R}\to\mathbb{R}^3$**  
   We use three Gaussian RBFs  
   $\phi_i(x)=\exp\bigl(-\tfrac{(x-c_i)^2}{2\sigma^2}\bigr)$  
   centered at $c_1=-2$, $c_2=0$, $c_3=2$.

2. **Jacobian / derivative**  
   Since the input is one–dimensional, the Jacobian reduces to a row vector  
   $\tfrac{d\phi}{dx}(x) = [\,\phi_1'(x),\phi_2'(x),\phi_3'(x)\,]$.  
   We compute $\phi_i'(x)=\phi_i(x)\cdot\bigl[-(x-c_i)/\sigma^2\bigr]$.

3. **Visualization**  
   - **Top panel:** Each RBF feature $\phi_i(x)$ vs. $x$, with the value at $x_0=0$ highlighted.  
   - **Bottom panel:** The corresponding derivatives $\phi_i'(x)$ vs. $x$, again marking $\phi_i'(0)$.

This example illustrates how the Jacobian of a nonlinear basis expansion quantifies the **sensitivity of each feature** to changes in the input. In a model $f(x)=w^\top\phi(x)$, these derivatives appear whenever one needs to understand or optimize how $f$ itself changes with $x$.


```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# Define a vector-valued function f: R2 -> R2
def f(x, y):
    # Example nonlinear function
    return np.array([x**2 - y, y**2 + x])

# Define the Jacobian of f
def jacobian(x, y):
    # df1/dx = 2x, df1/dy = -1
    # df2/dx = 1,  df2/dy = 2y
    return np.array([[2*x, -1],
                     [1,   2*y]])

# Set up grid for visualization
x_vals = np.linspace(-2, 2, 25)
y_vals = np.linspace(-2, 2, 25)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute Jacobian norm (Frobenius) at each grid point
J_norm = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        J = jacobian(X[i,j], Y[i,j])
        J_norm[i,j] = np.linalg.norm(J, ord='fro')

# Plot 1: Heatmap of Jacobian norm
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, J_norm, levels=20, cmap='plasma')
plt.colorbar(label='||J_f(x,y)||_F')
plt.title('Jacobian Norm of $f(x,y)$')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, linestyle=':')

# Plot 2: Local linear approximation at selected points
points = [(-1, -1), (0, 0), (1, 1)]
scale = 0.5  # scaling for visualization
plt.subplot(1, 2, 2)
plt.title('Local Linear Approximation via Jacobian')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, linestyle=':')

# Plot original basis vectors at each point and their images under J
for (x0, y0) in points:
    J = jacobian(x0, y0)
    # Basis vectors
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    # Mapped vectors
    Je1 = J @ e1
    Je2 = J @ e2
    # Plot the point
    plt.scatter(x0, y0, color='black')
    # Plot original basis arrows
    plt.quiver(x0, y0, e1[0]*scale, e1[1]*scale, 
               angles='xy', scale_units='xy', scale=1, color='blue', width=0.003)
    plt.quiver(x0, y0, e2[0]*scale, e2[1]*scale, 
               angles='xy', scale_units='xy', scale=1, color='green', width=0.003)
    # Plot transformed basis arrows
    plt.quiver(x0, y0, Je1[0]*scale, Je1[1]*scale, 
               angles='xy', scale_units='xy', scale=1, color='red', width=0.003)
    plt.quiver(x0, y0, Je2[0]*scale, Je2[1]*scale, 
               angles='xy', scale_units='xy', scale=1, color='orange', width=0.003)
    # Annotate
    plt.text(x0+0.1, y0+0.1, f"({x0},{y0})")
    
plt.legend(['point','e1','e2','J*e1','J*e2'], loc='upper left')
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()
```

1. **Plots a heatmap** of the Frobenius norm of the Jacobian $\mathbf{J}_f(x,y)$ for 
   
   $$
     f(x,y) = \begin{bmatrix} x^2 - y \\ y^2 + x \end{bmatrix},
   $$
   revealing regions where the mapping changes most rapidly.

2. **Displays the local linear approximation** at three sample points $(-1,-1)$, $(0,0)$, and $(1,1)$. At each point:
   - The **blue and green arrows** are the standard basis vectors $\mathbf{e}_1$ and $\mathbf{e}_2$.
   - The **red and orange arrows** are their images under the Jacobian, $\mathbf{J}_f \mathbf{e}_1$ and $\mathbf{J}_f \mathbf{e}_2$.
   - These illustrate how the Jacobian linearly approximates the action of $f$ in a neighborhood: small steps in input space are mapped to (approximately) linear steps in output space.

## Jacobian in Machine Learning

In machine learning, most of the time we’re optimizing a scalar “loss,” but internally the models we train are often vector-valued mappings.

Wherever you see a function  

$$
  f: \mathbb{R}^n \;\longrightarrow\; \mathbb{R}^m
$$
you’ll want its Jacobian.  Here are some of the most common places it shows up:

1. **Neural network layers**  
   Every layer in a feed-forward network maps its input vector to another vector of activations.  During back-propagation you repeatedly apply the chain rule, which involves multiplying by the Jacobian (or its transpose) of each layer’s transformation.  

2. **Multi-output regression & classification**  
   - **Softmax / logistic layer:**  
     $\;f(\mathbf{x}) = \mathrm{softmax}(W\mathbf{x}+b)\in\mathbb{R}^K$  
     The Jacobian tells you how each logit or class probability changes if you perturb the inputs.  
   - **Vector-valued regression:**  
     Predicting, say, 3D positions $\mathbf{y}\in\mathbb{R}^3$ from features $\mathbf{x}\in\mathbb{R}^n$; the Jacobian is crucial for uncertainty propagation, local linear approximations, or Gauss-Newton updates.

3. **Auto­encoders, sequence‐to‐sequence, and any encoder–decoder**  
   You have an “encoder” $f(\mathbf{x})\in\mathbb{R}^d$ whose Jacobian tells you how latent codes shift when the input changes—useful for manifold learning, interpretability, or Jacobian regularization.

4. **Normalizing flows & density models**  
   Invertible flows define  
   $\mathbf{z}=f(\mathbf{x})$  
   and require the **log-determinant of the Jacobian**, $\log|\det J_f(\mathbf{x})|$, to compute exact likelihoods.

5. **Adversarial robustness and regularization**  
   Penalizing the norm of the input–output Jacobian (or its largest singular value) makes your model less sensitive (more “flat”) around the training points, improving robustness.

6. **Meta-learning & fast adaptation (MAML)**  
   Inner‐loop gradient steps involve second derivatives—i.e.\ Jacobians of gradients—so you end up computing Jacobians of vector‐valued gradient updates.

7. **Sensitivity analysis & Taylor expansions**  
   Whenever you want a first-order approximation  

   $$
     f(\mathbf{x}+\delta\mathbf{x})
     \approx f(\mathbf{x}) + J_f(\mathbf{x})\,\delta\mathbf{x},
   $$
   the Jacobian is exactly that linear map.

---

### When do we “see” these vector‐valued functions?

- **Every hidden layer** in a deep net—its activations are vectors.  
- The **final output** in multi‐class classification or multi‐task regression is a vector of length $m$.  
- **Batched inputs**: if you stack $B$ examples into a matrix, your network is really computing  
  $\mathbb{R}^{B\times n}\to\mathbb{R}^{B\times m}$.  The Jacobian then becomes a block matrix or a tensor.  
- **Spatial outputs**: semantic segmentation or image-to-image models output an $(H\times W\times C)$ tensor, which you can flatten to a giant vector.

In all these cases, the Jacobian tells you *how* small changes in the input propagate through the model to produce changes in the vector of outputs—fundamental both for training (via back-prop) and for understanding/modeling sensitivity or invertibility.


