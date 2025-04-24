---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# The Jacobian

In machine learning, most of the time we’re optimizing a scalar “loss,” but internally the models we train are often vector-valued mappings.

Wherever you see a function with multiple outputs, like  

$$
  f: \mathbb{R}^n \;\longrightarrow\; \mathbb{R}^m
$$
you’ll want its **Jacobian**.

The Jacobian is the matrix of first-order partial derivatives of each output of $f$ with respect to each of the input dimensions. It generalizes the gradient of a scalar-valued function to vector-valued functions.

$$\mathbf{J}_f = \begin{bmatrix}
    \frac{\partial f_1}{\partial x_1} & \dots & \frac{\partial f_1}{\partial x_n} \\
    \vdots & \ddots & \vdots \\
    \frac{\partial f_m}{\partial x_1} & \dots & \frac{\partial f_m}{\partial x_n}\end{bmatrix}
\hspace{0.5cm}\text{i.e.}\hspace{0.5cm}
[\mathbf{J}_f]_{ij} = \frac{\partial f_i}{\partial x_j}$$

Note the special case $m = 1$, where $\nabla f = \mathbf{J}_f^{\!\top\!}$.


## **Tanh Basis and Their Jacobians**

We have seen vector-valued functions in the context of **basis functions** such as the tanh basis functions that we used in the prediction of temperature as a function of the day in the year.


$$
\phi_i(x; a_i, b_i) = \tanh(a_i x + b_i)
$$
where $a_i$ and $b_i$ are the slope and bias hyperparameters, respectively.

While in principle, each tanh function has three input dimensions: the input $x$, the slope $a_i$, and the bias $b_i$, in practice, we are often interested in the Jacobian with respect to the hyperparameters $a_i$ and $b_i$ at a fixed input $x_0$.
In this case, the Jacobian is a matrix of partial derivatives with respect to the hyperparameters $a_i$ and $b_i$ at a fixed input value $x_0$.

The Jacobian with respect to the slope $a_i$ is given by:

$$
\frac{\partial \phi_i(x_0)}{\partial a_i} = x_0 (1 - \tanh^2(a_i x_0 + b_i))$$
And the Jacobian with respect to the bias $b_i$ is given by:

$$
\frac{\partial \phi_i(x_0)}{\partial b_i} = 1 - \tanh^2(a_i x_0 + b_i)
$$
This shows how the tanh function responds to changes in the slope and bias hyperparameters.

By our convention, the Jacobian is a matrix of partial derivatives, where each row corresponds to a given output dimension and each column to a different input dimension. In our case the $i$-th row corresponds to the output of the $i$-th tanh basis function and each column corresponds to one of the different slope or bias hyperparameters.

## **Visualizing the Jacobian of Tanh Basis Functions**
In the following code, we visualize the tanh basis functions and their Jacobians with respect to the slope $a$, and bias $b$.
We will plot the three tanh basis functions and their Jacobians with respect to the slope $a$, and bias $b$.

1. **∂φ/∂a** (Jacobian w.r.t. the slope hyperparameter $a$) at a fixed $x_0$  
2. **∂φ/∂b** (Jacobian w.r.t. the bias hyperparameter $b$) at the same fixed $x_0$  

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# Basis parameters
a = np.array([.1, .2, .3])
b = np.array([-10.0,-50.0,-100.0])

# tanh basis and its derivatives
def phi_tanh(x, a, b):
    return np.tanh(a * x + b)

def dphi_tanh_da(x, a, b):
    return x * (1 - np.tanh(a*x + b)**2)

def dphi_tanh_db(x, a, b):
    return 1 - np.tanh(a*x + b)**2

def phi_tanh_vector(x, a_values, b_values, add_bias=False):
    phi = np.array([
        phi_tanh_scalar(x, a_values[i], b_values[i])
        for i in range(len(a_values))
    ])
    # Add bias term if required
    if add_bias:
        phi = np.hstack((phi, np.ones_like(x)))
    return phi

def jacobian_phi_tanh(x, a_values, b_values, add_bias=False):
    jacobian_a = np.array([
        dphi_tanh_da(x, a_values[i], b_values[i])
        for i in range(len(a_values))
    ])
    jacobian_b = np.array([
        dphi_tanh_db(x, a_values[i], b_values[i])
        for i in range(len(a_values))
    ])
    jacobian = np.vstack((jacobian_a, jacobian_b)).T
    # Reshape to match the number of features
    jacobian = jacobian.reshape(len(x), len(a_values) * 2)
    # Add bias term if required
    if add_bias:
        jacobian = np.hstack((jacobian, np.ones_like(x)))
    return jacobian

# 1) Variation w.r.t. slope a at x0
x0 = 150.0
a_vals = np.linspace(0.0, 1, 400)
Phi_a  = np.stack([phi_tanh(x0, a_vals, b[i:i+1]) for i in range(3)], axis=1)
DPa    = np.stack([dphi_tanh_da(x0, a_vals, b[i:i+1]) for i in range(3)], axis=1)

# 2) Variation w.r.t. bias b at x0
b_vals = np.linspace(-60, 0, 400)
Phi_b  = np.stack([phi_tanh(x0, a[i:i+1], b_vals) for i in range(3)], axis=1)
DPb    = np.stack([dphi_tanh_db(x0, a[i:i+1], b_vals) for i in range(3)], axis=1)

# Plot
fig, axs = plt.subplots(4, 1, figsize=(8, 12))

# Slope hyperparameter derivative
for i in range(3):
    axs[0].plot(a_vals, Phi_a[:, i], label=f'φ{i}(a)=tanh(a·{x0}+{b[i]})')
    axs[1].plot(a_vals, DPa[:, i], '--', label=f'∂φ{i}/∂a')
axs[0].set_title(f'Variation w.r.t. slope a at x₀={x0}')
axs[1].set_title(f'Variation w.r.t. slope a at x₀={x0}')
axs[0].legend(); axs[0].grid(True)
axs[1].legend(); axs[1].grid(True)

# Bias hyperparameter derivative
for i in range(3):
    axs[2].plot(b_vals, Phi_b[:, i], label=f'φ{i}(b)=tanh({a[i]}·{x0}+b)')
    axs[3].plot(b_vals, DPb[:, i], '--', label=f'∂φ{i}/∂b')
axs[2].set_title(f'Variation w.r.t. bias b at x₀={x0}')
axs[2].legend(); axs[2].grid(True)
axs[3].set_title(f'Variation w.r.t. bias b at x₀={x0}')
axs[3].legend(); axs[3].grid(True)

plt.tight_layout()
plt.show()
```

We observe, that the slope and bias parameter of the $i$-th tanh basis function only have an effect on the $i$-th output dimension of the Jacobian for a given data point. This is a consequence of the fact that the tanh basis functions are independent of each other.

It follows that for $n$ data points, even though the Jacobian is a $3n$-by-$6$ matrix, each row of the Jacobian has only $2n$ non-zero entries, corresponding to the slope and bias hyperparameters of the $i$-th tanh basis function. Thus, we can think of the Jacobian as a $3n$-by-$2$ matrix, where each row corresponds to the slope and bias hyperparameters of the $i$-th tanh basis function.

This observation allows us to simplify the implementation, as we only need to keep track of the non-zero dimensions.
We organize all the hyperparameters in a $2$-by-$3$ matrix $\mathbf{W}$ and represent the non-zero part of the Jacobian as a $3$-by-$2$ matrix per data point $x$, or a $n$-by-$3$-by-$2$ tensor for the whole data set. Note that the full Jacobian can be obtained by zero-padding.

```{code-cell} ipython3
import numpy as np

class TanhBasis:
    def __init__(self, W):
        self.W = W

    def XW(self, x):
        """Compute the product of the input data and the weights."""
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return x @ self.W[:-1] + self.W[-1]

    def transform(self, x):
        """Compute the tanh basis functions."""
        return np.tanh(self.XW(x))

    def jacobian(self, x):
        """Compute the Jacobian of the tanh basis functions."""
        return self.dXW(x) * (1 - np.tanh(self.XW(x))**2)

    def dXW(self, x):
        """Compute the derivative of the product of the input data and the weights."""
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return np.hstack((x, np.ones((x.shape[0], 1))))
```

Before we can use the `TanhBasis` class with Jacobian to optimize over the hyperparameters in our ridge regression, we need to understand, how the Jacobian is used in the optimization process. The missing ingredient is the **chain rule** that we will discuss next.
