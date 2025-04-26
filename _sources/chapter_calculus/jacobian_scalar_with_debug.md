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

## Basis Functions and their Jacobians

We have seen vector-valued functions in the context of **basis functions** such as the tanh basis functions that we used in the prediction of temperature as a function of the day in the year.

In the context of basis functions, we have a function $\boldsymbol{\phi}$ that transforms the input data $\mathbf{x}$ into a new feature space. The basis functions are typically parameterized by weights and biases, and they can be thought of as nonlinear transformations of the input data.
The goal of using basis functions is to create a new feature space that can better capture the underlying structure of the data, allowing for more flexible modeling.

In order to optimize over the basis functions, we need to compute their Jacobian. What are the inputs and what are the outputs of the basis functions?

If we have $K$ basis functions, these transform a $D$-dimensional feature vector $\mathbf{x}$ into a $K$-dimensional vector $\boldsymbol{\phi}_\boldsymbol{\theta}(\mathbf{x})$.

$$
\boldsymbol{\phi}_\boldsymbol{\theta}(\mathbf{x}): \mathbb{R}^D \rightarrow \mathbb{R}^K
$$

Here, $\boldsymbol{\theta}$ is the vector of parameters of $\boldsymbol{\phi}$.

In machine learning our goal will be to optimize over these features based on a loss function computed over a fixed training data set of size $N$ that is transformed using $\boldsymbol{\phi}$. Thus, we need to understand how the transformed data features of the training data set change as we modify the parameters $\boldsymbol{\theta}$.

In this view, we have to treat the parameters $\boldsymbol{\theta}$ as the input and the transformed data set as the output of $\boldsymbol{\phi}$.

$$
\boldsymbol{\phi}_{\mathbf{X}}(\boldsymbol{\theta}): \mathbb{R}^{P}\rightarrow\mathbb{R}
^{NK},$$
where $P$ is the total number of parameters of all the basis functions.

Thus, the Jacobian will be a matrix of size $P$ times $(NK)$.

To better keep track of the dimensions, we can think of the Jacobian as a $P$-by-$K$ matrix for each data point $n$.

$$
\mathbf{J}_{\boldsymbol{\phi}}(\mathbf{x}_n) = \begin{bmatrix}
    \frac{\partial \phi_{n1}}{\partial \theta_1} & \dots & \frac{\partial \phi_{n1}}{\partial \theta_P} \\
    \vdots & \ddots & \vdots \\
    \frac{\partial \phi_{nK}}{\partial \theta_1} & \dots & \frac{\partial \phi_{nK}}{\partial \theta_P}\end{bmatrix}
\hspace{0.5cm}\text{i.e.}\hspace{0.5cm}
[\mathbf{J}_{\boldsymbol{\phi}}]_{n, k; p} = \frac{\partial \phi_{nk}}{\partial \theta_p}
$$

where $n$ is the index of the data point, $k$ is the index of the basis function, and $p$ is the index of the parameter.

## Jacobian of tanh basis functions

Let's apply this to an example, where we transform the data using $K$ tanh basis functions.

Let $\phi_{nk}$ be the $k$-th basis-function activation on the $n$-th sample.

$$
\phi_{nk}
=\tanh\!\bigl(z_{nk}\bigr)
\quad\text{with}\quad
z_{nk} \;=\;\sum_{d=1}^D a_{dk}\,x_{nd}\;+\;b_k
$$

So, the parameters of the basis functions are the weights $a_{dk}$ and the biases $b_k$.
The weights $a_{dk}$ are the slopes of the $k$-th basis function with respect to the $d$-th input feature, and the biases $b_k$ are the offsets of the $k$-th basis function.

### Partial derivative for the weights $a_{kd}$

$$
\frac{\partial\,\phi_{nk}}{\partial\,a_{d\ell}}
\quad,\quad
d\in\{1,\dots,D\},\;k,\ell\in\{1,\dots,K\}.
$$

---

### 1. Chain-rule decomposition

By the chain rule,

$$
\frac{\partial\,\phi_{nk}}{\partial\,a_{d\ell}}
=\underbrace{\frac{d}{dz}\tanh(z)\Big|_{z=z_{nk}}}_{\tanh'(z_{nk})}
\;\times\;
\frac{\partial\,z_{nk}}{\partial\,a_{d\ell}}.
$$

---

### 2. Compute each factor

1. **Derivative of tanh**  

   $$
   \frac{d}{dz}\tanh(z)
   =1 - \tanh^2(z)
   \;\;\;\Longrightarrow\;\;\;
   \tanh'(z_{nk}) = 1 - \tanh^2\bigl(z_{nk}\bigr)
   = 1 - \phi_{nk}^2.
   $$

2. **Derivative of the affine argument**  

   $$
   z_{nk}
   = \sum_{d=1}^D a_{dk}\,x_{nd} + b_k
   \quad\Longrightarrow\quad
   \frac{\partial z_{nk}}{\partial a_{d\ell}}
   = 
   \begin{cases}
     x_{nd}, & \ell = k,\\
     0,      & \ell \neq k.
   \end{cases}
   $$

---

### 3. Put it together

Hence

$$
\boxed{
\frac{\partial\,\phi_{nk}}{\partial\,a_{d\ell}}
=
\begin{cases}
\bigl(1 - \phi_{nk}^2\bigr)\;x_{nd}, 
& \ell=k,\\
0, 
& \ell\neq k.
\end{cases}
}
$$

Equivalently, in one expression using the Kronecker delta $\delta_{\ell k}$:

$$
\frac{\partial\,\phi_{nk}}{\partial\,a_{d\ell}}
= \delta_{\ell k}\;\bigl(1 - \phi_{nk}^2\bigr)\;x_{nd}.
$$

We can see that the Jacobian has plenty of zeros, as the parameter of the $k$-th basis function only have an effect on the output of the $k$-th basis function and thus the partial derivatives for all the other basis functions are zero. This is a consequence of the fact that the tanh basis functions are independent of each other.

### Matrix-form view

If we collect all $\phi_{nk}$ into an $N\times K$ matrix $\boldsymbol{\Phi}$, and all weights $a_{dk}$ into a $D\times K$ matrix $\mathbf{A}$, then the Jacobian $\tfrac{\partial\,\boldsymbol{\Phi}}{\partial\,\mathbf{A}}$ is a 4-tensor with entries

$$
\bigl[J_{\boldsymbol{\Phi},\mathbf{A}}\bigr]_{\,n,k\,;\,d,\ell}
=\frac{\partial\,\phi_{nk}}{\partial\,a_{d\ell}}
=\delta_{\ell k}\,(1-\phi_{nk}^2)\,x_{nd}.
$$

This observation allows us to simplify the implementation, as in practice, we only need to keep track of the non-zero part of the Jacobian.

For each basis unit $k$ you get the non-zero gradient

$$
\nabla_{a_{\cdot k}}\phi_{\cdot k}
=\bigl(1-\phi_{\cdot k}^2\bigr)\odot x_{\cdot\,:\,}
$$
(where “$\odot$” is element-wise multiplication of the vector $1-\phi_{nk}^2$ with each column of the design matrix).

## Partial derivative for the weights $b_k$

The non-zero part of the Jacobian with respect to the bias $b_k$ is a vector of size $N$ (one for each data point) and is given by:

$$
\frac{\partial\,\phi_{nk}}{\partial\,b_k}
=\frac{d}{dz}\tanh(z)\Big|_{z=z_{nk}}
=\frac{d}{dz}\tanh(z_{nk})
=1 - \tanh^2(z_{nk})
=1 - \phi_{nk}^2.
$$

This shows how the tanh function responds to changes in the bias hyperparameter $b_k$.

Let's implement the tanh basis with a function `jacobian(X)` that returns the Jacobian matrix.

The returned Jacobian `J` has shape $(N,\,D+1,\,k)$ with

$$
J[n,d,k] \;=\; \frac{\partial\,\phi_{n,k}}{\partial\,W_{d,k}}
\;=\;
\begin{cases}
(1-\phi_{n,k}^2)\,X_{n,d}, & d<D\quad(\text{zero-based indexing}),\\
(1-\phi_{n,k}^2), & d=D\quad(\text{bias term}).
\end{cases}
$$

Analytic derivations and implementations of Jacobians are always invovled and there are many ways to make errors.
To assert that the implementation is correct, we  added a `numerical_jacobian(X, eps)` method to the `TanhBasis` class. It perturbs each parameter $W_{d,k}$ by $\pm\varepsilon$, recomputes the activations, and uses a central difference to numerically approximate

$$
\frac{\partial \phi_{n,k}}{\partial W_{d,k}}
\equiv \frac{\phi_{n,k}(W_{d,k}+\varepsilon)\;-\;\phi_{n,k}(W_{d,k}-\varepsilon)}{2\varepsilon}.
$$

```{code-cell} ipython3
import numpy as np

class TanhBasis:
    def __init__(self, W):
        """
        W: array of shape (D+1, P), where
        - W[:D, k] are the slopes a_dk for each input dimension d and unit k
        - W[D, k] is the bias b_k for unit k.
        """
        self.W = W.copy()

    def Z(self, X):
        """Compute the product of the input data and the weights."""
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        return X @ self.W[:-1] + self.W[-1]

    def transform(self, X):
        """Compute the tanh basis functions."""
        return np.tanh(self.Z(X))

    def jacobian(self, X):
        """Compute the Jacobian of the tanh basis functions."""
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        dZ_dW = np.hstack((X, np.ones((X.shape[0], 1)))) # shape (N,D+1)
        dPhi_dz = (1 - np.tanh(self.Z(X))**2)   # shape (N,P)
        return dZ_dW[:,:,np.newaxis] * dPhi_dz[:,np.newaxis,:] # shape (N,D+1,P)

    def numerical_jacobian(self, X, eps=1e-6):
        """
        Numerically approximate the Jacobian of transform(X) wrt W.
        Returns an array of shape (n, d+1, p).
        """
        original_W = self.W.copy()
        N, D = X.shape
        K = self.W.shape[1]
        num_J = np.zeros((N, D+1, K))
        # iterate over each parameter j,k
        for d in range(D+1):
            for k in range(K):
                # perturb up
                self.W = original_W.copy()
                self.W[d, k] += eps
                phi_plus = self.transform(X)
                # perturb down
                self.W = original_W.copy()
                self.W[d, k] -= eps
                phi_minus = self.transform(X)
                # central difference
                num_J[:, d, k] = (phi_plus[:, k] - phi_minus[:, k]) / (2 * eps)
        # restore W
        self.W = original_W
        return num_J
```

A quick comparison on random data shows the max absolute difference between analytic and numeric Jacobians is on the order of $10^{-11}$, confirming that the analytic Jacobian is correctly implemented:

```{code-cell} ipython3
:tags: [hide-input]
np.random.seed(0)
X = np.random.randn(5, 3)           # 5 samples, 3 features
W_init = np.random.randn(4, 2)*0.1  # 3 slopes + 1 bias, 2 units
tanh_basis = TanhBasis(W_init)

# Analytical Jacobian
J_analytic = tanh_basis.jacobian(X)
# Numerical Jacobian
J_numeric  = tanh_basis.numerical_jacobian(X, eps=1e-6)

print("Max abs diff:", np.max(np.abs(J_analytic - J_numeric)))
```

---

## **Visualizing the Jacobian of Tanh Basis Functions**
In the following, we visualize the three tanh basis functions that we used in the previous Section for temperature prediction and their Jacobians with respect to the slope $a$, and bias $b$.
We will plot the three tanh basis functions and their Jacobians with respect to the slope $a$, and bias $b$.

1. **∂φ/∂a** (Jacobian w.r.t. the slope hyperparameter $a$) at a fixed 1-dimensional $x_0$  
2. **∂φ/∂b** (Jacobian w.r.t. the bias hyperparameter $b$) at the same fixed 1-dimensional $x_0$  

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# Basis parameters
a = np.array([.1, .2, .3])
b = np.array([-10.0,-50.0,-100.0])

# tanh basis for single dimensional x and its derivatives
def phi_tanh(x, a, b):
    return np.tanh(a * x + b)

def dphi_tanh_da(x, a, b):
    return x * (1 - np.tanh(a*x + b)**2)

def dphi_tanh_db(x, a, b):
    return 1 - np.tanh(a*x + b)**2

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

Before we can use the `TanhBasis` class with Jacobian to optimize over the hyperparameters in our ridge regression, we need to understand, how the Jacobian is used in the optimization process. The missing ingredient is the **chain rule** that we will discuss next.

---

# The chain rule

Most functions that we wish to optimize are not completely arbitrary
functions, but rather are composed of simpler functions which we know
how to handle. The chain rule gives us a way to calculate derivatives
for a composite function in terms of the derivatives of the simpler
functions that make it up.

The chain rule from single-variable calculus should be familiar:

$$(f \circ g)'(x) = f'(g(x))g'(x)$$ 

where $\circ$ denotes function
composition.

There is a natural generalization of this rule to multivariate functions.

:::{prf:theorem} Multivariate Chain Rule
:label: chain-rule
:nonumber:
Suppose $f : \mathbb{R}^m \to \mathbb{R}^k$ and
$g : \mathbb{R}^n \to \mathbb{R}^m$.

Then $f \circ g : \mathbb{R}^n \to \mathbb{R}^k$ and

$$\mathbf{J}_{f \circ g}(\mathbf{x}) = \mathbf{J}_f(g(\mathbf{x}))\mathbf{J}_g(\mathbf{x})$$
:::

In the special case $k = 1$ we have the following corollary since
$\nabla f = \mathbf{J}_f^{\!\top\!}$.

:::{prf:corollary} Chain Rule for Scalar-Valued Functions
:label: chain-rule-scalar
:nonumber:
Suppose $f : \mathbb{R}^m \to \mathbb{R}$ and
$g : \mathbb{R}^n \to \mathbb{R}^m$. Then
$f \circ g : \mathbb{R}^n \to \mathbb{R}$ and

$$\nabla (f \circ g)(\mathbf{x}) = \mathbf{J}_g(\mathbf{x})^{\!\top\!} \nabla f(g(\mathbf{x}))$$
:::

## Chain Rule for Basis Function Regression

Now we can apply the chain rule to optimize the hyperparameters of the tanh basis functions in the context of our temperature prediction example.
We still have to modify our ridge regression code to use the tanh basis function class and enable optimization over the hyperparameters using the chain rule.

So, let's derive the Jacobian of ridge regression with respect to the hyperparameter matrix $\mathbf{W}_\phi$ of the basis functions $\phi: \operatorname{dom}{x} \rightarrow \mathbb{R}^d$.

Let's have a look in how the basis functions affect the loss function.

Let $f$ be the loss function, which is a function of the weights $\mathbf{w}$ and the hyperparameters $\mathbf{W}_\phi$ be the matrix of hyperparameters of the tanh basis functions. The loss function is given by:

$$
L(\mathbf{w}, \mathbf{W}_\phi) = L(\mathbf{w}, \mathbf{W}_\phi) = \frac{1}{2n}\left(\sum_{i=1}^n l_i\right) + \frac{\lambda}{2}\|\mathbf{w}\|^2_2 = \frac{1}{2n}\left(\sum_{i=1}^n ({y}_i - \boldsymbol{\phi}(\mathbf{x}_i; \mathbf{W}_\phi)\mathbf{w})^2\right) + \frac{\lambda}{2}\|\mathbf{w}\|^2_2
$$

When combining all $n$ transformed input data points $\boldsymbol{\phi(\mathbf{x}_i;\mathbf{W}_\phi)}$ into the transformed design matrix $\boldsymbol{\Phi}(\mathbf{W}_\phi) \in\mathbb{R}^{n,d}$, and all the labels into the vector $\mathbf{y}\in\mathbb{R}^n$, we get

$$
L(\mathbf{w}, \mathbf{W}_\phi) = \frac{1}{2n}\left(\mathbf{y} - \boldsymbol{\Phi}(\mathbf{W}_\phi)\mathbf{w}\right)^\top\left(\mathbf{y} - \boldsymbol{\Phi}(\mathbf{W}_\phi)\mathbf{w}\right) + \frac{\lambda}{2}\|\mathbf{w}\|^2_2
$$

Let $\mathbf{J}_\phi$ be the Jacobian of the tanh basis functions with respect to the hyperparameters $\mathbf{W}_\phi$ at a fixed input value $x_0$, as we have derived in the last section.

As the loss is scalar, we can apply the Chain Rule for Scalar-Valued Functions.

$$\nabla_{\mathbf{W}_\phi} (L \circ \Phi)(\mathbf{W}_\phi) = \mathbf{J}_\Phi(\mathbf{W}_\phi)^{\!\top\!} \nabla_{\mathbf{W}_\phi} L(\mathbf{w}, \mathbf{W}_\phi)$$

We have derived $\mathbf{J}_\Phi(\mathbf{W}_\phi)$ in the last section. Thus, the missing ingredient that we need to derive is $\nabla_{\mathbf{W}_\phi} L(\mathbf{w}, \mathbf{W}_\phi)$. Note that if we change $\mathbf{W}_\phi$ we are changing the data representation that goes into the regression function. It follows that in contrast to the gradient that we used to optimize the ridge regression weights, we now have to take the gradient of the mean squared error with respect to the input data dimensions.

Let's start by computing the gradient of the squared error $l_i$ for the $i$-th data point only:

$$
\nabla_{\boldsymbol{\phi}_i} l_i = \nabla_{\boldsymbol{\phi}_i} (y_i - \boldsymbol{\phi}(\mathbf{x}_i; \mathbf{W}_\phi)^\top\mathbf{w})^2 = -2 \mathbf{w}(y_i - \boldsymbol{\phi}(\mathbf{x}_i; \mathbf{W}_\phi)^\top\mathbf{w})
$$

This gradient is a vector of length $d$. It follows that the gradient for the loss $L$ over the whole training data set, will be a vector of length $dn$, i.e. the concatenation of the $n$ gradient vectors of all the $l_i$. As for implementation purposes it is useful to keep track of the sample indices and the dimension indices, we write this gradient as the $n$-by-$d$ matrix $\nabla_{\boldsymbol{\Phi}} L(\mathbf{w}, \mathbf{W}_\phi)$

$$
\nabla_{\boldsymbol{\Phi}} L(\mathbf{w}, \mathbf{W}_\phi) = 
\frac{1}{2n} \begin{pmatrix} (\nabla_{\boldsymbol{\phi}} l_i)^\top \end{pmatrix}_{i=1}^n
$$

Alternatively, we could have used matrix derivatives to directly derive $\nabla_{\boldsymbol{\Phi}} L(\mathbf{w}, \mathbf{W}_\phi)$ as

$$
\nabla_{\boldsymbol{\Phi}} L(\mathbf{w}, \mathbf{W}_\phi) = \frac{-1}{n}\mathbf{w}\left(\mathbf{y} - \boldsymbol{\Phi}(\mathbf{W}_\phi)\mathbf{w}\right)
$$

Let's integrate this term into our ridge regression implementation:

```{code-cell} ipython3
import numpy as np

class BasisFunctionRidgeRegressionGD:
    def __init__(self, basis_function, learning_rate=0.01, num_iterations=1000, ridge=0.1):
        self.basis_function = basis_function
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.ridge = ridge
        self.ridge_basis = 0.1
        self.w = None

    def mse(self, X, y):
        # Mean Squared Error
        residuals = (y - self.pred(X))
        return np.mean(residuals*residuals)

    def d_loss_d_Phi(self, Phi, y):
        # gradient of the mean squared error w.r.t. Phi
        N = Phi.shape[0]
        if self.w is None:
            self.w = np.zeros(Phi.shape[1])
        residuals = (y - Phi @ self.w)
        return -residuals[:,np.newaxis] * self.w[np.newaxis,:] / N
    
    def loss(self, X, y):
        L = self.loss_Phi(self.basis_function.transform(X),y)
        #L = 0.5*self.mse(X,y) + 0.5*self.ridge*np.sum(self.w**2)
        # add penalty on basis‐params:
        # L += 0.5*self.ridge_basis*np.sum(self.basis_function.W**2)
        return L

    def loss_Phi(self, Phi, y):
        residuals = (y - Phi @ self.w)
        mse = np.mean(residuals*residuals)
        L = 0.5*mse + 0.5*self.ridge*np.sum(self.w**2)
        # add penalty on basis‐params:
        L += 0.5*self.ridge_basis*np.sum(self.basis_function.W**2)
        return L

    def gradient_w(self, X, y):
        # Gradient of the loss
        return -(self.basis_function.transform(X).T @ (y - self.pred(X))) / len(y) + self.ridge * self.w

    def gradient_basis_function(self, X, y):
        # Gradient of the loss
        return -self.w[np.newaxis,:] * (y - self.pred(X))[:,np.newaxis] / len(y)

    def gradient_basis_function_W(self, X, y):
        grad_loss_bf = self.gradient_basis_function(X, y)      # shape (N,P)
        jacobian_phi = self.basis_function.jacobian(X)     # shape (N,D+1,P)
        # chain‐rule: dL/dW_phi = sum_i grad_bf[i] • jacobian_phi[i]
        res = grad_loss_bf[:, None, :] * jacobian_phi          # (N, D+1, P)
        gW = res.sum(0)                                    # (D+1, P)
        # optional L2 on W:
        gW += self.ridge_basis * self.basis_function.W
        return gW

    def fit(self, X, y):
        Phi = self.basis_function.transform(X)
        # self.w = np.random.randn(Phi.shape[1])*0.001
        self.w = np.zeros(Phi.shape[1])
        for it in range(self.num_iterations):
            gW = self.gradient_basis_function_W(X, y)
            grad_w = self.gradient_w(X, y)
            # debug print every 100 iters
            if 0: # it % 100 == 
                ana = self.gradient_basis_function(X,y)
                num = self.numerical_grad_Phi(X, y)
                diff= np.max(np.abs(ana-num))
                print(f"iter {it}: max|ana−num| Phi= {diff:.3e}")
                if diff>1e-6:
                    print(ana)
                    print(num)

                ana = gW
                num = self.numerical_grad_W(X, y)
                diff=np.max(np.abs(ana-num))
                print(f"iter {it}: max|ana−num| W  = {diff:.3e}")
                if diff>1e-6:
                    print(ana)
                    print(num)
                    print (self.basis_function.W)
            # update basis function W and w
            self.basis_function.W -= self.learning_rate * gW
            self.w -= self.learning_rate * grad_w

    def numerical_grad_W(self, X, y, eps=1e-7):
        """
        Numerically approximate dL/dW by central finite differences.
        Returns an array of the same shape as self.basis_function.W.
        """
        W = self.basis_function.W
        num_grad = np.zeros_like(W)
        # flatten indices
        it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = W[idx]
            # f(W + eps)
            W[idx] = orig + eps
            loss_plus = self.loss(X, y)
            # f(W - eps)
            W[idx] = orig - eps
            loss_minus = self.loss(X, y)
            # central difference
            num_grad[idx] = (loss_plus - loss_minus) / (2*eps)
            # restore
            W[idx] = orig
            it.iternext()
        return num_grad

    def numerical_grad_Phi(self, X, y, eps=1e-7):
        """
        Numerically approximate dL/dPhi by central finite differences.
        Returns an array of the same shape as self.basis_function.transform(X)
        """
        Phi = self.basis_function.transform(X)
        num_grad = np.zeros_like(Phi)
        # flatten indices
        it = np.nditer(Phi, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = Phi[idx]
            # f(W + eps)
            Phi[idx] = orig + eps
            loss_plus = self.loss_Phi(Phi, y)
            # f(W - eps)
            Phi[idx] = orig - eps
            loss_minus = self.loss_Phi(Phi, y)
            # central difference
            num_grad[idx] = (loss_plus - loss_minus) / (2*eps)
            # restore
            Phi[idx] = orig
            it.iternext()
        return num_grad

    def pred(self, X):
        Phi = self.basis_function.transform(X)
        if self.w is None:
            self.w = np.zeros(Phi.shape[1])
        return Phi @ self.w
```

Now, we have all the pieces together, to apply our new linear regression implementation to the temperature prediction problem.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

YEAR = 1900
def load_weather_data(year = None):
    """
    load data from a weather station in Potsdam
    """

    names = ['station', 'date' , 'type', 'measurement', 'e1','e2', 'E', 'e3']
    data = pd.read_csv('../../datasets/weatherstations/GM000003342.csv', names = names, low_memory=False) # 47876 rows, 8 columns
    # convert the date column to datetime format
    data['date'] = pd.to_datetime(data['date'], format="%Y%m%d") # 47876 unique days
    types = data['type'].unique()

    tmax = data[data['type']=='TMAX'][['date','measurement']] # Maximum temperature (tenths of degrees C), 47876
    tmin = data[data['type']=='TMIN'][['date','measurement']] # Minimum temperature (tenths of degrees C), 47876
    prcp = data[data['type']=='PRCP'][['date','measurement']] # Precipitation (tenths of mm), 47876
    snwd = data[data['type']=='SNWD'][['date','measurement']] # Snow depth (mm), different shape
    tavg = data[data['type']=='TAVG'][['date','measurement']] # average temperature, different shape 1386
    arr = np.array([tmax.measurement.values,tmin.measurement.values, prcp.measurement.values]).T 

    df = pd.DataFrame(arr/10.0, index=tmin.date, columns=['TMAX', 'TMIN', 'PRCP']) # compile data in a dataframe and convert temperatures to degrees C, precipitation to mm

    if year is not None:
        df = df[pd.to_datetime(f'{year}-1-1'):pd.to_datetime(f'{year}-12-31')]
    
    df['days'] = (df.index - df.index.min()).days
    return df

# Load weather data for the year 2000
df = load_weather_data(year = YEAR)

np.random.seed(2)
idx = np.random.permutation(df.shape[0])

idx_train = idx[0:100]
idx_test = idx[100:]

data_train = df.iloc[idx_train]
data_test = df.iloc[idx_test]

N_train = 100

a = np.array([.1, .2, .3])
b = np.array([-10.0,-50.0,-100.0])
W = np.array([a, b])

# tanh_basis = TanhBasis(W)
tanh_basis = TanhBasis(W)

ridge = 0.1     # strength of the L2 penalty in ridge regression

x_train = data_train.days.values[:N_train][:,np.newaxis] * 1.0
# X_train = tanh_basis(x_train, a, b)
y_train = data_train.TMAX.values[:N_train]

# print(tanh_basis.W)

reg = BasisFunctionRidgeRegressionGD(basis_function=tanh_basis,ridge=ridge, learning_rate=0.00001, num_iterations=1000000)
reg.fit(x_train, y_train)

# print(tanh_basis.W)

x_days = np.arange(366)[:,np.newaxis]
# X_days = tanh_basis(x_days, a, b)
y_days_pred = reg.pred(x_days)

x_test = data_test.days.values[:,np.newaxis] * 1.0
# X_test = tanh_basis(x_test, a, b)
y_test = data_test.TMAX.values
y_test_pred = reg.pred(x_test)
print("training MSE : %.4f" % reg.mse(x_train, y_train))
print("test MSE     : %.4f" % reg.mse(x_test, y_test))


fig = plt.figure()
ax = plt.plot(x_train,y_train,'.')
ax = plt.plot(x_test,y_test,'.')
ax = plt.legend(["train MSE = %.2f" % reg.mse(x_train, y_train),"test MSE = %.2f" % reg.mse(x_test, y_test)])
ax = plt.plot(x_days,y_days_pred)
ax = plt.ylim([-27,39])
ax = plt.xlabel("day of the year")
ax = plt.ylabel("Maximum Temperature - degree C")
ax = plt.title("Year : %i        N : %i" % (YEAR, N_train))
```

## Chain rule and the back-propagation algorithm

Note that the model that we have derived represents a fully-connected neural network with a single hidden layer and the tanh activation function, or if we also count the output layer, this would be a 2-layer neural network.
In the neural network world, the transformed data $\boldsymbol\Phi(\mathbf{X}; \mathbf{W}_\phi)$ would correspond to the nodes on the hidden layer of the neural network and the parameter matrix $\mathbf{W}_\phi$ would correspond to the weights of the hidden layer.

We have used the chain rule to compute the the gradient of $L$ with respect to $\mathbf{W}_\phi$.
It turns out that neural networks use the back-propagation algorithm to compute such grtadients. The back-propagation is fundamentally based on the chain rule. However, it represents a  more efficient implementation than the one that we have used here, as it would organize all copmutations into a forward pass that computes all the network layers up to the loss functions, while caching all the intermediate computations, and a backward pass, where it traces the network back towards the input and re-using all cached terms. Our implementation does not make use of caching and thus would be slower than the back-propagation algorithm.

Now, from a deep learning perspective, our 2-layer neural network would still be considered fairly shallow.
If you would like to learn more about the back-propagation algorithm and how to build and train massively deep neural network architectures, check out the excellent [Dive into Deep Learning](https://d2l.ai) online book.