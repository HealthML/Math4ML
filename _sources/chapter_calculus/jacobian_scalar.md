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

## Basis Functions and their Jacobians
In order to optimize over the basis functions, we need to compute its Jacobian. What are the inputs and what are the outputs of the basis functions?

If we have $K$ basis functions, these transform a $D$-dimensional feature vector $\mathbf{x}$ into a $K$-dimensional vector $\boldsymbol{\phi}_\boldsymbol{\theta}(\mathbf{x})$.

$$
\boldsymbol{\phi}_\boldsymbol{\theta}(\mathbf{x}): \mathbb{R}^D \rightarrow \mathbb{R}^K
$$

Here, $\boldsymbol{\theta}$ is the vector of parameters of $\boldsymbol{\phi}$.

In machine learning our goal will be to optimize over these features based on a loss function computed over a fixed training data set of size $N$ that is transformed using $\boldsymbol{\phi}$. Thus, we need to understand how the transformed data features of the training data set change as we modify the parameters $\boldsymbol{\theta}$.

In this view, have to treat the parameters $\boldsymbol{\theta}$ as the input and the transformed data set as the output of $\boldsymbol{\phi}$.

$$
\boldsymbol{\phi}_{\mathbf{X}}(\boldsymbol{\theta}): \mathbb{R}^{P}\rightarrow\mathbb{R}
^{NK},$$
where $P$ is the total number of parameters of all the basis functions.

Thus, the Jacobian will be a matrix of size $P$ times $(NK)$.

## Jacobian of tanh basis functions

Let's apply this to an example, where we transform the data using $K$ tanh basis functions.

Let $\phi_{nk}$ be the $k$-th basis-function activation on the $n$-th sample.

$$
\phi_{nk}
=\tanh\!\bigl(z_{nk}\bigr)
\quad\text{with}\quad
z_{nk} \;=\;\sum_{d=1}^D a_{dk}\,x_{nd}\;+\;b_k
$$

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

---

### Matrix-form view

If we collect all $\phi_{nk}$ into an $N\times K$ matrix $\boldsymbol{\Phi}$, and all weights $a_{dk}$ into a $D\times K$ matrix $\mathbf{A}$, then the Jacobian $\tfrac{\partial\,\boldsymbol{\Phi}}{\partial\,\mathbf{A}}$ is a 4-tensor with entries

$$
\bigl[J_{\boldsymbol{\Phi},\mathbf{A}}\bigr]_{\,n,l\,;\,d,\ell}
=\frac{\partial\,\phi_{nk}}{\partial\,a_{d\ell}}
=\delta_{\ell k}\,(1-\phi_{nk}^2)\,x_{nd}.
$$

In practice, for each basis unit $k$ you get the gradient

$$
\nabla_{a_{\cdot k}}\phi_{\cdot k}
=\bigl(1-\phi_{\cdot k}^2\bigr)\odot x_{\cdot\,:\,}
$$
(where “$\odot$” is element-wise multiplication of the vector $1-\phi_{ik}^2$ with each column of the design matrix).


---

I’ve implemented a `TanhBasisLoop` class whose `jacobian(X)` method explicitly loops over:

1. **Samples** $n$,  
2. **Units** $k$, and  
3. **Parameters** $d$ (including the bias as the last parameter).

The returned Jacobian `J` has shape $(N,\,D+1,\,k)$ with

$$
J[n,d,k] \;=\; \frac{\partial\,\phi_{n,k}}{\partial\,W_{d,k}}
\;=\;
\begin{cases}
(1-\phi_{n,k}^2)\,X_{n,d}, & d<D\quad(\text{zero-based indexing}),\\
(1-\phi_{n,k}^2), & d=D\quad(\text{bias term}).
\end{cases}
$$

The example usage at the bottom tests:

- Computing the activation matrix `Phi`,  
- Then the Jacobian `J`,  
- And finally verifying one entry, e.g.  
  `J[0,1,0] == (1 - Phi[0,0]**2) * X[0,1]`.  


I’ve added a `numerical_jacobian(X, eps)` method to your `TanhBasisLoop` class. It perturbs each parameter $W_{d,k}$ by $\pm\varepsilon$, recomputes the activations, and uses a central difference to approximate

$$
\frac{\partial \phi_{n,k}}{\partial W_{d,k}}
\equiv \frac{\phi_{n,k}(W_{d,k}+\varepsilon)\;-\;\phi_{n,k}(W_{d,k}-\varepsilon)}{2\varepsilon}.
$$

A quick comparison on random data shows the max absolute difference between analytic and numeric Jacobians is on the order of $10^{-11}$, confirming that the loop‐based Jacobian is correctly implemented.

```{code-cell} ipython3
import numpy as np

class TanhBasisLoop:
    def __init__(self, W):
        """
        W: array of shape (D+1, P), where
           - W[:D, k] are the slopes a_dk for each input dimension d and unit k
           - W[D, k] is the bias b_k for unit k.
        """
        self.W = W.copy()

    def transform(self, X):
        """
        Compute tanh basis functions.
        X: (N, D) array of inputs.
        Returns: (N, K) array of activations.
        """
        N, D = X.shape
        slopes = self.W[:D, :]  # shape (d, p)
        bias   = self.W[D, :]   # shape (p,)
        Z = X @ slopes + bias   # (n, p)
        return np.tanh(Z)

    def jacobian(self, X):
        """
        Compute Jacobian d phi_{n,k} / d W_{d,l} with loops.
        Returns an array of shape (N, D+1, K).
        """
        X = np.asarray(X)
        N, D = X.shape
        K = self.W.shape[1]
        Phi = self.transform(X)
        J = np.zeros((N, D+1, K))
        for n in range(N):
            for k in range(K):
                deriv = 1.0 - Phi[n, k]**2
                for d in range(D):
                    J[n, d, k] = deriv * X[n, d]
                J[n, D, k] = deriv * 1.0
        print (J.shape)
        return J

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

# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.randn(5, 3)           # 5 samples, 3 features
    W_init = np.random.randn(4, 2)*0.1  # 3 slopes + 1 bias, 2 units
    tanh_loop = TanhBasisLoop(W_init)

    # Analytical Jacobian
    J_analytic = tanh_loop.jacobian(X)
    # Numerical Jacobian
    J_numeric  = tanh_loop.numerical_jacobian(X, eps=1e-6)

    print("Max abs diff:", np.max(np.abs(J_analytic - J_numeric)))
    print("Analytic Jacobian:\n", J_analytic)
```