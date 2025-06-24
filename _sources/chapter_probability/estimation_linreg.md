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
# Estimation for Linear Regression


## Maximum Likelihood Estimation for Linear Regression

The likelihood is:

$$
p(\mathbf{y} \mid \mathbf{w}) = \prod_{i=1}^n \mathcal{N}(y_i \mid \mathbf{x}_i^\top \mathbf{w}, \sigma^2)
$$

The **log-likelihood** is:

$$
\log p(\mathbf{y} \mid \mathbf{w}) = -\frac{1}{2\sigma^2} \|\mathbf{y} - \mathbf{Xw}\|^2 + \text{const}
$$

So the **MLE** is:

$$
\hat{\mathbf{w}}_{\text{MLE}} = \operatorname{argmin}_\mathbf{w} \|\mathbf{y} - \mathbf{Xw}\|^2
$$

### Example: MLE and MAP estimators for polynomial regression

Let's compare the MLE and MAP estimators for the polynomial regression model.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge

def sine_polynomial_regression_demo(n=15, degree=14, noise_std=0.1, alpha=1.0, seed=42):
    np.random.seed(seed)

    # === True function ===
    def true_func(x): return np.sin(2 * np.pi * x)

    # === Generate data ===
    X = np.sort(np.random.rand(n, 1), axis=0)
    y = true_func(X).ravel() + np.random.randn(n) * noise_std

    # === Generate test data ===
    X_test = np.linspace(0, 1, 500).reshape(-1, 1)
    y_true = true_func(X_test)

    # === Polynomial features ===
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    X_test_poly = poly.transform(X_test)

    # === MLE (Ordinary Least Squares) ===
    ols = LinearRegression()
    ols.fit(X_poly, y)
    y_pred_mle = ols.predict(X_test_poly)

    # === MAP (Ridge Regression) ===
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_poly, y)
    y_pred_map = ridge.predict(X_test_poly)

    # === Plot ===
    plt.figure(figsize=(10, 5))
    plt.plot(X_test, y_true, 'k--', label='True Function (sin)')
    plt.scatter(X, y, color='black', s=30, label='Noisy Samples')
    plt.plot(X_test, y_pred_mle, 'r-', label='MLE (degree {})'.format(degree))
    plt.plot(X_test, y_pred_map, 'b-', label='MAP (ridge, α={})'.format(alpha))
    plt.title(f'Polynomial Regression: Degree {degree}, n = {n}')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Try degree approaching number of points
sine_polynomial_regression_demo(n=15, degree=14, alpha=1.0)


```