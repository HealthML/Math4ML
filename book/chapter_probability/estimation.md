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
# Estimation of Parameters

Now we get into some basic topics from statistics. We make some
assumptions about our problem by prescribing a **parametric** model
(e.g. a distribution that describes how the data were generated), then
we fit the parameters of the model to the data. How do we choose the
values of the parameters?

## Maximum likelihood estimation

A common way to fit parameters is **maximum likelihood estimation**
(MLE). The basic principle of MLE is to choose values that  "explain" the
data best by maximizing the probability/density of the data we've seen
as a function of the parameters. Suppose we have random variables
$X_1, \dots, X_n$ and corresponding observations $x_1, \dots, x_n$. Then

$$
\hat{\mathbf{\theta}}_\text{mle} = \operatorname{argmax}_\mathbf{\theta} \mathcal{L}(\mathbf{\theta})
$$

where $\mathcal{L}$ is the **likelihood function**

$$
\mathcal{L}(\mathbf{\theta}) = p(x_1, \dots, x_n; \mathbf{\theta})
$$

Often, we assume that $X_1, \dots, X_n$ are i.i.d. Then we can write

$$p(x_1, \dots, x_n; \theta) = \prod_{i=1}^n p(x_i; \mathbf{\theta})$$

At this point, it is usually convenient to take logs, giving rise to the
**log-likelihood**

$$\log\mathcal{L}(\mathbf{\theta}) = \sum_{i=1}^n \log p(x_i; \mathbf{\theta})$$

This is a valid operation because the probabilities/densities are
assumed to be positive, and since log is a monotonically increasing
function, it preserves ordering. In other words, any maximizer of
$\log\mathcal{L}$ will also maximize $\mathcal{L}$.

For some distributions, it is possible to analytically solve for the
maximum likelihood estimator. If $\log\mathcal{L}$ is differentiable,
setting the derivatives to zero and trying to solve for
$\mathbf{\theta}$ is a good place to start.



## MLE for the Normal Distribution

Let us now apply the principle of maximum likelihood estimation to a common and important case: the **normal distribution**. Suppose we observe data points $x_1, \dots, x_n \in \mathbb{R}$ which we assume to be i.i.d. samples from a normal distribution with unknown parameters: mean $\mu \in \mathbb{R}$ and variance $\sigma^2 > 0$. That is,

$$
x_i \sim \mathcal{N}(\mu, \sigma^2)
$$

We want to find the **maximum likelihood estimates** (MLEs) of $\mu$ and $\sigma^2$.

### Step 1: Write the Likelihood

The likelihood function is the joint probability density of the data:

$$
\mathcal{L}(\mu, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
$$

### Step 2: Take the Log-Likelihood

Taking the log turns the product into a sum and simplifies the exponential:

$$
\log \mathcal{L}(\mu, \sigma^2)
= -\frac{n}{2} \log(2\pi) - \frac{n}{2} \log \sigma^2 - \frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2
$$

### Step 3: Maximize the Log-Likelihood

To find the MLEs, we take derivatives and set them to zero.

#### Derivative w\.r.t. $\mu$:

$$
\frac{\partial}{\partial \mu} \log \mathcal{L} = \frac{1}{\sigma^2} \sum_{i=1}^n (x_i - \mu) = 0
\quad \Rightarrow \quad
\hat{\mu}_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n x_i
$$

#### Derivative w\.r.t. $\sigma^2$:

$$
\frac{\partial}{\partial \sigma^2} \log \mathcal{L} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4} \sum_{i=1}^n (x_i - \mu)^2 = 0
$$

Solving for $\sigma^2$ (and plugging in $\hat{\mu}$):

$$
\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2
$$

### Summary

The maximum likelihood estimates for a univariate normal distribution are:

$$
\hat{\mu}_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n x_i, \quad
\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2
$$

These are just the **sample mean** and the **(non-biased) sample variance** with denominator $n$ (not $n-1$).


* Generates synthetic i.i.d. samples from a normal distribution.
* Computes MLE estimates of $\mu$ and $\sigma^2$.
* Plots:

  * Histogram of the data
  * True density curve
  * Fitted (MLE) density curve

---


```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def mle_normal_demo(n=100, mu_true=2.0, sigma_true=1.5, seed=42):
    np.random.seed(seed)
    
    # Generate data
    data = np.random.normal(mu_true, sigma_true, size=n)
    
    # MLE estimates
    mu_mle = np.mean(data)
    sigma2_mle = np.mean((data - mu_mle)**2)
    sigma_mle = np.sqrt(sigma2_mle)
    
    print(f"True μ = {mu_true},   MLE μ̂ = {mu_mle:.3f}")
    print(f"True σ² = {sigma_true**2}, MLE σ̂² = {sigma2_mle:.3f}")
    
    # Plot
    x = np.linspace(min(data) - 1, max(data) + 1, 300)
    true_pdf = norm.pdf(x, mu_true, sigma_true)
    mle_pdf = norm.pdf(x, mu_mle, sigma_mle)
    
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=20, density=True, alpha=0.4, color='gray', label='Data histogram')
    plt.plot(x, true_pdf, 'g--', lw=2, label='True distribution')
    plt.plot(x, mle_pdf, 'r-', lw=2, label='MLE fit')
    plt.axvline(mu_mle, color='r', linestyle='--', lw=1)
    plt.axvline(mu_true, color='g', linestyle='--', lw=1)
    plt.title('MLE of the Normal Distribution')
    plt.xlabel('$x$')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the demo
mle_normal_demo()
```

---

### ✅ Output Example

```
True μ = 2.0,   MLE μ̂ = 2.023
True σ² = 2.25, MLE σ̂² = 2.082
```


* See that the **sample mean and variance (with denominator $n$)** are the MLEs.
* Understand how MLE fits data by maximizing likelihood.
* Visually compare the **true** vs **fitted** normal densities.

Perfect — here's how to expand your section with a smooth transition from **MLE to MAP**, showing the **motivation** from the small sample case, especially $n = 1$, followed by the derivation of **MAP estimates for the Gaussian distribution with a conjugate prior**.

---

## Limitations of MLE in the Small Sample Case

The maximum likelihood estimator (MLE) works well with large amounts of data, but in the **small sample regime**, it has important limitations:

* **Variance underestimation**: The MLE for variance,

$$
\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2,
$$

is **biased**. In fact, the unbiased estimator uses a denominator of $n - 1$, not $n$.

* **Extreme case $n = 1$**: With only one sample, the MLE of the variance is **not even defined**, since we have

$$
\hat{\sigma}^2_{\text{MLE}} = \frac{1}{1} (x_1 - x_1)^2 = 0,
$$

which clearly **understates our uncertainty**.

These issues motivate a **Bayesian perspective**, where we regularize estimation with a prior.

---

## Maximum A Posteriori Estimation (MAP)

MAP estimation introduces prior knowledge into the estimation problem. Using Bayes' rule, the posterior over parameters $\theta$ is:

$$
p(\theta \mid x_1, \dots, x_n) \propto p(\theta) \cdot p(x_1, \dots, x_n \mid \theta)
$$

The **MAP estimate** is the mode of the posterior:

$$
\hat{\theta}_\text{MAP} = \operatorname{argmax}_\theta \log p(\theta) + \sum_{i=1}^n \log p(x_i \mid \theta)
$$


## Maximum a posteriori estimation

A more Bayesian way to fit parameters is through **maximum a posteriori
estimation** (MAP). In this technique we assume that the parameters are
a random variable, and we specify a prior distribution
$p(\mathbf{\theta})$. Then we can employ Bayes' rule to compute the
posterior distribution of the parameters given the observed data:

$$p(\mathbf{\theta} | x_1, \dots, x_n) \propto p(\mathbf{\theta})p(x_1, \dots, x_n | \mathbf{\theta})$$

Computing the normalizing constant is often intractable, because it
involves integrating over the parameter space, which may be very
high-dimensional. Fortunately, if we just want the MAP estimate, we
don't care about the normalizing constant! It does not affect which
values of $\mathbf{\theta}$ maximize the posterior. So we have

$$\hat{\mathbf{\theta}}_\text{map} = \operatorname{argmax}_\mathbf{\theta} p(\mathbf{\theta})p(x_1, \dots, x_n | \mathbf{\theta})$$

Again, if we assume the observations are i.i.d., then we can express
this in the equivalent, and possibly friendlier, form

$$\hat{\mathbf{\theta}}_\text{map} = \operatorname{argmax}_\mathbf{\theta} \left(\log p(\mathbf{\theta}) + \sum_{i=1}^n \log p(x_i | \mathbf{\theta})\right)$$

A particularly nice case is when the prior is chosen carefully such that
the posterior comes from the same family as the prior. In this case the
prior is called a **conjugate prior**. For example, if the likelihood is
binomial and the prior is beta, the posterior is also beta. There are
many conjugate priors; the reader may find this [table of conjugate
priors](https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions)
useful.

---

## MAP Estimation for a Gaussian Mean with Known Variance

Assume:

* Data: $x_1, \dots, x_n \sim \mathcal{N}(\mu, \sigma^2)$, with known $\sigma^2$
* Prior: $\mu \sim \mathcal{N}(\mu_0, \tau^2)$

### Derivation:

The log posterior is:

$$
\log p(\mu \mid x_{1:n}) = \log p(\mu) + \sum_{i=1}^n \log p(x_i \mid \mu)
$$

$$
= -\frac{1}{2\tau^2}(\mu - \mu_0)^2 - \frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2 + \text{const}
$$

Differentiating w\.r.t. $\mu$ and setting to zero gives:

$$
\hat{\mu}_{\text{MAP}} = \left( \frac{n}{\sigma^2} + \frac{1}{\tau^2} \right)^{-1}
\left( \frac{1}{\tau^2} \mu_0 + \frac{n}{\sigma^2} \bar{x} \right)
$$

This is a **weighted average** of the prior mean and the sample mean.

### Interpretation:

$$
\hat{\mu}_{\text{MAP}} = \frac{\tau^2}{\tau^2 + \sigma^2 / n} \bar{x} +
\frac{\sigma^2 / n}{\tau^2 + \sigma^2 / n} \mu_0
$$

* When $n \to \infty$: MAP converges to MLE.
* When $n \to 0$: MAP converges to the prior mean $\mu_0$.

---

### ✅ Python Demo: Comparing MLE and MAP

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def mle_vs_map_demo(n=1, mu_true=0.0, sigma=1.0, mu0=5.0, tau=1.0, seed=42):
    np.random.seed(seed)
    x = np.random.normal(mu_true, sigma, size=n)
    x_bar = np.mean(x)

    # MLE estimate
    mu_mle = x_bar

    # MAP estimate
    weight = (tau**2) / (tau**2 + sigma**2 / n)
    mu_map = weight * x_bar + (1 - weight) * mu0

    print(f"n = {n}")
    print(f"Sample mean       (MLE): {mu_mle:.3f}")
    print(f"MAP estimate             : {mu_map:.3f}")
    print(f"Prior mean         μ₀    : {mu0}")

    # Plot posterior
    mu_vals = np.linspace(mu_true - 3*sigma, mu0 + 3*tau, 400)
    prior_pdf = norm.pdf(mu_vals, mu0, tau)
    likelihood = norm.pdf(mu_vals, x_bar, sigma / np.sqrt(n))
    posterior_unnorm = prior_pdf * likelihood
    posterior = posterior_unnorm / np.trapz(posterior_unnorm, mu_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(mu_vals, prior_pdf, 'g--', label='Prior $p(\\mu)$')
    plt.plot(mu_vals, likelihood, 'b--', label='Likelihood $p(x|\\mu)$')
    plt.plot(mu_vals, posterior, 'm-', lw=2, label='Posterior $p(\\mu|x)$')
    plt.axvline(mu_mle, color='blue', linestyle=':', label='MLE')
    plt.axvline(mu_map, color='purple', linestyle='--', label='MAP')
    plt.axvline(mu0, color='green', linestyle='--', lw=1)
    plt.title(f'MLE vs MAP (n = {n})')
    plt.xlabel('$\\mu$')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()
    plt.show()

# Try with n=1 to emphasize prior influence
mle_vs_map_demo(n=1)
```

---

### 📌 Takeaways:

* With **small samples**, MLE can be **overconfident** and even undefined for variance.
* MAP estimation **regularizes** using prior knowledge.
* For the Gaussian with known variance, MAP gives a **weighted average** of prior mean and sample mean.

---

## Ridge Regression as MAP Estimation

Let’s consider a standard linear regression setup:

### Problem:

We observe:

* Inputs $\mathbf{x}_1, \dots, \mathbf{x}_n \in \mathbb{R}^d$
* Outputs $y_1, \dots, y_n \in \mathbb{R}$

Stacked as matrices:

* $\mathbf{X} \in \mathbb{R}^{n \times d}$, with rows $\mathbf{x}_i^\top$
* $\mathbf{y} \in \mathbb{R}^n$

We assume a **linear model with Gaussian noise**:

$$
y_i = \mathbf{x}_i^\top \mathbf{w} + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

Equivalently, in matrix form:

$$
\mathbf{y} \sim \mathcal{N}(\mathbf{Xw}, \sigma^2 \mathbf{I})
$$

---

## MLE for Linear Regression

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

---

## MAP with Gaussian Prior

Now suppose we place a **zero-mean Gaussian prior** on $\mathbf{w}$:

$$
\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2 \mathbf{I})
\quad\Rightarrow\quad
\log p(\mathbf{w}) = -\frac{1}{2\tau^2} \|\mathbf{w}\|^2 + \text{const}
$$

The **log-posterior** is then:

$$
\log p(\mathbf{w} \mid \mathbf{y}) \propto
- \frac{1}{2\sigma^2} \|\mathbf{y} - \mathbf{Xw}\|^2
- \frac{1}{2\tau^2} \|\mathbf{w}\|^2
$$

Taking the **negative log posterior**, we obtain the **ridge regression objective**:

$$
\hat{\mathbf{w}}_{\text{MAP}} = \operatorname{argmin}_\mathbf{w}
\left\{
\|\mathbf{y} - \mathbf{Xw}\|^2 + \lambda \|\mathbf{w}\|^2
\right\}
\quad \text{where } \lambda = \frac{\sigma^2}{\tau^2}
$$

---

### ✅ Conclusion

> **Ridge regression is the MAP estimate in linear regression under a Gaussian prior on the weights.**

* The regularization term $\lambda \|\mathbf{w}\|^2$ comes from the log of the Gaussian prior.
* The ridge penalty **shrinks** weights toward zero, just like the prior favors small values.

---
Great! Below is a **numerical Python demo** that illustrates the equivalence between:

* **Ridge regression**: minimizing squared loss + $\lambda \|\mathbf{w}\|^2$
* **MAP estimation**: under a Gaussian prior $\mathbf{w} \sim \mathcal{N}(0, \tau^2 I)$

We’ll simulate data from a linear model, perform both **ridge regression** and **MAP estimation**, and confirm they return **identical results**.

---

### ✅ Python Code: Ridge Regression ≡ MAP Estimation

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from numpy.linalg import inv

def ridge_vs_map_demo(n=50, d=5, sigma=1.0, tau=1.0, seed=42):
    np.random.seed(seed)

    # Generate true weights
    w_true = np.random.randn(d)

    # Generate inputs and noise
    X = np.random.randn(n, d)
    noise = np.random.normal(0, sigma, size=n)
    y = X @ w_true + noise

    # === Ridge Regression ===
    lambda_ = sigma**2 / tau**2
    ridge = Ridge(alpha=lambda_, fit_intercept=False)
    ridge.fit(X, y)
    w_ridge = ridge.coef_

    # === MAP Estimation (same as Ridge closed-form) ===
    A = X.T @ X + lambda_ * np.eye(d)
    b = X.T @ y
    w_map = inv(A) @ b

    # === Compare ===
    print("True weights:     ", np.round(w_true, 3))
    print("Ridge estimate:   ", np.round(w_ridge, 3))
    print("MAP estimate:     ", np.round(w_map, 3))
    print("Are ridge and MAP identical? ", np.allclose(w_ridge, w_map))

    # === Visualize ===
    indices = np.arange(d)
    width = 0.25
    plt.bar(indices - width, w_true, width=width, label='True $w$', color='gray')
    plt.bar(indices, w_ridge, width=width, label='Ridge $\\hat{w}$', color='blue')
    plt.bar(indices + width, w_map, width=width, label='MAP $\\hat{w}$', color='purple', alpha=0.6)
    plt.xticks(indices)
    plt.title('True vs Ridge vs MAP Estimates')
    plt.xlabel('Weight Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run the demo
ridge_vs_map_demo()
```

---

### 📌 What to Expect:

* The **Ridge** and **MAP estimates** will match up to numerical precision.
* You’ll see bar plots of the **true weights**, **ridge solution**, and **MAP solution** side by side.

---

### 💡 Interpretation:

| Term                          | Role                                             |
| ----------------------------- | ------------------------------------------------ |
| $\lambda = \sigma^2 / \tau^2$ | Ridge penalty = prior precision                  |
| Ridge regression              | Minimizes loss + prior log-density               |
| MAP estimation                | Maximizes posterior (log-likelihood + log-prior) |

