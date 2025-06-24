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
