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
# Bayesian Linear Regression

Let’s now place a **prior** and compute the **full posterior distribution** over the weights $\mathbf{w}$.

---

## 1. **Model Assumptions**

* Data: $\mathbf{y} \in \mathbb{R}^n$, $\mathbf{X} \in \mathbb{R}^{n \times d}$
* Likelihood:

  $$
  \mathbf{y} \mid \mathbf{w} \sim \mathcal{N}(\mathbf{Xw}, \sigma^2 \mathbf{I})
  $$
* Prior:

  $$
  \mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2 \mathbf{I})
  $$

---

## 2. **Posterior Over Weights**

Because the prior and likelihood are both Gaussian, the posterior is also Gaussian:

$$
\mathbf{w} \mid \mathbf{y}, \mathbf{X} \sim \mathcal{N}(\mathbf{w}_{\text{post}}, \mathbf{\Sigma}_{\text{post}})
$$

Where:

$$
\mathbf{\Sigma}_{\text{post}} = \left( \frac{1}{\sigma^2} \mathbf{X}^\top \mathbf{X} + \frac{1}{\tau^2} \mathbf{I} \right)^{-1}, \quad
\mathbf{w}_{\text{post}} = \frac{1}{\sigma^2} \mathbf{\Sigma}_{\text{post}} \mathbf{X}^\top \mathbf{y}
$$

---

## 3. **Posterior Predictive Distribution**

For a new input $\mathbf{x}_* \in \mathbb{R}^d$, the **predictive distribution** is:

$$
p(y_* \mid \mathbf{x}_*, \mathbf{X}, \mathbf{y}) = \mathcal{N}\left( \mathbf{x}_*^\top \mathbf{w}_{\text{post}}, \ \mathbf{x}_*^\top \mathbf{\Sigma}_{\text{post}} \mathbf{x}_* + \sigma^2 \right)
$$

---

Would you like me to now prepare a **Python demo** that shows:

* Posterior distribution over weights?
* Posterior predictive uncertainty?
* Comparisons to MAP or MLE?

Let me know your preferred dimension setting (e.g., 1D or 2D regression for plotting)!
