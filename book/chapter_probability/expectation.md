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
# Expected Value

If we have some random variable $X$, we might be interested in knowingwhat is the "average" value of $X$. 
This concept is captured by the
**expected value** (or **mean**) $\mathbb{E}[X]$, which is defined as

$$\mathbb{E}[X] = \sum_{x \in X(\Omega)} xp(x)$$ 

for discrete $X$ and as

$$\mathbb{E}[X] = \int_{-\infty}^\infty xp(x)\operatorname{d}{x}$$ 

for continuous
$X$.

In words, we are taking a weighted sum of the values that $X$ can take
on, where the weights are the probabilities of those respective values.
The expected value has a physical interpretation as the "center of mass"
of the distribution.

---

In our running example, the random variable $X$ (number of heads in two fair coin tosses) has the following distribution:

| $x$                 | 0    | 1   | 2    |
| ------------------- | ---- | --- | ---- |
| $\mathbb{P}(X = x)$ | 0.25 | 0.5 | 0.25 |

We compute the expected value as:

$$
\mathbb{E}[X] = \sum_{x=0}^2 x \cdot \mathbb{P}(X = x) = 0 \cdot 0.25 + 1 \cdot 0.5 + 2 \cdot 0.25 = 1.0
$$


* This means that **on average**, you expect to see **1 head** in two tosses of a fair coin.
* The expected value $\mathbb{E}[X]$ corresponds to the **center of mass** of the PMF, and aligns with our intuition about symmetry in a fair coin toss experiment.


## Properties of the expected value

### Linearity of expectation

A very useful property of the expected value is that of **linearity of expectation**:

$$\mathbb{E}\left[\sum_{i=1}^n \alpha_i X_i + \beta\right] = \sum_{i=1}^n \alpha_i \mathbb{E}[X_i] + \beta$$

Note that this holds even if the $X_i$ are not independent!

---

Let us see an example involving **two coin tosses**.

Suppose we toss a fair coin twice. Define:

* $X_1 = \mathbb{1}\{\text{first toss is heads}\}$
* $X_2 = \mathbb{1}\{\text{second toss is heads}\}$

Then:

* $\mathbb{E}[X_1] = \mathbb{E}[X_2] = 0.5$
* Let $S = X_1 + X_2$ be the **total number of heads**

By linearity:

$$
\mathbb{E}[S] = \mathbb{E}[X_1 + X_2] = \mathbb{E}[X_1] + \mathbb{E}[X_2] = 0.5 + 0.5 = 1.0
$$


---

### Product rule for expectation

But if the $X_i$ are independent, the **product rule for expectation** also holds:

$$\mathbb{E}\left[\prod_{i=1}^n X_i\right] = \prod_{i=1}^n \mathbb{E}[X_i]$$

Let's extend the coin toss example to illustrate the **product rule for expectation**, which holds **only if the random variables are independent**:


* Let
  $X_1 = \mathbb{1}\{\text{first toss is heads}\}$,
  $X_2 = \mathbb{1}\{\text{second toss is heads}\}$
* These are independent indicators, each with $\mathbb{E}[X_1] = \mathbb{E}[X_2] = 0.5$

Then:

$$
\mathbb{E}[X_1 \cdot X_2] = \mathbb{P}(\text{first toss is H and second toss is H}) = \mathbb{P}(\text{hh}) = 0.25
$$

$$
\mathbb{E}[X_1] \cdot \mathbb{E}[X_2] = 0.5 \cdot 0.5 = 0.25
$$

So the product rule holds here.


```{code-cell} ipython3
:tags: [hide-input]
import numpy as np

# Using same definitions from earlier
outcomes = ['hh', 'ht', 'th', 'tt']
X1 = np.array([1 if o[0] == 'h' else 0 for o in outcomes])
X2 = np.array([1 if o[1] == 'h' else 0 for o in outcomes])
X1_X2 = X1 * X2
probs = np.full_like(X1, 0.25, dtype=float)  # uniform distribution

E_X1 = np.sum(X1 * probs)
E_X2 = np.sum(X2 * probs)
E_product = np.sum(X1_X2 * probs)
product_of_expectations = E_X1 * E_X2

print(f"E[X₁]           = {E_X1:.2f}")
print(f"E[X₂]           = {E_X2:.2f}")
print(f"E[X₁·X₂]        = {E_product:.2f}")
print(f"E[X₁]·E[X₂]     = {product_of_expectations:.2f}")
```

* Since $X_1$ and $X_2$ are **independent**, we observe:

  $$
  \mathbb{E}[X_1 \cdot X_2] = \mathbb{E}[X_1] \cdot \mathbb{E}[X_2]
  $$
* This would **not** hold if the tosses were somehow dependent (e.g., if the second toss always matched the first).
