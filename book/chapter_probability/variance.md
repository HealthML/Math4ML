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
# Variance

Expectation provides a measure of the "center" of a distribution, but frequently we are also interested in what the "spread" is about that center. 
We define the variance $\operatorname{Var}(X)$ of a random variable $X$ by

$$\operatorname{Var}(X) = \mathbb{E}\left[\left(X - \mathbb{E}[X]\right)^2\right]$$

In words, this is the average squared deviation of the values of $X$ from the mean of $X$. 
Using a little algebra and the linearity of expectation, it is straightforward to show that

$$\operatorname{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$

---

So the **variance** of a random variable $X$ is the expected value of the squared deviation of $X$ from its mean.

So, the variance of the number of heads in a single toss of a fair coin is

$$
\mathrm{Var}(X) = \mathbb{E}[X^2] - \left( \mathbb{E}[X] \right)^2 = \sum_x x^2 \cdot \mathbb{P}(X = x) - 0.5^2 = 0\cdot 0.5 + 1^2\cdot 0.5 - 0.5^2 = 0.25
$$


```{code-cell} ipython3
:tags: [remove-cell]
import numpy as np

# x and PMF as before
x_vals = np.array([0, 1, 2])
pmf_vals = np.array([0.25, 0.5, 0.25])

# E[X]
E_X = np.sum(x_vals * pmf_vals)

# E[X^2]
E_X2 = np.sum((x_vals**2) * pmf_vals)

# Var(X) = E[X^2] - (E[X])^2
var_X = E_X2 - E_X**2
std_X = np.sqrt(var_X)

print(f"E[X]     = {E_X:.2f}")
print(f"E[X^2]   = {E_X2:.2f}")
print(f"Var(X)   = {var_X:.2f}")
print(f"Std(X)   = {std_X:.2f}")
```


## Properties of variance

Variance is not linear (because of the squaring in the definition), but one can show the following:

$$\operatorname{Var}(\alpha X + \beta) = \alpha^2 \operatorname{Var}(X)$$


Basically, multiplicative constants become squared when they are pulled out, and additive constants disappear (since the variance contributed by a constant is zero).


---

Furthermore, if $X_1, \dots, X_n$ are uncorrelated, then

$$\operatorname{Var}(X_1 + \dots + X_n) = \operatorname{Var}(X_1) + \dots + \operatorname{Var}(X_n)$$

---
It follows that in our example of two tosses of a fair coin, the variance of the total number of heads can be computed from the variance of the individual tosses:

$$\operatorname{Var}(X_1 + X_2) = \operatorname{Var}(X_1) + \operatorname{Var}(X_2) = 0.25 + 0.25 = 1.0$$

---
The variance of the total number of heads in $n$ tosses of a fair coin is $n$ times the variance of the number of heads in a single toss, i.e. $n \cdot 0.25$.

## Standard deviation

Variance is a useful notion, but it suffers from that fact the units of variance are not the same as the units of the random variable (again because of the squaring). 

To overcome this problem we can use **standard deviation**, which is defined as $\sqrt{\operatorname{Var}(X)}$. 
The standard deviation of $X$ has the same units as $X$.

---
The standard deviation of the total number of heads in $n$ tosses of a fair coin is $\sqrt{n \cdot 0.25}$.
