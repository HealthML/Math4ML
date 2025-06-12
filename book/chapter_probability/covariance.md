# Covariance

Covariance is a measure of the linear relationship between two random
variables. We denote the covariance between $X$ and $Y$ as
$\operatorname{Cov}(X, Y)$, and it is defined to be

$$\operatorname{Cov}(X, Y) = \mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]$$

Note that the outer expectation must be taken over the joint
distribution of $X$ and $Y$.

Again, the linearity of expectation allows us to rewrite this as

$$\operatorname{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

Comparing these formulas to the ones for variance, it is not hard to see
that $\operatorname{Var}(X) = \operatorname{Cov}(X, X)$.

A useful property of covariance is that of **bilinearity**:

$$\begin{aligned}
\operatorname{Cov}(\alpha X + \beta Y, Z) &= \alpha\operatorname{Cov}(X, Z) + \beta\operatorname{Cov}(Y, Z) \\
\operatorname{Cov}(X, \alpha Y + \beta Z) &= \alpha\operatorname{Cov}(X, Y) + \beta\operatorname{Cov}(X, Z)
\end{aligned}$$

## Correlation

Normalizing the covariance gives the **correlation**:

$$\rho(X, Y) = \frac{\operatorname{Cov}(X, Y)}{\sqrt{\operatorname{Var}(X)\operatorname{Var}(Y)}}$$

Correlation also measures the linear relationship between two variables,
but unlike covariance always lies between $-1$ and $1$.

Two variables are said to be **uncorrelated** if
$\operatorname{Cov}(X, Y) = 0$ because $\operatorname{Cov}(X, Y) = 0$
implies that $\rho(X, Y) = 0$. If two variables are independent, then
they are uncorrelated, but the converse does not hold in general.

