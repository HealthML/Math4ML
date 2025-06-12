# Variance

Expectation provides a measure of the "center" of a distribution, but
frequently we are also interested in what the "spread" is about that
center. We define the variance $\operatorname{Var}(X)$ of a random
variable $X$ by

$$\operatorname{Var}(X) = \mathbb{E}\left[\left(X - \mathbb{E}[X]\right)^2\right]$$

In words, this is the average squared deviation of the values of $X$
from the mean of $X$. Using a little algebra and the linearity of
expectation, it is straightforward to show that

$$\operatorname{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$

## Properties of variance

Variance is not linear (because of the squaring in the definition), but
one can show the following:

$$\operatorname{Var}(\alpha X + \beta) = \alpha^2 \operatorname{Var}(X)$$

Basically, multiplicative constants become squared when they are pulled
out, and additive constants disappear (since the variance contributed by
a constant is zero).

Furthermore, if $X_1, \dots, X_n$ are uncorrelated[^14], then

$$\operatorname{Var}(X_1 + \dots + X_n) = \operatorname{Var}(X_1) + \dots + \operatorname{Var}(X_n)$$

## Standard deviation

Variance is a useful notion, but it suffers from that fact the units of
variance are not the same as the units of the random variable (again
because of the squaring). To overcome this problem we can use **standard
deviation**, which is defined as $\sqrt{\operatorname{Var}(X)}$. The
standard deviation of $X$ has the same units as $X$.

