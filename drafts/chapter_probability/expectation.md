## Great Expectations

If we have some random variable $X$, we might be interested in knowing
what is the "average" value of $X$. This concept is captured by the
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

### Properties of expected value

A very useful property of expectation is that of linearity:

$$\mathbb{E}\left[\sum_{i=1}^n \alpha_i X_i + \beta\right] = \sum_{i=1}^n \alpha_i \mathbb{E}[X_i] + \beta$$

Note that this holds even if the $X_i$ are not independent!

But if they are independent, the product rule also holds:

$$\mathbb{E}\left[\prod_{i=1}^n X_i\right] = \prod_{i=1}^n \mathbb{E}[X_i]$$

## Variance

Expectation provides a measure of the "center" of a distribution, but
frequently we are also interested in what the "spread" is about that
center. We define the variance $\operatorname{Var}(X)$ of a random
variable $X$ by

$$\operatorname{Var}(X) = \mathbb{E}\left[\left(X - \mathbb{E}[X]\right)^2\right]$$

In words, this is the average squared deviation of the values of $X$
from the mean of $X$. Using a little algebra and the linearity of
expectation, it is straightforward to show that

$$\operatorname{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$

### Properties of variance

Variance is not linear (because of the squaring in the definition), but
one can show the following:

$$\operatorname{Var}(\alpha X + \beta) = \alpha^2 \operatorname{Var}(X)$$

Basically, multiplicative constants become squared when they are pulled
out, and additive constants disappear (since the variance contributed by
a constant is zero).

Furthermore, if $X_1, \dots, X_n$ are uncorrelated[^14], then

$$\operatorname{Var}(X_1 + \dots + X_n) = \operatorname{Var}(X_1) + \dots + \operatorname{Var}(X_n)$$

### Standard deviation

Variance is a useful notion, but it suffers from that fact the units of
variance are not the same as the units of the random variable (again
because of the squaring). To overcome this problem we can use **standard
deviation**, which is defined as $\sqrt{\operatorname{Var}(X)}$. The
standard deviation of $X$ has the same units as $X$.

