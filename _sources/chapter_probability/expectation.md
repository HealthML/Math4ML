# Expected Value

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

## Properties of the expected value

A very useful property of expectation is that of linearity:

$$\mathbb{E}\left[\sum_{i=1}^n \alpha_i X_i + \beta\right] = \sum_{i=1}^n \alpha_i \mathbb{E}[X_i] + \beta$$

Note that this holds even if the $X_i$ are not independent!

But if they are independent, the product rule also holds:

$$\mathbb{E}\left[\prod_{i=1}^n X_i\right] = \prod_{i=1}^n \mathbb{E}[X_i]$$
