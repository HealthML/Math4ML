## Covariance

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

### Correlation

Normalizing the covariance gives the **correlation**:

$$\rho(X, Y) = \frac{\operatorname{Cov}(X, Y)}{\sqrt{\operatorname{Var}(X)\operatorname{Var}(Y)}}$$

Correlation also measures the linear relationship between two variables,
but unlike covariance always lies between $-1$ and $1$.

Two variables are said to be **uncorrelated** if
$\operatorname{Cov}(X, Y) = 0$ because $\operatorname{Cov}(X, Y) = 0$
implies that $\rho(X, Y) = 0$. If two variables are independent, then
they are uncorrelated, but the converse does not hold in general.

## Random vectors

So far we have been talking about **univariate distributions**, that is,
distributions of single variables. But we can also talk about
**multivariate distributions** which give distributions of **random
vectors**:

$$\mathbf{X} = \begin{bmatrix}X_1 \\ \vdots \\ X_n\end{bmatrix}$$ 

The
summarizing quantities we have discussed for single variables have
natural generalizations to the multivariate case.

Expectation of a random vector is simply the expectation applied to each
component:

$$\mathbb{E}[\mathbf{X}] = \begin{bmatrix}\mathbb{E}[X_1] \\ \vdots \\ \mathbb{E}[X_n]\end{bmatrix}$$

The variance is generalized by the **covariance matrix**:

$$\mathbf{\Sigma} = \mathbb{E}[(\mathbf{X} - \mathbb{E}[\mathbf{X}])(\mathbf{X} - \mathbb{E}[\mathbf{X}])^{\!\top\!}] = \begin{bmatrix}
\operatorname{Var}(X_1) & \operatorname{Cov}(X_1, X_2) & \dots & \operatorname{Cov}(X_1, X_n) \\
\operatorname{Cov}(X_2, X_1) & \operatorname{Var}(X_2) & \dots & \operatorname{Cov}(X_2, X_n) \\
\vdots & \vdots & \ddots & \vdots \\
\operatorname{Cov}(X_n, X_1) & \operatorname{Cov}(X_n, X_2) & \dots & \operatorname{Var}(X_n)
\end{bmatrix}$$ 

That is, $\Sigma_{ij} = \operatorname{Cov}(X_i, X_j)$.
Since covariance is symmetric in its arguments, the covariance matrix is
also symmetric. It's also positive semi-definite: for any $\mathbf{x}$,

$$\mathbf{x}^{\!\top\!}\mathbf{\Sigma}\mathbf{x} = \mathbf{x}^{\!\top\!}\mathbb{E}[(\mathbf{X} - \mathbb{E}[\mathbf{X}])(\mathbf{X} - \mathbb{E}[\mathbf{X}])^{\!\top\!}]\mathbf{x} = \mathbb{E}[\mathbf{x}^{\!\top\!}(\mathbf{X} - \mathbb{E}[\mathbf{X}])(\mathbf{X} - \mathbb{E}[\mathbf{X}])^{\!\top\!}\mathbf{x}] = \mathbb{E}[((\mathbf{X} - \mathbb{E}[\mathbf{X}])^{\!\top\!}\mathbf{x})^2] \geq 0$$

The inverse of the covariance matrix, $\mathbf{\Sigma}^{-1}$, is
sometimes called the **precision matrix**.

## Estimation of Parameters

Now we get into some basic topics from statistics. We make some
assumptions about our problem by prescribing a **parametric** model
(e.g. a distribution that describes how the data were generated), then
we fit the parameters of the model to the data. How do we choose the
values of the parameters?

### Maximum likelihood estimation

A common way to fit parameters is **maximum likelihood estimation**
(MLE). The basic principle of MLE is to choose values that "explain" the
data best by maximizing the probability/density of the data we've seen
as a function of the parameters. Suppose we have random variables
$X_1, \dots, X_n$ and corresponding observations $x_1, \dots, x_n$. Then

$$\hat{\mathbf{\theta}}_\text{mle} = \operatorname{argmax}_\mathbf{\theta} \mathcal{L}(\mathbf{\theta})$$

where $\mathcal{L}$ is the **likelihood function**

$$\mathcal{L}(\mathbf{\theta}) = p(x_1, \dots, x_n; \mathbf{\theta})$$

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

### Maximum a posteriori estimation

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

## The Gaussian distribution

There are many distributions, but one of particular importance is the
**Gaussian distribution**, also known as the **normal distribution**. It
is a continuous distribution, parameterized by its mean
$\boldsymbol\mu \in \mathbb{R}^d$ and positive-definite covariance matrix
$\mathbf{\Sigma} \in \mathbb{R}^{d \times d}$, with density

$$p(\mathbf{x}; \boldsymbol\mu, \mathbf{\Sigma}) = \frac{1}{\sqrt{(2\pi)^d \det(\mathbf{\Sigma})}}\exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol\mu)^{\!\top\!}\mathbf{\Sigma}^{-1}(\mathbf{x} - \boldsymbol\mu)\right)$$

Note that in the special case $d = 1$, the density is written in the
more recognizable form

$$p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

We write $\mathbf{X} \sim \mathcal{N}(\boldsymbol\mu, \mathbf{\Sigma})$ to
denote that $\mathbf{X}$ is normally distributed with mean $\boldsymbol\mu$ and
variance $\mathbf{\Sigma}$.

### The geometry of multivariate Gaussians

The geometry of the multivariate Gaussian density is intimately related
to the geometry of positive definite quadratic forms, so make sure the
material in that section is well-understood before tackling this
section.

First observe that the p.d.f. of the multivariate Gaussian can be
rewritten as

$$p(\mathbf{x}; \boldsymbol\mu, \mathbf{\Sigma}) = g(\tilde{\mathbf{x}}^{\!\top\!}\mathbf{\Sigma}^{-1}\tilde{\mathbf{x}})$$

where $\tilde{\mathbf{x}} = \mathbf{x} - \boldsymbol\mu$ and
$g(z) = [(2\pi)^d \det(\mathbf{\Sigma})]^{-\frac{1}{2}}\exp\left(-\frac{z}{2}\right)$.
Writing the density in this way, we see that after shifting by the mean
$\boldsymbol\mu$, the density is really just a simple function of its precision
matrix's quadratic form.

Here is a key observation: this function $g$ is **strictly monotonically
decreasing** in its argument. That is, $g(a) > g(b)$ whenever $a < b$.
Therefore, small values of
$\tilde{\mathbf{x}}^{\!\top\!}\mathbf{\Sigma}^{-1}\tilde{\mathbf{x}}$
(which generally correspond to points where $\tilde{\mathbf{x}}$ is
closer to $\mathbf{0}$, i.e. $\mathbf{x} \approx \boldsymbol\mu$) have
relatively high probability densities, and vice-versa. Furthermore,
because $g$ is *strictly* monotonic, it is injective, so the
$c$-isocontours of $p(\mathbf{x}; \boldsymbol\mu, \mathbf{\Sigma})$ are the
$g^{-1}(c)$-isocontours of the function
$\mathbf{x} \mapsto \tilde{\mathbf{x}}^{\!\top\!}\mathbf{\Sigma}^{-1}\tilde{\mathbf{x}}$.
That is, for any $c$,

$$\{\mathbf{x} \in \mathbb{R}^d : p(\mathbf{x}; \boldsymbol\mu, \mathbf{\Sigma}) = c\} = \{\mathbf{x} \in \mathbb{R}^d : \tilde{\mathbf{x}}^{\!\top\!}\mathbf{\Sigma}^{-1}\tilde{\mathbf{x}} = g^{-1}(c)\}$$

In words, these functions have the same isocontours but different
isovalues.

Recall the executive summary of the geometry of positive definite
quadratic forms: the isocontours of
$f(\mathbf{x}) = \mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}$ are
ellipsoids such that the axes point in the directions of the
eigenvectors of $\mathbf{A}$, and the lengths of these axes are
proportional to the inverse square roots of the corresponding
eigenvalues. Therefore in this case, the isocontours of the density are
ellipsoids (centered at $\boldsymbol\mu$) with axis lengths proportional to the
inverse square roots of the eigenvalues of $\mathbf{\Sigma}^{-1}$, or
equivalently, the square roots of the eigenvalues of $\mathbf{\Sigma}$.

[^1]: More generally, vector spaces can be defined over any **field**
    $\mathbb{F}$. We take $\mathbb{F} = \mathbb{R}$ in this document to
    avoid an unnecessary diversion into abstract algebra.

[^2]: It is sometimes called the **kernel** by algebraists, but we
    eschew this terminology because the word "kernel" has another
    meaning in machine learning.

[^3]: If a normed space is complete with respect to the distance metric
    induced by its norm, we say that it is a **Banach space**.

[^4]: If an inner product space is complete with respect to the distance
    metric induced by its inner product, we say that it is a **Hilbert
    space**.

[^5]: Recall that $\mathbf{A}^{\!\top\!}\mathbf{A}$ and
    $\mathbf{A}\mathbf{A}^{\!\top\!}$ are positive semi-definite, so
    their eigenvalues are nonnegative, and thus taking square roots is
    always well-defined.

[^6]: A **neighborhood** about $\mathbf{x}$ is an open set which
    contains $\mathbf{x}$.

[^7]: $\mathcal{F}$ is required to be a $\sigma$-algebra for technical
    reasons; see [@rigorousprob].

[^8]: Note that a probability space is simply a measure space in which
    the measure of the whole space equals 1.

[^9]: This is a probabilist's version of the measure-theoretic term
    *almost everywhere*.

[^10]: In some cases it is possible to define conditional probability on
    events of probability zero, but this is significantly more technical
    so we omit it.

[^11]: The function must be measurable.

[^12]: More generally, the codomain can be any measurable space, but
    $\mathbb{R}$ is the most common case by far and sufficient for our
    purposes.

[^13]: Random variables that are continuous but not absolutely
    continuous are called **singular random variables**. We will not
    discuss them, assuming rather that all continuous random variables
    admit a density function.

[^14]: We haven't defined this yet; see the Correlation section below