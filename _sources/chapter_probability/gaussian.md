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