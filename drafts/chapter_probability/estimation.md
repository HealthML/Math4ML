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

