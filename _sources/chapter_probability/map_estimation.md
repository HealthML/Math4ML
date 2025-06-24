## Maximum A Posteriori Estimation (MAP)

MAP estimation introduces prior knowledge into the estimation problem.
Using Bayes' rule, the posterior over parameters $\theta$ is:

$$
p(\theta \mid x_1, \dots, x_n) \propto p(\theta) \cdot p(x_1, \dots, x_n \mid \theta)
$$

The **MAP estimate** is the mode of the posterior:

$$
\hat{\theta}_\text{MAP} = \operatorname{argmax}_\theta \log p(\theta) + \sum_{i=1}^n \log p(x_i \mid \theta)
$$


## Maximum a posteriori estimation

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

---
