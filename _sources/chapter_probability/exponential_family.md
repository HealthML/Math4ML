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
# The Exponential Family and Conjugate Priors

Many common distributions (Gaussian, Bernoulli, Poisson, etc.) belong to the **exponential family**, which has a convenient structure for Bayesian analysis. 
A probability distribution belongs to the exponential family if it can be written in the form:

$$
p(x \mid \theta) = h(x) \exp\left( \eta(\theta)^\top T(x) - A(\theta) \right)
$$

Where:

* $\theta$: natural (canonical) parameters
* $\eta(\theta)$: natural parameter function
* $T(x)$: sufficient statistics
* $A(\theta)$: log-partition function (normalizer)
* $h(x)$: base measure

---

### Why the Exponential Family Matters for MAP

If we choose a **prior that is conjugate** to the exponential family likelihood, the posterior has the **same functional form** as the prior — this makes both **analysis and computation much easier**.

In particular:

* The posterior is often interpretable as a **prior+data update**.
* It allows for **analytical MAP estimation**.
* The prior can be viewed as **pseudo-observations**, guiding estimation when data is scarce.

A particularly nice case is when the prior is chosen carefully such that
the posterior comes from the same family as the prior. 
In this case the prior is called a **conjugate prior**. 

For example, if the likelihood is
binomial and the prior is beta, the posterior is also beta. There are
many conjugate priors; the reader may find this [table of conjugate
priors](https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions)
useful.

