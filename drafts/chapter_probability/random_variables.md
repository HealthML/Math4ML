## Random variables

A **random variable** is some uncertain quantity with an associated
probability distribution over the values it can assume.

Formally, a random variable on a probability space
$(\Omega, \mathcal{F}, \mathbb{P})$ is a function[^11]
$X: \Omega \to \mathbb{R}$.[^12]

We denote the range of $X$ by
$X(\Omega) = \{X(\omega) : \omega \in \Omega\}$. To give a concrete
example (taken from [@pitman]), suppose $X$ is the number of heads in
two tosses of a fair coin. The sample space is

$$\Omega = \{hh, tt, ht, th\}$$ 

and $X$ is determined completely by the
outcome $\omega$, i.e. $X = X(\omega)$. For example, the event $X = 1$
is the set of outcomes $\{ht, th\}$.

It is common to talk about the values of a random variable without
directly referencing its sample space. The two are related by the
following definition: the event that the value of $X$ lies in some set
$S \subseteq \mathbb{R}$ is

$$X \in S = \{\omega \in \Omega : X(\omega) \in S\}$$ 

Note that special
cases of this definition include $X$ being equal to, less than, or
greater than some specified value. For example

$$\mathbb{P}(X = x) = \mathbb{P}(\{\omega \in \Omega : X(\omega) = x\})$$

A word on notation: we write $p(X)$ to denote the entire probability
distribution of $X$ and $p(x)$ for the evaluation of the function $p$ at
a particular value $x \in X(\Omega)$. Hopefully this (reasonably
standard) abuse of notation is not too distracting. If $p$ is
parameterized by some parameters $\theta$, we write
$p(X; \mathbf{\theta})$ or $p(x; \mathbf{\theta})$, unless we are in a
Bayesian setting where the parameters are considered a random variable,
in which case we condition on the parameters.

### The cumulative distribution function

The **cumulative distribution function** (c.d.f.) gives the probability
that a random variable is at most a certain value:

$$F(x) = \mathbb{P}(X \leq x)$$ 

The c.d.f. can be used to give the
probability that a variable lies within a certain range:

$$\mathbb{P}(a < X \leq b) = F(b) - F(a)$$

### Discrete random variables

A **discrete random variable** is a random variable that has a countable
range and assumes each value in this range with positive probability.
Discrete random variables are completely specified by their
**probability mass function** (p.m.f.) $p : X(\Omega) \to [0,1]$ which
satisfies 

$$\sum_{x \in X(\Omega)} p(x) = 1$$ 

For a discrete $X$, the
probability of a particular value is given exactly by its p.m.f.:

$$\mathbb{P}(X = x) = p(x)$$

### Continuous random variables

A **continuous random variable** is a random variable that has an
uncountable range and assumes each value in this range with probability
zero. Most of the continuous random variables that one would encounter
in practice are **absolutely continuous random variables**[^13], which
means that there exists a function $p : \mathbb{R} \to [0,\infty)$ that
satisfies 

$$F(x) \equiv \int_{-\infty}^x p(z)\operatorname{d}{z}$$ 

The function $p$
is called a **probability density function** (abbreviated p.d.f.) and
must satisfy 

$$\int_{-\infty}^\infty p(x)\operatorname{d}{x} = 1$$ 

The values of this
function are not themselves probabilities, since they could exceed 1.
However, they do have a couple of reasonable interpretations. One is as
relative probabilities; even though the probability of each particular
value being picked is technically zero, some points are still in a sense
more likely than others.

One can also think of the density as determining the probability that
the variable will lie in a small range about a given value. This is
because, for small $\epsilon > 0$,

$$\mathbb{P}(x-\epsilon \leq X \leq x+\epsilon) = \int_{x-\epsilon}^{x+\epsilon} p(z)\operatorname{d}{z} \approx 2\epsilon p(x)$$

using a midpoint approximation to the integral.

Here are some useful identities that follow from the definitions above:

$$\begin{aligned}
\mathbb{P}(a \leq X \leq b) &= \int_a^b p(x)\operatorname{d}{x} \\
p(x) &= F'(x)
\end{aligned}$$

### Other kinds of random variables

There are random variables that are neither discrete nor continuous. For
example, consider a random variable determined as follows: flip a fair
coin, then the value is zero if it comes up heads, otherwise draw a
number uniformly at random from $[1,2]$. Such a random variable can take
on uncountably many values, but only finitely many of these with
positive probability. We will not discuss such random variables because
they are rather pathological and require measure theory to analyze.

