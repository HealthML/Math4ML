# Joint distributions

Often we have several random variables and we would like to get a
distribution over some combination of them. A **joint distribution** is
exactly this. For some random variables $X_1, \dots, X_n$, the joint
distribution is written $p(X_1, \dots, X_n)$ and gives probabilities
over entire assignments to all the $X_i$ simultaneously.

## Independence of random variables

We say that two variables $X$ and $Y$ are **independent** if their joint
distribution factors into their respective distributions, i.e.

$$p(X, Y) = p(X)p(Y)$$ 

We can also define independence for more than two
random variables, although it is more complicated. Let
$\{X_i\}_{i \in I}$ be a collection of random variables indexed by $I$,
which may be infinite. Then $\{X_i\}$ are independent if for every
finite subset of indices $i_1, \dots, i_k \in I$ we have

$$p(X_{i_1}, \dots, X_{i_k}) = \prod_{j=1}^k p(X_{i_j})$$ 

For example,
in the case of three random variables, $X, Y, Z$, we require that
$p(X,Y,Z) = p(X)p(Y)p(Z)$ as well as $p(X,Y) = p(X)p(Y)$,
$p(X,Z) = p(X)p(Z)$, and $p(Y,Z) = p(Y)p(Z)$.

It is often convenient (though perhaps questionable) to assume that a
bunch of random variables are **independent and identically
distributed** (i.i.d.) so that their joint distribution can be factored
entirely: 

$$p(X_1, \dots, X_n) = \prod_{i=1}^n p(X_i)$$ 

where
$X_1, \dots, X_n$ all share the same p.m.f./p.d.f.

## Marginal distributions

If we have a joint distribution over some set of random variables, it is
possible to obtain a distribution for a subset of them by "summing out"
(or "integrating out" in the continuous case) the variables we don't
care about: 

$$p(X) = \sum_{y} p(X, y)$$

