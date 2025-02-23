# Notation

  Notation                     Meaning
  ---------------------------- ----------------------------------------------------------------------------------------
  $\mathbb{R}$                 set of real numbers
  $\mathbb{R}^n$               set (vector space) of $n$-tuples of real numbers, endowed with the usual inner product
  $\mathbb{R}^{m \times n}$    set (vector space) of $m$-by-$n$ matrices
  $\delta_{ij}$                Kronecker delta, i.e. $\delta_{ij} = 1$ if $i = j$, $0$ otherwise
  $\nabla f(\mathbf{x})$       gradient of the function $f$ at $\mathbf{x}$
  $\nabla^2 f(\mathbf{x})$     Hessian of the function $f$ at $\mathbf{x}$
  $\mathbf{A}^{\!\top\!}$      transpose of the matrix $\mathbf{A}$
  $\Omega$                     sample space
  $\mathbb{P}(A)$              probability of event $A$
  $p(X)$                       distribution of random variable $X$
  $p(x)$                       probability density/mass function evaluated at $x$
  $A^\text{c}$                 complement of event $A$
  $A \mathbin{\dot{\cup}} B$   union of $A$ and $B$, with the extra requirement that $A \cap B = \varnothing$
  $\mathbb{E}[X]$              expected value of random variable $X$
  $\operatorname{Var}(X)$      variance of random variable $X$
  $\operatorname{Cov}(X, Y)$   covariance of random variables $X$ and $Y$

Other notes:

-   Vectors and matrices are in bold (e.g. $\mathbf{x}, \mathbf{A}$).
    This is true for vectors in $\mathbb{R}^n$ as well as for vectors in
    general vector spaces. We generally use Greek letters for scalars
    and capital Roman letters for matrices and random variables.

-   To stay focused at an appropriate level of abstraction, we restrict
    ourselves to real values. In many places in this document, it is
    entirely possible to generalize to the complex case, but we will
    simply state the version that applies to the reals.

-   We assume that vectors are column vectors, i.e. that a vector in
    $\mathbb{R}^n$ can be interpreted as an $n$-by-$1$ matrix. As such,
    taking the transpose of a vector is well-defined (and produces a row
    vector, which is a $1$-by-$n$ matrix).


```{tableofcontents}
```
