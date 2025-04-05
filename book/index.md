# Preface

Machine learning (ML) and data science involve algorithms that automatically learn patterns and relationships directly from data. At their core, many ML methods rely heavily on fundamental mathematical concepts—especially linear algebra, calculus, probability, and optimization—to understand, analyze, and make predictions from data.

Our assumption is that the reader is already familiar with the basic concepts of multivariable calculus, linear algebra and probability theory (at the level of Mathematik 2 and Mathematik 3 in the ITSE Bachelor).

This work is based on Mathematics for Machine Learning ([math4ml](https://github.com/gwthomas/math4ml)) by Garrett Thomas, Department of Electrical Engineering and Computer Sciences, University of California, Berkeley, downloaded on February 23, 2025. The original note by Garrett Thomas, which includes the permissive phrase "You are free to distribute this document as you wish.", has been used as a frame that provides the mathematical topics. However, it has been significantly expanded to include more explanations, more detailed proofs, and applications from the machine learning domain.

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://healthml.github.io/Math4ML/">Mathematics for Machine Learning (Math4ML)</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://hpi.de/lippert">Christoph Lippert</a> is licensed under <a href="https://creativecommons.org/licenses/by-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>

Please also note that this book is work in progress. The book is serving as course material for the Mathematics for Machine Learning lecture in the summer semester in 2025 and will be updated weekly during the course duration.

## Why Mathematical Foundations?

This book carefully introduces and builds upon essential mathematical foundations necessary for modern machine learning:

- **Linear Algebra** (Chapters on vector spaces, linear maps, norms, inner products, eigen-decompositions, and matrix factorizations) provides the language and tools to represent and manipulate data efficiently.
- **Calculus and Optimization** (Chapters on gradients, Hessians, Taylor approximations, and convexity) are essential for understanding how machine learning models are trained through iterative optimization processes.
- **Probability and Statistics** (Chapters on probability basics, random variables, covariance, and estimation) are fundamental for modeling uncertainty, interpreting results, and developing robust algorithms.

Throughout the chapters, each mathematical topic is motivated by practical machine learning examples implemented in Python. By seeing theory in action—such as predicting cancer diagnoses, analyzing weather patterns, understanding genetic data, or recognizing handwritten digits—you'll gain deeper intuition and appreciation for these mathematical tools.

[Jupyter Book](https://jupyterbook.org/) allows us to place the theory directly in the context of practical applications. Interactive Jupyter Notebooks illustrate concepts clearly and concretely, allowing you to view the code, modify it, and explore these concepts on your own.

## Notation

| Notation                  | Meaning |
|---------------------------|--------------------------------------------------------------------------|
| $\mathbb{R}$              | set of real numbers |
| $\mathbb{R}^n$            | set (vector space) of $n$-tuples of real numbers, endowed with the usual inner product |
| $\mathbb{R}^{m \times n}$ | set (vector space) of $m$-by-$n$ matrices |
|  $\delta_{ij}$            | Kronecker delta, i.e. $\delta_{ij} = 1$ if $i = j$, $0$ otherwise |
| $\nabla f(\mathbf{x})$    | gradient of the function $f$ at $\mathbf{x}$ |
| $\nabla^2 f(\mathbf{x})$  | Hessian of the function $f$ at $\mathbf{x}$ |
| $\mathbf{A}^{\!\top\!}$   | transpose of the matrix $\mathbf{A}$ |
| $\Omega$                  | sample space |
| $\mathbb{P}(A)$           | probability of event $A$ |
| $p(X)$                    | distribution of random variable $X$ |
| $p(x)$                    | probability density/mass function evaluated at $x$ |
| $A^\text{c}$              | complement of event $A$ |
| $A \mathbin{\dot{\cup}} B$| union of $A$ and $B$, with the extra requirement that $A \cap B = \varnothing$ |
| $\mathbb{E}[X]$           | expected value of random variable $X$ |
| $\operatorname{Var}(X)$   | variance of random variable $X$ |
| $\operatorname{Cov}(X, Y)$| covariance of random variables $X$ and $Y$ |

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

## Table of Contents

We invite you to explore the table of contents below to discover the mathematical foundations that will empower you to master machine learning and data science:

```{tableofcontents}
```