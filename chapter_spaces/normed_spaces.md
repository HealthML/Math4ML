## Normed spaces

Norms generalize the notion of length from Euclidean space.

A **norm** on a real vector space $V$ is a function
$\|\cdot\| : V \to \mathbb{R}$ that satisfies

(i) $\|\mathbf{x}\| \geq 0$, with equality if and only if
    $\mathbf{x} = \mathbf{0}$

(ii) $\|\alpha\mathbf{x}\| = |\alpha|\|\mathbf{x}\|$

(iii) $\|\mathbf{x}+\mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$
      (the **triangle inequality** again)

for all $\mathbf{x}, \mathbf{y} \in V$ and all $\alpha \in \mathbb{R}$.
A vector space endowed with a norm is called a **normed vector space**,
or simply a **normed space**.

Note that any norm on $V$ induces a distance metric on $V$:

$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x}-\mathbf{y}\|$

One can verify that the axioms for metrics are satisfied under this definition and
follow directly from the axioms for norms. Therefore any normed space is
also a metric space.[^3]

We will typically only be concerned with a few specific norms on
$\mathbb{R}^n$:

$$\begin{aligned}
\|\mathbf{x}\|_1 &= \sum_{i=1}^n |x_i| \\
\|\mathbf{x}\|_2 &= \sqrt{\sum_{i=1}^n x_i^2} \\
\|\mathbf{x}\|_p &= \left(\sum_{i=1}^n |x_i|^p\right)^\frac{1}{p} \hspace{0.5cm}\hspace{0.5cm} (p \geq 1) \\
\|\mathbf{x}\|_\infty &= \max_{1 \leq i \leq n} |x_i|
\end{aligned}$$

Note that the 1- and 2-norms are special cases of the
$p$-norm, and the $\infty$-norm is the limit of the $p$-norm as $p$
tends to infinity. We require $p \geq 1$ for the general definition of
the $p$-norm because the triangle inequality fails to hold if $p < 1$.
(Try to find a counterexample!)

Here's a fun fact: for any given finite-dimensional vector space $V$,
all norms on $V$ are equivalent in the sense that for two norms
$\|\cdot\|_A, \|\cdot\|_B$, there exist constants $\alpha, \beta > 0$
such that

$$\alpha\|\mathbf{x}\|_A \leq \|\mathbf{x}\|_B \leq \beta\|\mathbf{x}\|_A$$

for all $\mathbf{x} \in V$. Therefore convergence in one norm implies
convergence in any other norm. This rule may not apply in
infinite-dimensional vector spaces such as function spaces, though.



## Normed Spaces in Machine Learning

Normed spaces generalize the idea of **length** and thus naturally appear whenever machine learning algorithms quantify vector magnitude or enforce regularization.

### Examples:

1. **Regularization (Ridge and Lasso)**  
Regularization methods in machine learning, such as **ridge regression** (L2 regularization) and **lasso regression** (L1 regularization), explicitly use norms on parameter vectors:

- **L2 Regularization (Ridge):**

$$\text{Loss}_{ridge}(\mathbf{w}) = \text{MSE}(\mathbf{w}) + \lambda \|\mathbf{w}\|_2^2$$

(penalizes the squared Euclidean norm, encouraging small parameter values.)

- **L1 Regularization (Lasso):**
     
$$\text{Loss}_{lasso}(\mathbf{w}) = \text{MSE}(\mathbf{w}) + \lambda \|\mathbf{w}\|_1$$

(penalizes the sum of absolute parameter values, promoting sparsity in the solution.)

2. **Measuring Errors and Convergence (Gradient Descent)**  
When running optimization algorithms such as **gradient descent**, one commonly uses norms to measure how far parameter updates move between iterations:
     
$$\|\mathbf{w}_{t+1} - \mathbf{w}_{t}\|_2 \quad \text{or} \quad \|\nabla f(\mathbf{w}_t)\|_2$$

The algorithm stops when the magnitude (norm) of parameter updates or gradients becomes sufficiently small.

---
### Summary of ML Examples:

| Concept           | ML Examples                                             |
|-------------------|---------------------------------------------------------|
| Metric Space      | k-NN classifier, Clustering (k-means, DBSCAN), Text similarity (Levenshtein) |
| Normed Space      | L1/L2 regularization (ridge, lasso), Gradient descent convergence |

These examples highlight the practical and foundational role of metrics and norms in machine learning, illustrating how abstract mathematical concepts directly influence algorithm design and performance.