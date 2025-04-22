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
# Representer Theorem for Linear Functions

In many machine learning problems, especially those involving $l2$-regularized risk minimization over linear functions, the **representer theorem** guarantees that the solution to the optimization problem can be written as a finite linear combination of the training samples. Consider the following regularized formulation:

$$
\operatorname{argmin}_{\mathbf{w}, b} \; \sum_{i=1}^n L(y_i, \mathbf{w}^\top \mathbf{x}_i + b) + \lambda\, \|\mathbf{w}\|^2,
$$

where:
- $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ are the training examples with $\mathbf{x}_i \in \mathbb{R}^d$ and corresponding labels $y_i$,
- $L$ is a loss function (such as the logistic loss for logistic regression, or the squared loss for linear regression),
- $\lambda > 0$ is a regularization parameter,
- $\|\mathbf{w}\|^2$ represents the squared Euclidean norm, which penalizes the complexity of the classifier.

The representer theorem asserts that there exists a solution $(\mathbf{w}^*, b^*)$ where the optimal weight vector $\mathbf{w}^*$ lies in the span of the training data. That is, one can express

$$
\mathbf{w}^* = \sum_{i=1}^n \alpha_i \mathbf{x}_i,
$$

for some coefficients $\alpha_1, \alpha_2, \dots, \alpha_n \in \mathbb{R}$. Consequently, the decision function for a new example $\mathbf{x}^*$ becomes

$$
f(\mathbf{x}^*) = \mathbf{w}^{*\top} \mathbf{x} + b^* 
= \left(\sum_{i=1}^n \alpha_i \mathbf{x}_i^\top\right) \mathbf{x} + b^*
= \sum_{i=1}^n \alpha_i\, \langle \mathbf{x}_i, \mathbf{x}^* \rangle + b^*.
$$

This representation shows that the decision function is a linear combination of the inner products between the training examples and the new input $\mathbf{x}^*$, plus a bias term $b^*$.
This is particularly useful in kernelized settings, where the inner products can be computed using a kernel function $k(\mathbf{x}_i, \mathbf{x}^*)$ that implicitly maps the data into a higher-dimensional feature space without explicitly computing the mapping.

This representation has profound implications:

- **Finite Representation:** Even if the underlying hypothesis space (e.g., an RKHS in the kernelized setting) is infinite-dimensional, regularization forces the solution to lie in the finite-dimensional span of the training examples.
- **Computational Efficiency:** The optimization problem is effectively reduced to finding the finite set of coefficients $\alpha_i$ (and $b$), which can be computed efficiently using kernel methods or convex optimization techniques.
- **Interpretability:** In linear classifiers, this representation reveals that the learned decision boundary is entirely determined by the training data. In the case of support vector machines, for instance, only a subset of the training points (the support vectors) will have nonzero coefficients $\alpha_i$, directly indicating which examples are critical for classification.

Thus, for linear classifiers with decision boundaries of the form $\mathbf{w}^\top \mathbf{x} + b$, the representer theorem not only ensures that the problem has a solution in the span of the data but also provides practical and theoretical benefits by reducing the complexity of the learning task.

Similarly, for linear regression with regularization, the representer theorem guarantees that the optimal solution is in the span of the training inputs and can be written entirely in terms of inner products between these inputs, providing both theoretical insight and practical advantages in computation and model interpretation.

--- 
## Representer Theorem for Linear Classifiers and Linear Regression
::: {prf:theorem} Representer Theorem
:label: thm-representer
:nonumber:
Let $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ be a set of training examples in $\mathbb{R}^d$ and let $L$ be a loss function. Consider the following optimization problem:

$$
\min_{\mathbf{w}, b} \; \sum_{i=1}^n L(y_i, \mathbf{w}^\top \mathbf{x}_i + b) + \lambda\, \|\mathbf{w}\|^2,
$$

where $\lambda > 0$ is a regularization parameter. Then, there exists an optimal solution $(\mathbf{w}^*, b^*)$ such that

$$
\mathbf{w}^* = \sum_{i=1}^n \alpha_i \mathbf{x}_i,
$$
for some coefficients $\alpha_i \in \mathbb{R}$, where $\mathbf{x}_i$ are the training examples.
:::

:::{prf:proof}
Assume that we are solving a regularized risk minimization problem over $\mathbf{w} \in \mathbb{R}^d$ and $b \in \mathbb{R}$ of the form

$$
\min_{\mathbf{w}, b} \; \sum_{i=1}^n L\bigl(y_i, \mathbf{w}^\top \mathbf{x}_i + b\bigr) + \lambda\, \|\mathbf{w}\|^2,
$$

where $L$ is a loss function, and $\lambda>0$ is the regularization parameter. Let 
$$
V = \operatorname{span}\{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\}
$$
be the subspace spanned by the training data.

For any vector $\mathbf{w} \in \mathbb{R}^d$, we can decompose it uniquely as

$$
\mathbf{w} = \mathbf{w}_0 + \mathbf{w}_1,
$$

where $\mathbf{w}_0 \in V$ and $\mathbf{w}_1 \in V^\perp$ (the orthogonal complement of $V$). Notice that by construction, for every training example $\mathbf{x}_i$ (which lies in $V$), we have

$$
\mathbf{w}_1^\top\mathbf{x}_i = 0.
$$

Thus, the prediction for each training example is

$$
\mathbf{w}^\top \mathbf{x}_i + b = (\mathbf{w}_0 + \mathbf{w}_1)^\top \mathbf{x}_i + b = \mathbf{w}_0^\top \mathbf{x}_i + \mathbf{w}_1^\top \mathbf{x}_i + b = \mathbf{w}_0^\top \mathbf{x}_i + b.
$$

This shows that the component $\mathbf{w}_1$ (lying in $V^\perp$) does not affect the predictions on the training data.

Now, consider the regularization term $\|\mathbf{w}\|^2$. 

$$
\|\mathbf{w}\|^2 = \mathbf{w}^\top\mathbf{w} = (\mathbf{w}_0 + \mathbf{w}_1)^\top(\mathbf{w}_0 + \mathbf{w}_1) = \mathbf{w}_0^\top\mathbf{w}_0 + \mathbf{w}_1^\top\mathbf{w}_1 + 2\mathbf{w}_0^\top\mathbf{w}_1
$$

### TODO:

$$
\mathbf{w}_0^\top\mathbf{w}_1 = 0
$$

Because $\mathbf{w}_0$ and $\mathbf{w}_1$ are orthogonal, $\mathbf{w}_0^\top\mathbf{w}_1=0$ and we have

$$
\|\mathbf{w}\|^2 = \|\mathbf{w}_0\|^2 + \|\mathbf{w}_1\|^2.
$$

If $\mathbf{w}_1 \neq \mathbf{0}$, then $\|\mathbf{w}\|^2 > \|\mathbf{w}_0\|^2$ but the predictions remain the same.
Since the objective includes the regularization term $\lambda\,\|\mathbf{w}\|^2$, any solution can be improved (or at least not worsened) by setting $\mathbf{w}_1 = \mathbf{0}$. In other words, there is no benefit to having a component in $V^\perp$.

Thus, we can always find an optimal weight vector $\mathbf{w}^*$ that lies entirely in $V$; that is,

$$
\mathbf{w}^* = \mathbf{w}_0 \quad \text{with} \quad \mathbf{w}^* \in V.
$$

Because $V$ is the span of $\{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\}$, there exist scalars $\alpha_1, \alpha_2, \dots, \alpha_n$ such that

$$
\mathbf{w}^* = \sum_{i=1}^n \alpha_i \mathbf{x}_i.
$$

This completes the proof that the optimal solution can be expressed as a linear combination of the training data.
:::
---

### Explanation:

- **Key Observation:** The loss term depends only on $\mathbf{w}^\top \mathbf{x}_i$. Since any component of $\mathbf{w}$ perpendicular to all training examples (i.e. in $V^\perp$) does not affect the loss, including it only increases the regularization penalty.
- **Conclusion:** We can always set the $V^\perp$ component to zero without changing the predictions on the training data, thereby arriving at an optimal solution that lies in the span of the training data. Consequently, the optimal weight vector can be written as

$$
\mathbf{w}^* = \sum_{i=1}^n \alpha_i \mathbf{x}_i.
$$

This representer theorem is central in kernel methods and many linear models because it reduces an infinite-dimensional search (if one considers a feature space mapping) to a finite-dimensional problem based solely on the training samples.

## Application in Kernel Methods

### TODO