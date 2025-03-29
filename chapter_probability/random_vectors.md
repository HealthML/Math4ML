## Random vectors

So far we have been talking about **univariate distributions**, that is,
distributions of single variables. But we can also talk about
**multivariate distributions** which give distributions of **random
vectors**:

$$\mathbf{X} = \begin{bmatrix}X_1 \\ \vdots \\ X_n\end{bmatrix}$$ 

The
summarizing quantities we have discussed for single variables have
natural generalizations to the multivariate case.

Expectation of a random vector is simply the expectation applied to each
component:

$$\mathbb{E}[\mathbf{X}] = \begin{bmatrix}\mathbb{E}[X_1] \\ \vdots \\ \mathbb{E}[X_n]\end{bmatrix}$$

The variance is generalized by the **covariance matrix**:

$$\mathbf{\Sigma} = \mathbb{E}[(\mathbf{X} - \mathbb{E}[\mathbf{X}])(\mathbf{X} - \mathbb{E}[\mathbf{X}])^{\!\top\!}] = \begin{bmatrix}
\operatorname{Var}(X_1) & \operatorname{Cov}(X_1, X_2) & \dots & \operatorname{Cov}(X_1, X_n) \\
\operatorname{Cov}(X_2, X_1) & \operatorname{Var}(X_2) & \dots & \operatorname{Cov}(X_2, X_n) \\
\vdots & \vdots & \ddots & \vdots \\
\operatorname{Cov}(X_n, X_1) & \operatorname{Cov}(X_n, X_2) & \dots & \operatorname{Var}(X_n)
\end{bmatrix}$$ 

That is, $\Sigma_{ij} = \operatorname{Cov}(X_i, X_j)$.
Since covariance is symmetric in its arguments, the covariance matrix is
also symmetric. It's also positive semi-definite: for any $\mathbf{x}$,

$$\mathbf{x}^{\!\top\!}\mathbf{\Sigma}\mathbf{x} = \mathbf{x}^{\!\top\!}\mathbb{E}[(\mathbf{X} - \mathbb{E}[\mathbf{X}])(\mathbf{X} - \mathbb{E}[\mathbf{X}])^{\!\top\!}]\mathbf{x} = \mathbb{E}[\mathbf{x}^{\!\top\!}(\mathbf{X} - \mathbb{E}[\mathbf{X}])(\mathbf{X} - \mathbb{E}[\mathbf{X}])^{\!\top\!}\mathbf{x}] = \mathbb{E}[((\mathbf{X} - \mathbb{E}[\mathbf{X}])^{\!\top\!}\mathbf{x})^2] \geq 0$$

The inverse of the covariance matrix, $\mathbf{\Sigma}^{-1}$, is
sometimes called the **precision matrix**.

