#### Linear Regression

In linear regression, the predicted values $\hat{\mathbf{y}}$ of a linear model form a **column space** of the data matrix $\mathbf{X}$:

- Given:

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{\beta}, \quad \mathbf{X} \in \mathbb{R}^{n\times d}, \quad \mathbf{\beta} \in \mathbb{R}^{d\times 1}$$

- **Subspace property:**  
  The set of all possible predictions $\hat{\mathbf{y}}$ for different coefficients $\mathbf{\beta}$ is the column space of $\mathbf{X}$, a subspace of $\mathbb{R}^n$.
  It contains the zero vector (achieved by setting all $\mathbf{\beta}$ to zero), and is closed under vector addition and scalar multiplication.
