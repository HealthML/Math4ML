# Generalized Least Squares for Logistic Regression

It frames each IRLS step as solving a **generalized least squares problem** involving a symmetric positive definite matrix. This unifies the treatment with Ridge Regression and provides a nice segue to Newton's method (which you revisit later).
---

## IRLS as a Sequence of Generalized Least Squares Problems

In each iteration of **Iteratively Reweighted Least Squares (IRLS)** for logistic regression, we solve a **weighted least squares problem** of the form:

$$
\min_{\mathbf{w}} \frac{1}{2} (\mathbf{z}^{(t)} - \mathbf{Xw})^\top \mathbf{S}^{(t)} (\mathbf{z}^{(t)} - \mathbf{Xw})
$$

where:

* $\mathbf{X} \in \mathbb{R}^{n \times d}$ is the design matrix
* $\mathbf{z}^{(t)}$ is a "pseudo-response" vector derived from the current model predictions
* $\mathbf{S}^{(t)}$ is a diagonal **weight matrix** with entries:

$$
\mathbf{S}^{(t)}_{ii} = \hat{y}_i^{(t)} (1 - \hat{y}_i^{(t)}), \quad \text{where } \hat{\mathbf{y}}^{(t)} = \sigma(\mathbf{Xw}^{(t)})
$$

---

### IRLS as Quadratic Optimization

At iteration $t$, the objective function is quadratic in $\mathbf{w}$, and can be written in the general form:

$$
f^{(t)}(\mathbf{w}) = \frac{1}{2} \mathbf{w}^\top \mathbf{A}^{(t)} \mathbf{w} - \mathbf{b}^{(t)\top} \mathbf{w}
$$

with:

* $\mathbf{A}^{(t)} = \mathbf{X}^\top \mathbf{S}^{(t)} \mathbf{X}$, a symmetric positive semi-definite matrix
* $\mathbf{b}^{(t)} = \mathbf{X}^\top \mathbf{S}^{(t)} \mathbf{z}^{(t)}$

The minimum of this function is found by setting the gradient to zero:

$$
\nabla f^{(t)}(\mathbf{w}) = \mathbf{A}^{(t)} \mathbf{w} - \mathbf{b}^{(t)} = 0 \quad \Rightarrow \quad \boxed{
\mathbf{w}^{(t+1)} = (\mathbf{X}^\top \mathbf{S}^{(t)} \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{S}^{(t)} \mathbf{z}^{(t)}
}
$$

This is the weight update used in IRLS. Each step is a **generalized least squares problem**, and thanks to the diagonal structure of $\mathbf{S}^{(t)}$, the linear system can be solved efficiently.

---

### Summary and Outlook

* At each iteration, IRLS solves a **quadratic minimization problem** involving a symmetric positive definite matrix $\mathbf{X}^\top \mathbf{S} \mathbf{X}$
* This mirrors the structure we saw in Ridge Regression, where the matrix $\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}$ also defines a quadratic form
* The analogy helps us understand logistic regression optimization as a **sequence of local least squares problems**, adapted to fit a nonlinear model

> 📝 In a later section, we will revisit these ideas in the context of **Newton's method**, and explain why this sequence of updates corresponds to Newton’s steps on the logistic loss function.





In this section, we introduce **logistic regression**, a linear classifier that predicts probabilities.
We derive an iterative solution for **logistic regression** by solving for the zero of the gradient.
The derivation involves linearizing the loss function and solving for a weight update that zeros the gradient at each step. This eventually leads to what's known as **Iteratively Reweighted Least Squares (IRLS)**.

## Iterative Solution for Logistic Regression via Gradient Zeros

In logistic regression, the predicted probabilities are given by the **sigmoid function**:

$$
\hat{\mathbf{y}} = \sigma(\mathbf{Xw}) = \frac{1}{1 + e^{-\mathbf{Xw}}}
$$

The negative log-likelihood (NLL) loss function for logistic regression is:

$$
\mathcal{L}(\mathbf{w}) = -\sum_{i=1}^n \left[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\right]
$$

This can also be written in vectorized form:

$$
\mathcal{L}(\mathbf{w}) = -\mathbf{y}^\top \log(\sigma(\mathbf{Xw})) - (\mathbf{1} - \mathbf{y})^\top \log(1 - \sigma(\mathbf{Xw}))
$$

The gradient of this loss with respect to $\mathbf{w}$ is:

$$
\nabla \mathcal{L}(\mathbf{w}) = \mathbf{X}^\top (\sigma(\mathbf{Xw}) - \mathbf{y})
$$

We now seek a weight vector $\mathbf{w}$ such that this gradient is zero:

$$
\mathbf{X}^\top (\sigma(\mathbf{Xw}) - \mathbf{y}) = 0
$$

But because $\sigma(\mathbf{Xw})$ is a nonlinear function of $\mathbf{w}$, we **cannot solve this equation directly**. Instead, we linearize the function around the current weight vector and solve an approximate linear system at each iteration.

---

## Step-by-step derivation of the iterative update

1. **Let** $\hat{\mathbf{y}} = \sigma(\mathbf{Xw})$

2. **Linearize** $\hat{\mathbf{y}}$ around the current weight $\mathbf{w}^{(t)}$ using a first-order Taylor expansion:

$$
\hat{\mathbf{y}} \approx \hat{\mathbf{y}}^{(t)} + \mathbf{S} \mathbf{X}(\mathbf{w} - \mathbf{w}^{(t)})
$$

where $\hat{\mathbf{y}}^{(t)} = \sigma(\mathbf{Xw}^{(t)})$, and $\mathbf{S}$ is a diagonal matrix with entries:

$$
\mathbf{S}_{ii} = \hat{y}_i^{(t)} (1 - \hat{y}_i^{(t)})
$$

3. **Substitute** this into the gradient and set it to zero:

$$
\mathbf{X}^\top (\hat{\mathbf{y}}^{(t)} + \mathbf{S} \mathbf{X}(\mathbf{w} - \mathbf{w}^{(t)}) - \mathbf{y}) = 0
$$

4. **Rearrange terms** to isolate the weight update:

$$
\mathbf{X}^\top \mathbf{S} \mathbf{X} (\mathbf{w} - \mathbf{w}^{(t)}) = \mathbf{X}^\top (\mathbf{y} - \hat{\mathbf{y}}^{(t)})
$$

5. **Solve** for $\mathbf{w}$:

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + (\mathbf{X}^\top \mathbf{S} \mathbf{X})^{-1} \mathbf{X}^\top (\mathbf{y} - \hat{\mathbf{y}}^{(t)})
$$

---

## Interpretation and Aftermath

This gives us an **iterative procedure** where at each step:

* You compute predicted probabilities $\hat{\mathbf{y}}^{(t)} = \sigma(\mathbf{Xw}^{(t)})$
* Construct the weight matrix $\mathbf{S}$
* Solve a weighted least squares problem to get the next $\mathbf{w}^{(t+1)}$

This is known as **Iteratively Reweighted Least Squares (IRLS)**.

> 💡 In hindsight, this is an instance of **Newton’s method** applied to the logistic regression loss. Specifically, it's Newton's method using the Hessian of the log-likelihood, which turns out to be $\mathbf{X}^\top \mathbf{S} \mathbf{X}$.

## Connection to the Hessian: Why the Iterative Update Works

To understand why the update

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + (\mathbf{X}^\top \mathbf{S} \mathbf{X})^{-1} \mathbf{X}^\top (\mathbf{y} - \hat{\mathbf{y}}^{(t)})
$$

makes sense, we examine the second derivative (the **Hessian**) of the logistic regression loss function.

Recall the loss:

$$
\mathcal{L}(\mathbf{w}) = -\mathbf{y}^\top \log(\sigma(\mathbf{Xw})) - (\mathbf{1} - \mathbf{y})^\top \log(1 - \sigma(\mathbf{Xw}))
$$

Let $\hat{\mathbf{y}} = \sigma(\mathbf{Xw})$. The gradient is:

$$
\nabla \mathcal{L}(\mathbf{w}) = \mathbf{X}^\top (\hat{\mathbf{y}} - \mathbf{y})
$$

Now we compute the **Hessian**, which is the matrix of second partial derivatives:

$$
\nabla^2 \mathcal{L}(\mathbf{w}) = \mathbf{X}^\top \mathbf{S} \mathbf{X}
$$

where $\mathbf{S}$ is a diagonal matrix with entries:

$$
\mathbf{S}_{ii} = \hat{y}_i (1 - \hat{y}_i)
$$

This expression comes from differentiating the sigmoid function and applying the chain rule:

* The derivative of $\sigma(z)$ is $\sigma(z)(1 - \sigma(z))$
* Because $\hat{\mathbf{y}} = \sigma(\mathbf{Xw})$, the Jacobian is a diagonal matrix $\mathbf{S}$
* Then, the second derivative of the loss involves multiplying by $\mathbf{X}^\top$ on the left and $\mathbf{X}$ on the right

Thus:

$$
\nabla^2 \mathcal{L}(\mathbf{w}) = \sum_{i=1}^n \hat{y}_i (1 - \hat{y}_i) \mathbf{x}_i \mathbf{x}_i^\top = \mathbf{X}^\top \mathbf{S} \mathbf{X}
$$

This is exactly the matrix that appears in our iterative update rule.

---

## Summary and Outlook

* The update rule

  $$
  \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + (\mathbf{X}^\top \mathbf{S} \mathbf{X})^{-1} \mathbf{X}^\top (\mathbf{y} - \hat{\mathbf{y}}^{(t)})
  $$

  solves a linear system that approximates the zero of the gradient using second-order curvature information.

* The matrix $\mathbf{X}^\top \mathbf{S} \mathbf{X}$ is the **Hessian** of the loss function.

* This makes the method an instance of **Newton’s method** applied to logistic regression.

> 📝 We will revisit the general form and theoretical properties of Newton’s method in a later section. For now, it's enough to know that this procedure uses both gradient and curvature information to efficiently find the optimum.
