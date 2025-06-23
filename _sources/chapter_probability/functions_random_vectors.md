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
# Functions of Random Vectors

Just as a function applied to a univariate random variable results in a new random variable, applying a function to a random **vector** yields a new random **variable** (or vector).

Let $\mathbf{X} \in \mathbb{R}^n$ be a random vector, and let $f : \mathbb{R}^n \to \mathbb{R}^m$ be a measurable function. 

Then the composition

$$
\mathbf{Y} = f(\mathbf{X})
$$

defines a new random vector $\mathbf{Y} \in \mathbb{R}^m$.

### Scalar-Valued Functions

If $f : \mathbb{R}^n \to \mathbb{R}$, then $Y = f(\mathbf{X})$ is a scalar random variable.


### Example: Sum of Dice Rolls

Let $\mathbf{X} = \begin{bmatrix} X_1 \\ X_2 \end{bmatrix}$, where $X_1, X_2$ are the results of rolling two fair six-sided dice. 

Each component has values in $\{1, 2, 3, 4, 5, 6\}$.

Define a new random variable:

$$
Y = f(\mathbf{X}) = X_1 + X_2
$$

Then $Y$ is a random variable whose possible values are $\{2, 3, \dots, 12\}$, and the distribution of $Y$ can be computed by **counting** all combinations $(x_1, x_2)$ such that $x_1 + x_2 = y$. 

For example:

* $\mathbb{P}(Y=2) = \mathbb{P}(X_1=1, X_2=1) = \frac{1}{36}$
* $\mathbb{P}(Y=7) = \frac{6}{36}$
* $\mathbb{P}(Y=12) = \frac{1}{36}$

So even though $Y$ is not a component of $\mathbf{X}$, it is still a random variable **induced by** a function of $\mathbf{X}$.

### Vector-Valued Functions

If $f : \mathbb{R}^n \to \mathbb{R}^m$, then $f(\mathbf{X})$ is a random vector in $\mathbb{R}^m$.

This is especially useful when modeling **nonlinear transformations** of multivariate data.


---

### Example: Vector-Valued Transformation

Let $\mathbf{X} = \begin{bmatrix} X_1 \\ X_2 \end{bmatrix}$, where $X_1$ and $X_2$ are independent random variables uniformly distributed on $[0,1]$. 

Define a function:

$$
f(\mathbf{x}) = \begin{bmatrix} x_1 + x_2 \\ x_1 \cdot x_2 \end{bmatrix}
$$

Then $\mathbf{Y} = f(\mathbf{X})$ is a **random vector** in $\mathbb{R}^2$, whose first component is the **sum** of $X_1$ and $X_2$, and whose second component is the **product**. 

The distribution of $\mathbf{Y}$ is induced by pushing forward the joint distribution of $(X_1, X_2)$ through $f$.

### Change of Variables (Jacobian Rule)

If $f : \mathbb{R}^n \to \mathbb{R}^n$ is **invertible and differentiable**, then the distribution of $\mathbf{Y} = f(\mathbf{X})$ can be computed using the **Jacobian determinant**:

$$
p_{\mathbf{Y}}(\mathbf{y}) = p_{\mathbf{X}}(f^{-1}(\mathbf{y})) \cdot \left| \det \left( \frac{\partial f^{-1}}{\partial \mathbf{y}} \right) \right|
$$

or equivalently, using the Jacobian of $f$:

$$
p_{\mathbf{Y}}(\mathbf{y}) = p_{\mathbf{X}}(\mathbf{x}) \cdot \left| \det \left( \frac{\partial f}{\partial \mathbf{x}} \right)^{-1} \right|, \quad \text{where } \mathbf{y} = f(\mathbf{x})
$$

---

Let’s compute the distribution of $\mathbf{Y} = f(\mathbf{X}) = \begin{bmatrix} X_1 + X_2 \\ X_1 \cdot X_2 \end{bmatrix}$ using the **change-of-variables formula** for a transformation of continuous random variables.

---

## Step 1: Define the input distribution

Let $X_1, X_2 \sim \text{Uniform}[0,1]$, and independent.

Then the joint density of $\mathbf{X} = (X_1, X_2)$ is

$$
p_{\mathbf{X}}(x_1, x_2) = 1 \quad \text{for } 0 \le x_1, x_2 \le 1
$$

---

## Step 2: Define the transformation

Let:

$$
\begin{aligned}
Y_1 &= f_1(x_1, x_2) = x_1 + x_2 \\
Y_2 &= f_2(x_1, x_2) = x_1 x_2
\end{aligned}
$$

We need to find the joint density $p_{\mathbf{Y}}(y_1, y_2)$. 

To do so, we must:

1. Express $(x_1, x_2)$ in terms of $(y_1, y_2)$
2. Compute the determinant of the Jacobian of the inverse transformation

---

## Step 3: Invert the transformation

We solve for $x_1, x_2$ from:

$$
\begin{aligned}
y_1 &= x_1 + x_2 \\
y_2 &= x_1 x_2
\end{aligned}
$$

This gives the quadratic equation:

$$
x^2 - y_1 x + y_2 = 0
$$

with solutions:

$$
x_1, x_2 = \frac{y_1 \pm \sqrt{y_1^2 - 4y_2}}{2}
$$

This only yields real values if $y_1^2 \ge 4y_2$. 

Let’s denote the roots as:

$$
x_1 = \frac{y_1 + \sqrt{y_1^2 - 4y_2}}{2}, \quad x_2 = \frac{y_1 - \sqrt{y_1^2 - 4y_2}}{2}
$$

or vice versa (symmetry doesn't matter since the Jacobian determinant will handle it).

---

## Step 4: Jacobian determinant

We compute the Jacobian matrix of $f$:

$$
J_f =
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2}
\end{bmatrix}
=
\begin{bmatrix}
1 & 1 \\
x_2 & x_1
\end{bmatrix}
$$

So the determinant is:

$$
\det J_f = x_1 - x_2
$$

We then get the change-of-variables density:

$$
p_{\mathbf{Y}}(y_1, y_2) = \sum_{\text{roots}} p_{\mathbf{X}}(x_1, x_2) \cdot \left| \det J_f(x_1, x_2) \right|^{-1}
$$

Because $p_{\mathbf{X}}(x_1, x_2) = 1$ in the domain $[0,1]^2$, we must restrict to the region where both $x_1, x_2 \in [0,1]$, and hence where $y_1 \in [0,2]$, $y_2 \in [0,1]$, and $y_1^2 \ge 4 y_2$.

Thus, the final density is:

$$
p_{\mathbf{Y}}(y_1, y_2) =
\begin{cases}
\frac{1}{|x_1 - x_2|}, & \text{if } 0 \le x_1, x_2 \le 1 \text{ and } x_1 + x_2 = y_1,\ x_1 x_2 = y_2 \\
0, & \text{otherwise}
\end{cases}
$$

We can now write this entirely in terms of $y_1$ and $y_2$, using:

$$
|x_1 - x_2| = \sqrt{y_1^2 - 4y_2}
$$

So:

$$
p_{\mathbf{Y}}(y_1, y_2) =
\begin{cases}
\frac{2}{\sqrt{y_1^2 - 4y_2}}, & \text{if } 0 \le x_1, x_2 \le 1 \text{ and } y_1^2 \ge 4 y_2 \\
0, & \text{otherwise}
\end{cases}
$$

The factor 2 accounts for the two symmetric roots (since $(x_1,x_2)$ and $(x_2,x_1)$ both map to the same $(y_1,y_2)$).

---



