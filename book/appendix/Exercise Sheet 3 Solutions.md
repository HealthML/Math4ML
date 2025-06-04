# Exercise Sheet 3 Solutions

### 1.
#### (a)
Let

$$
f : \mathbb{R}^2 \to \mathbb{R}, \quad f(x, y) = 9x^2 - y^3 + 9xy
$$

We are asked to compute the **Hessian matrix** at the point $ (x, y) = (3, -3) $.


*Step 1: Compute second-order partial derivatives*

To compute the Hessian matrix, we first compute the first-order partial derivatives:

$$
\frac{\partial f}{\partial x} = 18x + 9y, \quad
\frac{\partial f}{\partial y} = -3y^2 + 9x
$$

Then we compute the second-order partial derivatives:

$$
\frac{\partial^2 f}{\partial x^2} = 18, \quad
\frac{\partial^2 f}{\partial y^2} = -6y, \quad
\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x} = 9
$$

At the point $ (3, -3) $, we evaluate:

$$
\frac{\partial^2 f}{\partial x^2}(3, -3) = 18, \quad
\frac{\partial^2 f}{\partial y^2}(3, -3) = -6(-3) = 18, \quad
\frac{\partial^2 f}{\partial x \partial y}(3, -3) = 9
$$

*Step 2: Form the Hessian matrix*

```math
H_{(3, -3)} =
\begin{bmatrix}
\frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\
\frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2}
\end{bmatrix}
=
\begin{bmatrix}
18 & 9 \\
9 & 18
\end{bmatrix}
```

#### (b)
We recall the following definitions and propositions:

- **Definition**: A symmetric matrix $ A $ is **positive definite** if for all non-zero vectors $ a $, we have $ a^T A a > 0 $.
- **Proposition**: A symmetric matrix is positive definite **if and only if** all its **eigenvalues** are positive.


To compute the eigenvalues, solve the characteristic equation:

$$
\det(H - \lambda I) = 0
\Rightarrow
\begin{vmatrix}
18 - \lambda & 9 \\
9 & 18 - \lambda
\end{vmatrix}
= (18 - \lambda)^2 - 81 = 0
$$

Simplifying:

$$
(18 - \lambda)^2 = 81 \Rightarrow 18 - \lambda = \pm 9
\Rightarrow \lambda = 9, \ 27
$$


Since both eigenvalues are **positive**, the Hessian matrix at the point $ (3, -3) $ is **positive definite**. Therefore, $ f(x, y) $ has a **local minimum** at this point.


### 2.

Let $ f(x) = x \cdot \ln(x) $ defined on the interval $ [1, e^2] $.

#### (a)
To apply the Mean Value Theorem (MVT), we must verify that:

- $ f $ is **continuous** on $ [1, e^2] $
- $ f $ is **differentiable** on $ (1, e^2) $

Since $ f(x) = x \ln(x) $ is a product of continuous and differentiable functions for $ x > 0 $, both conditions are satisfied.


#### (b)

We compute:

$$
f(e^2) = e^2 \cdot \ln(e^2) = e^2 \cdot 2 = 2e^2
$$

$$
f(1) = 1 \cdot \ln(1) = 0
$$

Hence, the average rate of change is:

$$
\frac{f(e^2) - f(1)}{e^2 - 1} = \frac{2e^2}{e^2 - 1}
$$

Next, compute the derivative:

$$
f'(x) = \frac{d}{dx}[x \ln(x)] = \ln(x) + 1
$$

We solve:

$$
f'(c) = \ln(c) + 1 = \frac{2e^2}{e^2 - 1}
\Rightarrow \ln(c) = \frac{2e^2}{e^2 - 1} - 1 = \frac{e^2 + 1}{e^2 - 1}
\Rightarrow c = \exp\left( \frac{e^2 + 1}{e^2 - 1} \right)
$$

#### (c)
The Mean Value Theorem states that there exists a point $ c \in (1, e^2) $ where the **instantaneous rate of change** $ f'(c) $ equals the **average rate of change** over the interval:

$$
f'(c) = \frac{f(e^2) - f(1)}{e^2 - 1}
$$

Geometrically, this means the **tangent line** to the curve at $ x = c $ is **parallel** to the **secant line** connecting the endpoints $ (1, f(1)) $ and $ (e^2, f(e^2)) $.


### 3.
#### (a)

We compute the derivatives at $ x = 0 $:

- $ f(x) = \arctan(x) $
- $ f'(x) = \frac{1}{1+x^2} \Rightarrow f'(0) = 1 $
- $ f''(x) = \frac{-2x}{(1+x^2)^2} \Rightarrow f''(0) = 0 $
- $ f^{(3)}(x) = \frac{6x^2 - 2}{(1 + x^2)^3} \Rightarrow f^{(3)}(0) = -2 $

Hence, the third-degree Taylor polynomial is:

$$
P_3(x) = f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \frac{f^{(3)}(0)}{3!}x^3
= 0 + x + 0 - \frac{2}{6}x^3 = x - \frac{1}{3}x^3
$$

#### (b)
The Lagrange remainder is:

$$
R_3(x) = \frac{f^{(4)}(c)}{4!} x^4 = \frac{f^{(4)}(c)}{24} x^4 \quad \text{for some } c \in (0, x)
$$

We previously found:

$$
f^{(4)}(c) = \frac{24c(1 - c^2)}{(1 + c^2)^4}
$$

So,

$$
R_3(x) = \frac{c(1 - c^2)}{(1 + c^2)^4} x^4 \quad \text{for some } c \in (0, x)
$$

#### (c)

Our goal is to show:

```math
|R_3(x)| = \left| \frac{c(1 - c^2)}{(1 + c^2)^4} x^4 \right| \leq \frac{x^4}{4(1 + c^2)^2}
\quad \text{for } c \in (0, 1)
```

We simplify the absolute value of the remainder:

```math
|R_3(x)| = \frac{c(1 - c^2)}{(1 + c^2)^4} x^4
```

So we now want to **prove** that:

```math
\frac{c(1 - c^2)}{(1 + c^2)^4} \leq \frac{1}{4(1 + c^2)^2}
```

Now we multiply both sides by $ (1 + c^2)^4 $ (which is strictly positive):

```math
4c(1 - c^2) \leq (1 + c^2)^2
```

Now expand both sides:

**Left-hand side:**
```math
4c(1 - c^2) = 4c - 4c^3
```

**Right-hand side:**
```math
(1 + c^2)^2 = 1 + 2c^2 + c^4
```

So we want to show:
```math
4c - 4c^3 \leq 1 + 2c^2 + c^4
\quad \Leftrightarrow \quad
c^4 + 4c^3 + 2c^2 - 4c + 1 \geq 0
```

Now factor the left-hand side:
```math
c^4 + 4c^3 + 2c^2 - 4c + 1 = (c^2 + 2c - 1)^2 \geq 0
```

This inequality clearly holds for all c including $ c \in (0, 1) $, so the bound is valid.
