# Symmetry of Mixed Partial Derivatives (Clairaut’s Theorem)

:::{prf:theorem} Clairaut Schwarz
:label: thm-Clairaut-appendix
:nonumber:

Let $f: \mathbb{R}^2 \to \mathbb{R}$ be a function such that both mixed partial derivatives $\frac{\partial^2 f}{\partial x \partial y}$ and $\frac{\partial^2 f}{\partial y \partial x}$ exist and are **continuous** on an open set containing a point $(x_0, y_0)$

Then:

$$
\boxed{
\frac{\partial^2 f}{\partial x \partial y}(x_0, y_0) = \frac{\partial^2 f}{\partial y \partial x}(x_0, y_0)
}
$$

That is, **the order of differentiation can be interchanged**.
:::

## Intuition

If a function is smooth enough (specifically, if the second-order partial derivatives exist and are continuous), then the "curvature" in the $x$ direction after differentiating in the $y$ direction is the same as the curvature in the $y$ direction after differentiating in the $x$ direction.

---

## Proof Sketch

We will sketch a proof using the **mean value theorem** and the definition of partial derivatives. Let’s assume that $f$ has continuous second partial derivatives in an open rectangle around the point $(x_0, y_0)$.

Define:

$$
F(h,k) = \frac{f(x_0 + h, y_0 + k) - f(x_0 + h, y_0) - f(x_0, y_0 + k) + f(x_0, y_0)}{hk}
$$

Then, as $h, k \to 0$, $F(h,k) \to \frac{\partial^2 f}{\partial y \partial x}(x_0, y_0)$ and also $F(h,k) \to \frac{\partial^2 f}{\partial x \partial y}(x_0, y_0)$, provided the second partial derivatives are continuous.

### Step-by-step:

1. By the **Mean Value Theorem**, the numerator of $F(h,k)$ can be interpreted as a finite difference approximation to a mixed partial derivative.
2. Using Taylor’s Theorem with remainder, or via integral representations of derivatives, one can show that:

   $$
   \lim_{(h,k) \to (0,0)} F(h,k) = \frac{\partial^2 f}{\partial x \partial y}(x_0, y_0)
   $$

   and also

   $$
   \lim_{(h,k) \to (0,0)} F(h,k) = \frac{\partial^2 f}{\partial y \partial x}(x_0, y_0)
   $$

   due to continuity of the second derivatives.
3. Hence, the limits agree and the mixed partials are equal.

Therefore:

$$
\frac{\partial^2 f}{\partial x \partial y}(x_0, y_0) = \frac{\partial^2 f}{\partial y \partial x}(x_0, y_0)
$$

---

## When Clairaut's Theorem **Does Not Apply**

If the second-order mixed partial derivatives exist but are **not continuous**, the symmetry may fail. A classic counterexample is:

$$
f(x, y) =
\begin{cases}
\frac{xy(x^2 - y^2)}{x^2 + y^2}, & \text{if } (x, y) \neq (0, 0) \\
0, & \text{if } (x, y) = (0, 0)
\end{cases}
$$

This function has both mixed partial derivatives at the origin, but they are not equal because they are not continuous there.

