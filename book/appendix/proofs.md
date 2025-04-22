# Detailed Proofs  

Here we list more detailed proofs of theorems and lemmas that are not included in the main text.


::: {prf:theorem} Cauchy–Schwarz Inequality
For all $\mathbf{x}, \mathbf{y} \in V$,

$$
|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \|\mathbf{x}\| \cdot \|\mathbf{y}\|,
$$
with equality if and only if $\mathbf{x}$ and $\mathbf{y}$ are linearly dependent.
:::

::: {prf:proof}
Consider any scalar $t \in \mathbb{R}$. By the properties of inner products, the norm squared of the vector $\mathbf{x} - t\mathbf{y}$ is non-negative:

$$
\|\mathbf{x} - t\mathbf{y}\|^2 = \langle \mathbf{x} - t\mathbf{y}, \mathbf{x} - t\mathbf{y} \rangle \ge 0.
$$
Expanding the inner product using linearity gives:

$$
\|\mathbf{x} - t\mathbf{y}\|^2 = \langle \mathbf{x}, \mathbf{x} \rangle - 2t \langle \mathbf{x}, \mathbf{y} \rangle + t^2 \langle \mathbf{y}, \mathbf{y} \rangle = \|\mathbf{x}\|^2 - 2t \langle \mathbf{x}, \mathbf{y} \rangle + t^2\|\mathbf{y}\|^2.
$$
This expression is a quadratic function in $t$:

$$
f(t) = \|\mathbf{x}\|^2 - 2t \langle \mathbf{x}, \mathbf{y} \rangle + t^2\|\mathbf{y}\|^2.

$$
Since $f(t) \ge 0$ for all $t \in \mathbb{R}$, the quadratic must have a non-positive discriminant. The discriminant is:

$$
\Delta = (-2 \langle \mathbf{x}, \mathbf{y} \rangle)^2 - 4\|\mathbf{x}\|^2 \|\mathbf{y}\|^2 = 4\langle \mathbf{x}, \mathbf{y} \rangle^2 - 4\|\mathbf{x}\|^2 \|\mathbf{y}\|^2.
$$
Thus, we have:

$$
4\langle \mathbf{x}, \mathbf{y} \rangle^2 - 4\|\mathbf{x}\|^2 \|\mathbf{y}\|^2 \le 0.
$$
Dividing both sides by 4 yields:

$$
\langle \mathbf{x}, \mathbf{y} \rangle^2 \le \|\mathbf{x}\|^2 \|\mathbf{y}\|^2.
$$
Taking the square root of both sides (and noting that both norms are nonnegative) gives the desired inequality:

$$
|\langle \mathbf{x}, \mathbf{y} \rangle| \le \|\mathbf{x}\| \cdot \|\mathbf{y}\|.
$$

Equality holds if and only if the discriminant $\Delta = 0$, which happens exactly when there exists a scalar $t$ such that $\mathbf{x} = t\,\mathbf{y}$, i.e., when $\mathbf{x}$ and $\mathbf{y}$ are linearly dependent.  
◻
:::

The **discriminant** is a term that comes from the study of quadratic equations. When you have a quadratic equation in the form

$$
at^2 + bt + c = 0,
$$

the discriminant is defined as

$$
\Delta = b^2 - 4ac.
$$

The value of the discriminant tells you important information about the solutions (or roots) of the quadratic equation:

- **If $\Delta > 0$:** There are two distinct real roots.
- **If $\Delta = 0$:** There is exactly one real root (a repeated or double root).
- **If $\Delta < 0$:** There are no real roots; instead, the roots are complex conjugates.

In the context of the Cauchy–Schwarz inequality proof, we examined the quadratic function

$$
f(t) = \|\mathbf{x}\|^2 - 2t \langle \mathbf{x}, \mathbf{y} \rangle + t^2\|\mathbf{y}\|^2,
$$

which represents the squared norm $\|\mathbf{x} - t\mathbf{y}\|^2$ and is nonnegative for all real $t$. For this quadratic function to be nonnegative for all $t$, its discriminant must be non-positive (i.e., $\Delta \leq 0$); otherwise, the quadratic would cross the horizontal axis, implying negative values for some $t$. In our derivation, after computing the discriminant

$$
\Delta = 4\langle \mathbf{x}, \mathbf{y} \rangle^2 - 4\|\mathbf{x}\|^2 \|\mathbf{y}\|^2,
$$

requiring $\Delta \le 0$ gives us the inequality

$$
\langle \mathbf{x}, \mathbf{y} \rangle^2 \leq \|\mathbf{x}\|^2 \|\mathbf{y}\|^2,
$$

which is exactly the Cauchy–Schwarz inequality after taking square roots.

Thus, the discriminant in this context helps us conclude that the quadratic cannot have two distinct real roots (which would indicate a region where the function dips below zero) and thus must satisfy the inequality.