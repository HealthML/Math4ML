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
# Orthogonal projections

We now consider a particular kind of optimization problem that is
particularly well-understood and can often be solved in closed form:
given some point $\mathbf{x}$ in an inner product space $V$, find the
closest point to $\mathbf{x}$ in a subspace $S$ of $V$. This process is
referred to as **projection onto a subspace**.

The following diagram should make it geometrically clear that, at least
in Euclidean space, the solution is intimately related to orthogonality
and the Pythagorean theorem:

```{code-cell} ipython3
:tags: [hide-input]
# Re-import required packages due to kernel reset
import numpy as np
import matplotlib.pyplot as plt

# Define subspace S spanned by vector e1
e1 = np.array([1, 2])
e1 = e1 / np.linalg.norm(e1)  # normalize to make it orthonormal

# Define arbitrary point x not in the subspace
x = np.array([2, 1])

# Compute projection of x onto the subspace spanned by e1
x_proj = np.dot(x, e1) * e1

# Define a second point y in the subspace (for triangle)
y = 3 * e1

# Set up plot
fig, ax = plt.subplots(figsize=(6, 6))

# Draw vectors
origin = np.array([0, 0])
ax.quiver(*origin, *x, angles='xy', scale_units='xy', scale=1, color='blue', label=r'$\mathbf{x}$')
ax.quiver(*origin, *x_proj, angles='xy', scale_units='xy', scale=1, color='green', label=r'$\mathbf{y}^* = P\mathbf{x}$')
ax.quiver(*origin, *y, angles='xy', scale_units='xy', scale=1, color='gray', alpha=0.5, label=r'$\mathbf{y} \in S$')

# Draw dashed lines to form triangle
ax.plot([x[0], x_proj[0]], [x[1], x_proj[1]], 'k--', lw=1)
ax.plot([y[0], x[0]], [y[1], x[1]], 'k--', lw=1)
ax.plot([y[0], x_proj[0]], [y[1], x_proj[1]], 'k--', lw=1)

# Annotate
ax.text(*(x + 0.2), r'$\mathbf{x}$', fontsize=12)
ax.text(*(x_proj + 0.2), r'$\mathbf{y}^*$', fontsize=12)
ax.text(*(y + 0.2), r'$\mathbf{y}$', fontsize=12)

# Draw subspace line
line_extent = np.linspace(-3, 3, 100)
s_line = np.outer(line_extent, e1)
ax.plot(s_line[:, 0], s_line[:, 1], 'r-', lw=1, label=r'Subspace $S$')

# Formatting
ax.set_xlim(-1, 4)
ax.set_ylim(-1, 4)
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
ax.set_title(r"Orthogonal Projection of $\mathbf{x}$ onto Subspace $S$")

plt.tight_layout()
plt.show()
```
In this diagram, the blue vector $\mathbf{x}$ is an arbitrary point in the
inner product space $V$, the green vector $\mathbf{y}^* = P\mathbf{x}$ is
the projection of $\mathbf{x}$ onto the subspace $S$, and the gray vector
$\mathbf{y}$ is an arbitrary point in $S$. The dashed lines form a right
triangle with $\mathbf{x}$, $\mathbf{y}^*$, and $\mathbf{y}$ as vertices.
The right triangle formed by these three points illustrates the
relationship between the projection and orthogonality: the line segment
from $\mathbf{x}$ to $\mathbf{y}^*$ is perpendicular to the subspace $S$,
and the distance from $\mathbf{x}$ to $\mathbf{y}^*$ is the shortest
distance from $\mathbf{x}$ to any point in $S$. This is a direct
consequence of the Pythagorean theorem, which states that in a right
triangle, the square of the length of the hypotenuse (in this case,
$\|\mathbf{x}-\mathbf{y}\|$) is equal to the sum of the squares of the
lengths of the other two sides (in this case, $\|\mathbf{x}-\mathbf{y}^*\|$ and $\|\mathbf{y}^*-\mathbf{y}\|$).


Here $\mathbf{y}$ is an arbitrary element of the subspace $S$, and
$\mathbf{y}^*$ is the point in $S$ such that $\mathbf{x}-\mathbf{y}^*$
is perpendicular to $S$. The hypotenuse of a right triangle (in this
case $\|\mathbf{x}-\mathbf{y}\|$) is always longer than either of the
legs (in this case $\|\mathbf{x}-\mathbf{y}^*\|$ and
$\|\mathbf{y}^*-\mathbf{y}\|$), and when $\mathbf{y} \neq \mathbf{y}^*$
there always exists such a triangle between $\mathbf{x}$, $\mathbf{y}$,
and $\mathbf{y}^*$.

Our intuition from Euclidean space suggests that the closest point to
$\mathbf{x}$ in $S$ has the perpendicularity property described above,
and we now show that this is indeed the case.

:::{prf:proposition}
:label: prop-unique-minimizer
:nonumber:
Suppose $\mathbf{x} \in V$ and $\mathbf{y} \in S$. Then $\mathbf{y}^*$
is the unique minimizer of $\|\mathbf{x}-\mathbf{y}\|$ over
$\mathbf{y} \in S$ if and only if $\mathbf{x}-\mathbf{y}^* \perp S$.
:::


:::{prf:proof}
$(\implies)$ Suppose $\mathbf{y}^*$ is the unique minimizer of
$\|\mathbf{x}-\mathbf{y}\|$ over $\mathbf{y} \in S$. That is,
$\|\mathbf{x}-\mathbf{y}^*\| \leq \|\mathbf{x}-\mathbf{y}\|$ for all
$\mathbf{y} \in S$, with equality only if $\mathbf{y} = \mathbf{y}^*$.
Fix $\mathbf{v} \in S$ and observe that 

$$\begin{aligned}
g(t) &:= \|\mathbf{x}-\mathbf{y}^*+t\mathbf{v}\|^2 \\
&= \langle \mathbf{x}-\mathbf{y}^*+t\mathbf{v}, \mathbf{x}-\mathbf{y}^*+t\mathbf{v} \rangle \\
&= \langle \mathbf{x}-\mathbf{y}^*, \mathbf{x}-\mathbf{y}^* \rangle - 2t\langle \mathbf{x}-\mathbf{y}^*, \mathbf{v} \rangle + t^2\langle \mathbf{v}, \mathbf{v} \rangle \\
&= \|\mathbf{x}-\mathbf{y}^*\|^2 - 2t\langle \mathbf{x}-\mathbf{y}^*, \mathbf{v} \rangle + t^2\|\mathbf{v}\|^2
\end{aligned}$$ 

must have a minimum at $t = 0$ as a consequence of this
assumption. Thus

$$0 = g'(0) = \left.-2\langle \mathbf{x}-\mathbf{y}^*, \mathbf{v} \rangle + 2t\|\mathbf{v}\|^2\right|_{t=0} = -2\langle \mathbf{x}-\mathbf{y}^*, \mathbf{v} \rangle$$

giving $\mathbf{x}-\mathbf{y}^* \perp \mathbf{v}$. Since $\mathbf{v}$
was arbitrary in $S$, we have $\mathbf{x}-\mathbf{y}^* \perp S$ as
claimed.

$(\impliedby)$ Suppose $\mathbf{x}-\mathbf{y}^* \perp S$. Observe that
for any $\mathbf{y} \in S$, $\mathbf{y}^*-\mathbf{y} \in S$ because
$\mathbf{y}^* \in S$ and $S$ is closed under subtraction. Under the
hypothesis, $\mathbf{x}-\mathbf{y}^* \perp \mathbf{y}^*-\mathbf{y}$, so
by the Pythagorean theorem,

$$\|\mathbf{x}-\mathbf{y}\| = \|\mathbf{x}-\mathbf{y}^*+\mathbf{y}^*-\mathbf{y}\| = \|\mathbf{x}-\mathbf{y}^*\| + \|\mathbf{y}^*-\mathbf{y}\| \geq \|\mathbf{x} - \mathbf{y}^*\|$$

and in fact the inequality is strict when $\mathbf{y} \neq \mathbf{y}^*$
since this implies $\|\mathbf{y}^*-\mathbf{y}\| > 0$. Thus
$\mathbf{y}^*$ is the unique minimizer of $\|\mathbf{x}-\mathbf{y}\|$
over $\mathbf{y} \in S$. ◻
:::

Since a unique minimizer in $S$ can be found for any $\mathbf{x} \in V$,
we can define an operator

$$P\mathbf{x} = \operatorname{argmin}_{\mathbf{y} \in S} \|\mathbf{x}-\mathbf{y}\|$$

Observe that $P\mathbf{y} = \mathbf{y}$ for any $\mathbf{y} \in S$,
since $\mathbf{y}$ has distance zero from itself and every other point
in $S$ has positive distance from $\mathbf{y}$. Thus
$P(P\mathbf{x}) = P\mathbf{x}$ for any $\mathbf{x}$ (i.e., $P^2 = P$)
because $P\mathbf{x} \in S$. The identity $P^2 = P$ is actually one of
the defining properties of a **projection**, the other being linearity.

An immediate consequence of the previous result is that
$\mathbf{x} - P\mathbf{x} \perp S$ for any $\mathbf{x} \in V$, and
conversely that $P$ is the unique operator that satisfies this property
for all $\mathbf{x} \in V$. For this reason, $P$ is known as an
**orthogonal projection**.

If we choose an orthonormal basis for the target subspace $S$, it is
possible to write down a more specific expression for $P$.

:::{prf:proposition}
:label: prop-orthonormal-basis-projection
:nonumber:

If $\mathbf{e}_1, \dots, \mathbf{e}_m$ is an orthonormal basis for $S$,
then

$$P\mathbf{x} = \sum_{i=1}^m \langle \mathbf{x}, \mathbf{e}_i \rangle\mathbf{e}_i$$
:::


:::{prf:proof} 
Let $\mathbf{e}_1, \dots, \mathbf{e}_m$ be an orthonormal basis
for $S$, and suppose $\mathbf{x} \in V$. Then for all $j = 1, \dots, m$,

$$\begin{aligned}
\left\langle \mathbf{x}-\sum_{i=1}^m \langle \mathbf{x}, \mathbf{e}_i \rangle\mathbf{e}_i, \mathbf{e}_j \right\rangle &= \langle \mathbf{x}, \mathbf{e}_j \rangle - \sum_{i=1}^m \langle \mathbf{x}, \mathbf{e}_i \rangle\underbrace{\langle \mathbf{e}_i, \mathbf{e}_j \rangle}_{\delta_{ij}} \\
&= \langle \mathbf{x}, \mathbf{e}_j \rangle - \langle \mathbf{x}, \mathbf{e}_j \rangle \\
&= 0
\end{aligned}$$ 

We have shown that the claimed expression, call it
$\tilde{P}\mathbf{x}$, satisfies
$\mathbf{x} - \tilde{P}\mathbf{x} \perp \mathbf{e}_j$ for every element
$\mathbf{e}_j$ of the orthonormal basis for $S$. It follows (by
linearity of the inner product) that
$\mathbf{x} - \tilde{P}\mathbf{x} \perp S$, so the previous result
implies $P = \tilde{P}$. ◻
:::

The fact that $P$ is a linear operator (and thus a proper projection, as
earlier we showed $P^2 = P$) follows readily from this result.
