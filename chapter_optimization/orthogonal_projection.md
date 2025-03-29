## Orthogonal projections

We now consider a particular kind of optimization problem that is
particularly well-understood and can often be solved in closed form:
given some point $\mathbf{x}$ in an inner product space $V$, find the
closest point to $\mathbf{x}$ in a subspace $S$ of $V$. This process is
referred to as **projection onto a subspace**.

The following diagram should make it geometrically clear that, at least
in Euclidean space, the solution is intimately related to orthogonality
and the Pythagorean theorem:

 ::: center
![image](../figures/orthogonal-projection.png)
:::

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

*Proposition.*
Suppose $\mathbf{x} \in V$ and $\mathbf{y} \in S$. Then $\mathbf{y}^*$
is the unique minimizer of $\|\mathbf{x}-\mathbf{y}\|$ over
$\mathbf{y} \in S$ if and only if $\mathbf{x}-\mathbf{y}^* \perp S$.



*Proof.* $(\implies)$ Suppose $\mathbf{y}^*$ is the unique minimizer of
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

*Proposition.*
If $\mathbf{e}_1, \dots, \mathbf{e}_m$ is an orthonormal basis for $S$,
then

$$P\mathbf{x} = \sum_{i=1}^m \langle \mathbf{x}, \mathbf{e}_i \rangle\mathbf{e}_i$$


*Proof.* Let $\mathbf{e}_1, \dots, \mathbf{e}_m$ be an orthonormal basis
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


The fact that $P$ is a linear operator (and thus a proper projection, as
earlier we showed $P^2 = P$) follows readily from this result.
