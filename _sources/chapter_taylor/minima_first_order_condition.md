## Conditions for local minima

proposition
:::{prf:theorem} First-order condition 
:label: thm-first-order-condition
:nonumber:
If $\mathbf{x}^*$ is a local minimum of $f$ and $f$ is continuously
differentiable in a neighborhood of $\mathbf{x}^*$, then
$\nabla f(\mathbf{x}^*) = \mathbf{0}$.
:::

:::{prf:proof}
Let $\mathbf{x}^*$ be a local minimum of $f$, and suppose towards a contradiction that $\nabla f(\mathbf{x}^*) \neq \mathbf{0}$.
Let $\mathbf{h} = -\nabla f(\mathbf{x}^*)$, noting that by the continuity of $\nabla f$ we have

$$\lim_{t \to 0} -\nabla f(\mathbf{x}^* + t\mathbf{h}) = -\nabla f(\mathbf{x}^*) = \mathbf{h}$$

Hence

$$\lim_{t \to 0} \mathbf{h}^{\!\top\!}\nabla f(\mathbf{x}^* + t\mathbf{h}) = \mathbf{h}^{\!\top\!}\nabla f(\mathbf{x}^*) = -\|\mathbf{h}\|_2^2 < 0$$

Thus there exists $T > 0$ such that $\mathbf{h}^{\!\top\!}\nabla f(\mathbf{x}^* + t\mathbf{h}) < 0$ for all $t \in [0,T]$.

Now we apply Taylor's theorem: for any $t \in (0,T]$, there exists $t' \in (0,t)$ such that

$$f(\mathbf{x}^* + t\mathbf{h}) = f(\mathbf{x}^*) + t\mathbf{h}^{\!\top\!} \nabla f(\mathbf{x}^* + t'\mathbf{h}) < f(\mathbf{x}^*)$$

whence it follows that $\mathbf{x}^*$ is not a local minimum, a contradiction.
Hence $\nabla f(\mathbf{x}^*) = \mathbf{0}$. ◻
:::

The proof shows us why the vanishing gradient is necessary for an extremum:
if $\nabla f(\mathbf{x})$ is nonzero, there always exists a sufficiently small step $\alpha > 0$ such that $f(\mathbf{x} - \alpha\nabla f(\mathbf{x}))) < f(\mathbf{x})$.
For this reason, $-\nabla f(\mathbf{x})$ is called a **descent direction**.

Points where the gradient vanishes are called **stationary points**.

Note that not all stationary points are extrema.
Consider $f : \mathbb{R}^2 \to \mathbb{R}$ given by $f(x,y) = x^2 - y^2$. We have
$\nabla f(\mathbf{0}) = \mathbf{0}$, but the point $\mathbf{0}$ is the
minimum along the line $y = 0$ and the maximum along the line $x = 0$.
Thus it is neither a local minimum nor a local maximum of $f$. Points
such as these, where the gradient vanishes but there is no local
extremum, are called **saddle points**.