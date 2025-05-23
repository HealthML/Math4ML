# Convex sets

::: center
![image](../figures/convex-set.png)
A convex set
:::

::: center
![image](../figures/nonconvex-set.png)
A non-convex set
:::

A set $\mathcal{X} \subseteq \mathbb{R}^d$ is **convex** if

$$t\mathbf{x} + (1-t)\mathbf{y} \in \mathcal{X}$$

for all
$\mathbf{x}, \mathbf{y} \in \mathcal{X}$ and all $t \in [0,1]$.

Geometrically, this means that all the points on the line segment
between any two points in $\mathcal{X}$ are also in $\mathcal{X}$.

See Figure [1](#fig:convexset){reference-type="ref"
reference="fig:convexset"} for a visual.

Why do we care whether or not a set is convex? We will see later that the nature of minima can depend greatly on whether or not the feasible set is convex.
Undesirable pathological results can occur when we allow
the feasible set to be arbitrary, so for proofs we will need to assume that it is convex. 

Fortunately, we often want to minimize over all of
$\mathbb{R}^d$, which is easily seen to be a convex set.

