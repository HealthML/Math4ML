### Subspaces

Vector spaces can contain other vector spaces.
If $V$ is a vector space, then $S \subseteq V$ is said to be a **subspace** of $V$ if

(i) $\mathbf{0} \in S$

(ii) $S$ is closed under addition: $\mathbf{x}, \mathbf{y} \in S$
     implies $\mathbf{x}+\mathbf{y} \in S$

(iii) $S$ is closed under scalar multiplication:
      $\mathbf{x} \in S, \alpha \in \mathbb{R}$ implies
      $\alpha\mathbf{x} \in S$

Note that $V$ is always a subspace of $V$, as is the trivial vector
space which contains only $\mathbf{0}$.

As a concrete example, a line passing through the origin is a subspace
of Euclidean space.

Some of the most important subspaces are those induced by linear maps.
If $T : V \to W$ is a linear map, we define the **nullspace** (or **kernel**) of $T$
as

$\operatorname{null}(T) = \{\mathbf{x} \in V \mid T\mathbf{x} = \mathbf{0}\}.$

and
the **range** (or the **columnspace** if we are considering the matrix
form) of $T$ as

$\operatorname{range}(T) = \{\mathbf{y} \in W \mid \exists \mathbf{x} \in V : T\mathbf{x} = \mathbf{y}\}.$

It is a good exercise to verify that the nullspace and range of a linear
map are always subspaces of its domain and codomain, respectively.