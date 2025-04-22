## Taylor's theorem

Taylor's theorem has natural generalizations to functions of more than
one variable.
We give the version presented in [@numopt].

:::{prf:theorem} Taylor theorem
:label: thm-taylor-theorem-numopt
:nonumber:
Suppose $f : \mathbb{R}^d \to \mathbb{R}$ is
continuously differentiable, and let $\mathbf{h} \in \mathbb{R}^d$.
Then there exists $t \in (0,1)$ such that

$$f(\mathbf{x} + \mathbf{h}) = f(\mathbf{x}) + \nabla f(\mathbf{x} + t\mathbf{h})^{\!\top\!}\mathbf{h}$$

Furthermore, if $f$ is twice continuously differentiable, then

$$\nabla f(\mathbf{x} + \mathbf{h}) = \nabla f(\mathbf{x}) + \int_0^1 \nabla^2 f(\mathbf{x} + t\mathbf{h})\mathbf{h} \operatorname{d}{t}$$

and there exists $t \in (0,1)$ such that

$$f(\mathbf{x} + \mathbf{h}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^{\!\top\!}\mathbf{h} + \frac{1}{2}\mathbf{h}^{\!\top\!}\nabla^2f(\mathbf{x}+t\mathbf{h})\mathbf{h}$$
:::

This theorem is used in proofs about conditions for local minima of
unconstrained optimization problems. Some of the most important results
are given in the next section.



