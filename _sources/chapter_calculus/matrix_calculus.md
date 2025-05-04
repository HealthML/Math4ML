# Matrix calculus

Since a lot of optimization reduces to finding points where the gradient
vanishes, it is useful to have differentiation rules for matrix and
vector expressions. We give some common rules here. Probably the two
most important for our purposes are 

$$\begin{aligned}
\nabla_\mathbf{x} &(\mathbf{a}^{\!\top\!}\mathbf{x}) = \mathbf{a} \\
\nabla_\mathbf{x} &(\mathbf{x}^{\!\top\!}\mathbf{A}\mathbf{x}) = (\mathbf{A} + \mathbf{A}^{\!\top\!})\mathbf{x}
\end{aligned}$$ 

Note that this second rule is defined only if
$\mathbf{A}$ is square. Furthermore, if $\mathbf{A}$ is symmetric, we
can simplify the result to $2\mathbf{A}\mathbf{x}$.