## Metric spaces

Metrics generalize the notion of distance from Euclidean space (although
metric spaces need not be vector spaces).

A **metric** on a set $S$ is a function $d : S \times S \to \mathbb{R}$
that satisfies

(i) $d(x,y) \geq 0$, with equality if and only if $x = y$

(ii) $d(x,y) = d(y,x)$

(iii) $d(x,z) \leq d(x,y) + d(y,z)$ (the so-called **triangle
      inequality**)

for all $x, y, z \in S$.

A key motivation for metrics is that they allow limits to be defined for
mathematical objects other than real numbers. We say that a sequence
$\{x_n\} \subseteq S$ converges to the limit $x$ if for any
$\epsilon > 0$, there exists $N \in \mathbb{N}$ such that
$d(x_n, x) < \epsilon$ for all $n \geq N$. Note that the definition for
limits of sequences of real numbers, which you have likely seen in a
calculus class, is a special case of this definition when using the
metric $d(x, y) = |x-y|$.
