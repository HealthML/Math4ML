# The chain rule

Most functions that we wish to optimize are not completely arbitrary
functions, but rather are composed of simpler functions which we know
how to handle. The chain rule gives us a way to calculate derivatives
for a composite function in terms of the derivatives of the simpler
functions that make it up.

The chain rule from single-variable calculus should be familiar:

$$(f \circ g)'(x) = f'(g(x))g'(x)$$ 

where $\circ$ denotes function
composition.

There is a natural generalization of this rule to multivariate functions.

:::{prf:theorem} Multivariate Chain Rule
:label: chain-rule
:nonumber:
Suppose $f : \mathbb{R}^m \to \mathbb{R}^k$ and
$g : \mathbb{R}^n \to \mathbb{R}^m$.

Then $f \circ g : \mathbb{R}^n \to \mathbb{R}^k$ and

$$\mathbf{J}_{f \circ g}(\mathbf{x}) = \mathbf{J}_f(g(\mathbf{x}))\mathbf{J}_g(\mathbf{x})$$
:::

In the special case $k = 1$ we have the following corollary since
$\nabla f = \mathbf{J}_f^{\!\top\!}$.

:::{prf:corollary} Chain Rule for Scalar-Valued Functions
:label: chain-rule-scalar
:nonumber:
Suppose $f : \mathbb{R}^m \to \mathbb{R}$ and
$g : \mathbb{R}^n \to \mathbb{R}^m$. Then
$f \circ g : \mathbb{R}^n \to \mathbb{R}$ and

$$\nabla (f \circ g)(\mathbf{x}) = \mathbf{J}_g(\mathbf{x})^{\!\top\!} \nabla f(g(\mathbf{x}))$$
:::