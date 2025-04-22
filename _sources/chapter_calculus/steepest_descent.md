## Steepest Descent


**Normalized steepest descent direction** (at $x$, for norm $\|\dot\|$):

$$\Delta x_{\operatorname{nsd}} = \operatorname{argmin}\{\nabla f(x)^\top v | \|v\| = 1\}$$

interpretation: for small $v$, $f(x+)\approx f(x)+\nabla f(x)^\top v$;

direction $\Delta x_{\operatorname{nsd}}$ is unit-norm step with most negative directional derivative

**(unnormalized) steepest descent direction**

$$\Delta x_{\operatorname{sd}} = \|\nabla f(x)\|_* \Delta x_{\operatorname{nsd}}$$

satisfies $\nabla f(x)^\top\Delta x_{\operatorname{sd}}=-\|\Delta f(x)\|^2_*$

**steepest descent method**
- general descent method with $\Delta x = \Delta x_{\operatorname{sd}}$
- Covergence properties similar to gradient descent


#### Examples


- Euclidean norm: $\Delta x_{\operatorname{sd}} = -\nabla f(x)$
- quadratic norm $\|x\|_P = \left(x^\top P x\right)^{1/2}$, with symmetric and positive definite $n$ by $n$ matrix $P$: $\Delta x_{\operatorname{sd}} = -P^{-1}\nabla f(x) $
- $\ell_1$-norm: $\Delta x_{\operatorname{sd}} = -\frac{\partial f(x)}{\partial x_i} e_i$, where $|\frac{\partial f(x)}{\partial x_i} | = \|\nabla f(x)\|_\infty$

unit balls and normalized steepest descent directions for

quadratic norm

<!-- \includegraphics[width=0.5\textwidth]{figures/15_optimiztion/example_quadratic_norm.PNG} -->

$\ell_1$-norm

<!-- \includegraphics[width=0.5\textwidth]{figures/15_optimiztion/example_l1_norm.PNG} -->


### Choice of norm for steepest descent

<!-- \includegraphics[width=0.8\textwidth]{figures/15_optimiztion/example_choice_quadratic_norm.PNG} -->
- steepest descent with backtracking line search for two quadratic norms
- ellipses show $\{x| \|x-x^{(k)}\|_P = 1\}$
- equivalent interpretation of steepest descent with quadratic norm $\|\dot\|_P$:

gradient descent after change of variables $\tilde x =P^{1/2}x$

shows that choice of $P$ has strong effect on speed of convergence
