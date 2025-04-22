## Unconstrained Minimization

$$\operatorname{minimize}f(x)$$

- $f$ twice continuously differentiable
- optimal value $p^\star=\inf_x f(x)$ is attained (and finite)


### Unconstrained minimization methods

- produce a sequence of points $x^{(k)}\in\operatorname{dom}f, k=0,1,\dots$ with

$$f(x^{(k)})\to p^{\star}$$

- can be interpreted as iterative methods for solving the optimality condition

$$\nabla f(x^\star)=0$$

**Initial point**
- $x^{(0)}\in\operatorname{dom}f$

### Descent Methods

$$x^{(k+1)} = x^{(k)}+t^{(k)}\Delta x^{(k)} \;\;\; \text{with}\;\;f(x^{(k+1)})<f(x^{(k)})$$

- alternative notation: $x:= x+t \Delta x$
- $\Delta$ is the **step direction**, or **search direction**;  $t$ is the **step size**, or **step length**
- typically, chose $\nabla f(x)^\top\Delta x < 0$ (i.e., $\Delta x$ is a **descent direction**.

## General Descent method:

**given** a starting point $x\in \operatorname{dom}f$.

repeat:
1. Determine **descent direction** $\Delta x$
2. **Line search**. Chose a step size $t>0$.
3. **Update**. $x:=x+t\Delta x$

until stopping criterion is satisfied.

## Gradient Descent

general descent method with \(\Delta x = -\nabla f(x)\)

**given** a starting point $x\in \operatorname{dom}f$.

**repeat:**
1. $\Delta x := -\nabla f(x)$
2. **Line search**. Chose a step size $t$ via exact or backtracking line search.
3.  **Update**. $x:=x+t\Delta x$
**until** stopping criterion is satisfied.


- stopping criterion usually in the form $\|\nabla f(x)\|_2\leq\epsilon$
- very simple, but often very slow.

#### Quadratic problem in $\mathbb{R}^2$

$$f(x) = \frac{1}{2}(x_1^2 + \gamma x_2^2) \;\;\;\;\;(\gamma>0)$$

with exact line search, starting at $x^{(0)} = (\gamma,1)$:

$$x_1^{(k)} = \gamma\left(\frac{\gamma-1}{\gamma+1}\right)^k, \;\;\; x_2 = \left(-\frac{\gamma-1}{\gamma+1}\right)^k$$

1. very slow if $\gamma \gg 1$ or $\gamma \ll 1$.
2. example for $\gamma = 10$:
<!-- \includegraphics[width=0.5\textwidth]{figures/15_optimiztion/quadratic_problem_example.PNG} -->


#### Nonquadratic example

$$f(x) = \exp(x_1 + 3x_2-0.1)  + \exp(x_1-3x_2-0.1) +\exp(-x_1-0.1)$$

<!-- \includegraphics[width=0.7\textwidth]{figures/15_optimiztion/nonquadratic_example.PNG} -->


#### A problem in $\mathbb{R}^{100}$

$$f(x) = c^\top x  -\sum_{i=1}^{500} \log(b_i -a_i^\top x)$$

<!-- \includegraphics[width=0.5\textwidth]{figures/15_optimiztion/R100_example.PNG} -->


`linear' convergence, i.e., a straight line on a semi-log plot

