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

### Line Search

#### Line Search Types

##### exact line search: 
$t=\operatorname{argmin}_{t>0} f(x+t \Delta x)$

##### backtracking line search
(with parameters $\alpha\in(0,1/2), \beta\in(0,1)$)
- starting at $t=1$, repeat $t:=\beta t$ until

$$f(x+t\Delta x)<f(x) + \alpha t\nabla f(x)^\top \Delta x$$

- graphical interpretation: backtrack until $t\leq t_0$

<!-- \includegraphics[width=0.7\textwidth]{figures/15_optimiztion/backtracking_line_search.PNG} -->

#### Inexact line search
In the unconstrained minimization problem, the **Wolfe conditions** are a set of inequalities for performing *inexact line search*, especially in quasi-Newton methods.

Each step often involves approximately solving the subproblem

$$\min_{\alpha} f(\mathbf{x}_k + \alpha \Delta_k)$$

where $\mathbf{x}_k$ is the current best guess, $\Delta_k \in \mathbb{R}^n$ is a search direction, and $\alpha \in \mathbb{R}$ is the step length.

The inexact line searches provide an efficient way of computing an acceptable step length $\alpha$ that reduces $f$ \emph{sufficiently}.

A line search algorithm can use **Wolfe conditions** as a requirement for any guessed $\alpha$, before finding a new search direction $\Delta_k$.

A step length $\alpha_k$ is said to satisfy the **Wolfe conditions**, restricted to the direction $\Delta_k$, if the following two inequalities hold:


i)  $f(\mathbf{x}_k+\alpha_k\Delta_k)\leq f(\mathbf{x}_k) + c_1\alpha_k \Delta_k^{\mathrm T} \nabla f(\mathbf{x}_k)$

ii)  $-\Delta_k^\top\nabla f(\mathbf{x}_k+\alpha_k\Delta_k) \leq -c_2\Delta_k^\top\nabla f(\mathbf{x}_k)$

where $c_1$ and $c_2$ are constants in $(0,1)$, i.e., $0<c_1<c_2<1$.


(In examining condition (ii), recall that to ensure that $\Delta_k$ is a descent direction, we have $\Delta_k^\top\nabla f(\mathbf{x}_k) < 0$.)

$c_1$ is usually chosen to be quite small while $c_2$ is much larger; 

<!-- % Nocedal<ref>{{cite book | title = Numerical Optimization | last1=Nocedal |first1=Jorge | last2=Wright |first2=Stephen | url = https://books.google.com/books?id=epc5fX0lqRIC&lpg=PP1&pg=PA38#v=onepage&q | year=1999}}</ref> gives example values of <math>c_1=10^{-4}</math> -->
<!-- % and <math>c_2=0.9</math> for Newton or quasi-Newton methods and <math>c_2=0.1</math> for the nonlinear [[conjugate gradient method]]. Inequality i) is known as the '''Armijo rule'''<ref>{{cite journal | last =  Armijo | first = Larry | year = 1966 | title = Minimization of functions having Lipschitz continuous first partial derivatives | journal = Pacific J. Math. | volume = 16 | issue = 1 | pages = 1–3 | url = http://projecteuclid.org/euclid.pjm/1102995080 | doi=10.2140/pjm.1966.16.1}}</ref> and ii) as the '''curvature condition'''; i) ensures that the step length <math>\alpha_k</math> decreases <math>f</math>  'sufficiently', and ii) ensures that the slope has been reduced sufficiently. Conditions i) and ii) can be interpreted as respectively providing an upper and lower bound on the admissible step length values. -->

Denote a univariate function $\varphi$ restricted to the direction $\Delta_k$ as $\varphi(\alpha)=f(\mathbf{x}_k+\alpha\Delta_k)$. 

The Wolfe conditions can result in a value for the step length that is not close to a minimizer of $\varphi$. 

If we modify the curvature condition to the following,

$$ \textbf{iii)} \quad \big|\Delta_k^{\mathrm T}\nabla f(\mathbf{x}_k+\alpha_k\Delta_k)\big|\leq c_2\big|\Delta_k^{\mathrm T}\nabla f(\mathbf{x}_k)\big|$$

then i) and iii) together form the so-called \term{strong Wolfe conditions}, and force $\alpha_k$ to lie close to a **critical point** of $\varphi$.

Rationale:

The principal reason for imposing the Wolfe conditions in an optimization algorithm where  $\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha \Delta_k$ is to ensure convergence of the gradient to zero.  

In particular, if the cosine of the angle between $\Delta_k$ and the gradient,

$$\cos \theta_k = \frac {\nabla f(\mathbf{x}_k)^{\mathrm T}\Delta_k }{\| \nabla f(\mathbf{x}_k)\| \|\Delta_k\| } $$

is bounded away from zero and the i) and ii) conditions hold, then $\nabla f(\mathbf{x}_k) \rightarrow 0$.


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
