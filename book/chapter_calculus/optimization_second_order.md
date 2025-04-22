## Second order Optimization Methods

## Newton's Method
**Newton step**

$$\Delta x_{\operatorname{nt}} = -\nabla^2 f(x)^{-1}\nabla f(x)$$

Interpretations:
- $x + \Delta x_{\operatorname{nt}}$ minimizes second order approximation
	$$\hat f(x+v) = f(x) + \nabla f(x)^\top v + \frac{1}{2}v^\top\nabla^2 f(x)v$$
- $x + \Delta x_{\operatorname{nt}}$ solves linearized optimality condition

$$\nabla f(x+v) \approx \nabla \hat f(x+v) = \nabla f(x) + \nabla^2 f(x)v = 0$$

<!-- \includegraphics[width=1.0\textwidth]{figures/15_optimiztion/newton_second_order_min.PNG} -->
<!-- \includegraphics[width=1.0\textwidth]{figures/15_optimiztion/newton_linear_optimality_conditions.PNG} -->

### Newton step

$$\Delta x_{\operatorname{nt}} = -\nabla^2 f(x)^{-1}\nabla f(x)$$

Interpretations:
- $x + \Delta x_{\operatorname{nt}}$ is the steepest descent direction at $x$ in local Hessian norm

$$\|u\|_{\nabla^2 f(x)} = \left(u^\top\nabla^2f(x)u\right)^{1/2}$$

<!-- \includegraphics[width=0.7\textwidth]{figures/15_optimiztion/newton_Hessian_norm.PNG} -->

- dashed lines are contour lines of $f$
- ellipse is $\{x+v | v^\top\nabla^2f(x)v=1\}$
- arrow shows $-\nabla f(x)$

## Newton decrement

$$\lambda(x)=\sqrt{\nabla f(x)^\top\nabla^2 f(x)^{-1}\nabla f(x)}$$

is a measure of proximity of $x$ to $x^\star$

**Properties**
- gives an estimation of $f(x)-f(x^\star)$, using the quadratic approximation $\hat f$:

$$f(x)-\inf_y\hat f(y)=\frac{1}{2}\lambda(x)^2$$

- equal to the norm of the Newton step in the quadratic Hessian norm

$$\lambda(x)=\Delta x_{\operatorname{nt}}\nabla^2 f(x)\Delta x_{\operatorname{nt}}$$

- directional derivative in the Newton direction:

$$\nabla f(x)^\top\Delta x_{\operatorname{nt}}=-\lambda(x)^2$$

- affine invariant (unlike $\|\nabla f(x)\|_2$)

## Newton's method

**given** a starting point $x\in \operatorname{dom}f$, tolerance $\epsilon>0$.

**repeat:**
1. Compute the \term{Newton step} and  \term{Newton decrement}.

$$\Delta x_{\operatorname{nt}}:=-\nabla^2 f(x)^{-1}\nabla f(x); \;\;\lambda^2:=\nabla f(x)\nabla^2 f(x)^{-1}\nabla f(x)$$

2. \term{Stopping criterion}. \emph{quit} if $\frac{\lambda^2}{2}\leq\epsilon$.
3. \term{Line search}. Choose step size $t$ by backtracking line search
4. \term{Update}. $x:=x+t\Delta x_{\operatorname{nt}}$

affine invariant
- independent of linear changes of coordinates
- Newton iterates for $\tilde f(y)=f(Ty)$ with starting point $y^{(0)}=T^{-1}x^{(0)}$ are 
	$$y^{(k)}=T^{-1}x^{(k)}$$

Newton's method requires an invertible Hessian matrix.
- When $\nabla^2 f(x)$ is not p.d., $\Delta x_{\operatorname{nt}}$ may not even be defined.
- Even when it is defined, $\Delta x_{\operatorname{nt}}$ may not be a descent direction, if

$$\nabla f(x)^\top\Delta x_{\operatorname{nt}}<0$$

- In this case, line search methods needs to modify the search direction.