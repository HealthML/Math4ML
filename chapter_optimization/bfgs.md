## Quasi Newton methods

quadratic model of the objective at iterate $x^k$:

$$m_k(p)=f_k+\nabla f_k^\top p + \frac{1}{2}p^\top B_k p$$

Here, $B_k$ is an $n$ by $n$ symmetric p.d. matrix that will be updated at every iteration.

The search direction $\Delta_{\operatorname{qn}}= -B_k^{-1}\nabla f_k$ and the next iterate is $x^k+\alpha\Delta_{\operatorname{qn}}$.

$$m_{k+1}(p)=f_{k+1}+\nabla f_{k+1}^\top p + \frac{1}{2}p^\top B_{k+1} p$$

We would like to avoid computing $B_{k+1}$ afresh after the update.

Instead, put constraints on function approximation $m_{k+1}$:
1. gradient matches at current iterate $x^{k+1}$. \onslide<+->(trivially fullfilled)

$$\nabla m_{k+1}(0) = \nabla f(x^{k+1})$$ 

2.  gradient matches at last iterate $x^{k}$. (before step)\onslide<+->

$$\underbrace{\nabla m_{k+1}(-\alpha_k\Delta^{(k)})}_{=\nabla f_{k+1}-\alpha_k B_{k+1}\Delta^{(k)}} =\nabla f(x^{k})$$ 

Rearranging, we obtain

$$B_{k+1}\alpha_k\Delta^{(k)} = \nabla f_{k+1} - \nabla f_k$$

to simplify, we introduce

$$s_k=x^{k+1}-x^k = \alpha_k\Delta^{(k)}\;\;\;\text{and}\;\;y_k=\nabla f_{k+1} - \nabla f_k$$

So that we get the \term{secant equation}

$$B_{k+1}s_k = y_k$$

Additionally, we require the \term{curvature condition} for the steps (via line search for $\alpha_k$)

$$s_k^\top y_k>0$$

Adding and subtracting $\nabla^2 f(x)p$ from Taylor's theorem:

$$\nabla f (x+p) = \nabla f(x) + \nabla^2 f(x)p + \underbrace{\int_0^1 \left[ \nabla^2 f(x+tp) - \nabla^2 f(x)\right]p \operatorname{d} t}_{o(\|p\|)} $$

Setting $x=x^{(k)}$ and $p=x^{(k+1)}-x^{(k)}$, we obtain

$$\nabla f_{k+1} = \nabla f_k + \nabla^2 f_k(x^{(k+1)}-x^{(k)}) + o(\|x^{(k+1)}-x^{(k)}\|) $$

When $x^{(k)}$ and $x^{(k+1)}$ lie near a solution $x^\star$, within  which $\nabla^2f$ is p.d., this expansion is dominated by $ \nabla^2 f_k(x^{(k+1)}-x^{(k)})$

Thus,

$$\nabla^2f_k(x^{(k+1)}-x^{(k)})\approx \nabla f_{k+1} - \nabla f_k$$

Idea: Choose Hessian approximation $B_{k+1}$, such that it fulfills the \term{secant equation}:% $(B_k s_k=y_k)$:

$$B_{k+1} \underbrace{(x^{(k+1)}-x^{(k)})}_{s_k} = \underbrace{\nabla f_{k+1} - \nabla f_k}_{y_k}$$


### SR1
**Symmetric Rank 1**

**secant equation**: $(B_k s_k=y_k)$:

$$B_{k+1} \underbrace{(x^{(k+1)}-x^{(k)})}_{s_k} = \underbrace{\nabla f_{k+1} - \nabla f_k}_{y_k}$$

The \term{SR1} update formula for to the Hessian approximation is

$$B_{k+1} = B_k + \frac{(y_k - B_ks_k)(y_k - B_ks_k)^\top}{(y_k - B_ks_k)^\top s_k}$$

In each iteration, this formula performs a symmetric rank 1 update to the Hessian approximation.

If $B_k$ is p.d., then $B_{k+1}$ is also p.d.

### BFGS

**secant equation**: $(B_k s_k=y_k)$:

$$B_{k+1} \underbrace{(x^{(k+1)}-x^{(k)})}_{s_k} = \underbrace{\nabla f_{k+1} - \nabla f_k}_{y_k}$$

The \term{BFGS} update formula for to the Hessian approximation is

$$B_{k+1} = B_k - \frac{B_k s_k s_k^\top B_k}{s_k^\top B_k s_k} + \frac{y_ky_k^\top}{y_k^\top s_k}$$

In each iteration, this formula performs a symmetric rank 2 update to the Hessian approximation.

If $B_k$ is p.d., the Wolfe condition ii) implies $B_{k+1}$ is also p.d.

The descent direction in quasi-Newton methods is obtained by using $B_k$ instead of $\nabla^2 f$

$$\Delta x_{\operatorname{qn}} = -B^{-1}\nabla f_k$$

Note that practical implementations of the Newton method typically update the inverse approximation $H_k:=B_k^{-1}$ using the \term{ Woodbury matrix identity} for computing the inverse of a matrix $A$ after a rank-$k$ update $UCV$

$$\left(A+UCV^\top\right)^{-1} = A^{-1} - A^{-1}U\left(C^{-1} + VA^{-1}U\right)VA^{-1}$$

where

- $U$ is $N$ by $k$
- $C$ is $k$ by $k$ and invertible
- $V$ is $k$ by $N$ 


## LBFGS

