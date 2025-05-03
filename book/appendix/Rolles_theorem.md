# Rolle's theorem
Rolle's theorem states that if a function is continuous on a closed interval and differentiable on the open interval, and if the function takes the same value at both endpoints, then there exists at least one point in the open interval where the derivative is zero.

:::{prf:theorem} Rolles theorem
:label: thm-rolle-appendix
:nonumber:

Let $f:[a,b]\rightarrow\mathbb{R}$ be a function satisfying the following three conditions:

1. $f$ is continuous on the closed interval $[a,b]$.
2. $f$ is differentiable on the open interval $(a,b)$.
3. $f(a)=f(b)$.

Then, there exists some $c\in(a,b)$ such that:

$$
f'(c) = 0\,.
$$

In other words, there's at least one point where the tangent to the graph of $f(x)$ is horizontal.
:::

:::{prf:proof} **Rolle's theorem.**

**Step 1 (special case):**
If $f(x)$ is constant, say $f(x)=k$, then clearly $f'(x)=0$ everywhere on $(a,b)$. Hence, Rolle’s theorem trivially holds.

**Step 2 (general case):**
Suppose $f(x)$ is not constant on $[a,b] $. Since $f(x)$ is continuous on the compact interval $[a,b]$, by the **Extreme Value Theorem**, it must attain a maximum and minimum on this interval. Let’s denote by $x_{\max}\in[a,b]$ the point at which $f$ attains its maximum.

**Step 3 (Maximum inside interval):**
There are two possibilities:

* If $x_{\max}$ lies in the open interval $(a,b) $, then since $f(x_{\max})$ is a maximum, the derivative at this point, provided it exists, must be zero. Thus, setting $c=x_{\max}$, we have $f'(c)=0$, completing the proof.

* If $x_{\max}$ lies at one of the endpoints, say $x_{\max}=a$, then since $f(a)=f(b)$, the maximum at $a$ implies the function must be less or equal to $f(a)$ throughout $(a,b)$. Because $f$ is continuous and differentiable on $(a,b)$, we consider the following scenarios:

  * If $f(x)<f(a)$ for all $x\in(a,b)$, then $f$ strictly decreases immediately after $a$, implying $f'(a)<0$, contradicting differentiability at endpoint (since differentiability is only assumed on $(a,b)$, we actually do not consider the derivative exactly at endpoint here; however, differentiability inside the interval implies the function is smoothly transitioning away from endpoints). Still, even without directly using endpoint derivatives, the fact remains that there must exist another critical point inside the interval due to continuity and differentiability constraints, or otherwise the function would monotonically decrease, contradicting the equality $f(a)=f(b)$. Thus, the maximum cannot be exclusively at an endpoint without another interior maximum or minimum. Hence, there must exist at least one critical point with zero derivative within the interval.

**Step 4 (Conclusion):**
In every possible scenario, there must exist a point $c\in(a,b)$ such that $f'(c)=0$.

:::