# Detailed Proofs  

Here we list more detailed proofs of theorems and lemmas that are not included in the main text.

::: {prf:theorem} Cauchy–Schwarz Inequality
For all $\mathbf{x}, \mathbf{y} \in V$,

$$
|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \|\mathbf{x}\| \cdot \|\mathbf{y}\|,
$$
with equality if and only if $\mathbf{x}$ and $\mathbf{y}$ are linearly dependent.
:::

::: {prf:proof}
Consider any scalar $t \in \mathbb{R}$. By the properties of inner products, the norm squared of the vector $\mathbf{x} - t\mathbf{y}$ is non-negative:

$$
\|\mathbf{x} - t\mathbf{y}\|^2 = \langle \mathbf{x} - t\mathbf{y}, \mathbf{x} - t\mathbf{y} \rangle \ge 0.
$$
Expanding the inner product using linearity gives:

$$
\|\mathbf{x} - t\mathbf{y}\|^2 = \langle \mathbf{x}, \mathbf{x} \rangle - 2t \langle \mathbf{x}, \mathbf{y} \rangle + t^2 \langle \mathbf{y}, \mathbf{y} \rangle = \|\mathbf{x}\|^2 - 2t \langle \mathbf{x}, \mathbf{y} \rangle + t^2\|\mathbf{y}\|^2.
$$
This expression is a quadratic function in $t$:

$$
f(t) = \|\mathbf{x}\|^2 - 2t \langle \mathbf{x}, \mathbf{y} \rangle + t^2\|\mathbf{y}\|^2.

$$
Since $f(t) \ge 0$ for all $t \in \mathbb{R}$, the quadratic must have a non-positive discriminant. The discriminant is:

$$
\Delta = (-2 \langle \mathbf{x}, \mathbf{y} \rangle)^2 - 4\|\mathbf{x}\|^2 \|\mathbf{y}\|^2 = 4\langle \mathbf{x}, \mathbf{y} \rangle^2 - 4\|\mathbf{x}\|^2 \|\mathbf{y}\|^2.
$$
Thus, we have:

$$
4\langle \mathbf{x}, \mathbf{y} \rangle^2 - 4\|\mathbf{x}\|^2 \|\mathbf{y}\|^2 \le 0.
$$
Dividing both sides by 4 yields:

$$
\langle \mathbf{x}, \mathbf{y} \rangle^2 \le \|\mathbf{x}\|^2 \|\mathbf{y}\|^2.
$$
Taking the square root of both sides (and noting that both norms are nonnegative) gives the desired inequality:

$$
|\langle \mathbf{x}, \mathbf{y} \rangle| \le \|\mathbf{x}\| \cdot \|\mathbf{y}\|.
$$

Equality holds if and only if the discriminant $\Delta = 0$, which happens exactly when there exists a scalar $t$ such that $\mathbf{x} = t\,\mathbf{y}$, i.e., when $\mathbf{x}$ and $\mathbf{y}$ are linearly dependent.  
◻
:::

The **discriminant** is a term that comes from the study of quadratic equations. When you have a quadratic equation in the form

$$
at^2 + bt + c = 0,
$$

the discriminant is defined as

$$
\Delta = b^2 - 4ac.
$$

The value of the discriminant tells you important information about the solutions (or roots) of the quadratic equation:

- **If $\Delta > 0$:** There are two distinct real roots.
- **If $\Delta = 0$:** There is exactly one real root (a repeated or double root).
- **If $\Delta < 0$:** There are no real roots; instead, the roots are complex conjugates.

In the context of the Cauchy–Schwarz inequality proof, we examined the quadratic function

$$
f(t) = \|\mathbf{x}\|^2 - 2t \langle \mathbf{x}, \mathbf{y} \rangle + t^2\|\mathbf{y}\|^2,
$$

which represents the squared norm $\|\mathbf{x} - t\mathbf{y}\|^2$ and is nonnegative for all real $t$. For this quadratic function to be nonnegative for all $t$, its discriminant must be non-positive (i.e., $\Delta \leq 0$); otherwise, the quadratic would cross the horizontal axis, implying negative values for some $t$. In our derivation, after computing the discriminant

$$
\Delta = 4\langle \mathbf{x}, \mathbf{y} \rangle^2 - 4\|\mathbf{x}\|^2 \|\mathbf{y}\|^2,
$$

requiring $\Delta \le 0$ gives us the inequality

$$
\langle \mathbf{x}, \mathbf{y} \rangle^2 \leq \|\mathbf{x}\|^2 \|\mathbf{y}\|^2,
$$

which is exactly the Cauchy–Schwarz inequality after taking square roots.

Thus, the discriminant in this context helps us conclude that the quadratic cannot have two distinct real roots (which would indicate a region where the function dips below zero) and thus must satisfy the inequality.


::: {prf:theorem} scalar-scalar chain rule
Let $f: \mathbb{R} \to \mathbb{R}$ and $g: \mathbb{R} \to \mathbb{R}$ be differentiable functions. If $f$ is differentiable at $u_0 = g(x_0)$ and $g$ is differentiable at $x_0$, then the composition $( f \circ g)(x) = f\bigl(g(x)\bigr)$ is differentiable at $x_0$, and we have

$$
( f \circ g)'(x_0) = f'(g(x_0)) \cdot g'(x_0).
$$

where $\circ$ denotes function composition.

where $\circ$ denotes function composition.
:::

:::{prf:proof} **Scalar–Scalar Chain Rule.**  

Let $g$ be differentiable at $x_0$ and $f$ be differentiable at $u_0=g(x_0)$.  
Define

$$
h(x)=f\bigl(g(x)\bigr).
$$
Then

$$
h'(x_0)
=\lim_{\Delta x\to0}\frac{h(x_0+\Delta x)-h(x_0)}{\Delta x}
=\lim_{\Delta x\to0}\frac{f\bigl(g(x_0+\Delta x)\bigr)-f\bigl(g(x_0)\bigr)}{\Delta x}.
$$
Set 

$$
\Delta u = g(x_0+\Delta x)-g(x_0),
$$
so that $\Delta u\to0$ and $\tfrac{\Delta u}{\Delta x}\to g'(x_0)$ by differentiability of $g$.  We now write

$$
\frac{f\bigl(g(x_0+\Delta x)\bigr)-f\bigl(g(x_0)\bigr)}{\Delta x}
\;=\;
\frac{f(u_0+\Delta u)-f(u_0)}{\Delta u}
\;\times\;
\frac{\Delta u}{\Delta x}.
$$
By the **Mean Value Theorem** applied to $f$ on the interval $[u_0,u_0+\Delta u]$, there exists some $\xi$ between $u_0$ and $u_0+\Delta u$ so that

$$
\frac{f(u_0+\Delta u)-f(u_0)}{\Delta u}
= f'(\xi).
$$
As $\Delta x\to0$, we have $\xi\to u_0$, and hence $f'(\xi)\to f'(u_0)$.  Therefore

$$
h'(x_0)
=\lim_{\Delta x\to0}
\underbrace{\frac{f(u_0+\Delta u)-f(u_0)}{\Delta u}}_{\to f'(u_0)}
\;\times\;
\underbrace{\frac{\Delta u}{\Delta x}}_{\to g'(x_0)}
= f'(g(x_0))\,g'(x_0).
$$
This completes the proof of the Chain Rule.
◻
:::

## Mean value theorem

The Mean Value Theorem (MVT) says that for any smooth curve connecting two points, there is at least one point in between where the instantaneous slope (the derivative) matches the average slope over the whole interval.
The MVT is a special case of the **Fundamental Theorem of Calculus** that links the derivative of a function to its integral.

:::{prf:theorem} Mean value theorem
:label: thm-mean-value-theorem
:nonumber:

Let $f:[a,b]\to\mathbb{R}$ be a continuous function on the closed interval $[a,b]$, and differentiable on the open interval $(a,b)$, where $a\neq b$.

Then there exists some $c \in (a,b)$ such that

$$f'(c)=\frac{f(b)-f(a)}{b-a}.$$
:::

In other words, the tangent line at $x=c$ is exactly parallel to the secant line joining $(a,f(a))$ and $(b,f(b))$.

:::{prf:proof} Mean value theorem

1. **Construct an auxiliary function.**  
   Define
   
   $$
     g(x) \;=\; f(x)\;-\;\Bigl(\underbrace{\tfrac{f(b)-f(a)}{b-a}}_{m}\Bigr)\,(x-a)\,.
   $$
   By construction, $g$ is continuous on $[a,b]$ and differentiable on $(a,b)$, and
   
   $$
     g(a) = f(a) - m\,(a-a) = f(a), 
     \quad
     g(b) = f(b) - m\,(b-a) = f(a).
   $$
   Hence $g(a)=g(b)$.

2. **Apply Rolle’s theorem.**  
   Rolle’s theorem states that if a function $g$ is continuous on $[a,b]$, differentiable on $(a,b)$, and $g(a)=g(b)$, then there exists $c\in(a,b)$ with $g'(c)=0$.  

3. **Compute $g'(x)$.**  

   $$
     g'(x) \;=\; f'(x) \;-\; m.
   $$
   Rolle’s theorem gives $g'(c)=0$, so $f'(c)-m=0$, i.e.

   $$
     f'(c) = m = \frac{f(b)-f(a)}{b-a}.
   $$
   This completes the proof.

:::

## Fundamental Theorem of Calculus

**First Fundamental Theorem of Calculus**  

:::{prf:theorem} Fundamental Theorem of Calculus I
:label: thm-ftc-i
:nonumber:

**Theorem (FTC I).**  
Let $f$ be continuous on $[a,b]$, and define

$$
F(x)\;=\;\int_{a}^{\,x} f(t)\,\mathrm{d}t,\quad x\in[a,b].
$$
Then $F$ is differentiable on $(a,b)$ and  

$$
F'(x) \;=\; f(x)
\quad\text{for every }x\in(a,b).
$$
:::

:::{prf:proof} **Fundamental Theorem of Calculus I.**

Let $f$ be continuous on $[a,b]$, and define

$$
F(x)\;=\;\int_{a}^{\,x} f(t)\,\mathrm{d}t,\quad x\in[a,b].
$$
We will show that $F$ is differentiable on $(a,b)$ and that $F'(x)=f(x)$ for every $x\in(a,b)$.
To show that $F$ is differentiable, we need to compute the difference quotient

Fix any point $x\in(a,b)$.  We compute the difference quotient for $F$ at $x$:

$$
\frac{F(x+h)-F(x)}{h}
\;=\;
\frac{1}{h}\Bigl(\int_a^{\,x+h}f(t)\,\mathrm{d}t \;-\;\int_a^{\,x}f(t)\,\mathrm{d}t\Bigr)
\;=\;
\frac{1}{h}\,\int_{x}^{\,x+h} f(t)\,\mathrm{d}t.
$$

Because $f$ is continuous at $x$, on the tiny interval $[x,x+h]$ it attains both a minimum $m_h$ and a maximum $M_h$, and these both converge to $f(x)$ as $h\to0$.  
Thus

$$
m_h \,\le\, \frac{1}{h}\int_{x}^{x+h}f(t)\,\mathrm{d}t \,\le\, M_h,
$$
and since $m_h\to f(x)$ and $M_h\to f(x)$, the Squeeze Theorem gives

$$
\lim_{h\to0}\frac{F(x+h)-F(x)}{h}
\;=\;
f(x).
$$
In other words, $F'(x)=f(x)$.  Because $x$ was arbitrary in $(a,b)$, $F$ is differentiable there with $F'=f$.  This completes the proof. ◻
:::

We provide astandard proof of the second part of the Fundamental Theorem of Calculus II (FTC II):

:::{prf:theorem} Fundamental Theorem of Calculus II
:label: thm-ftc-ii
:nonumber:

Let $f$ be continuous on $[a,b]$, and suppose $F$ is an antiderivative of $f$ there; that is, $F'(x)=f(x)$ for all $x\in[a,b]$.  Then

$$
\int_{a}^{b} f(x)\,\mathrm{d}x \;=\; F(b)\;-\;F(a).
$$

:::

:::{prf:proof} **Fundamental Theorem of Calculus II.**
Let $F$ be an antiderivative of $f$ on $[a,b]$.  We will show that

$$
  \int_a^b f(x)\,\mathrm{d}x \;=\; F(b) - F(a).
$$

This is equivalent to showing that the difference $F(b)-F(a)$ is equal to the Riemann integral $\int_a^b f(x)\,\mathrm{d}x$.

To do this, we will use the following steps:
1. **Partition the interval.**  
   Choose any partition

   $$
     a = x_0 < x_1 < \dots < x_{n-1} < x_n = b
   $$
   of $[a,b]$, and in each subinterval $[x_{i-1},x_i]$ pick an arbitrary sample point $\xi_i \in [x_{i-1},x_i]$.

2. **Riemann sum for $\int_a^b f$.**  
   Because $f$ is continuous, the Riemann sums

   $$
     \sum_{i=1}^n f(\xi_i)\,\bigl(x_i - x_{i-1}\bigr)
   $$
   converge (as the mesh $\max_i(x_i - x_{i-1})\to0$) to $\int_a^b f(x)\,\mathrm{d}x$.

3. **Mean value on each subinterval.**  
   On each $[x_{i-1},x_i]$, apply the ordinary mean value theorem to $F$: since $F$ is differentiable, there exists $c_i\in(x_{i-1},x_i)$ with
   
   $$
     F(x_i)-F(x_{i-1}) \;=\; F'(c_i)\,\bigl(x_i - x_{i-1}\bigr)
     \;=\; f(c_i)\,\bigl(x_i - x_{i-1}\bigr).
   $$

4. **Compare sums.**  
   Thus the telescoping sum

   $$
     F(b)-F(a)
     \;=\;
     \sum_{i=1}^n \bigl[F(x_i)-F(x_{i-1})\bigr]
     \;=\;
     \sum_{i=1}^n f(c_i)\,(x_i - x_{i-1}).
   $$
   Notice each $c_i$ lies in the corresponding subinterval, just like the Riemann sample points $\xi_i$.

5. **Refine the partition.**  
   As we let the mesh of the partition go to zero, continuity of $f$ implies that both sums

   $$
     \sum_{i=1}^n f(c_i)\,(x_i - x_{i-1})
     \quad\text{and}\quad
     \sum_{i=1}^n f(\xi_i)\,(x_i - x_{i-1})
   $$
   converge to the same limit, namely $\int_a^b f(x)\,\mathrm{d}x$.  But the first sum is exactly $F(b)-F(a)$ for every partition.

6. **Conclude.**  
   Therefore

   $$
     F(b)-F(a)
     \;=\;
     \lim_{\text{mesh}\to0}\sum_{i=1}^n f(c_i)\,(x_i - x_{i-1})
     \;=\;
     \int_a^b f(x)\,\mathrm{d}x,
   $$
   which completes the proof. ◻

:::

**Key idea:** by slicing $[a,b]$ into tiny pieces, on each little piece the average rate of change of $F$ equals $f$ at some interior point; summing those up exactly telescopes to $F(b)-F(a)$, and the same sums approximate the integral $\int_a^b f$.