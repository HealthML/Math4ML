# The Chain Rule for Scalar-Scalar Functions

The **Chain Rule** is a fundamental theorem in calculus that describes how to differentiate composite functions. It states that if you have two functions, $f$ and $g$, and you want to differentiate their composition $f(g(x))$, you can do so by multiplying the derivative of $f$ evaluated at $g(x)$ by the derivative of $g$ evaluated at $x$.
This is particularly useful when dealing with functions that are composed of other functions, as it allows us to break down the differentiation process into manageable parts. 

::: {prf:theorem} scalar-scalar chain rule
:label: thm-scalar-scalar-chain-rule-appendix
:nonumber:

Let $f: \mathbb{R} \to \mathbb{R}$ and $g: \mathbb{R} \to \mathbb{R}$ be differentiable functions. If $f$ is differentiable at $u_0 = g(x_0)$ and $g$ is differentiable at $x_0$, then the composition $( f \circ g)(x) = f\bigl(g(x)\bigr)$ is differentiable at $x_0$, and we have

$$
( f \circ g)'(x_0) = f'(g(x_0)) \cdot g'(x_0).
$$

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
so that $\Delta u\to0$ and $\tfrac{\Delta u}{\Delta x}\to g'(x_0)$ by differentiability of $g$.

We now write

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
As $\Delta x\to0$, we have $\xi\to u_0$, and hence $f'(\xi)\to f'(u_0)$.

Therefore

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
