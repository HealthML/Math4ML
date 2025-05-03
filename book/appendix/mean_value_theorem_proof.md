# Mean value theorem

The Mean Value Theorem (MVT) says that for any smooth curve connecting two points, there is at least one point in between where the instantaneous slope (the derivative) matches the average slope over the whole interval.
The MVT is a special case of the **Fundamental Theorem of Calculus** that links the derivative of a function to its integral.

:::{prf:theorem} Mean value theorem
:label: thm-mean-value-theorem-appendix
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
