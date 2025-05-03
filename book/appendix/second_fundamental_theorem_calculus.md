# second fundamental theorem of calculus

We provide astandard proof of the second part of the Fundamental Theorem of Calculus II (FTC II):

:::{prf:theorem} Fundamental Theorem of Calculus II
:label: thm-ftc-ii-appendix
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