# First Fundamental Theorem of Calculus

:::{prf:theorem} Fundamental Theorem of Calculus I
:label: thm-ftc-i-appendix
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