# Extreme Value Theorem
The Extreme Value Theorem states that if a function is continuous on a closed interval, then it attains both a maximum and a minimum value on that interval. This theorem is crucial in optimization problems, as it guarantees the existence of optimal solutions within bounded domains.

:::{prf:theorem} Extreme Value Theorem
:label: thm-extreme-value-theorem-appendix
:nonumber:

If a function $f:[a,b] \to \mathbb{R}$ is continuous on a closed and bounded interval $[a,b]$, then $f$ attains both a minimum and a maximum value on this interval. That is, there exist points $x_{\min}, x_{\max} \in [a,b]$ such that:

$$
f(x_{\min}) \leq f(x) \leq f(x_{\max}), \quad \text{for all } x \in [a,b].
$$

:::


:::{prf:proof} **Extreme Value Theorem.**

The proof relies on two important properties: continuity and compactness (closedness and boundedness) of the interval.

### **Step 1: Compactness of the interval**

Consider the interval $[a,b]\subset\mathbb{R}$.

* It is **bounded**, because $a$ and $b$ are finite.
* It is **closed**, because it contains its endpoints.

Thus, the interval $[a,b]$ is **compact** (closed and bounded).

### **Step 2: Boundedness of the continuous function on a compact interval**

* Assume for contradiction that the continuous function $f$ is not bounded above.
* Then, for every integer $n$, there exists some $x_n \in [a,b]$ with $f(x_n) > n$. This defines a sequence $(x_n)\subseteq[a,b]$.
* Since $[a,b]$ is compact, by the **Bolzano–Weierstrass theorem**, the sequence $(x_n)$ has a convergent subsequence $(x_{n_k})$ converging to some limit $x^*\in[a,b]$.

By continuity of $f$:

$$
\lim_{k\to\infty} f(x_{n_k}) = f(x^*).
$$

But we know $f(x_{n_k})\to\infty$ as $k\to\infty$, contradicting the fact that the limit must be finite. Hence, $f$ must be bounded above.

The argument for boundedness below is similar: if $f$ were unbounded below, we would similarly reach a contradiction. Thus, $f$ is bounded both above and below.

### **Step 3: Existence of maximum and minimum**

Let’s first prove existence of a maximum value (the proof for minimum is symmetric):

* Define the supremum of the set $f([a,b])$:

$$
M = \sup\{f(x): x \in [a,b]\}.
$$

Since $f$ is bounded above, $M$ is finite.

* To show $f$ attains the value $M$, assume for contradiction that it does not. Then, for all $x\in[a,b]$, we have $f(x)<M$. Define the function $g(x)=\frac{1}{M - f(x)}$. Clearly, $g$ is continuous on $[a,b]$ (because the denominator is never zero by our assumption).

* Since $[a,b]$ is compact and $g$ is continuous, $g$ must be bounded. However, as $f(x)$ approaches $M$, the denominator $M - f(x)$ approaches zero, making $g(x)$ grow without bound. Thus, there must exist a sequence $x_n$ in $[a,b]$ such that $f(x_n)\to M$.

* Again, by compactness (Bolzano–Weierstrass theorem), there exists a subsequence $x_{n_k}\to x^*\in[a,b]$. By continuity of $f$, we have:

$$
f(x^*)=\lim_{k\to\infty}f(x_{n_k})=M.
$$

This shows that the function indeed attains its maximum at some point $x^*\in[a,b]$.

By symmetry, $f$ also attains its minimum in a similar manner.

:::