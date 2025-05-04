# Bolzano-Weierstrass theorem
Below is a rigorous statement and proof of the **Bolzano–Weierstrass theorem**:

:::{prf:theorem} Bolzano–Weierstrass theorem
:label: thm-bolzano-weierstrass-appendix
:nonumber:

Every bounded sequence $(x_n)\subseteq\mathbb{R}$ has a convergent subsequence.

Formally: if $(x_n)$ is bounded, there exist a subsequence $(x_{n_k})$ and a limit $ x^*\in\mathbb{R}$ such that:

$$
\lim_{k\to\infty} x_{n_k} = x^*.
$$

:::

:::{prf:proof} **Bolzano–Weierstrass theorem.**

The proof of the Bolzano–Weierstrass theorem is based on the completeness property of the real numbers, which states that every bounded sequence has a least upper bound (supremum) and greatest lower bound (infimum).

We use the method of repeated bisection intervals (also known as the "nested interval argument").

**Step 1 (boundedness):**
Let $(x_n)\subseteq\mathbb{R}$ be a bounded sequence. Then there exist real numbers $m, M$ with $m\le x_n\le M$ for all $n\in\mathbb{N}$.

Define the initial interval:

$$
[a_1,b_1] = [m,M].
$$

This interval contains all terms $x_n$.

---

**Step 2 (interval bisection):**
We construct nested intervals by repeatedly halving intervals:

* Divide the initial interval $[a_1,b_1]$ into two equal halves:

$$
[a_1,c_1], \quad [c_1,b_1] \quad \text{where } c_1 = \frac{a_1+b_1}{2}.
$$

At least one of these intervals contains infinitely many terms of the sequence (otherwise, each interval would contain only finitely many terms, contradicting the infinite nature of the sequence).

* Choose the interval (say $[a_2,b_2]$) containing infinitely many terms. Clearly, the length of this interval is exactly half the original length:

$$
|b_2 - a_2| = \frac{1}{2}(b_1 - a_1).
$$

* Again, divide this chosen interval into two equal subintervals and pick the one containing infinitely many terms.

We continue indefinitely, obtaining a nested sequence of intervals:

$$
[a_1,b_1]\supseteq[a_2,b_2]\supseteq[a_3,b_3]\supseteq\cdots
$$

such that:

* Each interval contains infinitely many terms of $(x_n)$.
* The length of each interval shrinks by a factor of $1/2$ at each step, hence:

$$
|b_k - a_k| = \frac{1}{2^{k-1}}(b_1 - a_1)\quad\rightarrow\quad 0\quad\text{as}\quad k\to\infty.
$$

---

**Step 3 (intersection of intervals):**
By the **Nested Interval Property**, the intersection of these intervals is nonempty and contains exactly one point. Let:

$$
\{x^*\} = \bigcap_{k=1}^{\infty}[a_k,b_k].
$$

This follows from completeness of $\mathbb{R}$. Such a point $x^*$ clearly exists and is unique because interval lengths shrink to 0.

---

**Step 4 (construction of the convergent subsequence):**
We now construct a subsequence $(x_{n_k})$ converging to $x^*$:

* From the interval $[a_1,b_1]$, select a term $x_{n_1}$.
* From interval $[a_2,b_2]$, select a term $x_{n_2}$ with $n_2>n_1$.
* Continue this way: from each interval $[a_k,b_k]$, choose a term $x_{n_k}$ with $n_k>n_{k-1}$. Such a choice is always possible since each interval contains infinitely many terms.

Thus, by construction, the subsequence $(x_{n_k})$ satisfies:

* $x_{n_k}\in[a_k,b_k]$, and the intervals $[a_k,b_k]$ shrink to the single point $x^*$.

Since the intervals shrink to length zero and each contains $x_{n_k}$, we must have:

$$
\lim_{k\to\infty} x_{n_k} = x^*.
$$

---

## **Conclusion:**

Thus, we have shown rigorously that every bounded sequence in $\mathbb{R}$ has a convergent subsequence. This completes the proof of the Bolzano–Weierstrass theorem.
:::