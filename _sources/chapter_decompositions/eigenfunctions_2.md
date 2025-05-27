Yes — and this is a **key insight** of Mercer's theorem and functional analysis:

> Just like we represent a function in a basis via its **coefficients**, we can represent functions in terms of **eigenfunctions** of an integral operator — and this lets us work in $\mathbb{R}^N$, even if the original space is infinite-dimensional.

---

### 🎯 Analogy with Basis Expansion

Let’s recall what you mentioned:

If $\{b_i\}_{i=1}^\infty$ is an orthonormal basis (e.g. Fourier basis) of $L^2(\mathcal{X})$, then any $f \in L^2(\mathcal{X})$ can be written as:

$$
f(x) = \sum_{i=1}^\infty \alpha_i b_i(x)
\quad \text{with } \alpha_i = \langle f, b_i \rangle
$$

The function is now represented by a **sequence** of coefficients $\boldsymbol{\alpha} = (\alpha_1, \alpha_2, \dots) \in \ell^2$. Truncating to the first $N$ gives an approximation in $\mathbb{R}^N$.

---

### ✅ Eigenfunctions Work the Same Way

Now suppose $T$ is the integral operator:

$$
(Tf)(x) = \int_\mathcal{X} k(x,x') f(x') dx'
$$

and $\{ \phi_i \}_{i=1}^\infty$ are the **orthonormal eigenfunctions** of $T$ with eigenvalues $\lambda_i \geq 0$. Then:

$$
f(x) = \sum_{i=1}^\infty \alpha_i \phi_i(x)
\quad \text{with } \alpha_i = \langle f, \phi_i \rangle
$$

That is, $\{ \phi_i \}$ gives a new basis of $L^2(\mathcal{X})$, adapted to the kernel.

But there’s more:

---

### 🌈 Mercer Kernel Defines a Feature Map via Eigenfunctions

Define the **feature map** $\phi(x)$ into $\ell^2$ via:

$$
\phi(x) = \left( \sqrt{\lambda_1} \phi_1(x), \sqrt{\lambda_2} \phi_2(x), \dots \right)
$$

Then for any $x, x' \in \mathcal{X}$:

$$
k(x,x') = \langle \phi(x), \phi(x') \rangle_{\ell^2}
$$

So this infinite-dimensional object is encoded by an **(infinite) vector of coefficients**, just like a Fourier expansion — but using *eigenfunctions of the kernel*.

In practice, we often truncate this expansion:

$$
\phi_N(x) = \left( \sqrt{\lambda_1} \phi_1(x), \dots, \sqrt{\lambda_N} \phi_N(x) \right) \in \mathbb{R}^N
$$

This gives a **finite-dimensional approximation** of the infinite feature space.

---

### 📌 Summary

| Concept        | Basis Functions               | Kernel Eigenfunctions                         |
| -------------- | ----------------------------- | --------------------------------------------- |
| Object         | $b_i(x)$                      | $\phi_i(x)$                                   |
| Coefficients   | $\langle f, b_i \rangle$      | $\langle f, \phi_i \rangle$                   |
| Representation | $f(x) = \sum \alpha_i b_i(x)$ | $f(x) = \sum \alpha_i \phi_i(x)$              |
| Feature Map    | N/A                           | $\phi(x) = (\sqrt{\lambda_i} \phi_i(x))$      |
| Kernel         | N/A                           | $k(x,x') = \langle \phi(x), \phi(x') \rangle$ |

So yes — **eigenfunctions let you represent infinite-dimensional functions by vectors in $\mathbb{R}^N$** — just like standard orthonormal basis functions do.

Would you like a small numerical example (e.g. a discretized Mercer kernel with eigenvectors as “eigenfunctions”)?
