# Fundamental Equivalences for Square matrices
Yes, you're absolutely right — these statements are all **equivalent** and form a **core set of if-and-only-if conditions** that characterize invertibility of a square matrix. Let's list them more formally and then prove that they are all equivalent:

---
:::{prf:theorem} Fundamental Equivalences
:label: trm-fundamental-equivalences
:nonumber:

Let $\mathbf{A} \in \mathbb{R}^{n \times n}$. The following statements are **equivalent** — that is, they are all true or all false together:

1. $\mathbf{A}$ is **invertible**
2. $\det(\mathbf{A}) \neq 0$
3. $\mathbf{A}$ is **full-rank**, i.e., $\operatorname{rank}(\mathbf{A}) = n$
4. The **columns** of $\mathbf{A}$ are **linearly independent**
5. The **rows** of $\mathbf{A}$ are **linearly independent**
6. $\mathbf{A}$ is **row-equivalent** to the identity matrix
7. The system $\mathbf{A}\mathbf{x} = \mathbf{b}$ has a **unique solution for every $\mathbf{b} \in \mathbb{R}^n$**

:::


:::{prf:proof}

We'll prove the chain of implications in a **circular fashion**, which implies all are equivalent.

---

**(1) ⇒ (2): Invertible ⇒ Determinant nonzero**

If $\mathbf{A}^{-1}$ exists, then

$$
\det(\mathbf{A} \mathbf{A}^{-1}) = \det(\mathbf{I}) = 1 = \det(\mathbf{A}) \det(\mathbf{A}^{-1}) \Rightarrow \det(\mathbf{A}) \neq 0
$$

---

**(2) ⇒ (3): $\det(\mathbf{A}) \neq 0$ ⇒ Full-rank**

A square matrix has full rank $\iff$ its rows/columns span $\mathbb{R}^n$, and this happens exactly when $\det(\mathbf{A}) \neq 0$.

If $\operatorname{rank}(\mathbf{A}) < n$, then one row or column is linearly dependent, making $\det(\mathbf{A}) = 0$.

---

**(3) ⇒ (4) and (5): Full-rank ⇒ Linear independence of rows and columns**

A matrix with rank $n$ must have linearly independent rows and columns by the definition of rank.

---

**(4) ⇒ (6): Independent columns ⇒ Row-equivalent to identity**

If the columns are linearly independent, Gaussian elimination can reduce $\mathbf{A}$ to the identity matrix $\mathbf{I}$ using row operations.

This means $\mathbf{A}$ is row-equivalent to $\mathbf{I}$.

---

**(6) ⇒ (7): Row-equivalent to $\mathbf{I}$ ⇒ Unique solution for all $\mathbf{b}$**

If $\mathbf{A} \sim \mathbf{I}$, then solving $\mathbf{A} \mathbf{x} = \mathbf{b}$ is equivalent to solving $\mathbf{I} \mathbf{x} = \mathbf{b}'$, which always has the unique solution $\mathbf{x} = \mathbf{b}'$.

---

**(7) ⇒ (1): Unique solution for all $\mathbf{b}$ ⇒ $\mathbf{A}$ is invertible**

If $\mathbf{A} \mathbf{x} = \mathbf{b}$ has a **unique solution for all** $\mathbf{b}$, then the inverse mapping $\mathbf{b} \mapsto \mathbf{x}$ is well-defined and linear, so $\mathbf{A}^{-1}$ exists.

---

**Conclusion**

All these statements are **equivalent**:

$$
\boxed{
\mathbf{A} \text{ invertible } \iff \det(\mathbf{A}) \neq 0 \iff \operatorname{rank}(\mathbf{A}) = n \iff \text{cols/rows lin. independent} \iff \mathbf{A} \sim \mathbf{I} \iff \text{unique solution for all } \mathbf{b}
}
$$

:::