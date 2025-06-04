# Exercise Sheet 4 Solutions

### 1.
 
$$
A = \begin{pmatrix} 3 & 2 \\ -2 & -1 \end{pmatrix}
$$

**Characteristic Polynomial:**

We compute the characteristic polynomial:

$$
p(\lambda) = \det(A - \lambda I) 
= \det \begin{pmatrix} 3 - \lambda & 2 \\ -2 & -1 - \lambda \end{pmatrix}
$$

$$
= (3 - \lambda)(-1 - \lambda) - (-2)(2)
= \lambda^2 - 2\lambda + 1 = (\lambda - 1)^2
$$

**Eigenvalues:**

Solving the characteristic polynomial:

$$
(\lambda - 1)^2 = 0 \Rightarrow \lambda_1 = \lambda_2 = 1
$$

**Eigenvectors:**

We solve:

$$
(A - \lambda I)\vec{x} = 0 \Rightarrow
\begin{pmatrix} 2 & 2 \\ -2 & -2 \end{pmatrix} \vec{x} = \vec{0}
$$

This reduces to:

$$
x_1 + x_2 = 0 \Rightarrow \vec{x} = \alpha \begin{pmatrix} 1 \\ -1 \end{pmatrix}
$$

So, the eigenvector is any scalar multiple of:

$$
\vec{x} = \begin{pmatrix} 1 \\ -1 \end{pmatrix}
$$


**Now, we solve them for B:**

$$
B = \begin{pmatrix}
0 & 1 & 0 \\
1 & 0 & 1 \\
1 & 1 & 0
\end{pmatrix}
$$

**Characteristic Polynomial:**

We compute the characteristic polynomial:

$$
p(\lambda) = \det(B - \lambda I) = 
\det \begin{pmatrix}
-\lambda & 1 & 0 \\
1 & -\lambda & 1 \\
1 & 1 & -\lambda
\end{pmatrix}
$$

Expanding along the first row:

```math
= -\lambda \cdot \det \begin{pmatrix}
-\lambda & 1 \\
1 & -\lambda
\end{pmatrix}
- 1 \cdot \det \begin{pmatrix}
1 & 1 \\
1 & -\lambda
\end{pmatrix}
+ 0
```

$$
= -\lambda(\lambda^2 - 1) - (-\lambda - 1)
= -\lambda^3 + \lambda + \lambda + 1
= -\lambda^3 + 2\lambda + 1
$$

So the characteristic polynomial is:

$$
p(\lambda) = -\lambda^3 + 2\lambda + 1
$$

**Eigenvalues:**

Solving the characteristic polynomial:

$$
-\lambda^3 + 2\lambda + 1 = 0
$$

```math
-\left( \lambda + 1 \right)\left( \lambda - \frac{1 + \sqrt{5}}{2} \right)\left( \lambda - \frac{1 - \sqrt{5}}{2} \right) = 0
```

so:

$$
\lambda_1 \approx -0.62, \quad
\lambda_2 = -1, \quad
\lambda_3 \approx 1.62
$$

**Eigenvectors:**

To find the eigenvectors, we solve:

$$
(B - \lambda I)\vec{x} = 0
$$

**For** $$ \lambda_1 \approx -0.62 $$:

$$
\vec{v}_1 = \begin{pmatrix}
1 \\
-0.62 \\
-0.62
\end{pmatrix}
$$

**For** $$ \lambda_2 = -1 $$:

$$
\vec{v}_2 = \begin{pmatrix}
-1 \\
1 \\
0
\end{pmatrix}
$$

**For** $$ \lambda_3 \approx 1.62 $$:

$$
\vec{v}_3 = \begin{pmatrix}
1 \\
1.62 \\
1.62
\end{pmatrix}
$$

$$\blacksquare$$

### 2.
We start by expressing the trace of $$ ABC $$:

```math
\mathrm{tr}(ABC) = \sum_{i=1}^{n} (ABC)_{ii}
```

**Computing $$ (ABC)_{ii} $$:**

We want to compute the entry in the $$ i $$-th row and $$ i $$-th column of the matrix product $$ ABC $$.

1. First, compute the product $$ AB $$, which gives:

```math
(AB)_{il} = \sum_{j=1}^{m} A_{ij} B_{jl}
```

2. Then compute $$ ABC = (AB)C $$. The $$ (i, i) $$-th element of $$ ABC $$ is:

```math
(ABC)_{ii} = \sum_{l=1}^{k} (AB)_{il} C_{li}
```

3. Substitute the expression for $$ (AB)_{il} $$:

```math
(ABC)_{ii} = \sum_{l=1}^{k} \left( \sum_{j=1}^{m} A_{ij} B_{jl} \right) C_{li}
```

4. Rearranging the summation order:

```math
(ABC)_{ii} = \sum_{j=1}^{m} \sum_{l=1}^{k} A_{ij} B_{jl} C_{li}
```

So, the trace becomes:

```math
\mathrm{tr}(ABC) = \sum_{i=1}^{n} \sum_{j=1}^{m} \sum_{k=1}^{k} A_{ij} B_{jk} C_{ki}
```


**Computing the trace of $$ BCA $$:**

```math
\mathrm{tr}(BCA) = \sum_{j=1}^{m} (BCA)_{jj}
```

Expand the product:

1. First compute $$ BC $$:

```math
(BC)_{ji} = \sum_{k=1}^{k} B_{jk} C_{ki}
```

2. Then compute:

```math
(BCA)_{jj} = \sum_{i=1}^{n} (BC)_{ji} A_{ij} = \sum_{i=1}^{n} \sum_{k=1}^{k} B_{jk} C_{ki} A_{ij}
```

3. So the trace is:

```math
\mathrm{tr}(BCA) = \sum_{j=1}^{m} \sum_{k=1}^{k} \sum_{i=1}^{n} B_{jk} C_{ki} A_{ij}
```


**Final Step:**

Since scalar multiplication is commutative:

```math
\sum_{i=1}^{n} \sum_{j=1}^{m} \sum_{k=1}^{k} A_{ij} B_{jk} C_{ki}
= \sum_{j=1}^{m} \sum_{k=1}^{k} \sum_{i=1}^{n} B_{jk} C_{ki} A_{ij}
```

Therefore:

```math
\mathrm{tr}(ABC) = \mathrm{tr}(BCA)
```

$$\blacksquare$$

### 3.

**Eigenvalues are the same**

Let $$ p(\lambda) $$ be the characteristic polynomial of $$ A $$. By definition:

```math
p(\lambda) = \det(A - \lambda I)
```

Now consider the transpose:

```math
\det(A^\top - \lambda I)
= \det((A - \lambda I)^\top)
= \det(A - \lambda I)
= p(\lambda)
```

So both $$ A $$ and $$ A^\top $$ have the **same characteristic polynomial**, which means they have the **same eigenvalues**, including their algebraic multiplicities.


**Eigenvectors may differ**
Although $$ A $$ and $$ A^\top $$ have the same eigenvalues, their eigenvectors **need not** be the same.

To see this, consider a concrete example:

Let

```math
A = \begin{pmatrix}
1 & 1 \\
0 & 1
\end{pmatrix}
```

Then:

```math
A^\top = \begin{pmatrix}
1 & 0 \\
1 & 1
\end{pmatrix}
```

The characteristic polynomial of both is:

```math
\det(A - \lambda I) = (1 - \lambda)^2
```

So they both have a **repeated eigenvalue** $$ \lambda = 1 $$.

Now compute eigenvectors:

- For $$ A $$, we solve:

```math
(A - I)\vec{x} = \begin{pmatrix}
0 & 1 \\
0 & 0
\end{pmatrix} \vec{x} = 0
\Rightarrow x_2 = 0 \Rightarrow \vec{x} = \begin{pmatrix}
1 \\
0
\end{pmatrix}
```

- For $$ A^\top $$, we solve:

```math
(A^\top - I)\vec{x} = \begin{pmatrix}
0 & 0 \\
1 & 0
\end{pmatrix} \vec{x} = 0
\Rightarrow x_1 = 0 \Rightarrow \vec{x} = \begin{pmatrix}
0 \\
1
\end{pmatrix}
```

Thus, the **eigenvectors are different**, even though the eigenvalues are the same.

**So:**

- $$ A $$ and $$ A^\top $$ always have the **same eigenvalues**, including multiplicities.
- However, they may have **different eigenvectors**, especially when the matrix is **not symmetric**.

$$\blacksquare$$

### 4.
#### (a) The rank of $$ B $$

We observe that each row (and column) of $$ B $$ is a linear combination of the others:

- The second row is:  
  ```math
  \text{Row}_2 = 2 \cdot \text{Row}_1
  ```
- The third row is:  
  ```math
  \text{Row}_3 = 3 \cdot \text{Row}_1
  ```

So all rows lie in the span of the first row.

Let’s do row reduction to confirm:

```math
\begin{pmatrix}
1 & 2 & 3 \\
2 & 4 & 6 \\
3 & 6 & 9
\end{pmatrix}
\rightarrow
\begin{pmatrix}
1 & 2 & 3 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{pmatrix}
```

Only one non-zero row remains after Gaussian elimination.

Therefore, the rank of $$ B $$ is:

```math
\mathrm{rank}(B) = 1
```

#### (b) Are the columns of $$ B $$ linearly independent?

Recall that the number of linearly independent columns is equal to the rank of the matrix.

Since:

```math
\mathrm{rank}(B) = 1 < 3
```

This means that the columns are **linearly dependent**.

In fact:

```math
\text{Col}_2 = 2 \cdot \text{Col}_1 \\
\text{Col}_3 = 3 \cdot \text{Col}_1
```
$$\blacksquare$$
