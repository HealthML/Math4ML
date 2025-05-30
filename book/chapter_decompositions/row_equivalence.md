---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Gaussian Elimination and the PLU Decomposition

Gaussian elimination is one of the most fundamental algorithms in linear algebra.
It provides a systematic procedure for simplifying matrices using elementary row operations and lies at the heart of solving linear systems, computing inverses, determining rank, and understanding matrix structure.

This section introduces the core concepts and forms related to Gaussian elimination:

* **Row Echelon Form (REF)**: A simplified form of a matrix that resembles an upper-triangular structure. REF is sufficient for solving linear systems using **back substitution**.
* **Reduced Row Echelon Form (RREF)**: A further simplified and canonical form where each pivot is 1 and the only nonzero entry in its column. RREF enables direct reading of solutions to linear systems.
* **Row Equivalence**: The idea that matrices related through row operations preserve important properties such as solvability and rank.
* **Gaussian Elimination**: The algorithm used to transform matrices into REF, using a sequence of elementary row operations.

Throughout this section, we will define these forms, illustrate them with examples, and demonstrate how they relate to one another and to solving equations of the form $\mathbf{A}\mathbf{x} = \mathbf{b}$.
We will also discuss when a matrix is invertible based on its row-reduced form, and how to use back substitution after performing Gaussian elimination.

This foundation is essential for understanding many areas of applied linear algebra, from numerical methods to machine learning.


---

## Elementary Row Operations

One of the most important facts underlying **Gaussian elimination** is that the following **elementary row operations** do not change the solution set of a linear system.
That is, if we apply these operations to both the matrix $\mathbf{A}$ and the right-hand side $\mathbf{b}$ in the system $\mathbf{A}\mathbf{x} = \mathbf{b}$, we obtain an **equivalent system** with the **same solutions**.

1. **Swap** two rows
   $R_i \leftrightarrow R_j$

2. **Scale** a row by a nonzero scalar
   $R_i \to \alpha R_i$, $\alpha \neq 0$

3. **Add a multiple** of one row to another
   $R_i \to R_i + \beta R_j$

---

:::{prf:theorem} Elementary Row Operations Preserve Solution Sets
:label: trm-elementary-rop-operations

Let $\mathbf{A} \mathbf{x} = \mathbf{b}$ be a system of linear equations.

If we apply a finite sequence of **elementary row operations** to both $\mathbf{A}$ and $\mathbf{b}$, the resulting system has the **same solution set** as the original.

That is, $\mathbf{A} \sim \mathbf{A}'$ implies:

$$
\mathbf{A} \mathbf{x} = \mathbf{b} \iff \mathbf{A}' \mathbf{x} = \mathbf{b}'
$$

:::

:::{prf:proof}

Each elementary row operation corresponds to **left-multiplication** of both sides of the equation by an **invertible matrix** $\mathbf{C}$:

$$
\mathbf{A} \mathbf{x} = \mathbf{b} \iff \mathbf{C}\mathbf{A} \mathbf{x} = \mathbf{C}\mathbf{b}
$$

* Swapping rows ↔ permutation matrix
* Scaling a row ↔ diagonal matrix
* Adding a multiple of one row to another ↔ elementary row matrix

Since invertible matrices preserve linear equivalence, applying these operations preserves the solution set.
:::

## Row Echelon Form (REF) (🇩🇪 **Zeilen-Stufenform**)

A matrix is said to be in **row echelon form (REF)** if it satisfies the following conditions:

1. **All zero rows** (if any) appear **at the bottom** of the matrix.
2. The **leading entry** (or pivot) of each nonzero row is strictly to the **right** of the leading entry of the row above it.
3. All entries **below** a pivot are **zero**.

---

The following is a matrix in row echelon form:

$$
\begin{bmatrix}
1 & 2 & 3 \\
0 & 1 & 4 \\
0 & 0 & 5
\end{bmatrix}
$$

But this is **not** in row echelon form (pivot in row 3 is not to the right of the pivot in row 2):

$$
\begin{bmatrix}
1 & 2 & 3 \\
0 & 0 & 1 \\
0 & 1 & 0
\end{bmatrix}
$$

A matrix in **REF** is a "staircase" matrix with each nonzero row starting further to the right, and all entries below each pivot are zero:

$$
\boxed{
\text{REF = Upper triangular-like form from which back substitution is possible}
}
$$


## Reduced Row Echelon Form (RREF)

A matrix is in **reduced row echelon form (RREF)** if it satisfies **all the conditions of row echelon form (REF)**, *plus* two additional conditions:

---

### Conditions for RREF

1. **REF conditions**:

   * All nonzero rows are above any all-zero rows.
   * Each leading (nonzero) entry of a row is strictly to the right of the leading entry of the row above it.
   * All entries below a pivot are zero.

2. **Additional conditions**:

   * Each **pivot is equal to 1** (i.e. all leading entries are 1).
   * Each pivot is the **only nonzero entry in its column**.

---

$$
\begin{bmatrix}
1 & 0 & 2 \\
0 & 1 & -3 \\
0 & 0 & 0
\end{bmatrix}
$$

This is in **RREF** because:

* Each pivot is 1.
* Each pivot column contains only one nonzero entry (the pivot itself).
* Pivots step to the right as you go down the rows.
* Zero row is at the bottom.

---

$$
\begin{bmatrix}
2 & 4 & 6 \\
0 & 3 & 9 \\
0 & 0 & 1
\end{bmatrix}
$$

This is in **REF**, but not RREF:

* It satisfies the "staircase" structure.
* But pivots are not 1.
* There are other nonzero entries in pivot columns.

---

## REF vs. RREF: Key Differences

| Feature                          | REF              | RREF |
| -------------------------------- | ---------------- | ---- |
| Zero rows at bottom              | ✅                | ✅    |
| Pivots step to the right         | ✅                | ✅    |
| Zeros below pivots               | ✅                | ✅    |
| Pivots are 1                     | ❌ (not required) | ✅    |
| Zeros **above and below** pivots | ❌                | ✅    |

---

## Gaussian Elimination

**Gaussian elimination** is a method for solving systems of linear equations by systematically transforming the coefficient matrix into **row echelon form** (REF) using the **elementary row operations** defined above.

It is one of the fundamental algorithms in linear algebra and underlies techniques such as solving $\mathbf{A}\mathbf{x} = \mathbf{b}$, computing the rank, and inverting matrices.
If we track the elementary operations of Gaussian Elimination, we obtain the PLU decomposition of a $\mathbf{A}$.

---

### Goal

Transform a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ into an **upper triangular matrix** (or REF), such that all entries below the pivot positions (leading entries in each row) are zero.

---

### Steps of Gaussian Elimination

1. **Identify the leftmost column** that contains a nonzero entry (the pivot column).
2. **Swap rows** (if necessary) so that the pivot entry is at the top of the current submatrix.
3. **Normalize** the pivot row so the pivot equals 1 (optional — standard Gaussian elimination doesn’t require this).
4. **Eliminate below the pivot**: Subtract suitable multiples of the pivot row from rows below to make all entries in the pivot column below the pivot zero.
5. **Move to the submatrix** that lies below and to the right, and repeat until the entire matrix is in row echelon form.


---


### Example: Gaussian Elimination by Hand

Start with:

$$
\mathbf{A} =
\begin{bmatrix}
2 & 4 & -2 \\
4 & 9 & -3 \\
-2 & -1 & 7
\end{bmatrix}
$$

---

**Step 1: Normalize the first pivot**

Pivot is $A_{11} = 2$. We'll eliminate below it.

Eliminate row 2:

$$
R_2 \leftarrow R_2 - 2 \cdot R_1
$$

$$
\begin{bmatrix}
2 & 4 & -2 \\
0 & 1 & 1 \\
-2 & -1 & 7
\end{bmatrix}
$$

Eliminate row 3:

$$
R_3 \leftarrow R_3 + R_1
$$

$$
\begin{bmatrix}
2 & 4 & -2 \\
0 & 1 & 1 \\
0 & 3 & 5
\end{bmatrix}
$$

---

**Step 2: Eliminate below second pivot**

Pivot at $A_{22} = 1$. Eliminate below.

$$
R_3 \leftarrow R_3 - 3 \cdot R_2
$$

$$
\begin{bmatrix}
2 & 4 & -2 \\
0 & 1 & 1 \\
0 & 0 & 2
\end{bmatrix}
$$

---

**Step 3: Normalize all pivots (optional, for RREF)**

We can divide each pivot row to make the pivots equal to 1:

$$
R_1 \leftarrow \frac{1}{2} R_1, \quad
R_3 \leftarrow \frac{1}{2} R_3
$$

$$
\begin{bmatrix}
1 & 2 & -1 \\
0 & 1 & 1 \\
0 & 0 & 1
\end{bmatrix}
$$

---

Final Result: Row Echelon Form (REF)

$$
\boxed{
\begin{bmatrix}
1 & 2 & -1 \\
0 & 1 & 1 \\
0 & 0 & 1
\end{bmatrix}
}
$$

This is the **upper triangular matrix** resulting from Gaussian elimination.

## Solving Linear Systems via Gaussian Elimination

To solve a linear system $\mathbf{A}\mathbf{x} = \mathbf{b}$ using **Gaussian elimination followed by back substitution**, it's not enough to row-reduce $\mathbf{A}$ alone — we must also apply the **same row operations** to the right-hand side vector $\mathbf{b}$. This gives a consistent system that we can solve in the transformed space.

Gaussian elimination turns the system:

$$
\mathbf{A} \mathbf{x} = \mathbf{b}
$$

into an equivalent upper-triangular system:

$$
\mathbf{U} \mathbf{x} = \mathbf{c}
$$

where:

* $\mathbf{U}$ is the row-reduced form of $\mathbf{A}$ (usually REF)
* $\mathbf{c}$ is the result of applying the **same row operations** to $\mathbf{b}$

Only with this consistent transformation can back substitution be applied.

### Example Linear System

Let’s extend the example to **solve a system** $\mathbf{A} \mathbf{x} = \mathbf{b}$ using **Gaussian elimination + back substitution**.

Given:

$$
\mathbf{A} =
\begin{bmatrix}
2 & 4 & -2 \\
4 & 9 & -3 \\
-2 & -1 & 7
\end{bmatrix}, \quad
\mathbf{b} =
\begin{bmatrix}
2 \\
8 \\
10
\end{bmatrix}
$$

We want to solve:

$$
\mathbf{A} \mathbf{x} = \mathbf{b}
$$

---

**Step 1: Augmented Matrix**

Form the augmented matrix $[\mathbf{A} \mid \mathbf{b}]$:

$$
\begin{bmatrix}
2 & 4 & -2 & \big| & 2 \\
4 & 9 & -3 & \big| & 8 \\
-2 & -1 & 7 & \big| & 10
\end{bmatrix}
$$

---

**Step 2: Apply Gaussian Elimination**

**Eliminate below pivot (row 1)**

* $R_2 \leftarrow R_2 - 2 \cdot R_1$
* $R_3 \leftarrow R_3 + R_1$

$$
\begin{bmatrix}
2 & 4 & -2 & \big| & 2 \\
0 & 1 & 1 & \big| & 4 \\
0 & 3 & 5 & \big| & 12
\end{bmatrix}
$$

**Eliminate below pivot (row 2)**

* $R_3 \leftarrow R_3 - 3 \cdot R_2$

$$
\begin{bmatrix}
2 & 4 & -2 & \big| & 2 \\
0 & 1 & 1 & \big| & 4 \\
0 & 0 & 2 & \big| & 0
\end{bmatrix}
$$

**Normalize pivots (optional)**

* $R_1 \leftarrow \frac{1}{2} R_1$
* $R_3 \leftarrow \frac{1}{2} R_3$

$$
\boxed{
\begin{bmatrix}
1 & 2 & -1 & \big| & 1 \\
0 & 1 & 1 & \big| & 4 \\
0 & 0 & 1 & \big| & 0
\end{bmatrix}
}
$$

This is the system in **row echelon form**.

---

**Step 3: Back Substitution**

Let the system be:

$$
\begin{aligned}
x_1 + 2x_2 - x_3 &= 1 \quad \text{(Row 1)} \\
\quad\;\;\;\; x_2 + x_3 &= 4 \quad \text{(Row 2)} \\
\quad\quad\quad\quad\; x_3 &= 0 \quad \text{(Row 3)}
\end{aligned}
$$

Back-substitute from bottom to top:

1. $x_3 = 0$
2. $x_2 + x_3 = 4 \Rightarrow x_2 = 4$
3. $x_1 + 2x_2 - x_3 = 1 \Rightarrow x_1 + 8 = 1 \Rightarrow x_1 = -7$

---

Final Solution

$$
\boxed{
\mathbf{x} =
\begin{bmatrix}
-7 \\
4 \\
0
\end{bmatrix}
}
$$



---

## Interpretation of Solving Systems by Gaussian Elimination

Think of it as row-reducing the **augmented matrix**:

$$
[\mathbf{A} \mid \mathbf{b}] \quad \longrightarrow \quad [\mathbf{U} \mid \mathbf{c}]
$$

You solve the simplified system $\mathbf{U} \mathbf{x} = \mathbf{c}$, not the original one.

---

$$
\boxed{
\text{Gaussian elimination modifies both } \mathbf{A} \text{ and } \mathbf{b} \text{ together.}
}
$$


Then you can solve $\mathbf{A} \mathbf{x} = \mathbf{b}$ via **back substitution**.

---

## Back Substitution

Once a matrix has been transformed into **row echelon form (REF)** using Gaussian elimination, we can solve a system of equations $\mathbf{A}\mathbf{x} = \mathbf{b}$ using **back substitution**.

This method proceeds from the bottom row of the triangular system upward, solving for each variable one at a time.

---

### General Idea

Suppose, after Gaussian elimination, we have the augmented system:

$$
\begin{aligned}
x_3 &= c_3 \\
x_2 + a_{23}x_3 &= c_2 \\
x_1 + a_{12}x_2 + a_{13}x_3 &= c_1
\end{aligned}
$$

We can compute the solution in reverse order:

1. Solve for $x_3$ from the last equation.
2. Plug $x_3$ into the second equation and solve for $x_2$.
3. Plug $x_2$ and $x_3$ into the first equation to solve for $x_1$.

---

### Back Substitution Example

Let’s solve:

$$
\begin{bmatrix}
1 & 2 & -1 \\
0 & 1 & 3 \\
0 & 0 & 2
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2 \\ x_3
\end{bmatrix}
=
\begin{bmatrix}
5 \\ 4 \\ 6
\end{bmatrix}
$$

We solve from the bottom up:

* $x_3 = \frac{6}{2} = 3$
* $x_2 + 3 \cdot 3 = 4 \Rightarrow x_2 = 4 - 9 = -5$
* $x_1 + 2(-5) - 3 = 5 \Rightarrow x_1 = 5 + 10 + 3 = 18$

So the solution is:

$$
\boxed{
\mathbf{x} =
\begin{bmatrix}
18 \\
-5 \\
3
\end{bmatrix}
}
$$


---

## Pivot Columns and Free Variables

When we reduce a matrix to **row echelon form (REF)** or **reduced row echelon form (RREF)**, the position of **pivots** in the matrix gives us direct insight into the structure of the solution set of the system $\mathbf{A}\mathbf{x} = \mathbf{b}$.

---

### ✅ Pivot Columns and Basic Variables

* A **pivot** is the first nonzero entry in a row of REF or RREF.
* The **columns** of the original matrix $\mathbf{A}$ that contain pivots are called **pivot columns**.
* The variables corresponding to pivot columns are called **basic variables**.

  * These are the variables you solve for directly using back substitution.

---

### 🆓 Non-Pivot Columns and Free Variables

* The **columns that do not contain a pivot** are called **free columns**.
* The variables corresponding to these columns are called **free variables**.

  * They can take on arbitrary values.
  * The values of basic variables depend on the free variables.

---

### 🧠 Solution Structure

The presence or absence of pivot positions determines the nature of the solution:

| Situation                                                                         | Interpretation                                 |
| --------------------------------------------------------------------------------- | ---------------------------------------------- |
| Pivot in every column of $\mathbf{A}$                                             | **Unique solution** (if consistent)            |
| Some columns with no pivot                                                        | **Infinitely many solutions** (free variables) |
| Inconsistent system (e.g., row of zeros in $\mathbf{A}$, nonzero in $\mathbf{b}$) | **No solution**                                |

---

### 🔢 Example

Suppose we reduce the augmented matrix to RREF:

$$
\left[
\begin{array}{ccc|c}
1 & 0 & 2 & 3 \\
0 & 1 & -1 & 1 \\
0 & 0 & 0 & 0
\end{array}
\right]
$$

* Pivots in columns 1 and 2 → $x_1$ and $x_2$ are **basic**
* No pivot in column 3 → $x_3$ is a **free variable**
* The solution has the form:

  $$
  \begin{aligned}
  x_1 &= 3 - 2x_3 \\
  x_2 &= 1 + x_3 \\
  x_3 &\text{ is free}
  \end{aligned}
  $$

This system has **infinitely many solutions**, parameterized by $x_3$.

---

### 🧩 Summary

$$
\boxed{
\text{Pivot columns } \leftrightarrow \text{ basic variables}, \quad
\text{Non-pivot columns } \leftrightarrow \text{ free variables}
}
$$

Understanding this structure helps us:

* Determine how many solutions exist
* Express the solution set explicitly
* Identify the degrees of freedom in underdetermined systems

---


## Row-Equivalent Matrices

Two matrices $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$ are called **row-equivalent** if one can be transformed into the other using a finite sequence of the **elementary row operations** defined above.

---

### Notation

We write:

$$
\mathbf{A} \sim \mathbf{B}
$$

to denote that $\mathbf{A}$ is **row-equivalent** to $\mathbf{B}$.

---

### Intuition

* Row-equivalence preserves the **solution set** of the linear system $\mathbf{A} \mathbf{x} = \mathbf{b}$.
* It **does not** change the **row space**, and hence **preserves the rank**.
* A matrix is row-equivalent to the **identity matrix** $\mathbf{I}$ if and only if it is **invertible** (for square matrices).

---

### Summary

$$
\boxed{
\mathbf{A} \sim \mathbf{B} \iff \text{you can get from one to the other by row operations}
}
$$

### Example
Here are the step-by-step row operations showing that the matrix

$$
\mathbf{A} = \begin{bmatrix}
2 & 1 & -1 \\
-3 & -1 & 2 \\
-2 & 1 & 2
\end{bmatrix}
$$

is **row-equivalent** to the identity matrix $\mathbf{I}$.
This confirms that $\mathbf{A}$ is **invertible**.

Here are the steps of the row reduction process rendered with corresponding **elementary row operations** and resulting matrices:

---

**Step 0**: Start with matrix $\mathbf{A}$

$$
\mathbf{A} =
\begin{bmatrix}
2 & 1 & -1 \\
-3 & -1 & 2 \\
-2 & 1 & 2
\end{bmatrix}
$$

---

**Step 1**: Normalize row 1

$R_1 \leftarrow \frac{1}{2} R_1$

$$
\begin{bmatrix}
1 & 0.5 & -0.5 \\
-3 & -1 & 2 \\
-2 & 1 & 2
\end{bmatrix}
$$

---

**Step 2**: Eliminate entries below pivot in column 1

$R_2 \leftarrow R_2 + 3 R_1$
$R_3 \leftarrow R_3 + 2 R_1$

$$
\begin{bmatrix}
1 & 0.5 & -0.5 \\
0 & 0.5 & 0.5 \\
0 & 2 & 1
\end{bmatrix}
$$

---

**Step 3**: Normalize row 2

$R_2 \leftarrow \frac{1}{0.5} R_2 = 2 R_2$

$$
\begin{bmatrix}
1 & 0.5 & -0.5 \\
0 & 1 & 1 \\
0 & 2 & 1
\end{bmatrix}
$$

---

**Step 4**: Eliminate below pivot in column 2

$R_3 \leftarrow R_3 - 2 R_2$

$$
\begin{bmatrix}
1 & 0.5 & -0.5 \\
0 & 1 & 1 \\
0 & 0 & -1
\end{bmatrix}
$$

---

**Step 5**: Normalize row 3

$R_3 \leftarrow -1 \cdot R_3$

$$
\begin{bmatrix}
1 & 0.5 & -0.5 \\
0 & 1 & 1 \\
0 & 0 & 1
\end{bmatrix}
$$

---

We have row-reduced $\mathbf{A}$ to **row echelon form**, which is the identity matrix after further elimination above the pivots (not shown).

Hence:

$$
\mathbf{A} \sim \mathbf{I} \quad \Rightarrow \quad \textbf{A is invertible.}
$$

---

## Matrix Inversion via Gaussian Elimination

Gaussian elimination can not only be used to solve systems $\mathbf{A}\mathbf{x} = \mathbf{b}$, but also to **compute the inverse** of a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, **if it exists**.

This is done by augmenting $\mathbf{A}$ with the identity matrix $\mathbf{I}$, and applying row operations to reduce $\mathbf{A}$ to $\mathbf{I}$.

The operations that convert $\mathbf{A}$ into $\mathbf{I}$ will simultaneously convert $\mathbf{I}$ into $\mathbf{A}^{-1}$. 
This approach is a **constructive proof** of invertibility.

---

### Procedure

1. Form the augmented matrix $[\mathbf{A} \mid \mathbf{I}]$.
2. Apply **Gaussian elimination** to row-reduce the left side to the identity matrix.
3. If successful, the right side will become $\mathbf{A}^{-1}$.
4. If the left side cannot be reduced to identity (e.g. a zero row appears), $\mathbf{A}$ is **not invertible**.

---

$$
\boxed{
[\mathbf{A} \mid \mathbf{I}] \longrightarrow [\mathbf{I} \mid \mathbf{A}^{-1}] \quad \text{via Gaussian elimination}.
}
$$

---

## PLU Decomposition

Every square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ has a **PLU decomposition** (or **LU decomposition with partial pivoting**):

$$
\boxed{
\mathbf{A} = \mathbf{P} \mathbf{L} \mathbf{U}
}
$$

* $\mathbf{P}^\top$ is a **permutation matrix**  (a rearrangement of the identity matrix) that tracks the row swaps
* $\mathbf{L}$ is **lower triangular** with unit diagonal (contains the elimination multipliers).
* $\mathbf{U}$ is **upper triangular** (result of Gaussian elimination).

As $\mathbf{P}$ is a permuation matrix $\mathbf{P}^{-1} = \mathbf{P}^{\top}$, we can alternatively write

$$
\mathbf{P}^\top \mathbf{A} = \mathbf{L} \mathbf{U}
$$

The PLU decomposition
* always exists for any square matrix.
* is used in stable numerical solvers.
* is efficient for solving systems and computing inverses.

---

### Example PLU Decomposition

Let

$$
\mathbf{A} =
\begin{bmatrix}
0 & 2 \\
1 & 4
\end{bmatrix}
$$

To eliminate below the pivot, we need to **swap rows**, since $A_{11} = 0$.
The permutation matrix is:

$$
\mathbf{P} =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}, \quad
\mathbf{P}^\top \mathbf{A} =
\begin{bmatrix}
1 & 4 \\
0 & 2
\end{bmatrix}
$$

Then:

$$
\mathbf{L} =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}, \quad
\mathbf{U} =
\begin{bmatrix}
1 & 4 \\
0 & 2
\end{bmatrix}
$$

So:

$$
\mathbf{P}^\top \mathbf{A} = \mathbf{L} \mathbf{U}
$$

---

```{code-cell} ipython3
import numpy as np
from scipy.linalg import lu

# Example matrix
A = np.array([[0, 2, 1],
              [1, 1, 0],
              [2, 1, 1]], dtype=float)

# Perform PLU decomposition
P, L, U = lu(A)

print("P.T @ A:\n", P.T @ A)
print("L @ U:\n", L @ U)
```

This uses **SciPy's `lu` function**, returning:

* $\mathbf{P}$: permutation matrix
* $\mathbf{L}$: unit lower triangular matrix
* $\mathbf{U}$: upper triangular matrix
  such that:

$$
\text{LU decomposition with pivoting gives } \mathbf{P}, \mathbf{L}, \mathbf{U} \text{ such that }  \mathbf{A} = \mathbf{P}\mathbf{L} \mathbf{U}
$$

---

To solve a linear system $\mathbf{A} \mathbf{x} = \mathbf{b}$ **given the PLU decomposition** of $\mathbf{A}$, that is:

$$
\boxed{
\mathbf{P}^\top \mathbf{A} = \mathbf{L} \mathbf{U}
}
$$

You solve the system in **three steps**:

**1. Permute the right-hand side**

Multiply both sides by $\mathbf{P}^\top$ to align with the decomposition:

$$
\mathbf{P}^\top \mathbf{A} \mathbf{x} = \mathbf{P}^\top \mathbf{b}
\Rightarrow \mathbf{L} \mathbf{U} \mathbf{x} = \mathbf{P}^\top \mathbf{b}
$$

Let:

$$
\mathbf{c} = \mathbf{P}^\top \mathbf{b}
$$

---

**2. Forward substitution**

Solve for intermediate vector $\mathbf{y}$ in:

$$
\mathbf{L} \mathbf{y} = \mathbf{c}
$$

Since $\mathbf{L}$ is lower triangular, this can be done **top-down**.

---

**3. Backward substitution**

Solve for final solution $\mathbf{x}$ in:

$$
\mathbf{U} \mathbf{x} = \mathbf{y}
$$

Since $\mathbf{U}$ is upper triangular, this can be done **bottom-up**.

---

```{code-cell} ipython3
from scipy.linalg import solve_triangular
# Right Hand Side of Equation
b = np.array([4, 2, 6], dtype=float)

# Step 1: permute b
c = P @ b

# Step 2: solve L y = P b
y = solve_triangular(L, c, lower=True)

# Step 3: solve U x = y
x = solve_triangular(U, y)

print("Solution x:", x)
```

---

$$
\boxed{
\mathbf{A} \mathbf{x} = \mathbf{b} \quad \Rightarrow \quad
\mathbf{P}^\top \mathbf{A} \mathbf{x} = \mathbf{L} \mathbf{U} \mathbf{x} = \mathbf{P}^\top \mathbf{b}
}
$$

Solve via:

* $\mathbf{L} \mathbf{y} = \mathbf{P}^\top \mathbf{b}$ (forward)
* $\mathbf{U} \mathbf{x} = \mathbf{y}$ (backward)

This is numerically efficient and stable, especially for **repeated solves** with the same $\mathbf{A}$.

## Summary: Why Gaussian Elimination and Matrix Forms Matter in Machine Learning

Understanding Gaussian elimination and matrix forms like **REF** and **RREF** is more than a theoretical exercise — it builds essential intuition and computational tools for many areas of **machine learning**.

Here’s how these concepts directly relate:

### Solving Linear Systems

Many machine learning algorithms boil down to solving systems of linear equations. For example:

* In **linear regression**, the optimal weights minimize a quadratic loss and satisfy the **normal equations**, which are linear:

  $$
  (\mathbf{X}^\top \mathbf{X}) \mathbf{w} = \mathbf{X}^\top \mathbf{y}
  $$

  Gaussian elimination provides an efficient way to solve these equations, especially for small- to medium-scale problems.

---

### Understanding Rank and Feature Spaces

* The **rank** of a data matrix $\mathbf{X}$ tells us the number of **linearly independent features**.
* Low-rank matrices appear naturally in **dimensionality reduction** (e.g. PCA), **collaborative filtering**, and **matrix completion**.
* Detecting whether features are redundant, or whether a system is under- or overdetermined, comes down to understanding row operations and rank.

---

### Interpreting Model Structure

* Matrices in **RREF** reveal directly which variables (features) are **pivotal** to a system — a perspective that underlies **feature selection**, **interpretability**, and **symbolic regression**.
* Understanding when systems have **unique**, **infinite**, or **no solutions** helps us reason about **well-posedness** and **overfitting** in models.

---

### Numerical Stability and Preconditioning

* Even when not done directly, Gaussian elimination underpins many **numerical algorithms** (e.g., LU decomposition, QR factorization).
* These are used in optimization, iterative solvers, and deep learning libraries for computing gradients, inverses, and solving systems in a stable way.

---

### Big Picture

> While machine learning often uses **high-level abstractions** and **automatic solvers**, understanding how these methods work at the matrix level helps build **intuition**, **debugging skills**, and **mathematical fluency** for real-world modeling.
