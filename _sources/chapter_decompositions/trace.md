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
# Trace

The **trace** of a square matrix is the sum of its diagonal entries:

$$\operatorname{tr}(\mathbf{A}) = \sum_{i=1}^n A_{ii}$$

The trace has several nice algebraic properties:

(i) $\operatorname{tr}(\mathbf{A}+\mathbf{B}) = \operatorname{tr}(\mathbf{A}) + \operatorname{tr}(\mathbf{B})$

(ii) $\operatorname{tr}(\alpha\mathbf{A}) = \alpha\operatorname{tr}(\mathbf{A})$

(iii) $\operatorname{tr}(\mathbf{A}^{\!\top\!}) = \operatorname{tr}(\mathbf{A})$

(iv) $\operatorname{tr}(\mathbf{A}\mathbf{B}\mathbf{C}\mathbf{D}) = \operatorname{tr}(\mathbf{B}\mathbf{C}\mathbf{D}\mathbf{A}) = \operatorname{tr}(\mathbf{C}\mathbf{D}\mathbf{A}\mathbf{B}) = \operatorname{tr}(\mathbf{D}\mathbf{A}\mathbf{B}\mathbf{C})$

The first three properties follow readily from the definition.
The last is known as **invariance under cyclic permutations**.
Note that the matrices cannot be reordered arbitrarily, for example
$\operatorname{tr}(\mathbf{A}\mathbf{B}\mathbf{C}\mathbf{D}) \neq \operatorname{tr}(\mathbf{B}\mathbf{A}\mathbf{C}\mathbf{D})$
in general.
Also, there is nothing special about the product of four matrices -- analogous rules hold for more or fewer matrices.


```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# Set up matrices for demonstration
A = np.array([[2, 1],
              [0, 3]])
B = np.array([[1, -1],
              [2, 0]])
alpha = 2.5

# Compute traces
trace_A = np.trace(A)
trace_B = np.trace(B)
trace_A_plus_B = np.trace(A + B)
trace_alphaA = np.trace(alpha * A)
trace_AT = np.trace(A.T)

# Cyclic permutation example
C = np.array([[0, 2],
              [1, 1]])
D = np.array([[1, 1],
              [0, -1]])

product_1 = A @ B @ C @ D
product_2 = B @ C @ D @ A
product_3 = C @ D @ A @ B
product_4 = D @ A @ B @ C

traces = [
    np.trace(product_1),
    np.trace(product_2),
    np.trace(product_3),
    np.trace(product_4),
]

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (i) Linearity
axes[0, 0].bar(['tr(A)', 'tr(B)', 'tr(A+B)'], [trace_A, trace_B, trace_A_plus_B],
               color=['blue', 'green', 'purple'])
axes[0, 0].set_title("Linearity: tr(A + B) = tr(A) + tr(B)")
axes[0, 0].axhline(trace_A + trace_B, color='gray', linestyle='--', label='Expected tr(A) + tr(B)')
axes[0, 0].legend()

# (ii) Scalar multiplication
axes[0, 1].bar(['tr(A)', 'tr(αA)'], [trace_A, trace_alphaA], color=['blue', 'orange'])
axes[0, 1].axhline(alpha * trace_A, color='gray', linestyle='--', label='Expected α·tr(A)')
axes[0, 1].set_title("Scaling: tr(αA) = α·tr(A)")
axes[0, 1].legend()

# (iii) Transpose invariance
axes[1, 0].bar(['tr(A)', 'tr(Aᵀ)'], [trace_A, trace_AT], color=['blue', 'red'])
axes[1, 0].set_title("Transpose: tr(Aᵀ) = tr(A)")

# (iv) Cyclic permutation invariance
axes[1, 1].bar(['ABCD', 'BCDA', 'CDAB', 'DABC'], traces, color='teal')
axes[1, 1].axhline(traces[0], color='gray', linestyle='--', label='Invariant trace')
axes[1, 1].set_title("Cyclic Permutation: tr(ABCD) = tr(BCDA) = ...")
axes[1, 1].legend()

plt.suptitle("Visualizing the Properties of the Trace Operator", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

```

Interestingly, the trace of a matrix is equal to the sum of its eigenvalues (repeated according to multiplicity):

$$\operatorname{tr}(\mathbf{A}) = \sum_i \lambda_i(\mathbf{A})$$
