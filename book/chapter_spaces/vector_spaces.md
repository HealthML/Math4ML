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

## Vector spaces

**Vector spaces** are the basic setting in which linear algebra happens.
A vector space $V$ is a set (the elements of which are called
**vectors**) on which two operations are defined: vectors can be added
together, and vectors can be multiplied by real numbers called **scalars**.
$V$ must satisfy

(i) There exists an additive identity (written $\mathbf{0}$) in $V$ such
    that $\mathbf{x}+\mathbf{0} = \mathbf{x}$ for all $\mathbf{x} \in V$

(ii) For each $\mathbf{x} \in V$, there exists an additive inverse
     (written $\mathbf{-x}$) such that
     $\mathbf{x}+(\mathbf{-x}) = \mathbf{0}$

(iii) There exists a multiplicative identity (written $1$) in
      $\mathbb{R}$ such that $1\mathbf{x} = \mathbf{x}$ for all
      $\mathbf{x} \in V$

(iv) Commutativity: $\mathbf{x}+\mathbf{y} = \mathbf{y}+\mathbf{x}$ for
     all $\mathbf{x}, \mathbf{y} \in V$

(v) Associativity:
    $(\mathbf{x}+\mathbf{y})+\mathbf{z} = \mathbf{x}+(\mathbf{y}+\mathbf{z})$
    and $\alpha(\beta\mathbf{x}) = (\alpha\beta)\mathbf{x}$ for all
    $\mathbf{x}, \mathbf{y}, \mathbf{z} \in V$ and
    $\alpha, \beta \in \mathbb{R}$

(vi) Distributivity:
     $\alpha(\mathbf{x}+\mathbf{y}) = \alpha\mathbf{x} + \alpha\mathbf{y}$
     and $(\alpha+\beta)\mathbf{x} = \alpha\mathbf{x} + \beta\mathbf{x}$
     for all $\mathbf{x}, \mathbf{y} \in V$ and
     $\alpha, \beta \in \mathbb{R}$

### Euclidean space

The quintessential vector space is **Euclidean space**, which we denote
$\mathbb{R}^n$. The vectors in this space consist of $n$-tuples of real
numbers:

$$\mathbf{x} = (x_1, x_2, \dots, x_n)$$

For our purposes, it
will be useful to think of them as $n \times 1$ matrices, or **column
vectors**:

$$\mathbf{x} = \begin{bmatrix}x_1 \\ x_2 \\ \vdots \\ x_n\end{bmatrix}$$

Addition and scalar multiplication are defined component-wise on vectors
in $\mathbb{R}^n$:

$$\mathbf{x} + \mathbf{y} = \begin{bmatrix}x_1 + y_1 \\ \vdots \\ x_n + y_n\end{bmatrix}, \hspace{0.5cm} \alpha\mathbf{x} = \begin{bmatrix}\alpha x_1 \\ \vdots \\ \alpha x_n\end{bmatrix}$$

Euclidean space is used to mathematically represent physical space, with notions such as distance, length, and angles.
Although it becomes hard to visualize for $n > 3$, these concepts generalize mathematically in obvious ways. 
Even when you're working in more general settings than $\mathbb{R}^n$, it is often useful to visualize vector addition and scalar multiplication in terms of 2D vectors in the plane or 3D vectors in space.

## Visualizing Vector Addition

```{code-cell} ipython3
:tags: [hide-input]
import matplotlib.pyplot as plt
import numpy as np

#  Below is a Python script using Matplotlib to visualize vector addition in 2D space. This script illustrates how vectors combine graphically.

# Define vectors
vector_a = np.array([2, 3])
vector_b = np.array([4, 1])

# Vector addition
vector_sum = vector_a + vector_b

# Plotting vectors
plt.figure(figsize=(6, 6))
ax = plt.gca()

# Plot vector a
ax.quiver(0, 0, vector_a[0], vector_a[1], angles='xy', scale_units='xy', scale=1, color='blue', label='$\\mathbf{a}$')
ax.text(vector_a[0]/2, vector_a[1]/2, '$\\mathbf{a}$', color='blue', fontsize=14)

# Plot vector b starting from the tip of vector a
ax.quiver(vector_a[0], vector_a[1], vector_b[0], vector_b[1], angles='xy', scale_units='xy', scale=1, color='green', label='$\\mathbf{b}$')
ax.text(vector_a[0] + vector_b[0]/2, vector_a[1] + vector_b[1]/2, '$\\mathbf{b}$', color='green', fontsize=14)

# Plot resultant vector
ax.quiver(0, 0, vector_sum[0], vector_sum[1], angles='xy', scale_units='xy', scale=1, color='red', label='$\\mathbf{a} + \\mathbf{b}$')
ax.text(vector_sum[0]/2, vector_sum[1]/2, '$\\mathbf{a}+\\mathbf{b}$', color='red', fontsize=14)

# Set limits and grid
ax.set_xlim(0, 7)
ax.set_ylim(0, 5)
plt.grid()

# Axes labels and title
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Visualization of Vector Addition')

# Aspect ratio
plt.gca().set_aspect('equal', adjustable='box')

plt.legend(loc='lower right')
plt.show()
```

### Explanation:

- **Blue arrow** represents vector $\mathbf{a}$.
- **Green arrow** represents vector $\mathbf{b}$ placed at the tip of vector $\mathbf{a}$.
- **Red arrow** shows the resulting vector $\mathbf{a} + \mathbf{b}$.

This visualization intuitively demonstrates how vectors combine to produce a resultant vector in Euclidean space.




Here's how you can neatly integrate the proof that polynomials form a vector space, and then connect that intuitively to an ML example using polynomial features:

## Example: Polynomials as a Vector Space

Consider the set $P_n$ of all real-valued polynomials with degree at most $n$:


$$P_n = \{ a_0 + a_1x + a_2x^2 + \dots + a_nx^n \mid a_i \in \mathbb{R} \}.$$

We show that $P_n$ is a vector space by verifying the vector space axioms:

### Proof:

Let $p(x), q(x), r(x) \in P_n$ be arbitrary polynomials:

- **Closure under addition:**  
  The sum $p(x) + q(x)$ is:
  
$$(p+q)(x) = (a_0 + b_0) + (a_1 + b_1)x + \dots + (a_n + b_n)x^n.$$

  Clearly, this is also a polynomial of degree at most $n$, so $p(x) + q(x) \in P_n$.

- **Closure under scalar multiplication:**  
  For any scalar $\alpha \in \mathbb{R}$, the scalar multiplication $\alpha p(x)$ is:
  
$$(\alpha p)(x) = \alpha a_0 + \alpha a_1 x + \dots + \alpha a_n x^n,$$

  which remains in $P_n$.

- **Existence of additive identity:**  
  The zero polynomial $0(x) = 0 + 0x + \dots + 0x^n$ serves as the additive identity:
  
$$p(x) + 0(x) = p(x).$$

- **Existence of additive inverse:**  
  For every polynomial $p(x) = a_0 + a_1 x + \dots + a_n x^n$, there exists $-p(x)$:
  
$$-p(x) = -a_0 - a_1 x - \dots - a_n x^n,$$
  
  such that $p(x) + (-p(x)) = 0(x)$.

- **Commutativity and associativity:**  
  Addition of polynomials and scalar multiplication clearly satisfy commutativity and associativity due to the commutativity and associativity of real numbers.

- **Distributivity:**  
  Scalar multiplication distributes over polynomial addition, and addition of scalars distributes over scalar multiplication, directly inherited from real numbers.

Thus, all vector space axioms are satisfied, and $P_n$ is indeed a vector space.

## Polynomial Features in Machine Learning

Using polynomial vector spaces, we can enhance simple machine learning algorithms by explicitly representing complex, nonlinear relationships.

### Example: Polynomial Features in Linear Regression and Nearest Centroid Classifier

Consider a simple ML task—fitting or classifying data that's clearly nonlinear:

- **Original Data**: $\mathbf{x} \in \mathbb{R}$, one-dimensional feature.
- **Polynomial Feature Map**: Transform the input into a polynomial vector space, e.g.:
  
$$\phi(x) = [1, x, x^2, \dots, x^n]^\top.$$

#### Linear Regression with Polynomial Features:
Instead of fitting a line $y = w_0 + w_1 x$, we fit:

$$y = w_0 + w_1 x + w_2 x^2 + \dots + w_n x^n = \mathbf{w}^\top \phi(x)$$

This can model more complex curves while still being linear in the parameters $\mathbf{w}$.

#### Nearest Centroid Classifier with Polynomial Features:
Instead of measuring Euclidean distance in the original feature space, we measure it in the polynomial feature space:

- Centroids become averages of polynomial features:

$$\mathbf{c}_k = \frac{1}{N_k}\sum_{i:y_i=k} \phi(x_i).$$

- Classification uses distances in this polynomial space:

$$\hat{y} = \arg\min_k \|\phi(x)-\mathbf{c}_k\|.$$

This simple feature mapping enables the classifier to separate nonlinear boundaries (circles, ellipses, curves) easily and effectively.


### Insights for Students:

- Recognizing polynomials as vector spaces gives a clear mathematical justification for methods using polynomial features.
- Polynomial vector spaces allow linear methods to handle nonlinear patterns by embedding data into higher-dimensional spaces.
- Even simple classifiers or regressors gain expressive power through such explicit polynomial feature expansions.
