# Proofs: Multivariate polynomials form a vector space

:::{prf:theorem} Multivariate polynomials form a vector space
:label: thm-multivariate-polynomial-vector-space
:nonumber:

Let $\mathbf{x} = [x_1, x_2, \dots, x_d] \in \mathbb{R}^d$ be a Euclidean vector.  
The set $P_n^d$ of all real-valued multivariate polynomials in $\mathbf{x}$ of total degree at most $n$ is a vector space.

That is,

$$
P_n^d = \left\{ p(\mathbf{x}) = \sum_{|\alpha| \leq n} a_\alpha \mathbf{x}^\alpha \;\middle|\; a_\alpha \in \mathbb{R},\ \alpha \in \mathbb{N}_0^d \right\},
$$

where $\alpha = [\alpha_1, \dots, \alpha_d]$ is a multi-index,  
$|\alpha| = \alpha_1 + \dots + \alpha_d$, and  
$\mathbf{x}^\alpha = x_1^{\alpha_1} \cdots x_d^{\alpha_d}$.
:::

We verify that $P_n^d$ satisfies all vector space axioms.

:::{prf:proof}
Let $p(\mathbf{x}), q(\mathbf{x}) \in P_n^d$ be two multivariate polynomials:

- **Closure under addition:**  
  Their sum is

  $$
  (p + q)(\mathbf{x}) = \sum_{|\alpha| \leq n} (a_\alpha + b_\alpha)\mathbf{x}^\alpha.
  $$

  This is again a polynomial of total degree ≤ $n$, so $p + q \in P_n^d$.

- **Closure under scalar multiplication:**  
  For any scalar $\lambda \in \mathbb{R}$,

  $$
  (\lambda p)(\mathbf{x}) = \sum_{|\alpha| \leq n} (\lambda a_\alpha)\mathbf{x}^\alpha \in P_n^d.
  $$

- **Zero polynomial:**  
  The zero function

  $$
  0(\mathbf{x}) = \sum_{|\alpha| \leq n} 0 \cdot \mathbf{x}^\alpha
  $$

  is in $P_n^d$, and serves as the additive identity.

- **Additive inverse:**  
  For each $p(\mathbf{x}) \in P_n^d$, define

  $$
  (-p)(\mathbf{x}) = \sum_{|\alpha| \leq n} (-a_\alpha)\mathbf{x}^\alpha.
  $$

  Then $p + (-p) = 0$.

- **Commutativity and associativity of addition:**  
  Follows from properties of addition in $\mathbb{R}$ for coefficients.

- **Distributivity of scalar multiplication:**  
  Also follows from distributivity in $\mathbb{R}$.

Therefore, $P_n^d$ satisfies all vector space axioms over $\mathbb{R}$, and is a vector space.
:::