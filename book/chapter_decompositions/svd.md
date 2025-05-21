# Singular value decomposition

Singular value decomposition (SVD) is a widely applicable tool in linear
algebra. Its strength stems partially from the fact that *every matrix*
$\mathbf{A} \in \mathbb{R}^{m \times n}$ has an SVD (even non-square
matrices)! The decomposition goes as follows:

$$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\!\top\!}$$

where
$\mathbf{U} \in \mathbb{R}^{m \times m}$ and
$\mathbf{V} \in \mathbb{R}^{n \times n}$ are orthogonal matrices and
$\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$ is a diagonal matrix with
the **singular values** of $\mathbf{A}$ (denoted $\sigma_i$) on its
diagonal.

By convention, the singular values are given in non-increasing order,
i.e.

$$\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_{\min(m,n)} \geq 0$$

Only the first $r$ singular values are nonzero, where $r$ is the rank of
$\mathbf{A}$.

Observe that the SVD factors provide eigendecompositions for
$\mathbf{A}^{\!\top\!}\mathbf{A}$ and $\mathbf{A}\mathbf{A}^{\!\top\!}$:

$$\begin{aligned}
\mathbf{A}^{\!\top\!}\mathbf{A} &= (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\!\top\!})^{\!\top\!}\mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\!\top\!} = \mathbf{V}\mathbf{\Sigma}^{\!\top\!}\mathbf{U}^{\!\top\!}\mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\!\top\!} = \mathbf{V}\mathbf{\Sigma}^{\!\top\!}\mathbf{\Sigma}\mathbf{V}^{\!\top\!} \\
\mathbf{A}\mathbf{A}^{\!\top\!} &= \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\!\top\!}(\mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\!\top\!})^{\!\top\!} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\!\top\!}\mathbf{V}\mathbf{\Sigma}^{\!\top\!}\mathbf{U}^{\!\top\!} = \mathbf{U}\mathbf{\Sigma}\mathbf{\Sigma}^{\!\top\!}\mathbf{U}^{\!\top\!}
\end{aligned}$$

It follows immediately that the columns of $\mathbf{V}$
(the **right-singular vectors** of $\mathbf{A}$) are eigenvectors of
$\mathbf{A}^{\!\top\!}\mathbf{A}$, and the columns of $\mathbf{U}$ (the
**left-singular vectors** of $\mathbf{A}$) are eigenvectors of
$\mathbf{A}\mathbf{A}^{\!\top\!}$.

The matrices $\mathbf{\Sigma}^{\!\top\!}\mathbf{\Sigma}$ and
$\mathbf{\Sigma}\mathbf{\Sigma}^{\!\top\!}$ are not necessarily the same
size, but both are diagonal with the squared singular values
$\sigma_i^2$ on the diagonal (plus possibly some zeros). Thus the
singular values of $\mathbf{A}$ are the square roots of the eigenvalues
of $\mathbf{A}^{\!\top\!}\mathbf{A}$ (or equivalently, of
$\mathbf{A}\mathbf{A}^{\!\top\!}$)[^5].

## Some useful matrix identities

In the following, we present a number of important identities for the SVD.

### Matrix-vector product as linear combination of matrix columns

*Proposition.* 
Let $\mathbf{x} \in \mathbb{R}^n$ be a vector and
$\mathbf{A} \in \mathbb{R}^{m \times n}$ a matrix with columns
$\mathbf{a}_1, \dots, \mathbf{a}_n$. Then

$$\mathbf{A}\mathbf{x} = \sum_{i=1}^n x_i\mathbf{a}_i$$

This identity is extremely useful in understanding linear operators in
terms of their matrices' columns. The proof is very simple (consider
each element of $\mathbf{A}\mathbf{x}$ individually and expand by
definitions) but it is a good exercise to convince yourself.

### Sum of outer products as matrix-matrix product

An **outer product** is an expression of the form
$\mathbf{a}\mathbf{b}^{\!\top\!}$, where $\mathbf{a} \in \mathbb{R}^m$
and $\mathbf{b} \in \mathbb{R}^n$. By inspection it is not hard to see
that such an expression yields an $m \times n$ matrix such that

$$[\mathbf{a}\mathbf{b}^{\!\top\!}]_{ij} = a_ib_j$$

It is not
immediately obvious, but the sum of outer products is actually
equivalent to an appropriate matrix-matrix product! We formalize this
statement as

*Proposition.* 
Let $\mathbf{a}_1, \dots, \mathbf{a}_k \in \mathbb{R}^m$ and
$\mathbf{b}_1, \dots, \mathbf{b}_k \in \mathbb{R}^n$. Then

$$\sum_{\ell=1}^k \mathbf{a}_\ell\mathbf{b}_\ell^{\!\top\!} = \mathbf{A}\mathbf{B}^{\!\top\!}$$

where

$$\mathbf{A} = \begin{bmatrix}\mathbf{a}_1 & \cdots & \mathbf{a}_k\end{bmatrix}, \hspace{0.5cm} \mathbf{B} = \begin{bmatrix}\mathbf{b}_1 & \cdots & \mathbf{b}_k\end{bmatrix}$$

*Proof.* For each $(i,j)$, we have

$$\left[\sum_{\ell=1}^k \mathbf{a}_\ell\mathbf{b}_\ell^{\!\top\!}\right]_{ij} = \sum_{\ell=1}^k [\mathbf{a}_\ell\mathbf{b}_\ell^{\!\top\!}]_{ij} = \sum_{\ell=1}^k [\mathbf{a}_\ell]_i[\mathbf{b}_\ell]_j = \sum_{\ell=1}^k A_{i\ell}B_{j\ell}$$

This last expression should be recognized as an inner product between
the $i$th row of $\mathbf{A}$ and the $j$th row of $\mathbf{B}$, or
equivalently the $j$th column of $\mathbf{B}^{\!\top\!}$. Hence by the
definition of matrix multiplication, it is equal to
$[\mathbf{A}\mathbf{B}^{\!\top\!}]_{ij}$. ◻

