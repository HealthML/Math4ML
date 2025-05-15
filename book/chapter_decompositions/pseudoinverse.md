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
# Moore-Penrose Pseudoinverse
The Moore-Penrose pseudoinverse is a generalization of the matrix inverse that can be applied to non-square or singular matrices. It is denoted as $ A^+ $ for a matrix $ A $. The pseudoinverse satisfies the following properties:
1. **Existence**: The pseudoinverse exists for any matrix $ A $.
2. **Uniqueness**: The pseudoinverse is unique.
3. **Properties**:
   - $ A A^+ A = A $
   - $ A^+ A A^+ = A^+ $
   - $ (A A^+)^\top = A A^+ $
   - $ (A^+ A)^\top = A^+ A $
4. **Rank**: The rank of $ A^+ $ is equal to the rank of $ A $.
5. **Singular Value Decomposition (SVD)**: The pseudoinverse can be computed using the singular value decomposition of $ A $. If $ A = U \Sigma V^\top $, where $ U $ and $ V $ are orthogonal matrices and $ \Sigma $ is a diagonal matrix with singular values, then:
   
   $$
   A^+ = V \Sigma^+ U^\top
   $$
   where $ \Sigma^+ $ is obtained by taking the reciprocal of the non-zero singular values in $ \Sigma $ and transposing the resulting matrix.
6. **Applications**: The pseudoinverse is used in various applications, including solving linear systems, least squares problems, and in machine learning algorithms such as linear regression.
7. **Least Squares Solution**: The pseudoinverse provides a least squares solution to the equation $ Ax = b $ when $ A $ is not square or has no unique solution. The least squares solution is given by:
   
   $$
   x = A^+ b
   $$
8. **Geometric Interpretation**: The pseudoinverse can be interpreted geometrically as the projection of a vector onto the column space of $ A $.
9. **Computational Considerations**: The computation of the pseudoinverse can be done efficiently using numerical methods, such as the SVD, especially for large matrices.
10. **Limitations**: The pseudoinverse may not be suitable for all applications, especially when the matrix is ill-conditioned or has a high condition number.
