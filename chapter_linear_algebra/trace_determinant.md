## Trace

The **trace** of a square matrix is the sum of its diagonal entries:

$$\operatorname{tr}(\mathbf{A}) = \sum_{i=1}^n A_{ii}$$

The trace has several nice
algebraic properties:

(i) $\operatorname{tr}(\mathbf{A}+\mathbf{B}) = \operatorname{tr}(\mathbf{A}) + \operatorname{tr}(\mathbf{B})$

(ii) $\operatorname{tr}(\alpha\mathbf{A}) = \alpha\operatorname{tr}(\mathbf{A})$

(iii) $\operatorname{tr}(\mathbf{A}^{\!\top\!}) = \operatorname{tr}(\mathbf{A})$

(iv) $\operatorname{tr}(\mathbf{A}\mathbf{B}\mathbf{C}\mathbf{D}) = \operatorname{tr}(\mathbf{B}\mathbf{C}\mathbf{D}\mathbf{A}) = \operatorname{tr}(\mathbf{C}\mathbf{D}\mathbf{A}\mathbf{B}) = \operatorname{tr}(\mathbf{D}\mathbf{A}\mathbf{B}\mathbf{C})$

The first three properties follow readily from the definition. The last
is known as **invariance under cyclic permutations**. Note that the
matrices cannot be reordered arbitrarily, for example
$\operatorname{tr}(\mathbf{A}\mathbf{B}\mathbf{C}\mathbf{D}) \neq \operatorname{tr}(\mathbf{B}\mathbf{A}\mathbf{C}\mathbf{D})$
in general. Also, there is nothing special about the product of four
matrices -- analogous rules hold for more or fewer matrices.

Interestingly, the trace of a matrix is equal to the sum of its
eigenvalues (repeated according to multiplicity):

$$\operatorname{tr}(\mathbf{A}) = \sum_i \lambda_i(\mathbf{A})$$

## Determinant

The **determinant** of a square matrix can be defined in several
different confusing ways, none of which are particularly important for
our purposes; go look at an introductory linear algebra text (or
Wikipedia) if you need a definition. But it's good to know the
properties:

(i) $\det(\mathbf{I}) = 1$

(ii) $\det(\mathbf{A}^{\!\top\!}) = \det(\mathbf{A})$

(iii) $\det(\mathbf{A}\mathbf{B}) = \det(\mathbf{A})\det(\mathbf{B})$

(iv) $\det(\mathbf{A}^{-1}) = \det(\mathbf{A})^{-1}$

(v) $\det(\alpha\mathbf{A}) = \alpha^n \det(\mathbf{A})$

Interestingly, the determinant of a matrix is equal to the product of
its eigenvalues (repeated according to multiplicity):

$$\det(\mathbf{A}) = \prod_i \lambda_i(\mathbf{A})$$