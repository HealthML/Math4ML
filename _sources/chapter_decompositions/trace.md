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



