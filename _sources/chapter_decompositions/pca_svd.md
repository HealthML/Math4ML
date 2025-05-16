---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: math4ml
  language: python
  name: python3
---
+++ {"slideshow": {"slide_type": "slide"}}

# Principal Components Analysis using the SVD
### Step 6: Compute PCA via SVD (Optional)

Rather than computing $\mathbf{X}^\top \mathbf{X}$, you can also directly compute the **Singular Value Decomposition** of $\mathbf{X}$:

$$
\mathbf{X} = \mathbf{U} \Sigma \mathbf{V}^\top
$$

* $\mathbf{U} \in \mathbb{R}^{N \times N}$
* $\Sigma \in \mathbb{R}^{N \times D}$
* $\mathbf{V} \in \mathbb{R}^{D \times D}$

Then the principal components are the **first $k$ columns** of $\mathbf{V}$, and:

$$
\mathbf{Z} = \mathbf{X} \mathbf{V}_k
$$

is the reduced representation.

---

