## Metric spaces

Metrics generalize the notion of distance from Euclidean space (although
metric spaces need not be vector spaces).

A **metric** on a set $S$ is a function $d : S \times S \to \mathbb{R}$
that satisfies

(i) $d(x,y) \geq 0$, with equality if and only if $x = y$

(ii) $d(x,y) = d(y,x)$

(iii) $d(x,z) \leq d(x,y) + d(y,z)$ (the so-called **triangle
      inequality**)

for all $x, y, z \in S$.

A key motivation for metrics is that they allow limits to be defined for
mathematical objects other than real numbers. We say that a sequence
$\{x_n\} \subseteq S$ converges to the limit $x$ if for any
$\epsilon > 0$, there exists $N \in \mathbb{N}$ such that
$d(x_n, x) < \epsilon$ for all $n \geq N$. Note that the definition for
limits of sequences of real numbers, which you have likely seen in a
calculus class, is a special case of this definition when using the
metric $d(x, y) = |x-y|$.


## Metric Spaces in Machine Learning

Metric spaces provide a generalized framework for measuring **distances** or similarities between data points. Many machine learning methods rely heavily on choosing appropriate metrics.

### Examples:

1. **Nearest Neighbor Classifiers (k-NN)**  
In k-NN, classification depends entirely on distances between points. While typically the Euclidean distance (metric) is used:
     
$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2,$$
     
other metrics such as the Manhattan (1-norm) or Minkowski (p-norm) can also be employed.

2. **Clustering Algorithms (k-means, DBSCAN)**  
Clustering methods rely fundamentally on the notion of a "distance" or metric. For example, **DBSCAN** clusters points that lie within a certain "distance" radius (metric-defined neighborhoods).

3. **String or Text Similarity (non-vector metrics)**  
Metrics can also be defined on non-numeric data. For example, the **Levenshtein distance** (edit distance) is widely used to measure similarity between strings, DNA sequences, or words in natural language processing (NLP). The Levenshtein distance counts how many single-character edits (insertions, deletions, substitutions) are needed to transform one string into another.
