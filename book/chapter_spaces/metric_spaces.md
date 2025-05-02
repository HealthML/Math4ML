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
# Metric spaces

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
This choice can significantly affect the performance of algorithms, especially in tasks like clustering, classification, and nearest neighbor searches.

### **Example: Edit Distance (Levenshtein Distance)**

A common metric in machine learning—especially in natural language processing (NLP)—is the **edit distance**, which measures the minimum number of insertions, deletions, and substitutions required to transform one string into another.

Formally, for two strings $x, y \in \Sigma^*$ (the set of all finite strings over some alphabet $\Sigma$), the **edit distance** $d_{\text{edit}}(x, y)$ is defined as the minimum number of single-character operations (insert, delete, substitute) needed to transform $x$ into $y$.

- **Metric properties:**  
  - $d_{\text{edit}}(x, y) \geq 0$, and $= 0$ iff $x = y$  
  - $d_{\text{edit}}(x, y) = d_{\text{edit}}(y, x)$  
  - Satisfies the triangle inequality

- **Elements are no Vector Space:**  
  Strings are not elements of a vector space: there’s no well-defined vector addition or scalar multiplication. Still, the edit distance is a valid metric and is widely used in applications like spell checking, DNA sequence alignment, and fuzzy string matching.

```{code-cell} ipython3
import numpy as np

def edit_distance(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    dp = np.zeros((len_s1 + 1, len_s2 + 1), dtype=int)

    for i in range(len_s1 + 1):
        dp[i][0] = i
    for j in range(len_s2 + 1):
        dp[0][j] = j

    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i-1][j] + 1,     # deletion
                           dp[i][j-1] + 1,     # insertion
                           dp[i-1][j-1] + cost) # substitution
    return dp[len_s1][len_s2]

# Example usage
print("Edit distance between 'kitten' and 'sitting':", edit_distance("kitten", "sitting"))
```


**Nearest Neighbor Classifiers (k-NN)**  
In k-NN, classification depends entirely on distances between points. While typically the Euclidean distance (metric) $d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2,$ is used, other metrics can be applied. For example, in text classification, the edit distance is often more appropriate than Euclidean distance. The k-NN algorithm classifies a query point based on the labels of its k nearest neighbors in the training set, using a distance metric to determine "closeness." The edit distance is particularly useful for string data, where it quantifies how many single-character edits are needed to transform one string into another.

```{code-cell} ipython3
import numpy as np
from collections import Counter

def knn_classify(query, train_data, train_labels, k=3):
      """Classify a query string using k-NN based on edit distance."""
      distances = np.zeros(len(train_data))
      for i, sample in enumerate(train_data):
          distances[i] = edit_distance(query, sample)
      k_nearest = np.argsort(distances)[:k]
      labels = [train_labels[i] for i in k_nearest]
      return Counter(labels).most_common(1)[0][0]
```

```{code-cell} ipython3
:tags: [hide-input]
# Example of k-NN classification using edit distance
import numpy as np
from collections import Counter

# --- Edit distance function ---
def edit_distance(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    dp = np.zeros((len_s1 + 1, len_s2 + 1), dtype=int)

    for i in range(len_s1 + 1):
        dp[i][0] = i
    for j in range(len_s2 + 1):
        dp[0][j] = j

    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[len_s1][len_s2]

# --- Training data ---
train_strings = [
    "apple", "banana", "pear", "peach",     # fruits
    "tiger", "lion", "zebra", "monkey"      # animals
]
train_labels = [
    "fruit", "fruit", "fruit", "fruit",
    "animal", "animal", "animal", "animal"
]

# --- Query strings to classify ---
queries = ["appl", "pears", "tigre", "monke", "peeko", "leon", "panana"]

# --- Perform classification ---
print("k-NN classification using edit distance:\n")
for query in queries:
    prediction = knn_classify(query, train_strings, train_labels, k=3)
    print(f"'{query}' → predicted label: {prediction}")
```

**Density-Based Spatial Clustering of Applications with Noise (DBSCAN)**  
Clustering methods rely fundamentally on the notion of a "distance" or metric. For example, **DBSCAN** clusters points that lie within a certain "distance" radius (metric-defined neighborhoods).

DBSCAN is a density-based clustering algorithm that groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions. It requires two parameters:
- **Epsilon (eps)**: The maximum distance between two points for them to be considered as in the same neighborhood.
- **MinPts**: The minimum number of points required to form a dense region.

DBSCAN groups data points based on the idea of density connectivity. It defines clusters as areas of high point density separated by regions of low density. The algorithm uses two parameters: **ε (epsilon)**, which defines the radius of a neighborhood around a point, and **minPts**, the minimum number of points required to form a dense region. A point is a **core point** if there are at least *minPts* points (including itself) within its ε-neighborhood. A **border point** lies within ε of a core point but has fewer than *minPts* neighbors itself. Points that are neither core nor border are considered **noise** or **outliers**. DBSCAN starts by picking an unvisited point and checking if it is a core point; if so, it forms a new cluster by recursively visiting all density-connected points. This approach allows DBSCAN to discover clusters of arbitrary shape and to handle noise effectively, all without requiring the number of clusters to be specified in advance.


Let's look at an example where we use edit distance as a metric to cluster strings the strings  "apple", "appl", "appel", "banana", "bananna", "peach", "peaco", "zebra", "zebro", "zeebar", "lion" using DBSCAN. The edit distance is used to determine the similarity between strings, and we will cluster them based on a specified epsilon (eps) and minimum points (min_pts).

The output of the algorithm with `eps=2` and `min_pts=2` will show how the strings are grouped into clusters based on their edit distances. Strings that are similar (i.e., have a small edit distance) will be clustered together, while those that are more distant will be marked as noise or belong to different clusters.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
from collections import deque

# --- Edit distance function ---
def edit_distance(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    dp = np.zeros((len_s1 + 1, len_s2 + 1), dtype=int)

    for i in range(len_s1 + 1):
        dp[i][0] = i
    for j in range(len_s2 + 1):
        dp[0][j] = j

    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[len_s1][len_s2]

# --- DBSCAN-like clustering with edit distance ---
def dbscan_strings(strings, eps=2, min_pts=2):
    n = len(strings)
    labels = [-1] * n  # -1 means noise
    cluster_id = 0

    def region_query(i):
        return [j for j in range(n) if edit_distance(strings[i], strings[j]) <= eps]

    def expand_cluster(i, neighbors):
        labels[i] = cluster_id
        queue = deque(neighbors)
        while queue:
            j = queue.popleft()
            if labels[j] == -1:  # previously marked as noise
                labels[j] = cluster_id
            if labels[j] != None:
                continue
            labels[j] = cluster_id
            new_neighbors = region_query(j)
            if len(new_neighbors) >= min_pts:
                queue.extend(new_neighbors)

    for i in range(n):
        if labels[i] != -1:
            continue
        neighbors = region_query(i)
        if len(neighbors) < min_pts:
            labels[i] = -1  # noise
        else:
            expand_cluster(i, neighbors)
            cluster_id += 1

    return labels

# --- Example strings ---
strings = [
    "apple", "appl", "appel", "banana", "bananna",
    "peach", "peaco", "zebra", "zebro", "zeebar", "lion"
]

# --- Run clustering ---
labels = dbscan_strings(strings, eps=2, min_pts=2)

# --- Show results ---
print("DBSCAN-style clustering with edit distance:\n")
clusters = {}
for string, label in zip(strings, labels):
    clusters.setdefault(label, []).append(string)

for cid, members in sorted(clusters.items()):
    cname = f"Cluster {cid}" if cid != -1 else "Noise"
    print(f"{cname}: {members}")
```

