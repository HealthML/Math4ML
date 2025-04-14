Absolutely! Below is a Python script using `matplotlib` that visualizes the **DBSCAN algorithm in 2D Euclidean space**.

It performs the clustering step-by-step on synthetic data and shows:

- Core points (in bold)
- Border points
- Noise
- The ε-neighborhoods
- Clusters colored distinctly

---

### 📜 Python Script: Visualizing DBSCAN in \(\mathbb{R}^2\)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors
from collections import deque

# Generate synthetic 2D data (e.g., moons)
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# DBSCAN Parameters
eps = 0.2
min_pts = 5

# --- Step 1: Classify each point ---
n = len(X)
labels = [-1] * n  # -1 = noise
visited = [False] * n
cluster_id = 0

# Precompute distances
nbrs = NearestNeighbors(radius=eps).fit(X)
neighborhoods = nbrs.radius_neighbors(X, return_distance=False)

def expand_cluster(i, neighbors):
    global cluster_id
    labels[i] = cluster_id
    queue = deque(neighbors)
    while queue:
        j = queue.popleft()
        if not visited[j]:
            visited[j] = True
            j_neighbors = neighborhoods[j]
            if len(j_neighbors) >= min_pts:
                queue.extend(j_neighbors)
        if labels[j] == -1:
            labels[j] = cluster_id

for i in range(n):
    if visited[i]:
        continue
    visited[i] = True
    neighbors = neighborhoods[i]
    if len(neighbors) < min_pts:
        labels[i] = -1  # noise
    else:
        expand_cluster(i, neighbors)
        cluster_id += 1

# --- Plotting ---
plt.figure(figsize=(8, 6))
unique_labels = set(labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(sorted(unique_labels), colors):
    if k == -1:
        col = 'gray'  # noise
        label = "Noise"
    else:
        label = f"Cluster {k}"
    class_member_mask = np.array(labels) == k
    core_mask = np.array([len(neighborhoods[i]) >= min_pts for i in range(n)])
    border_mask = np.logical_and(class_member_mask, ~core_mask)
    core_points = X[np.logical_and(class_member_mask, core_mask)]
    border_points = X[border_mask]
    
    plt.plot(core_points[:, 0], core_points[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=8, label=label)
    plt.plot(border_points[:, 0], border_points[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=4)

# Draw eps-radius circles around a few core points
for i in range(0, n, 30):
    if len(neighborhoods[i]) >= min_pts:
        circle = plt.Circle((X[i, 0], X[i, 1]), eps, color='black', fill=False, linestyle='--', alpha=0.3)
        plt.gca().add_patch(circle)

plt.title("DBSCAN Clustering in Euclidean Space")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.legend()
plt.axis("equal")
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()
```

---

### 🧠 What This Script Does:

- Uses `sklearn.datasets.make_moons()` to generate non-convex clusters.
- Manually implements **DBSCAN** logic:
  - Classifies **core**, **border**, and **noise** points
  - Expands clusters from core points
- Visualizes:
  - **Core points**: large colored circles
  - **Border points**: smaller colored circles
  - **Noise points**: gray
  - **ε-neighborhoods** around selected points

This is a great way to *see* DBSCAN in action and understand how clusters form based on density, not shape.

Would you like an interactive slider version to explore different values of ε and `min_pts`?