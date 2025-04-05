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

# Vector spaces

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

## Euclidean space

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
plt.title('Vector Addition')

# Aspect ratio
plt.gca().set_aspect('equal', adjustable='box')

plt.legend(loc='lower right')
plt.show()
```
This visualization intuitively demonstrates how vectors combine to produce a resultant vector in Euclidean space by vector addition. The blue arrow represents vector $\mathbf{a}$, the green arrow represents vector $\mathbf{b}$ placed at the tip of vector $\mathbf{a}$, and the red arrow shows the resulting vector $\mathbf{a} + \mathbf{b}$.

```{code-cell} ipython3
:tags: [hide-input]
import matplotlib.pyplot as plt
import numpy as np

# Define original vector
vector_a = np.array([1.5, 2])

# Scalars to multiply with
scalars = [-1, 1.3]

# Colors for different scalars
colors = ['purple', 'green']
labels = [r'$-1 \cdot \mathbf{a}$', r'$1.3 \cdot \mathbf{a}$']

# Plotting
plt.figure(figsize=(6, 6))
ax = plt.gca()

# Plot original vector
ax.quiver(0, 0, vector_a[0], vector_a[1], angles='xy', scale_units='xy', scale=1, color='blue', label=r'$\mathbf{a}$')
ax.text(vector_a[0]/2, vector_a[1]/2, r'$\mathbf{a}$', color='blue', fontsize=14)

# Plot scaled vectors
for i, scalar in enumerate(scalars):
    scaled_vector = scalar * vector_a
    ax.quiver(0, 0, scaled_vector[0], scaled_vector[1], angles='xy', scale_units='xy', scale=1, color=colors[i], label=labels[i], alpha=0.5)
    ax.text(scaled_vector[0]/2, scaled_vector[1]/2, labels[i], color=colors[i], fontsize=14)

# Set limits and grid
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
plt.grid()

# Axes labels and title
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Scalar Multiplication of a Vector')

# Aspect ratio
ax.set_aspect('equal', adjustable='box')

plt.legend(loc='upper left')
plt.show()
```
This script shows the original vector $\mathbf{a}$ (in blue), and three scaled versions using scalars -1 and 0.5, and 2. The scaled vectors demonstrate inversion, shrinking, and stretching, respectively.

## $k$-means clustering in Euclidean space

Now, we have explored the basic properties of Euclidean space, we can apply these concepts to machine learning tasks.
We will discuss the $k$-means clustering algorithm in Euclidean space, which is a popular unsupervised learning method used to partition data into distinct groups based on their feature vectors. It only uses the operations of vector addition and scalar multiplication, which are the basic operations of a vector space.

The algorithm works as follows:

1. **Initialization**: Randomly select $k$ initial cluster centroids from the dataset.

2. Iterate over the following steps until convergence:
  - **Assignment Step**: For each data point, assign it to the nearest cluster centroid based on the Euclidean distance.

$$
\text{argmin}_k \|\mathbf{x} - \mathbf{c}_k\|^2
$$

  where $\mathbf{c}_k$ is the centroid of cluster $k$ and $\mathbf{x}$ is the data point.

  - **Update Step**: Recalculate the centroid vectors of the clusters by taking the mean of all data vectors assigned to each cluster. This uses the vector addition and scalar multiplication operations:

$$
\mathbf{c}_k = \frac{1}{N_k}\sum_{i:y_i=k} \mathbf{x}_i
$$

  where $N_k$ is the number of points assigned to cluster $k$ and $y_i$ is the label of the data point $\mathbf{x}_i$.

We will implement a python class for the $k$-means algorithm, which will include methods for fitting the model to the data and predicting cluster assignments for new data points.

```{code-cell} ipython3
import numpy as np

class KMeans:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    def fit(self, X, num_iterations=10):
        # Randomly initialize cluster centers
        self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        self.labels = np.zeros(X.shape[0])

        for _ in range(num_iterations):  # Iterate a fixed number of times
          old_labels = self.labels.copy()

          # Assign clusters based on closest center
          distances = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
          self.labels = np.argmin(distances, axis=1)

          # Update cluster centers
          for i in range(self.n_clusters):
              self.centers[i] = X[self.labels == i].mean(axis=0)
          
          # Check for convergence (optional)
          if np.all(self.labels == old_labels):
              break

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        return np.argmin(distances, axis=1)
```

Now we can generate a dataset of random vectors in $\mathbb{R}^2$ and apply the $k$-means algorithm to it to find the optimal cluster centers $\mathbf{c}_k$ and the cluster assignments for all data vectors.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0) 
n=50 # samples per c1uster
centers= [ [3,4], [8,3], [2,10], [9,9], ]
dataset=np.zeros((0,3))
sigmas = [ 0.5, 1, 1.5, 2 ]

# Generate clusters
for i in range(len(centers)):
    correlation=(np.random.rand()-0.5)*2
    center=centers[i]
    sigma=sigmas[i]
    cluster=np.random.multivariate_normal(center, [[sigma, correlation],[correlation, sigma]], n)
    label=np.zeros((n,1))+i
    cluster=np.hstack([cluster,label])
    dataset=np.vstack([dataset,cluster])

# Create a KMeans instance and fit it to the dataset
kmeans = KMeans(n_clusters=4)
kmeans.fit(dataset[:,:2], num_iterations=10)
cluster_assignments = kmeans.predict(dataset[:,:2])

# plot the cluster centers and the clustering of the dataset
plt.figure(figsize=(10,10))
ax = plt.scatter(kmeans.centers[:,0], kmeans.centers[:,1], c='red', s=200, alpha=0.5)
ax = plt.scatter(dataset[:,0], dataset[:,1], c=cluster_assignments, s=100, alpha=0.5)
ax = plt.title("$k$-means clustering of a dataset and cluster centers")
```
In the plot, the red dots represent the cluster centers $\mathbf{c}_k$ found by the $k$-means algorithm, while the colored points represent the data vectors $\mathbf{x}_n$ assigned to each cluster. The colors indicate which cluster each point belongs to.

