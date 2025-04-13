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
## Hyperplanes and Linear Classification

The Nearest Centroid Classifier belongs to a broader family of linear classifiers, meaning the decision boundary is a linear function separating the classes. For binary classification, such as distinguishing benign from malignant biopsies, the decision boundary separates the space into two **halfspaces**.

### A Hyperplane in Euclidean Space

In an $n$-dimensional Euclidean space $\mathbb{R}^n$, a **hyperplane** is a flat, $(n-1)$-dimensional subset that divides the space into two distinct halfspaces.

Formally, a hyperplane can be expressed as:

$$
\mathbf{w}^T \mathbf{x} + b = 0
$$

where:
- $\mathbf{w} \in \mathbb{R}^n$ is a normal vector perpendicular to the hyperplane.
- $b \in \mathbb{R}$ is a scalar offset determining the hyperplane's position relative to the origin.


A halfspace in $\mathbb{R}^n$ is defined by:

$$
\{\mathbf{x} \in \mathbb{R}^n \mid \mathbf{w}^T \mathbf{x} + b \geq 0\}
$$

Where:
- $\mathbf{w}$ is a vector perpendicular to the decision boundary
- $b$ is a scalar offset


## Decision Boundary for the Nearest Centroid Classifier
For the nearest centroid classifier, the boundary is a hyperplane precisely halfway between the centroids, perpendicular to the line connecting them.

### Deriving the Separating Hyperplane for a Two-Class Problem

Consider two classes with centroids $\mathbf{c}_1$ and $\mathbf{c}_2$.
The decision boundary (hyperplane) is the set of points equidistant from both centroids. 
Mathematically, we have:

$$
\|\mathbf{x}-\mathbf{c}_1\|^2 = \|\mathbf{x}-\mathbf{c}_2\|^2
$$

Expanding both sides and simplifying:

$$
\mathbf{x}^T\mathbf{x} - 2\mathbf{c}_1^T\mathbf{x} + \mathbf{c}_1^T\mathbf{c}_1 = \mathbf{x}^T\mathbf{x} - 2\mathbf{c}_2^T\mathbf{x} + \mathbf{c}_2^T\mathbf{c}_2
$$

Subtracting $\mathbf{x}^T\mathbf{x}$ from both sides:

$$
-2\mathbf{c}_1^T\mathbf{x} + \mathbf{c}_1^T\mathbf{c}_1 = -2\mathbf{c}_2^T\mathbf{x} + \mathbf{c}_2^T\mathbf{c}_2
$$

Rearranging to isolate $\mathbf{x}$:

$$
2(\mathbf{c}_2 - \mathbf{c}_1)^T \mathbf{x} = \mathbf{c}_2^T\mathbf{c}_2 - \mathbf{c}_1^T\mathbf{c}_1
$$

Thus, the separating hyperplane equation is:

$$
(\mathbf{c}_2 - \mathbf{c}_1)^T \mathbf{x} - \frac{1}{2}(\mathbf{c}_2^T\mathbf{c}_2 - \mathbf{c}_1^T\mathbf{c}_1) = 0
$$

Here, the normal vector $\mathbf{w}$ is given by $(\mathbf{c}_2 - \mathbf{c}_1)$, and the offset $b$ is $-\frac{1}{2}(\mathbf{c}_2^T\mathbf{c}_2 - \mathbf{c}_1^T\mathbf{c}_1)$.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
from numpy.linalg import norm    
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# load the dataset
df = pd.read_csv('data/breast-cancer-wisconsin.csv')
df = df.drop(columns=['Unnamed: 32', 'id'])

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
X = df.drop(columns=['diagnosis']).values
y = df['diagnosis'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# fit the model
clf = NearestCentroidClassifier()
clf.fit(X_train, y_train)
# make predictions
y_pred = clf.predict(X_test)
# calculate accuracy

# compute the hyperplane for nearest centroid classifier
def plot_hyperplane(X, y, clf):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=50)
    
    # Plot the centroids
    for i, centroid in enumerate(clf.centroids):
        plt.scatter(centroid[0], centroid[1], marker='x', color='black', s=100, label=f'Centroid {i}')
    
    # Plot the hyperplane
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary and Centroids')
    plt.legend()
    plt.show()

# Assuming clf is your trained NearestCentroidClassifier

#plot the normal vector w
w = clf.centroids[1] - clf.centroids[0]
plt.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='black')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.title('Normal Vector of the Hyperplane')
plt.grid()
plt.show()
plot_hyperplane(X_train, y_train, clf)

```

```{note}
The code above generates a synthetic dataset with two classes and visualizes the decision boundary of the nearest centroid classifier. The centroids are marked with black crosses, and the decision boundary is shown as a contour plot.
The decision boundary is the line where the classifier is uncertain about the class label, and it separates the two classes in the feature space.
```
