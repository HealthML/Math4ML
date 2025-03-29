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

## Example: Nearest Centroid Classifier with Different Metrics and Norms

Recall, the **nearest centroid classifier** assigns each new point to the class of its closest centroid according to a chosen distance measure. While we've typically used Euclidean distance (the 2-norm), many other metrics can be used, each potentially giving a different classification result.

### Different metrics and norms:

1. **Manhattan (L1) distance**:

$$d_{L_1}(\mathbf{x}, \mathbf{y}) = \|\mathbf{x}-\mathbf{y}\|_1 = \sum_{i=1}^{n}|x_i - y_i|$$

**Effect on classifier**: Sensitive to coordinate-wise differences, leading to axis-aligned decision boundaries that can look like diamonds or squares.

2. **Chebyshev (L∞) distance**:

$$d_{L_{\infty}}(\mathbf{x}, \mathbf{y}) = \|\mathbf{x}-\mathbf{y}\|_{\infty} = \max_i |x_i - y_i|$$

**Effect on classifier**: Classifier decisions are based on the largest difference across all dimensions, leading to box-like decision boundaries.

3. **Mahalanobis distance** (a metric derived from data covariance structure):

$$d_{Mahal}(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x}-\mathbf{y})^T\mathbf{\Sigma}^{-1}(\mathbf{x}-\mathbf{y})}$$

where \(\mathbf{\Sigma}\) is the covariance matrix.

**Effect on classifier**: Takes into account data correlation, stretching or rotating decision boundaries according to feature covariance.

---

### Python Implementation Example:

Below is a simple Python implementation of the nearest centroid classifier that lets you experiment with different metrics. You can demonstrate this practically:

```python
from scipy.spatial.distance import cdist
import numpy as np

class FlexibleNearestCentroidClassifier:
    def __init__(self, metric='euclidean'):
        self.metric = metric
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids = np.array([X[y==label].mean(axis=0) for label in self.classes_])
    
    def predict(self, X):
        distances = cdist(X, self.centroids, metric=self.metric)
        return self.classes_[np.argmin(distances, axis=1)]
```

**Using the classifier with different metrics:**

```python
# Example usage:
metrics = ['euclidean', 'cityblock', 'chebyshev', 'mahalanobis']
metric_params = {'mahalanobis': {'VI': np.linalg.inv(np.cov(x_train.T))}}

for metric in metrics:
    clf = FlexibleNearestCentroidClassifier(metric=metric)
    clf.fit(x_train.values, y_train.values)
    
    # Predicting
    if metric == 'mahalanobis':
        y_pred = clf.predict(x_test.values, **metric_params[metric])
    else:
        y_pred = clf.predict(x_test.values)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy using {metric}: {accuracy:.3f}")
```

---

### Insights and Discussion for Students:

- Changing metrics effectively changes the "geometry" of your classification.
- L1 norm encourages sparsity and gives axis-aligned boundaries.
- L∞ norm emphasizes the largest feature difference, often creating box-shaped boundaries.
- Mahalanobis distance adapts to data covariance, considering correlations between features.

**Thus, the choice of metric is a critical design decision in machine learning algorithms.**