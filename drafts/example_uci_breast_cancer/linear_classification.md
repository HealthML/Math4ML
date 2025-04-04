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


# Linear Classification

## Classification Problem

We will focus on the **classification problem**. As mentioned earlier, classification is a task where the model predicts the class label $y$ for a given feature vector $x$. 

The task will be to predict whether a given patient has breast cancer or not.

We will use the **Wisconsin Diagnostic Breast Cancer (WDBC, 1993)** dataset from the UCI Machine Learning Repository. The dataset consists of 569 samples from patients, with 357 benign and 212 malignant cases. Each sample is described by 30 features extracted from fine needle aspirate slides. These features were obtained using image processing algorithms.

## Nearest Centroid Classifier

The **Nearest Centroid Classifier** is a simple yet effective linear classification algorithm for solving classification problems, such as diagnosing breast cancer from biopsy data.

### How the Nearest Centroid Classifier Works

This classifier works by calculating the geometric center (**centroid**) of each class from the training data. The centroid represents the average feature vector of all samples belonging to a class. Once centroids are computed, new data points are classified based on their distance to these centroids.

### Training the Algorithm

During training, we calculate a centroid for each class as follows:

$$
\mathbf{c}_k = \frac{1}{N_k} \sum_{i=1}^{N_k} \mathbf{x}_i \quad \text{for class} \ k
$$

Where:
- $\mathbf{c}_k$ is the centroid for class $k$
- $N_k$ is the number of samples in class $k$
- $\mathbf{x}_i$ is the feature vector of the $i$-th sample in class $k$

### Making Predictions

Given a new feature vector $\mathbf{x}$, we predict its class by finding the centroid it is closest to, typically using Euclidean distance:

$$
\hat{y} = \arg\min_k \, \|\mathbf{x} - \mathbf{c}_k\|
$$


```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
from numpy.linalg import norm    
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt  # basic plotting

# some not so standard imports:
import plotting_util as util     # useful plotting tools for teaching (see plotting_utils.py)
```

## Implementing the Nearest Centroid Classifier

We will implement the **Nearest Centroid Classifier** with two methods: 

1. `fit()` – This method trains the model by calculating the centroid for each class.
2. `predict()` – This method makes predictions based on the trained centroids.

**Note:** Unlike linear regression, the Nearest Centroid Classifier does not involve minimizing a loss function. Instead, it calculates the centroid (mean of the feature vectors) for each class and assigns class labels based on the nearest centroid to the input observation.

```{code-cell} ipython3

class NearestCentroidClassifier:
    def __init__(self):
        self.centroids = None
        self.classes_ = None
        
    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.
        :param X: array-like, shape (n_samples, n_features) Training data.
        :param y: array-like, shape (n_samples,) Target values.
        """
        self.classes_ = np.unique(y)
        self.centroids = [ np.mean(X[y==class_], axis=0) for class_ in self.classes_]

    def predict(self, X):
        """
        Perform classification on samples in X.
        :param X: array-like, shape (n_samples, n_features) Input data.
        :return: array, shape (n_samples,) Predicted class label per sample.
        """
        differences_cent_0 = norm(X - self.centroids[0], axis=1)
        differences_cent_1 = norm(X - self.centroids[1], axis=1)
        output = np.where(differences_cent_1<differences_cent_0, self.classes_[1], self.classes_[0])
        return output
```

## Loading the dataset and splitting it into training and test sets

Let's load the dataset and split it into training and test sets. We will use 80% of the data for training and 20% for testing.

Let's load the data:
```{code-cell} ipython3
:tags: [hide-input]
X, y = util.load_data(columns=["concavity_mean", "texture_mean"])
print (X.drop(['bias'], axis=1).shape)
X.drop(['bias'], axis=1).head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

## Training the classifier and evaluating its performance

We will train the classifier using the training set and then evaluate its performance on the test set.

```{code-cell} ipython3

# Create and train the Nearest Centroid Classifier
classifier = NearestCentroidClassifier()
classifier.fit(X_train.values,y_train.values)
# Predict the classes for the test data
y_pred = classifier.predict(X_test.values)

# Calculate and print the accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
```


```{code-cell} ipython3
# Set plot style parameters
plt.title('Breast Cancer Data with Centroids (dim 0 and 1)')
plt.xlabel('First Feature')
plt.ylabel('Second Feature')
plt.scatter(X_train.concavity_mean[y_train=="M"], X_train.texture_mean[y_train=="M"], alpha=0.8, color="red")
plt.scatter(X_train.concavity_mean[y_train=="B"], X_train.texture_mean[y_train=="B"], alpha=0.8, color="blue")

plt.scatter(classifier.centroids[0][0],classifier.centroids[0][1] , c='blue', s=200, marker='*', label='Centroid Malignant', edgecolors='black')
plt.scatter(classifier.centroids[1][0], classifier.centroids[1][1], c='red', s=200, marker='*', label='Centroid Benign', edgecolors='black')
plt.legend()
```




