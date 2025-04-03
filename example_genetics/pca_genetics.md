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

# Principal Components Analysis of Genetic Data

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysnptools.snpreader import Bed
```

We will be looking at Principal Component Analysis (PCA) algorithm. The algorith is a technique used for applications like dimensionality reduction, lossy data compression, feature extraction and data visualization. 

+++ {"slideshow": {"slide_type": "subslide"}}

The PCA performs the orthogonal projection of the data onto a lower dimensional linear space. The goal is to find the directions (principal components) in which the variance of the data is maximized.
An alternative definition of PCA is based on minimizing the sum-of-sqares of the projection errors.

+++ {"slideshow": {"slide_type": "slide"}}

## Implementation

For the PCA algorithm we implement `empirical_covariance` method that would be usef do calculating the covariance of the data. We also impmlement `PCA` class with `fit`, `transform` and `reverse_transform` methods.

```{code-cell} ipython3

def empirical_covariance(X):
    """
    Calculates the empirical covariance matrix for a given dataset.
    
    Parameters:
    X (numpy.ndarray): A 2D numpy array where rows represent samples and columns represent features.
    
    Returns:
    tuple: A tuple containing the mean of the dataset and the covariance matrix.
    """
    N = X.shape[0]  # Number of samples
    mean = X.mean(axis=0)  # Calculate the mean of each feature
    X_centered = X - mean[np.newaxis, :]  # Center the data by subtracting the mean
    covariance = X_centered.T @ X_centered / (N - 1)  # Compute the covariance matrix
    return mean, covariance
```


+++ {"slideshow": {"slide_type": "subslide"}}


```{code-cell} ipython3

class PCA:
    def __init__(self, k=None):
        """
        Initializes the PCA class without any components.

        Parameters:
        k (int, optional): Number of principal components to use.
        """
        self.pc_variances = None  # Eigenvalues of the covariance matrix
        self.principal_components = None  # Eigenvectors of the covariance matrix
        self.mean = None  # Mean of the dataset
        self.k = k  # the number of dimensions

    def fit(self, X):
        """
        Fit the PCA model to the dataset by computing the covariance matrix and its eigen decomposition.
        
        Parameters:
        X (numpy.ndarray): The data to fit the model on.
        """
        self.mean, covariance = empirical_covariance(X=X)
        eig_values, eig_vectors = np.linalg.eigh(covariance)  # Compute eigenvalues and eigenvectors
        order = np.argsort(eig_values)[::-1]  # Get indices of eigenvalues in descending order
        self.pc_variances = eig_values[order]  # Sort the eigenvalues
        self.principal_components = eig_vectors[:, order]  # Sort the eigenvectors
        if self.k is not None:
            self.pc_variances = self.pc_variances[:self.k]
            self.principal_components = self.principal_components[:,:self.k]

    def transform(self, X):
        """
        Transform the data into the principal component space.
        
        Parameters:
        X (numpy.ndarray): Data to transform.
        
        Returns:
        numpy.ndarray: Transformed data.
        """
        X_centered = X - self.mean
        return X_centered @ self.principal_components

    def reverse_transform(self, Z):
        """
        Transform data back to its original space.
        
        Parameters:
        Z (numpy.ndarray): Transformed data to invert.
        
        Returns:
        numpy.ndarray: Data in its original space.
        """
        return Z @ self.principal_components.T + self.mean

    def variance_explained(self):
        """
        Returns the amount of variance explained by the first k principal components.
        
        Returns:
        numpy.ndarray: Variances explained by the first k components.
        """
        return self.pc_variances
```

+++ {"slideshow": {"slide_type": "slide"}}

In the example below, we will use the PCA algorithm to reduce the dimensionality of a genetic dataset from the 1000 genomes project [1,2].

[1] Auton, A. et al. A global reference for human genetic variation. Nature 526, 68–74 (2015)

[2] Altshuler, D. M. et al. Integrating common and rare genetic variation in diverse human populations. Nature 467, 52–58 (2010)

After reducing the dimensionality, we will plot the results and examine whether clusters of ancestries are visible.  

We consider five ancestries in the dataset:  

- **EUR** - European  
- **AFR** - African  
- **EAS** - East Asian  
- **SAS** - South Asian  
- **AMR** - Native American  


+++ {"slideshow": {"slide_type": "subslide"}}


```{code-cell} ipython3
snpreader = Bed('./genetic_data/example2.bed', count_A1=True)
data = snpreader.read()
print(data.shape)
# y includes our labels and x includes our features
labels = pd.read_csv("./genetic_data/1kg_annotations_edit.txt", sep="\t", index_col="Sample")
list1 = data.iid[:,1].tolist()  #list with the Sample numbers present in genetic dataset
labels = labels[labels.index.isin(list1)]  #filter labels DataFrame so it only contains the sampleIDs present in genetic data
y = labels.SuperPopulation  # EUR, AFR, AMR, EAS, SAS
X = data.val[:, ~np.isnan(data.val).any(axis=0)]  #load genetic data to X, removing NaN values
```


+++ {"slideshow": {"slide_type": "subslide"}}


```{code-cell} ipython3

pca = PCA()
pca.fit(X=X)

X_pc = pca.transform(X)
X_reconstruction_full = pca.reverse_transform(X_pc)
print("L1 reconstruction error for full PCA : %.4E " % (np.absolute(X - X_reconstruction_full).sum()))

for rank in range(5):    #more correct: X_pc.shape[1]+1
    pca_lowrank = PCA(k=rank)
    pca_lowrank.fit(X=X)
    X_lowrank = pca_lowrank.transform(X)
    X_reconstruction = pca_lowrank.reverse_transform(X_lowrank)
    print("L1 reconstruction error for rank %i PCA : %.4E " % (rank, np.absolute(X - X_reconstruction).sum()))
```


+++ {"slideshow": {"slide_type": "subslide"}}


```{code-cell} ipython3

fig = plt.figure()
plt.plot(X_pc[y=="EUR"][:,0], X_pc[y=="EUR"][:,1],'.', alpha = 0.3)
plt.plot(X_pc[y=="AFR"][:,0], X_pc[y=="AFR"][:,1],'.', alpha = 0.3)
plt.plot(X_pc[y=="EAS"][:,0], X_pc[y=="EAS"][:,1],'.', alpha = 0.3)
plt.plot(X_pc[y=="AMR"][:,0], X_pc[y=="AMR"][:,1],'.', alpha = 0.3)
plt.plot(X_pc[y=="SAS"][:,0], X_pc[y=="SAS"][:,1],'.', alpha = 0.3)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(["EUR", "AFR","EAS","AMR","SAS"])

fig2 = plt.figure()
plt.plot(X_pc[y=="EUR"][:,0], X_pc[y=="EUR"][:,2],'.', alpha = 0.3)
plt.plot(X_pc[y=="AFR"][:,0], X_pc[y=="AFR"][:,2],'.', alpha = 0.3)
plt.plot(X_pc[y=="EAS"][:,0], X_pc[y=="EAS"][:,2],'.', alpha = 0.3)
plt.plot(X_pc[y=="AMR"][:,0], X_pc[y=="AMR"][:,2],'.', alpha = 0.3)
plt.plot(X_pc[y=="SAS"][:,0], X_pc[y=="SAS"][:,2],'.', alpha = 0.3)
plt.xlabel("PC 1")
plt.ylabel("PC 3")
plt.legend(["EUR", "AFR","EAS","AMR","SAS"])


fig3 = plt.figure()
plt.plot(X_pc[y=="EUR"][:,1], X_pc[y=="EUR"][:,2],'.', alpha = 0.3)
plt.plot(X_pc[y=="AFR"][:,1], X_pc[y=="AFR"][:,2],'.', alpha = 0.3)
plt.plot(X_pc[y=="EAS"][:,1], X_pc[y=="EAS"][:,2],'.', alpha = 0.3)
plt.plot(X_pc[y=="AMR"][:,1], X_pc[y=="AMR"][:,2],'.', alpha = 0.3)
plt.plot(X_pc[y=="SAS"][:,1], X_pc[y=="SAS"][:,2],'.', alpha = 0.3)
plt.xlabel("PC 2")
plt.ylabel("PC 3")
plt.legend(["EUR", "AFR","EAS","AMR","SAS"])

fig4 = plt.figure()
plt.plot(pca.variance_explained())
plt.xlabel("PC dimension")
plt.ylabel("variance explained")

fig4 = plt.figure()
plt.plot(pca.variance_explained().cumsum() / pca.variance_explained().sum())
plt.xlabel("PC dimension")
plt.ylabel("cumulative fraction of variance explained")
plt.show()
```

