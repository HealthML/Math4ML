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

# Probabilistic PCA
We will present **Probabilistic PCA (PPCA)**.  

### Key Aspects of Probabilistic PCA  
- Unlike standard PCA, **Probabilistic PCA** is formulated as a probabilistic model, allowing us to **sample new data points** from the learned distribution.  
- We will use the **closed-form solution** for PPCA to compute the principal components.  
- An alternative approach to solving PPCA is using the **Expectation-Maximization (EM) algorithm**, which iteratively estimates the latent variables and model parameters.  

PPCA is particularly useful when working with noisy or missing data, as it naturally integrates probabilistic modeling into dimensionality reduction.  


## Math behind PPCA

We introduce an explicit laten variale $z$ corresponding to the principal-component subspace. 
The prior distribution of $z$ is given by:


$$
p(\mathbf{z}) = \mathcal{N}(\mathbf{z} \mid \mathbf{0}, \mathbf{I})
$$

We can define the conditional distrubution of the observed variable $\mathbf{x}$, conditioned on the laten variable $\mathbf{z}$ as:
$$
p(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x} \mid \mathbf{W}\mathbf{z} + \mathbf{\mu}, \sigma^2\mathbf{I}).
$$

Therefore we can express $\mathbf{x}$ as a linear transformation of the latent variable $z$ plus a Gaussian noise term $\epsilon$:

$$ 
\mathbf{x} = \mathbf{W}\mathbf{z} + \mathbf{\mu} + \mathbf{\epsilon}
$$

The marginal distribution $p(\mathbf{x})$ of the observed variable could be obtained from the sum and product rules of probability:
$$
p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})d\mathbf{z} 

$$

This distribution is Gaussian given by:

$$
p(\mathbf{x}) = \mathcal{N}(\mathbf{x} \mid \mathbf{\mu}, \mathbf{C})
$$

where 
$$
\mathbf{C} = \mathbf{W}\mathbf{W}^T + \sigma^2\mathbf{I}
$$

The last equation is the posterior distribution of the latent variable $z$ given the observed variable $x$:
$$
p(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z} \mid \mathbf{M}^{-1}\mathbf{W}^T(\mathbf{x}-\mathbf{\mu}), \sigma^{-2}\mathbf{M})
$$

where:
$$\mathbf{M} = \mathbf{W}^T\mathbf{W} + \sigma^2\mathbf{I}$$

To sum up, the PPCA model could be defined by following distributions:
- laten variable distribution $p(\mathbf{z}) $
- distribution of the observed variable conditioned on the latent variable $p(\mathbf{x}|\mathbf{z})$
- predictive distribution $p(\mathbf{x})$
- posterior distribution $p(\mathbf{z}|\mathbf{x})$

Note that all the distribution are multivariate Gaussian distributions.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.datasets import fetch_openml
```

```{code-cell} ipython3
class PPCA():
    '''
    X - dataset
    x - data point
    z - laten variable
    
    '''
    def __init__(self,X, M):
        self.D = X.shape[1] # dimension of oryginal data points   
        self.M = M # dimension of reduced data point
        self.X = X #dataset
        self.calculate_parameters()
    def calculate_parameters(self):
        '''
        Determine parameteres of the model (mean, variance and W matrix). 
        Have to be overriden in child classes
        '''
        raise NotImplementedError 
    def sample_x(self):
        '''
        Sample from p(x) distribution
        '''
        mean = self.mean
        C = np.dot(self.W_ML, self.W_ML.T) + self.sigma * np.eye(self.D)
        distribution = stats.multivariate_normal(mean, C)
        return distribution.rvs()
    def sample_z(self):
        '''
        Sample from p(z) distribution
        '''
        distribution = stats.multivariate_normal(np.zeros(shape = self.M), np.eye(self.M))
        return distribution.rvs()
    def sample_x_given_z(self, z):
        '''
        Sample from p(x|z) distribution'
        '''
        distribution = stats.multivariate_normal(np.dot(self.W_ML, z) + self.mean, self.sigma * np.eye(self.D))
        return distribution.rvs()
    def sample_z_given_x(self, x):
        '''
        Sample from p(z|x) distribution
        '''
        M_matrix = np.dot(self.W_ML.T, self.W_ML) + self.sigma * np.eye(self.M)
        M_matrix_inv = np.linalg.inv(M_matrix)
        mean = np.linalg.multi_dot([M_matrix_inv, self.W_ML.T, (x - self.mean)])
        variance = self.sigma * M_matrix_inv                                    
        distribution = stats.multivariate_normal(mean, variance)
        return distribution.rvs()                                    
```

## Closed form solution
The closed-form solution for PPCA is derived from the maximum likelihood estimation of the model parameters. The likelihood is represented by:
$$
p(\mathbf{X} \mid \mathbf{\mu}, \mathbf{W}, \sigma^2) 
$$
where $\mathbf{X}$ is the observed data matrix. 
We need to find the values of $\mathbf{\mu}$, $\mathbf{W}$, and $\sigma^2$ that maximize the likelihood function.

The solution for $\mathbf{\mu} = \mathbf{\bar{x}}$ where $\mathbf{\bar{x}}$ is the mean of the data.

The solution for $\mathbf{W}$ is given by:
$$
\mathbf{W_{ML}} = \mathbf{U}_M (\mathbf{L}_M - \sigma^2\mathbf{I})^{1/2}
$$
where $\mathbf{U}_M$ is the matrix of the eigenvectors of the data covariance matrix, and $\mathbf{L}_M$ is the diagonal matrix of the corresponding eigenvalues. We assume the arangement of the eigenvectors in order of decreasing values of the corresponding eigenvalues.

The solution for $\sigma^2$ is given by:
$$
\sigma^2 = \frac{1}{D-M} \sum_{i=M+1}^{D} \lambda_i

where $D$ is the number of dimensions of the data, $M$ is the number of principal components, and $\lambda_i$ are the eigenvalues of the data covariance matrix.

```{code-cell} ipython3
# ## Closed-form solution (CF)

class PPCA_CF(PPCA):
    '''
    X - dataset
    x - data point
    z - laten variable
    
    '''        
    def calculate_parameters(self):
        '''
        Determine parameteres of the model by optimizing likelihood function. 
        It involves caltulating mean, variance and W matrix.
        '''
        self.mean = self.X.mean(axis = 0)
        
        covariance = np.cov(self.X, rowvar = False)
        eig_val, eig_vec = np.linalg.eig(covariance)
        idx = np.argsort(eig_val)[::-1]
        eig_val = np.real(eig_val[idx])
        eig_vec = np.real(eig_vec[:, idx])
        
        self.sigma = 1/(self.D - self.M) * np.sum(eig_val[self.M+1:])
        
        U_M = eig_vec[:, :self.M]
        L_M = np.diag(eig_val[:self.M])
        self.W_ML = np.dot(U_M, np.sqrt(L_M - self.sigma*np.eye(self.M)))
```

## PPCA on MNIST dataset

```{code-cell} ipython3
# Fetch MNIST data
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Extract data and labels
x_train, y_train = mnist['data'], mnist['target']

x_train = x_train / 255
x_train = x_train.reshape(70000, -1)
x_train = x_train[((y_train == '8') + (y_train == '1')),:]
y_train = y_train[((y_train == '8') + (y_train == '1'))]

model = PPCA_CF(x_train, 2)

# sample from p(x)
plt.figure(figsize =(10,10))
for i in range(1,10):
    plt.subplot(3,3,i)
    plt.imshow(model.sample_x().reshape(28,28))
plt.suptitle('Sampling from p(x)', fontsize=20)

# Show the original image
plt.figure()
idx = np.random.randint(0, x_train[0].shape[0])
plt.imshow(x_train[idx,:].reshape(28,28))
plt.suptitle('Original image', fontsize=20)

# Show the reconstructions

z = model.sample_z_given_x(x_train[idx,:]) # get latent variable p(z|x)

plt.figure(figsize =(10,10))
for i in range(1,10):
    plt.subplot(3,3,i)
    image = model.sample_x_given_z(z)
    plt.imshow(image.reshape(28,28))
    plt.axis('off')
plt.tight_layout()
plt.suptitle('Image reconstruction p(x|z)', fontsize=20)
plt.show()
```

