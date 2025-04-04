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

# Bayesian Linear Regression

Having specified a mean vector and a covariance matrix for a multivariate normal distribution, we can sample from it using the univariate normal distribution. This can be achieved through the Cholesky decomposition.  

The Cholesky decomposition of a matrix decomposes the matrix into a lower triangular matrix and its conjugate transpose, such that:  

$$ A = LL^T $$  

where \( L \) is the lower triangular matrix.  

We can use this method to find the lower triangular matrix \( L \) for the covariance matrix:  

$$ \Sigma = LL^T $$  

Then, we sample from the univariate normal distribution:  

$$ x \sim N(0,1) $$  

Finally, we use the formula:  

$$ y = \mu + Lx $$  

to sample from the multivariate normal distribution.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# sns.set_theme()
```


```{code-cell} ipython3
if __name__ == "__main__":
    m = np.array([[0,0]]).T
    cov = np.array([[1,0.5],[0.2,2]])

    #solution

    observations = []
    for i in range(1000):
        x = np.random.normal(size=(2,1))
        L = np.linalg.cholesky(cov)
        u = m + L@x
        observations.append(u)

    observations = np.concatenate(observations, axis=1)
    # Check the covariance matrix
    print(np.cov(observations))
    sns.jointplot(x=observations[0,:], y = observations[1,:], kind="kde", space=0, fill=True, aspect=1)
    plt.show()


```


## Bayesian Linear Regression implementation

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal
```

In this chapter, we will explore **Linear Bayesian Regression**.

### Key Topics:  
- We will show how the **posterior distribution** over the weights can be computed in **closed form**.  
- We will demonstrate how to make **predictions** using the posterior distribution.  

Bayesian regression provides a probabilistic framework for linear modeling, allowing us to incorporate prior knowledge and quantify uncertainty in predictions.


We use a gaussian distribution as a prir for the wieghts:
p(w) = N (w| m0, S0)

The posterior distribution after observing data is given by:  

$$
p(\mathbf{w} | \mathbf{X}, \mathbf{y}) = \mathcal{N}(\mathbf{w} | \mathbf{m}_N, \mathbf{S}_N)
$$  

where:  

$$
\mathbf{m}_N = \mathbf{S}_N \left( \mathbf{S}_0^{-1} \mathbf{m}_0 + \beta \mathbf{\Phi}^T \mathbf{t} \right)
$$

$$
\mathbf{S}_N^{-1} = \mathbf{S}_0^{-1} + \beta \mathbf{\Phi}^T \mathbf{\Phi}
$$
Here:  
- $\mathbf{S}_0$ and $\mathbf{m}_0$ represent the prior covariance and mean, respectively.  
- $\beta$ is the precision (inverse variance) of the noise in the observations.  
- $\mathbf{\Phi}$ is the design matrix, and $\mathbf{t}$ is the observed data.  

This formulation provides a **closed-form Bayesian update** for the weight distribution in linear regression.  


```{code-cell} ipython3
alpha = 1 
beta = 25


def generate_dataset(w):
    x = np.linspace(-1,1, 30)
    y = w[0]*x + w[1] + 0.1 * np.random.normal(size=x.shape)

    x = x[:,None]
    y = y[:,None]

    X = np.hstack((x, np.ones_like(x)))
    return X, y

def plot_distribution(m, S):
    x, y = np.mgrid[-2:2:.01, -2:2:.01]
    pos = np.dstack((x, y))
    rv = multivariate_normal(m.squeeze(), S)
    plt.contourf(x, y, rv.pdf(pos), levels=70, cmap='viridis')
    plt.axis('equal')
    plt.xlim((-2,2))
    plt.ylim((-2,2))
    

def calculate_posterior_parameters(X, y):
    Sn_inv = alpha*np.eye(2) + beta*X.T@X
    Sn = np.linalg.inv(Sn_inv)
    mn = beta*Sn@X.T@y
    return mn, Sn


def plot_functions(mn, Sn, points = False):

    x = np.linspace(-1,1,30)
    for i in range(10):
        w = np.random.multivariate_normal(mn.squeeze(), Sn)
        y = w[0]*x + w[1]
        plt.plot(x,y, c='b', zorder=1)

    if points:
        X,y = points 
        plt.scatter(X[:,0], y, c ='r', zorder=2)

def calculate_predictive_distribution(x, mn, Sn):
    mean = mn.T@np.array([[2,1]]).T
    var = (1/beta) + x.T@Sn@x
    std = np.sqrt(var)
    return mean, std



w = np.array([[-0.6, 0.4]]).T
X, y = generate_dataset(w)
indexes = list(range(30))
random.shuffle(indexes)


plt.figure(figsize=(15,15))

# plot the prior
plt.subplot(221)
plot_distribution(np.array([0,0]), np.array([[1,0], [0,1]])) 
plt.title('0 observations')

# plot posteriors
nb_of_points = 1
X_new = X[ indexes[:nb_of_points],:]
y_new = y[indexes[:nb_of_points],:]
# calculate posterior distribution
mn, Sn = calculate_posterior_parameters(X_new,y_new)
plt.subplot(222)
plot_distribution(mn, Sn) 
plt.title('1 observation')

nb_of_points = 2
X_new = X[ indexes[:nb_of_points],:]
y_new = y[indexes[:nb_of_points],:]
# calculate posterior distribution
mn, Sn = calculate_posterior_parameters(X_new,y_new)
plt.subplot(223)
plot_distribution(mn, Sn) 
plt.title('2 observations')

nb_of_points = 10
X_new = X[ indexes[:nb_of_points],:]
y_new = y[indexes[:nb_of_points],:]
# calculate posterior distribution
mn, Sn = calculate_posterior_parameters(X_new,y_new)
plt.subplot(224)
plot_distribution(mn, Sn) 
plt.title('10 observations')


# functions
plt.figure(figsize=(15,15))
plt.subplot(221)
plot_functions(np.array([0,0]), np.array([[1,0],[0,1]]), points=(X,y)) 
plt.title('0 observations')


nb_of_points = 1
X_new = X[ indexes[:nb_of_points],:]
y_new = y[indexes[:nb_of_points],:]
# calculate posterior distribution
mn, Sn = calculate_posterior_parameters(X_new,y_new)
plt.subplot(222)
plot_functions(mn, Sn, points=(X_new,y_new)) 
plt.title(f'{nb_of_points} observations')





nb_of_points = 2
X_new = X[ indexes[:nb_of_points],:]
y_new = y[indexes[:nb_of_points],:]
# calculate posterior distribution
mn, Sn = calculate_posterior_parameters(X_new,y_new)
plt.subplot(223)
plot_functions(mn, Sn, points=(X_new,y_new)) 
plt.title(f'{nb_of_points} observations')


nb_of_points = 10
X_new = X[ indexes[:nb_of_points],:]
y_new = y[indexes[:nb_of_points],:]
# calculate posterior distribution
mn, Sn = calculate_posterior_parameters(X_new,y_new)
plt.subplot(224)
plot_functions(mn, Sn, points=(X,y)) 
plt.title(f'{nb_of_points} observations')

plt.show()


# calculate predictive distribution
x = np.array([[3,1]]).T
print(calculate_predictive_distribution(x, mn, Sn))
```
