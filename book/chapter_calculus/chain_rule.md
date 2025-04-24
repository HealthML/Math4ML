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
# The chain rule

Most functions that we wish to optimize are not completely arbitrary
functions, but rather are composed of simpler functions which we know
how to handle. The chain rule gives us a way to calculate derivatives
for a composite function in terms of the derivatives of the simpler
functions that make it up.

The chain rule from single-variable calculus should be familiar:

$$(f \circ g)'(x) = f'(g(x))g'(x)$$ 

where $\circ$ denotes function
composition.

There is a natural generalization of this rule to multivariate functions.

:::{prf:theorem} Multivariate Chain Rule
:label: chain-rule
:nonumber:
Suppose $f : \mathbb{R}^m \to \mathbb{R}^k$ and
$g : \mathbb{R}^n \to \mathbb{R}^m$.

Then $f \circ g : \mathbb{R}^n \to \mathbb{R}^k$ and

$$\mathbf{J}_{f \circ g}(\mathbf{x}) = \mathbf{J}_f(g(\mathbf{x}))\mathbf{J}_g(\mathbf{x})$$
:::

In the special case $k = 1$ we have the following corollary since
$\nabla f = \mathbf{J}_f^{\!\top\!}$.

:::{prf:corollary} Chain Rule for Scalar-Valued Functions
:label: chain-rule-scalar
:nonumber:
Suppose $f : \mathbb{R}^m \to \mathbb{R}$ and
$g : \mathbb{R}^n \to \mathbb{R}^m$. Then
$f \circ g : \mathbb{R}^n \to \mathbb{R}$ and

$$\nabla (f \circ g)(\mathbf{x}) = \mathbf{J}_g(\mathbf{x})^{\!\top\!} \nabla f(g(\mathbf{x}))$$
:::

## Chain Rule for Basis Function Regression

Now we can apply the chain rule to optimize the hyperparameters of the tanh basis functions in the context of our temperature prediction example.
We still have to modify our ridge regression code to use the tanh basis function class and enable optimization over the hyperparameters using the chain rule.

So, let's derive the Jacobian of ridge regression with respect to the hyperparameter matrix $\mathbf{W}_\phi$ of the basis functions $\phi: \operatorname{dom}{x} \rightarrow \mathbb \mathbb{R}^d$.

Let's have a look in how the basis functions affect the loss function.

Let $f$ be the loss function, which is a function of the weights $\mathbf{w}$ and the hyperparameters $\mathbf{W}_\phi$ be the matrix of hyperparameters of the tanh basis functions. The loss function is given by:

$$
f(\mathbf{w}, \mathbf{W}_\phi) = \frac{1}{2n}\sum_{i=1}^n (\mathbf{y}_i - \boldsymbol{\phi}(\mathbf{x}_i; \mathbf{W}_\phi)\mathbf{w}\|^2_2 + \frac{\lambda}{2}\|\mathbf{w}\|^2_2
\mathbf{y} - \boldsymbol{\phi}(\mathbf{X})\mathbf{w}\|^2_2 + \frac{\lambda}{2}\|\mathbf{w}\|^2_2
$$
where $\mathbf{X}$ is the design matrix, $\mathbf{y}$ is the response vector, $\mathbf{w}$ is the weight vector, and $\lambda$ is the regularization parameter.

Let $J_\phi$ be the Jacobian of the tanh basis functions with respect to the hyperparameters $a_i$ and $b_i$ at a fixed input value $x_0$, as we have derived in the last section.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np

class TanhBasis:
    def __init__(self, W):
        self.W = W

    def XW(self, x):
        """Compute the product of the input data and the weights."""
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return x @ self.W[:,:-1] + self.W[:,-1]

    def dXW(self, x):
        """Compute the derivative of the product of the input data and the weights."""
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return np.hstack((x, np.ones((x.shape[0], 1))))
        
    def phi(self, x):
        """Compute the tanh basis functions."""
        return np.tanh(self.XW(x))

    def jacobian(self, x):
        """Compute the Jacobian of the tanh basis functions."""
        return self.dXW(x) * (1 - np.tanh(self.XW(x))**2)
```

```{code-cell} ipython3
import numpy as np

class BasisFunctionRidgeRegressionGD:
    def __init__(self, basis_function, learning_rate=0.01, num_iterations=1000, ridge=0.1):
        self.basis_function = basis_function
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.ridge = ridge

    def mse(self, X, y):
        # Mean Squared Error
        return np.mean((y - self.pred(X)) ** 2)
    
    def loss(self, X, y):
        # Loss function (MSE + Ridge penalty)
        return self.mse(X, y) + self.ridge * np.sum(self.w ** 2)

    def gradient(self, X, y):
        # Gradient of the loss
        jacobian_phi = self.basis_function.jacobian(X)
        phi = self.basis_function.transform(X)
        return -X.T @ (y - self.pred(X)) / len(y) + self.ridge * self.w

    def fit(self, X, y):
        # Initialize weights
        self.w = np.zeros(X.shape[1])
        
        # Gradient descent loop
        for _ in range(self.num_iterations):   
            self.w -= self.learning_rate * self.gradient(X, y)

    def pred(self, X):
        return self.basis_function.transform(X) @ self.w
```

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
YEAR = 1900
def load_weather_data(year = None):
    """
    load data from a weather station in Potsdam
    """

    names = ['station', 'date' , 'type', 'measurement', 'e1','e2', 'E', 'e3']
    data = pd.read_csv('../../datasets/weatherstations/GM000003342.csv', names = names, low_memory=False) # 47876 rows, 8 columns
    # convert the date column to datetime format
    data['date'] = pd.to_datetime(data['date'], format="%Y%m%d") # 47876 unique days
    types = data['type'].unique()

    tmax = data[data['type']=='TMAX'][['date','measurement']] # Maximum temperature (tenths of degrees C), 47876
    tmin = data[data['type']=='TMIN'][['date','measurement']] # Minimum temperature (tenths of degrees C), 47876
    prcp = data[data['type']=='PRCP'][['date','measurement']] # Precipitation (tenths of mm), 47876
    snwd = data[data['type']=='SNWD'][['date','measurement']] # Snow depth (mm), different shape
    tavg = data[data['type']=='TAVG'][['date','measurement']] # average temperature, different shape 1386
    arr = np.array([tmax.measurement.values,tmin.measurement.values, prcp.measurement.values]).T 

    df = pd.DataFrame(arr/10.0, index=tmin.date, columns=['TMAX', 'TMIN', 'PRCP']) # compile data in a dataframe and convert temperatures to degrees C, precipitation to mm

    if year is not None:
        df = df[pd.to_datetime(f'{year}-1-1'):pd.to_datetime(f'{year}-12-31')]
    
    df['days'] = (df.index - df.index.min()).days
    return df

# Load weather data for the year 2000
df = load_weather_data(year = YEAR)

np.random.seed(2)
idx = np.random.permutation(df.shape[0])

idx_train = idx[0:100]
idx_test = idx[100:]

data_train = df.iloc[idx_train]
data_test = df.iloc[idx_test]

N_train = 100

def tanh_basis(x, a, b, add_bias=True):
    """
    tanh basis function
    x : input data
    a : slope of the tanh basis functions
    b : bias of the tanh basis functions
    add_bias : add a bias term to the output
    """
    phi = np.tanh(a[np.newaxis,:] * x + b[np.newaxis,:])
    if add_bias:
        phi = np.concatenate((phi, np.ones((phi.shape[0], 1))), axis=1)
    return phi

a = np.array([.1, .2, .3])
b = np.array([-10.0,-50.0,-100.0])

ridge = 0.1     # strength of the L2 penalty in ridge regression

x_train = data_train.days.values[:N_train][:,np.newaxis] * 1.0
X_train = tanh_basis(x_train, a, b)
y_train = data_train.TMAX.values[:N_train]

reg = RidgeRegressionGD(ridge=ridge, learning_rate=0.01, num_iterations=1000)
reg.fit(X_train, y_train)

x_days = np.arange(366)[:,np.newaxis]
X_days = tanh_basis(x_days, a, b)
y_days_pred = reg.pred(X_days)

x_test = data_test.days.values[:,np.newaxis] * 1.0
X_test = tanh_basis(x_test, a, b)
y_test = data_test.TMAX.values
y_test_pred = reg.pred(X_test)
print("training MSE : %.4f" % reg.mse(X_train, y_train))
print("test MSE     : %.4f" % reg.mse(X_test, y_test))


fig = plt.figure()
ax = plt.plot(x_train,y_train,'.')
ax = plt.plot(x_test,y_test,'.')
ax = plt.legend(["train MSE = %.2f" % reg.mse(X_train, y_train),"test MSE = %.2f" % reg.mse(X_test, y_test)])
ax = plt.plot(x_days,y_days_pred)
ax = plt.ylim([-27,39])
ax = plt.xlabel("day of the year")
ax = plt.ylabel("Maximum Temperature - degree C")
ax = plt.title("Year : %i        N : %i" % (YEAR, N_train))
```