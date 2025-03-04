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

+++ {"slideshow": {"slide_type": "slide"}}

## Diagnosing Breast cancer biopsies using Machine Learning

```{code-cell} ipython3
:tags: [hide-input]
### Some imports
# All packages are included in the Anaconda python distribution and integral part of a machine learning Python environment).
import numpy as np               # efficient matrix-vector operations
import numpy.linalg as la        # linear algebra (solvers etc.)
import pandas as pd              # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns            # data visualization  
sns.set_style("whitegrid")        # set the figure default style
sns.set_context("talk")
sns.set(font_scale=1.5)          # bigger fonts in images

import matplotlib.pyplot as plt  # basic plotting

# some not so standard imports:
# import importlib                 # enable reloading of libraries
import plotting_util as util     # useful plotting tools for teaching (see plotting_utils.py)
# importlib.reload(util)
# import time                      # timing (for example to benchmark an algorithm)
```

+++ {"slideshow": {"slide_type": "slide"}}


Wisconsin Diagnostic Breast Cancer (WDBC, 1993) data from UCI Machine Learning repository.

- 569 samples from patients with known diagnosis
- 357 benign
- 212 malignant
- 30 features extracted from fine needle aspirate slides


+++ {"slideshow": {"slide_type": "slide"}}

![title](uci_breast_cancer/papers/breast_cancer_nuclei_12938_2011_Article_597_Fig3_HTML.jpg)

+++

Let's load the data:
```{code-cell} ipython3
:tags: [hide-input]
X, y = util.load_data(columns=["concavity_mean", "texture_mean"])
print (X.drop(['bias'], axis=1).shape)
X.drop(['bias'], axis=1).head()
```

We are given a number of features that describe the nuclei that have been determined from image processing techniques [Street et al, 1992].
While the original data consists of 30 features and all presented methods work with 30 features, we restrict ourselves to 2 features, the average *concavity* and the *texture* of the nuclei for illustrative purposes. For each sample, we can represent these features as a vector in $\mathbb{R}^2$.

Given the information encoded in these **feature vectors**, we would like to diagnose wether the biopsies are malignant or benign, i.e., to **classify** the samples. Before we dive deeper into classification, we should have a deeper look into vectors, their properties, the operations that we can perform on them and the spaces they live in.


+++

## References
[Street et al, 1992] N. Street, W. Wolberg, O.L. Mangasarian:  Nuclear Feature Extraction For Breast Tumor Diagnosis. IS&T/SPIE 1993.
