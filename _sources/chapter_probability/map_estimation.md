## Maximum A Posteriori Estimation (MAP)

MAP estimation introduces prior knowledge into the estimation problem.
Using Bayes' rule, the posterior over parameters $\theta$ is:

$$
p(\theta \mid x_1, \dots, x_n) \propto p(\theta) \cdot p(x_1, \dots, x_n \mid \theta)
$$

The **MAP estimate** is the mode of the posterior:

$$
\hat{\theta}_\text{MAP} = \operatorname{argmax}_\theta \log p(\theta) + \sum_{i=1}^n \log p(x_i \mid \theta)
$$


