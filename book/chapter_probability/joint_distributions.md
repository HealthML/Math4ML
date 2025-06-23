# Joint distributions

Often we have several random variables and we would like to get a
distribution over some combination of them. 
A **joint distribution** is
exactly this. 

For some random variables $X_1, \dots, X_n$, the joint
distribution is written $p(X_1, \dots, X_n)$ and gives probabilities
over entire assignments to all the $X_i$ simultaneously.

## Independence
We say that two variables $X$ and $Y$ are **independent** if their joint
distribution factors into their respective distributions, i.e.

$$p(X, Y) = p(X)p(Y)$$ 

We can also define independence for more than two
random variables. 
Let
$\{X_i\}_{i \in I}$ be a collection of random variables indexed by $I$,
which may be infinite. 

Then $\{X_i\}$ are independent if for every
finite subset of indices $i_1, \dots, i_k \in I$ we have

$$p(X_{i_1}, \dots, X_{i_k}) = \prod_{j=1}^k p(X_{i_j})$$ 

For example, in the case of three random variables, $X, Y, Z$, we require that

$$p(X,Y,Z) = p(X)p(Y)p(Z)$$

as well as

$$p(X,Y) = p(X)p(Y)\quad\text{and}\quad p(X,Z) = p(X)p(Z) \quad\text{and}\quad p(Y,Z) = p(Y)p(Z)$$


## Independent and identically distributed random variables

In machine learning, we often assume that a bunch of random variables are **independent and identically distributed** (i.i.d.) so that their joint distribution can be factored entirely: 

$$p(X_1, \dots, X_n) = \prod_{i=1}^n p(X_i)$$ 

where $X_1, \dots, X_n$ all share the same p.m.f./p.d.f.


That means that for a parametrerized model $p_\boldsymbol{\theta}$, the distributions of all $X_i$ share the parameter vector $\boldsymbol{\theta}$ and are independent.

$$p_\boldsymbol{\theta}(X_1, \dots, X_n) = \prod_{i=1}^n p_\boldsymbol{\theta}(X_i)$$

## Example: Independent coin flips

Assume we have a coin with probability of heads $\theta$.

Let $X_1, \dots, X_n$ be the results of $n$ independent flips of the same coin, where $X_i = 1$ if the $i$-th flip is heads and $X_i = 0$ if the $i$-th flip is tails.

Then the joint distribution of $X_1, \dots, X_n$ is given by:

$$p(X_1, \dots, X_n) = \prod_{i=1}^n \theta^{X_i} (1-\theta)^{1-X_i} = \theta^{\sum_{i=1}^n X_i} (1-\theta)^{n-\sum_{i=1}^n X_i}$$

## Conditioning, the chain rule, and the sum rule for random variables

Previously, we have encountered conditioning, the chain rule, and the sum rule in the context of individual events. 
We now extend these foundational ideas to **random variables**, allowing us to reason about distributions and probabilities more systematically. 

### Conditional distributions

For random variables, the concept of conditioning generalizes naturally: the **conditional distribution** $p(X \mid Y=y)$ describes the probability distribution of $X$ given that $Y$ takes a specific value $y$. 

### The chain rule

Similarly, the **chain rule** generalizes to factorize joint distributions of multiple random variables, such as:

$$
p(X, Y) = p(X \mid Y)\, p(Y), \quad p(X, Y, Z) = p(X \mid Y, Z)\,p(Y \mid Z)\,p(Z),
$$

and so forth. 

### Marginal distributions and the sum rule

Finally, the **sum rule** for random variables states that we can obtain the marginal distribution of a variable by summing (or integrating) over all possible values of the other variables:

$$
p(X) = \sum_{y} p(X, y).
$$

These extensions provide a powerful framework to handle complex probabilistic scenarios involving multiple random variables.



### Example of Marginal and Conditional Distributions

Let's illustrate these concepts with a concrete scenario.

**Scenario: A Small Coding Bootcamp**

Imagine a small coding bootcamp with just **10 students**. 
We can define two random variables based on the students' properties:

*   **$X$ = Student's Experience Level**:
    *   $X=1$: Beginner (no prior experience)
    *   $X=2$: Intermediate (some prior experience)
*   **$Y$ = Number of Projects Completed**:
    *   $Y=0$: No projects
    *   $Y=1$: One project
    *   $Y=2$: Two projects

The 10 students in the bootcamp are composed as follows:

| Experience ($X$) | Projects ($Y$) | Number of Students |
| :--- | :--- | :--- |
| Beginner | 0 | 1 |
| Beginner | 1 | 1 |
| Beginner | 2 | 2 |
| Intermediate | 0 | 1 |
| Intermediate | 1 | 2 |
| Intermediate | 2 | 3 |
| **Total** | | **10** |

If we pick a student at random from this bootcamp, the joint probability distribution of their experience level ($X$) and project completions ($Y$) is given by the following table, where each entry is the number of students in that category divided by the total of 10.



$$
p(X, Y) = 
\begin{array}{c|ccc}
 & Y = 0 & Y = 1 & Y = 2 \\
\hline
X = 1 & 0.1 & 0.1 & 0.2 \\
X = 2 & 0.1 & 0.2 & 0.3 \\
X = 3 & 0.2 & 0.3 & 0.5
\end{array}
$$

#### Marginal Distributions

We first recall the marginal distributions, obtained by summing over the other variable.

Marginalizing over $ Y $, we get:

$$
\begin{aligned}
p(X = 1) &= 0.1 + 0.1 + 0.2 = 0.4 \\
p(X = 2) &= 0.1 + 0.2 + 0.3 = 0.6
\end{aligned}
$$

Marginalizing over $ X $, we get:

$$
\begin{aligned}
p(Y = 0) &= 0.1 + 0.1 = 0.2 \\
p(Y = 1) &= 0.1 + 0.2 = 0.3 \\
p(Y = 2) &= 0.2 + 0.3 = 0.5
\end{aligned}
$$

#### Conditional Distribution

Now, we introduce the conditional distribution. The conditional distribution of $ X $ given $ Y = y $ is defined as:

$$
p(X \mid Y = y) = \frac{p(X, Y = y)}{p(Y = y)}
$$

For example, let's compute $ p(X \mid Y = 2) $:

$$
\begin{aligned}
p(X = 1 \mid Y = 2) &= \frac{p(X = 1, Y = 2)}{p(Y = 2)} = \frac{0.2}{0.5} = 0.4 \\
p(X = 2 \mid Y = 2) &= \frac{p(X = 2, Y = 2)}{p(Y = 2)} = \frac{0.3}{0.5} = 0.6
\end{aligned}
$$

Thus, the conditional distribution $ p(X \mid Y = 2) $ is given by:

$$
p(X \mid Y = 2) =
\begin{cases}
0.4 & \text{if } X = 1 \\
0.6 & \text{if } X = 2
\end{cases}
$$

Similarly, one could compute the conditional distributions for $ Y = 0 $ and $ Y = 1 $.

This clearly illustrates how the conditional distribution relates directly to both joint and marginal distributions.
