# Probability Basics

Suppose we have some sort of randomized experiment (e.g. a coin toss,
die roll) that has a fixed set of possible outcomes. This set is called
the **sample space** and denoted $\Omega$.

We would like to define probabilities for some **events**, which are
subsets of $\Omega$. The set of events is denoted $\mathcal{F}$.[^7] The
**complement** of the event $A$ is another event,
$A^\text{c} = \Omega \setminus A$.

Then we can define a **probability measure**
$\mathbb{P} : \mathcal{F} \to [0,1]$ which must satisfy

(i) $\mathbb{P}(\Omega) = 1$

(ii) **Countable additivity**: for any countable collection of disjoint
     sets $\{A_i\} \subseteq \mathcal{F}$,

$$\mathbb{P}\bigg(\bigcup_i A_i\bigg) = \sum_i \mathbb{P}(A_i)$$

The triple $(\Omega, \mathcal{F}, \mathbb{P})$ is called a **probability
space**.[^8]

If $\mathbb{P}(A) = 1$, we say that $A$ occurs **almost surely** (often
abbreviated a.s.).[^9], and conversely $A$ occurs **almost never** if
$\mathbb{P}(A) = 0$.

From these axioms, a number of useful rules can be derived.

:::{prf:proposition} Probability axioms
:label: probability-axioms
:nonumber:

Let $A$ be an event. 

Then

(i) $\mathbb{P}(A^\text{c}) = 1 - \mathbb{P}(A)$.

(ii) If $B$ is an event and $B \subseteq A$, then
     $\mathbb{P}(B) \leq \mathbb{P}(A)$.

(iii) $0 = \mathbb{P}(\varnothing) \leq \mathbb{P}(A) \leq \mathbb{P}(\Omega) = 1$
:::


:::{prf:proof}

 (i) Using the countable additivity of $\mathbb{P}$, we have

$$\mathbb{P}(A) + \mathbb{P}(A^\text{c}) = \mathbb{P}(A \mathbin{\dot{\cup}} A^\text{c}) = \mathbb{P}(\Omega) = 1$$

To show (ii), suppose $B \in \mathcal{F}$ and $B \subseteq A$. Then

$$\mathbb{P}(A) = \mathbb{P}(B \mathbin{\dot{\cup}} (A \setminus B)) = \mathbb{P}(B) + \mathbb{P}(A \setminus B) \geq \mathbb{P}(B)$$

as claimed.

For (iii): the middle inequality follows from (ii) since
$\varnothing \subseteq A \subseteq \Omega$. We also have

$$\mathbb{P}(\varnothing) = \mathbb{P}(\varnothing \mathbin{\dot{\cup}} \varnothing) = \mathbb{P}(\varnothing) + \mathbb{P}(\varnothing)$$

by countable additivity, which shows $\mathbb{P}(\varnothing) = 0$.

:::


:::{prf:proposition}
:label: probability-union
:nonumber:

If $A$ and $B$ are events, then
$\mathbb{P}(A \cup B) = \mathbb{P}(A) + \mathbb{P}(B) - \mathbb{P}(A \cap B)$.

:::


:::{prf:proof}

The key is to break the events up into their various
overlapping and non-overlapping parts. 

$$\begin{aligned}
\mathbb{P}(A \cup B) &= \mathbb{P}((A \cap B) \mathbin{\dot{\cup}} (A \setminus B) \mathbin{\dot{\cup}} (B \setminus A)) \\
&= \mathbb{P}(A \cap B) + \mathbb{P}(A \setminus B) + \mathbb{P}(B \setminus A) \\
&= \mathbb{P}(A \cap B) + \mathbb{P}(A) - \mathbb{P}(A \cap B) + \mathbb{P}(B) - \mathbb{P}(A \cap B) \\
&= \mathbb{P}(A) + \mathbb{P}(B) - \mathbb{P}(A \cap B)
\end{aligned}$$

 ◻
:::

:::{prf:proposition} Union bound
:label: probability-union
:nonumber:

If $\{A_i\} \subseteq \mathcal{F}$ is a countable set of events,
disjoint or not, then

$$\mathbb{P}\bigg(\bigcup_i A_i\bigg) \leq \sum_i \mathbb{P}(A_i)$$

:::

This inequality is sometimes referred to as **Boole's inequality** or
the **union bound**.


:::{prf:proof}

Define $B_1 = A_1$ and
$B_i = A_i \setminus (\bigcup_{j < i} A_j)$ for $i > 1$, noting that
$\bigcup_{j \leq i} B_j = \bigcup_{j \leq i} A_j$ for all $i$ and the
$B_i$ are disjoint. Then

$$\mathbb{P}\bigg(\bigcup_i A_i\bigg) = \mathbb{P}\bigg(\bigcup_i B_i\bigg) = \sum_i \mathbb{P}(B_i) \leq \sum_i \mathbb{P}(A_i)$$

where the last inequality follows by monotonicity since
$B_i \subseteq A_i$ for all $i$.

:::


## Conditional probability

The **conditional probability** of event $A$ given that event $B$ has
occurred is written $\mathbb{P}(A | B)$ and defined as

$$\mathbb{P}(A | B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)}$$

assuming $\mathbb{P}(B) > 0$.[^10]

## Chain rule of probability

Another very useful tool, the **chain rule**, follows immediately from
this definition:

$$\mathbb{P}(A \cap B) = \mathbb{P}(A | B)\mathbb{P}(B) = \mathbb{P}(B | A)\mathbb{P}(A)$$

## Bayes' rule

Taking the equality from above one step further, we arrive at the simple
but crucial **Bayes' rule**:

$$\mathbb{P}(A | B) = \frac{\mathbb{P}(B | A)\mathbb{P}(A)}{\mathbb{P}(B)}$$

It is sometimes beneficial to omit the normalizing constant and write

$$
\mathbb{P}(A | B) \propto \mathbb{P}(A)\mathbb{P}(B | A)
$$ 

Under this
formulation, $\mathbb{P}(A)$ is often referred to as the **prior**,
$\mathbb{P}(A | B)$ as the **posterior**, and $\mathbb{P}(B | A)$ as the
**likelihood**.

In the context of machine learning, we can use Bayes' rule to update our
"beliefs" (e.g. values of our model parameters) given some data that
we've observed.

