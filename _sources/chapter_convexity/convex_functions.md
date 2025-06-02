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
# Basics of convex functions

In the remainder of this section, assume
$f : \mathbb{R}^d \to \mathbb{R}$ unless otherwise noted. We'll start
with the definitions and then give some results.

A function $f$ is **convex** if

$$f(t\mathbf{x} + (1-t)\mathbf{y}) \leq t f(\mathbf{x}) + (1-t)f(\mathbf{y})$$

for all $\mathbf{x}, \mathbf{y} \in \operatorname{dom} f$ and all $t \in [0,1]$.

If the inequality holds strictly (i.e. $<$ rather than $\leq$) for all
$t \in (0,1)$ and $\mathbf{x} \neq \mathbf{y}$, then we say that $f$ is
**strictly convex**.

A function $f$ is **strongly convex with parameter $m$** (or
**$m$-strongly convex**) if the function

$$\mathbf{x} \mapsto f(\mathbf{x}) - \frac{m}{2}\|\mathbf{x}\|_2^2$$ 

is convex.

These conditions are given in increasing order of strength; strong
convexity implies strict convexity which implies convexity.



## Geometric interpretation
The following figure illustrates the three types of convexity:

Geometrically, convexity means that the line segment between two points
on the graph of $f$ lies on or above the graph itself.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# Define a convex function
f = lambda x: x**2

# Define x values and compute y
x = np.linspace(-2, 2, 400)
y = f(x)

# Choose two points on the graph
x1, x2 = -1.5, 1.0
y1, y2 = f(x1), f(x2)

# Compute the line segment between the two points
t = np.linspace(0, 1, 100)
xt = t * x1 + (1 - t) * x2
yt_line = t * y1 + (1 - t) * y2

# Plot the function and the line segment
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$f(x) = x^2$', color='blue')
plt.plot(xt, yt_line, 'r--', label='Line segment')
plt.plot([x1, x2], [y1, y2], 'ro')  # endpoints
plt.title("Geometric Interpretation of Convexity")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

Strict convexity means that the graph of $f$ lies strictly above the
line segment, except at the segment endpoints. 
(So actually the function in the figure appears to be strictly convex.)


```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# Define x values
x = np.linspace(-2, 2, 400)

# Define three functions:
# 1. Convex with a flat part: piecewise constant in the center
def f1(x):
    return np.where(np.abs(x) < 0.2, 0.2, np.abs(x))

# 2. Strictly convex but not strongly convex
f2 = lambda x: x**4

# 3. Strongly convex
f3 = lambda x: x**2

# Evaluate functions
y1 = f1(x)
y2 = f2(x)
y3 = f3(x)

# Plot the functions
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label=r'$f(x) = \max(|x|, 0.2)$ (Convex, not strictly)', linestyle='--')
plt.plot(x, y2, label=r'$f(x) = x^4$ (Strictly Convex)', linestyle='-.')
plt.plot(x, y3, label=r'$f(x) = x^2$ (Strongly Convex)', linestyle='-')
plt.title("Examples of Convex, Strictly Convex, and Strongly Convex Functions")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.xlim(-0.6, 0.6)
plt.ylim(-0.02, 0.5)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

```

* **Dashed line**: $f(x) = \max(|x|, 0.2)$ — convex with a flat region near 0, so not strictly convex.
* **Dash-dotted line**: $f(x) = x^4$ — strictly convex but with vanishing curvature at 0 (not strongly convex).
* **Solid line**: $f(x) = x^2$ — strongly convex everywhere with constant positive curvature.



## Consequences of convexity

Why do we care if a function is (strictly/strongly) convex?

Basically, our various notions of convexity have implications about the
nature of minima. It should not be surprising that the stronger
conditions tell us more about the minima.

:::{prf:proposition} Minima of convex functions
:label: prop-convex-minima
:nonumber:
Let $\mathcal{X}$ be a convex set.

If $f$ is convex, then any local minimum of $f$ in $\mathcal{X}$ is also a global minimum.
:::

:::{prf:proof}

Suppose $f$ is convex, and let $\mathbf{x}^*$ be a local
minimum of $f$ in $\mathcal{X}$.

Then for some neighborhood $N \subseteq \mathcal{X}$ about $\mathbf{x}^*$, we have
$f(\mathbf{x}) \geq f(\mathbf{x}^*)$ for all $\mathbf{x} \in N$. 

Suppose
towards a contradiction that there exists
$\tilde{\mathbf{x}} \in \mathcal{X}$ such that
$f(\tilde{\mathbf{x}}) < f(\mathbf{x}^*)$.

Consider the line segment
$\mathbf{x}(t) = t\mathbf{x}^* + (1-t)\tilde{\mathbf{x}}, ~ t \in [0,1]$,
noting that $\mathbf{x}(t) \in \mathcal{X}$ by the convexity of
$\mathcal{X}$. 

Then by the convexity of $f$,

$$f(\mathbf{x}(t)) \leq tf(\mathbf{x}^*) + (1-t)f(\tilde{\mathbf{x}}) < tf(\mathbf{x}^*) + (1-t)f(\mathbf{x}^*) = f(\mathbf{x}^*)$$

for all $t \in (0,1)$.

We can pick $t$ to be sufficiently close to $1$ that
$\mathbf{x}(t) \in N$; 
then $f(\mathbf{x}(t)) \geq f(\mathbf{x}^*)$ by
the definition of $N,$ but $f(\mathbf{x}(t)) < f(\mathbf{x}^*)$ by the
above inequality, a contradiction.

It follows that $f(\mathbf{x}^*) \leq f(\mathbf{x})$ for all
$\mathbf{x} \in \mathcal{X}$, so $\mathbf{x}^*$ is a global minimum of
$f$ in $\mathcal{X}$. ◻
:::

:::{prf:proposition} Minima stricly convex functions
:label: prop-minima-striclty-convex
:nonumber:

Let $\mathcal{X}$ be a convex set.

If $f$ is strictly convex, then there
exists at most one local minimum of $f$ in $\mathcal{X}$. Consequently,
if it exists it is the unique global minimum of $f$ in $\mathcal{X}$.
:::

:::{prf:proof}

The second sentence follows from the first, so all we must show
is that if a local minimum exists in $\mathcal{X}$ then it is unique.

Suppose $\mathbf{x}^*$ is a local minimum of $f$ in $\mathcal{X}$, and
suppose towards a contradiction that there exists a local minimum
$\tilde{\mathbf{x}} \in \mathcal{X}$ such that
$\tilde{\mathbf{x}} \neq \mathbf{x}^*$.

Since $f$ is strictly convex, it is convex, so $\mathbf{x}^*$ and
$\tilde{\mathbf{x}}$ are both global minima of $f$ in $\mathcal{X}$ by
the previous result. Hence $f(\mathbf{x}^*) = f(\tilde{\mathbf{x}})$.
Consider the line segment
$\mathbf{x}(t) = t\mathbf{x}^* + (1-t)\tilde{\mathbf{x}}, ~ t \in [0,1]$,
which again must lie entirely in $\mathcal{X}$. By the strict convexity
of $f$,

$$f(\mathbf{x}(t)) < tf(\mathbf{x}^*) + (1-t)f(\tilde{\mathbf{x}}) = tf(\mathbf{x}^*) + (1-t)f(\mathbf{x}^*) = f(\mathbf{x}^*)$$

for all $t \in (0,1)$. 

But this contradicts the fact that $\mathbf{x}^*$
is a global minimum. Therefore if $\tilde{\mathbf{x}}$ is a local
minimum of $f$ in $\mathcal{X}$, then
$\tilde{\mathbf{x}} = \mathbf{x}^*$, so $\mathbf{x}^*$ is the unique
minimum in $\mathcal{X}$. ◻
:::

It is worthwhile to examine how the feasible set affects the
optimization problem. We will see why the assumption that $\mathcal{X}$
is convex is needed in the results above.

## Effects of Changing the Feasible Set

Consider the function $f(x) = x^2$, which is a strictly convex function.
The unique global minimum of this function in $\mathbb{R}$ is $x = 0$.

But let's see what happens when we change the feasible set
$\mathcal{X}$.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 400)
f = x**2

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
cases = [
    {"title": r"(i) $\mathcal{X} = \{1\}$", "feasible": [1], "color": "orange"},
    {"title": r"(ii) $\mathcal{X} = \mathbb{R} \setminus \{0\}$", "feasible": np.concatenate([x[x < 0], x[x > 0]]), "color": "orange"},
    {"title": r"(iii) $\mathcal{X} = (-\infty,-1] \cup [0,\infty)$", "feasible": np.concatenate([x[x <= -1], x[x >= 0]]), "color": "orange"},
    {"title": r"(iv) $\mathcal{X} = (-\infty,-1] \cup [1,\infty)$", "feasible": np.concatenate([x[x <= -1], x[x >= 1]]), "color": "orange"},
]

for ax, case in zip(axes.flat, cases):
    # Plot the function
    ax.plot(x, f, 'b-', label=r'$f(x) = x^2$')
    # Highlight feasible set
    if isinstance(case["feasible"], list):
        for pt in case["feasible"]:
            ax.plot(pt, pt**2, 'o', color=case["color"], markersize=10, label="feasible set" if pt == case["feasible"][0] else "")
    else:
        ax.plot(case["feasible"], case["feasible"]**2, color=case["color"], linewidth=6, alpha=0.3, label="feasible set")
    # Mark minima
    if case["title"].startswith("(i)"):
        ax.plot(1, 1, 'ro', label="minimum")
    elif case["title"].startswith("(ii)"):
        ax.text(0, 0.5, "No minimum", color="red", ha="center", fontsize=12)
    elif case["title"].startswith("(iii)"):
        ax.plot(-1, 1, 'ro', label="local min")
        ax.plot(0, 0, 'go', label="global min")
    elif case["title"].startswith("(iv)"):
        ax.plot(-1, 1, 'ro', label="global min")
        ax.plot(1, 1, 'ro')
    ax.set_title(case["title"], fontsize=14)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 9)
    ax.legend(loc="upper left")
    ax.grid(True)

plt.tight_layout()
plt.show()
```


(i) $\mathcal{X} = \{1\}$: This set is actually convex, so we still have
    a unique global minimum. But it is not the same as the unconstrained
    minimum!

(ii) $\mathcal{X} = \mathbb{R} \setminus \{0\}$: This set is non-convex,
     and we can see that $f$ has no minima in $\mathcal{X}$. For any
     point $x \in \mathcal{X}$, one can find another point
     $y \in \mathcal{X}$ such that $f(y) < f(x)$.

(iii) $\mathcal{X} = (-\infty,-1] \cup [0,\infty)$: This set is
      non-convex, and we can see that there is a local minimum
      ($x = -1$) which is distinct from the global minimum ($x = 0$).

(iv) $\mathcal{X} = (-\infty,-1] \cup [1,\infty)$: This set is
     non-convex, and we can see that there are two global minima
     ($x = \pm 1$).


## Showing that a function is convex

Hopefully the previous section has convinced the reader that convexity
is an important property. Next we turn to the issue of showing that a
function is (strictly/strongly) convex. It is of course possible (in
principle) to directly show that the condition in the definition holds,
but this is usually not the easiest way.

:::{prf:proposition} Norms
:label: prop-norms-convex
:nonumber:

Norms are convex.
:::


:::{prf:proof}

Let $\|\cdot\|$ be a norm on a vector space $V$. Then for all
$\mathbf{x}, \mathbf{y} \in V$ and $t \in [0,1]$,

$$\|t\mathbf{x} + (1-t)\mathbf{y}\| \leq \|t\mathbf{x}\| + \|(1-t)\mathbf{y}\| = |t|\|\mathbf{x}\| + |1-t|\|\mathbf{y}\| = t\|\mathbf{x}\| + (1-t)\|\mathbf{y}\|$$

where we have used respectively the triangle inequality, the homogeneity
of norms, and the fact that $t$ and $1-t$ are nonnegative. Hence
$\|\cdot\|$ is convex. ◻
:::

:::{prf:proposition} Gradient of Convex Functions
:label: prop-convex-functions-graph
:nonumber:

Suppose $f$ is differentiable.

Then $f$ is convex if and only if

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$$

for all $\mathbf{x}, \mathbf{y} \in \operatorname{dom} f$.
:::


:::{prf:proof}

(**"Only if" direction**)  
Suppose $f$ is convex and differentiable. By the definition of convexity, for any $t \in [0,1]$ and any $\mathbf{x}, \mathbf{y} \in \operatorname{dom} f$,

$$
f(t\mathbf{y} + (1-t)\mathbf{x}) \leq t f(\mathbf{y}) + (1-t) f(\mathbf{x}).
$$
Define $\varphi(t) = f(\mathbf{x} + t(\mathbf{y} - \mathbf{x}))$ for $t \in [0,1]$. Then $\varphi$ is convex as a function of $t$, and differentiable. The convexity of $\varphi$ implies

$$
\varphi(1) \geq \varphi(0) + \varphi'(0)(1-0) = f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle.
$$
But $\varphi(1) = f(\mathbf{y})$, so

$$
f(\mathbf{y}) \geq f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle.
$$

(**"If" direction**)  
Suppose the inequality

$$
f(\mathbf{y}) \geq f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle
$$
holds for all $\mathbf{x}, \mathbf{y} \in \operatorname{dom} f$. We want to show that $f$ is convex, i.e.,

$$
f(t\mathbf{y} + (1-t)\mathbf{x}) \leq t f(\mathbf{y}) + (1-t) f(\mathbf{x})
$$
for all $t \in [0,1]$.

Let $t \in (0,1)$ and define $\mathbf{z} = t\mathbf{y} + (1-t)\mathbf{x}$. 

By the assumption, applied at $\mathbf{x}$ and $\mathbf{y}$:

$$
\begin{aligned}
f(\mathbf{y}) &\geq f(\mathbf{z}) + \langle \nabla f(\mathbf{z}), \mathbf{y} - \mathbf{z} \rangle \\
f(\mathbf{x}) &\geq f(\mathbf{z}) + \langle \nabla f(\mathbf{z}), \mathbf{x} - \mathbf{z} \rangle
\end{aligned}
$$

Multiply the first inequality by $t$ and the second by $1-t$, then add:

$$
\begin{aligned}
t f(\mathbf{y}) + (1-t) f(\mathbf{x}) &\geq t f(\mathbf{z}) + t \langle \nabla f(\mathbf{z}), \mathbf{y} - \mathbf{z} \rangle \\
&\quad + (1-t) f(\mathbf{z}) + (1-t) \langle \nabla f(\mathbf{z}), \mathbf{x} - \mathbf{z} \rangle \\
&= f(\mathbf{z}) + \langle \nabla f(\mathbf{z}), t(\mathbf{y} - \mathbf{z}) + (1-t)(\mathbf{x} - \mathbf{z}) \rangle
\end{aligned}
$$

But

$$
\begin{aligned}
t(\mathbf{y} - \mathbf{z}) + (1-t)(\mathbf{x} - \mathbf{z}) &= t(\mathbf{y} - (t\mathbf{y} + (1-t)\mathbf{x})) + (1-t)(\mathbf{x} - (t\mathbf{y} + (1-t)\mathbf{x})) \\
&= t((1-t)(\mathbf{y} - \mathbf{x})) + (1-t)(-t(\mathbf{y} - \mathbf{x})) \\
&= t(1-t)(\mathbf{y} - \mathbf{x}) - t(1-t)(\mathbf{y} - \mathbf{x}) \\
&= 0
\end{aligned}
$$

So the inner product term vanishes, and we have

$$
t f(\mathbf{y}) + (1-t) f(\mathbf{x}) \geq f(\mathbf{z}) = f(t\mathbf{y} + (1-t)\mathbf{x})
$$
which is the definition of convexity.

◻
:::

:::{prf:proposition} Hessian of Convex Functions
:label: prop-Hessian-convex
:nonumber:

Suppose $f$ is twice differentiable. Then

(i) $f$ is convex if and only if $\nabla^2 f(\mathbf{x}) \succeq 0$ for
    all $\mathbf{x} \in \operatorname{dom} f$.

(ii) If $\nabla^2 f(\mathbf{x}) \succ 0$ for all
     $\mathbf{x} \in \operatorname{dom} f$, then $f$ is strictly convex.

(iii) $f$ is $m$-strongly convex if and only if
      $\nabla^2 f(\mathbf{x}) \succeq mI$ for all
      $\mathbf{x} \in \operatorname{dom} f$.
:::


:::{prf:proof}

We prove each part in turn.

**(i) $f$ is convex if and only if $\nabla^2 f(\mathbf{x}) \succeq 0$ for all $\mathbf{x}$.**

Recall that for a twice differentiable function $f$, the second-order Taylor expansion at $\mathbf{x}$ gives

$$
f(\mathbf{y}) = f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle + \frac{1}{2} (\mathbf{y} - \mathbf{x})^\top \nabla^2 f(\mathbf{z}) (\mathbf{y} - \mathbf{x})
$$
for some $\mathbf{z}$ on the line segment between $\mathbf{x}$ and $\mathbf{y}$.

- **($\implies$)** Suppose $f$ is convex. Then, by the first-order condition for convexity, for all $\mathbf{x}, \mathbf{y}$,

$$
f(\mathbf{y}) \geq f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle.
$$
Subtracting the right-hand side from the Taylor expansion, we get

$$
f(\mathbf{y}) - f(\mathbf{x}) - \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle = \frac{1}{2} (\mathbf{y} - \mathbf{x})^\top \nabla^2 f(\mathbf{z}) (\mathbf{y} - \mathbf{x}) \geq 0
$$
for all $\mathbf{x}, \mathbf{y}$ and some $\mathbf{z}$ between them. This is only possible if $\nabla^2 f(\mathbf{z})$ is positive semidefinite for all $\mathbf{z}$, and thus for all $\mathbf{x}$.

- **($\impliedby$)** Conversely, suppose $\nabla^2 f(\mathbf{x}) \succeq 0$ for all $\mathbf{x}$. Then, for any $\mathbf{x}, \mathbf{y}$, the function $g(t) = f(\mathbf{x} + t(\mathbf{y} - \mathbf{x}))$ is convex in $t$ because $g''(t) = (\mathbf{y} - \mathbf{x})^\top \nabla^2 f(\mathbf{x} + t(\mathbf{y} - \mathbf{x})) (\mathbf{y} - \mathbf{x}) \geq 0$. Therefore, $f$ is convex.

---

**(ii) If $\nabla^2 f(\mathbf{x}) \succ 0$ for all $\mathbf{x}$, then $f$ is strictly convex.**

If $\nabla^2 f(\mathbf{x})$ is positive definite for all $\mathbf{x}$, then for any $\mathbf{x} \neq \mathbf{y}$, the quadratic form $(\mathbf{y} - \mathbf{x})^\top \nabla^2 f(\mathbf{z}) (\mathbf{y} - \mathbf{x}) > 0$ for all $\mathbf{z}$ between $\mathbf{x}$ and $\mathbf{y}$. Thus, the Taylor expansion above is strictly greater than zero for $\mathbf{x} \neq \mathbf{y}$, so the convexity inequality is strict for $t \in (0,1)$, i.e., $f$ is strictly convex.

---

**(iii) $f$ is $m$-strongly convex if and only if $\nabla^2 f(\mathbf{x}) \succeq mI$ for all $\mathbf{x}$.**

Recall that $f$ is $m$-strongly convex if $f(\mathbf{x}) - \frac{m}{2}\|\mathbf{x}\|^2$ is convex. The Hessian of this function is $\nabla^2 f(\mathbf{x}) - mI$. By part (i), this is convex if and only if $\nabla^2 f(\mathbf{x}) - mI \succeq 0$, i.e., $\nabla^2 f(\mathbf{x}) \succeq mI$ for all $\mathbf{x}$.

◻
:::

:::{prf:proposition} Scaling Convex Functions
:label: prop-scaling-convex-functions
:nonumber:

If $f$ is convex and $\alpha \geq 0$, then $\alpha f$ is convex.
:::


:::{prf:proof}

Suppose $f$ is convex and $\alpha \geq 0$. Then for all
$\mathbf{x}, \mathbf{y} \in \operatorname{dom}(\alpha f) = \operatorname{dom} f$, 

$$\begin{aligned}
(\alpha f)(t\mathbf{x} + (1-t)\mathbf{y}) &= \alpha f(t\mathbf{x} + (1-t)\mathbf{y}) \\
&\leq \alpha\left(tf(\mathbf{x}) + (1-t)f(\mathbf{y})\right) \\
&= t(\alpha f(\mathbf{x})) + (1-t)(\alpha f(\mathbf{y})) \\
&= t(\alpha f)(\mathbf{x}) + (1-t)(\alpha f)(\mathbf{y})
\end{aligned}$$ 

so $\alpha f$ is convex. ◻
:::

:::{prf:proposition} Sum of Convex Functions
:label: prop-sum-convex-functions
:nonumber:

If $f$ and $g$ are convex, then $f+g$ is convex. 

Furthermore, if $g$ is
strictly convex, then $f+g$ is strictly convex, and if $g$ is
$m$-strongly convex, then $f+g$ is $m$-strongly convex.
:::


:::{prf:proof}

Suppose $f$ and $g$ are convex.
Then for all
$\mathbf{x}, \mathbf{y} \in \operatorname{dom} (f+g) = \operatorname{dom} f \cap \operatorname{dom} g$,

$$\begin{aligned}
(f+g)(t\mathbf{x} + (1-t)\mathbf{y}) &= f(t\mathbf{x} + (1-t)\mathbf{y}) + g(t\mathbf{x} + (1-t)\mathbf{y}) \\
&\leq tf(\mathbf{x}) + (1-t)f(\mathbf{y}) + g(t\mathbf{x} + (1-t)\mathbf{y}) & \text{convexity of $f$} \\
&\leq tf(\mathbf{x}) + (1-t)f(\mathbf{y}) + tg(\mathbf{x}) + (1-t)g(\mathbf{y}) & \text{convexity of $g$} \\
&= t(f(\mathbf{x}) + g(\mathbf{x})) + (1-t)(f(\mathbf{y}) + g(\mathbf{y})) \\
&= t(f+g)(\mathbf{x}) + (1-t)(f+g)(\mathbf{y})
\end{aligned}$$ 

so $f + g$ is convex.

If $g$ is strictly convex, the second inequality above holds strictly
for $\mathbf{x} \neq \mathbf{y}$ and $t \in (0,1)$, so $f+g$ is strictly
convex.

If $g$ is $m$-strongly convex, then the function
$h(\mathbf{x}) \equiv g(\mathbf{x}) - \frac{m}{2}\|\mathbf{x}\|_2^2$ is
convex, so $f+h$ is convex. 

But

$$(f+h)(\mathbf{x}) \equiv f(\mathbf{x}) + h(\mathbf{x}) \equiv f(\mathbf{x}) + g(\mathbf{x}) - \frac{m}{2}\|\mathbf{x}\|_2^2 \equiv (f+g)(\mathbf{x}) - \frac{m}{2}\|\mathbf{x}\|_2^2$$

so $f+g$ is $m$-strongly convex. ◻
:::

:::{prf:proposition} Weighted Sum of Convex Functions
:label: prop-convex-functions-weighted-sum
:nonumber:

If $f_1, \dots, f_n$ are convex and $\alpha_1, \dots, \alpha_n \geq 0$,
then 

$$\sum_{i=1}^n \alpha_i f_i$$ 

is convex.
:::

:::{prf:proof}
Follows from the previous two propositions by induction. ◻
:::

:::{prf:proposition} Combination of Affine and Convex Functions
:label: prop-linear-convex
:nonumber:

If $f$ is convex, then
$g(\mathbf{x}) \equiv f(\mathbf{A}\mathbf{x} + \mathbf{b})$ is convex
for any appropriately-sized $\mathbf{A}$ and $\mathbf{b}$.
:::


:::{prf:proof}

Suppose $f$ is convex and $g$ is defined like so. Then for all
$\mathbf{x}, \mathbf{y} \in \operatorname{dom} g$, 

$$\begin{aligned}
g(t\mathbf{x} + (1-t)\mathbf{y}) &= f(\mathbf{A}(t\mathbf{x} + (1-t)\mathbf{y}) + \mathbf{b}) \\
&= f(t\mathbf{A}\mathbf{x} + (1-t)\mathbf{A}\mathbf{y} + \mathbf{b}) \\
&= f(t\mathbf{A}\mathbf{x} + (1-t)\mathbf{A}\mathbf{y} + t\mathbf{b} + (1-t)\mathbf{b}) \\
&= f(t(\mathbf{A}\mathbf{x} + \mathbf{b}) + (1-t)(\mathbf{A}\mathbf{y} + \mathbf{b})) \\
&\leq tf(\mathbf{A}\mathbf{x} + \mathbf{b}) + (1-t)f(\mathbf{A}\mathbf{y} + \mathbf{b}) & \text{convexity of $f$} \\
&= tg(\mathbf{x}) + (1-t)g(\mathbf{y})
\end{aligned}$$ 

Thus $g$ is convex. ◻
:::

:::{prf:proposition} Maximum of Convex Functions
:label: prop-max-convex-functions
:nonumber:

If $f$ and $g$ are convex, then
$h(\mathbf{x}) \equiv \max\{f(\mathbf{x}), g(\mathbf{x})\}$ is convex.

:::

:::{prf:proof}

Suppose $f$ and $g$ are convex and $h$ is defined like so. Then
for all $\mathbf{x}, \mathbf{y} \in \operatorname{dom} h$, 

$$\begin{aligned}
h(t\mathbf{x} + (1-t)\mathbf{y}) &= \max\{f(t\mathbf{x} + (1-t)\mathbf{y}), g(t\mathbf{x} + (1-t)\mathbf{y})\} \\
&\leq \max\{tf(\mathbf{x}) + (1-t)f(\mathbf{y}), tg(\mathbf{x}) + (1-t)g(\mathbf{y})\} \\
&\leq \max\{tf(\mathbf{x}), tg(\mathbf{x})\} + \max\{(1-t)f(\mathbf{y}), (1-t)g(\mathbf{y})\} \\
&= t\max\{f(\mathbf{x}), g(\mathbf{x})\} + (1-t)\max\{f(\mathbf{y}), g(\mathbf{y})\} \\
&= th(\mathbf{x}) + (1-t)h(\mathbf{y})
\end{aligned}$$ 

Note that in the first inequality we have used convexity
of $f$ and $g$ plus the fact that $a \leq c, b \leq d$ implies
$\max\{a,b\} \leq \max\{c,d\}$. In the second inequality we have used
the fact that $\max\{a+b, c+d\} \leq \max\{a,c\} + \max\{b,d\}$.

Thus $h$ is convex. ◻
:::

### Examples

A good way to gain intuition about the distinction between convex,
strictly convex, and strongly convex functions is to consider examples
where the stronger property fails to hold.

Functions that are convex but not strictly convex:

(i) $f(\mathbf{x}) = \mathbf{w}^{\!\top\!}\mathbf{x} + \alpha$ for any
    $\mathbf{w} \in \mathbb{R}^d, \alpha \in \mathbb{R}$. Such a
    function is called an **affine function**, and it is both convex and
    concave.
    (In fact, a function is affine if and only if it is both convex and concave.)
    Note that linear functions and constant
    functions are special cases of affine functions.

(ii) $f(\mathbf{x}) = \|\mathbf{x}\|_1$

Functions that are strictly but not strongly convex:

(i) $f(x) = x^4$. 
This example is interesting because it is strictly
    convex but you cannot show this fact via a second-order argument
    (since $f''(0) = 0$).

(ii) $f(x) = \exp(x)$. 
This example is interesting because it's bounded
     below but has no local minimum.

(iii) $f(x) = -\log x$. 
This example is interesting because it's
      strictly convex but not bounded below.

Functions that are strongly convex:

(i) $f(\mathbf{x}) = \|\mathbf{x}\|_2^2$