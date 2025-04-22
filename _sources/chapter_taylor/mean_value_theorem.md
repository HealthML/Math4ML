## Mean value theorem

:::{prf:theorem} Mean value theorem
:label: thm-mean-value-theorem
:nonumber:
Let $f:[a,b]\to\mathbb{R}$ be a continuous function on the closed interval $[a,b]$, and differentiable on the open interval $(a,b)$, where $a\neq b$.

Then there exists some $c \in (a,b)$ such that

$$f'(c)=\frac{f(b)-f(a)}{b-a}.$$
:::


### Mean value theorem in several variables

Let $f:\mathbb{R}^n\to\mathbb{R}$ be a differentiable function.

Fix points $\mathbf{x},\mathbf{y}$ , and define $g(t)=f\Big((1-t)\mathbf{x}+t\mathbf{y}\Big)$.

Since $g$ is a differentiable function in one variable, the mean value theorem gives:

$$g(1)-g(0)=g'(c) \; \; \; \text{for some $c\in [0,1]$.}$$

But since $g(1)=f(\mathbf{y})$ and $g(0)=f(\mathbf{x})$, computing $g'(c)$ explicitly we have:

$$f(\mathbf{y})-f(\mathbf{x})=\nabla f\Big((1-c)\mathbf{x}+c\mathbf{y}\Big)^\top (\mathbf{y}-\mathbf{x})$$

By the **Cauchy–Schwarz inequality**, the equation gives the estimate:

$$\Bigg|f(\mathbf{y})-f(\mathbf{x})\Bigg|\le\Bigg|\nabla f\Big((1-c)\mathbf{x}+c\mathbf{y}\Big)\Bigg|\ \Big|\mathbf{y} - \mathbf{x}\Big|.$$

In particular, when the partial derivatives of $f$ are bounded, $f$ is \term{Lipschitz continuous}.

$$\Big|g(y)\Big|=\Bigg|g(y)-g(x)\Bigg|\le (0)\Big|y-x\Big|=0$$

### Mean value theorem for vector-valued functions

There is no exact analog of the mean value theorem for vector-valued functions.

$$f_i(x+h) - f_i(x) = \nabla f_i (x + t_ih)^\top h$$

Generally there will not be a \emph{single} $t$ that fullfils this for all $i$.

However a certain type of generalization of the mean value theorem to vector-valued functions is obtained as follows:

Let $f$ be a continuously differentiable real-valued function defined on an open interval $I$, and let $\mathbf{x}$ as well as $\mathbf{x} + h$ be points.

The mean value theorem in one variable tells us that there exists some $t^*$ between 0 and 1 such that

$$f(x+h)-f(x) = f'(x+t^*h)\cdot h.$$

On the other hand, we have, by the **fundamental theorem of calculus** followed by a change of variables,

$$f(x+h)-f(x) = \int_x^{x+h} f'(u) \, du = \left (\int_0^1 f'(x+th)\,dt\right)\cdot h.$$

Thus, the value $f'(x + t^*h)$ at the particular point $t^*$ has been replaced by the mean value
$\int_0^1 f'(x+th)\,dt$.

This last version can be generalized to vector valued functions:

:::{prf:theorem} Jacobian Lemma
:label: thm-Jacobian
:nonumber:
Let $\mathbf{f}:\mathbb{R}^n \rightarrow '\mathbb{R}^m$ continuously differentiable, and $x,h\in\mathbb{R}^n$ be vectors.

Then we have:

$$\mathbf{f}(\mathbf{x}+\mathbf{h})-f(\mathbf{x}) = \left (\int_0^1 \nabla \mathbf{f}(\mathbf{x}+th)\,dt\right)^\top h,$$

where $\nabla \mathbf{f}$ denotes the \term{Jacobian matrix} and the integral of a matrix is to be understood componentwise.
:::

If $f$ is twice continuously differentiable, then

$$f(\mathbf{x}_0 + \mathbf{h}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top\mathbf{h} + \frac{1}{2}\mathbf{h}^\top\nabla^2f(\mathbf{x}_0)\mathbf{h}$$

is a parabolic approximation to $f$ that has the same Gradient and Hessian as $f$ at $\mathbf{x}_0$.

