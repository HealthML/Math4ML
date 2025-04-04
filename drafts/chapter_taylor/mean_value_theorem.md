## Mean value theorem

<!-- % The function $f$ attains the slope of the secant between $a$ and $b$ as the derivative at the point $\mathbf{x}i\in(a,b)$. -->
<!-- % [[File:Mittelwertsatz6.svg|thumb|It is also possible that there are multiple tangents parallel to the secant.]] -->
:::{prf:theorem} Mean value theorem
Let $f:[a,b]\to\mathbb{R}$ be a continuous function on the closed interval $[a,b]$, and differentiable on the open interval $(a,b)$, where $a\neq b$.

Then there exists some $c \in (a,b)$ such that

$$f'(c)=\frac{f(b)-f(a)}{b-a}.$$
:::

<!-- # \includegraphics[width=1\textwidth]{img/Mittelwertsatz3.png} -->

<!-- # \includegraphics[width=1\textwidth]{img/Mittelwertsatz6.png} -->



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

:::{prf:lemma}
Let $\mathbf{f}:\mathbb{R}^n \rightarrow '\mathbb{R}^m$ continuously differentiable, and $x,h\in\mathbb{R}^n$ be vectors.

Then we have:

$$\mathbf{f}(\mathbf{x}+\mathbf{h})-f(\mathbf{x}) = \left (\int_0^1 \nabla \mathbf{f}(\mathbf{x}+th)\,dt\right)^\top h,$$

where $\nabla \mathbf{f}$ denotes the \term{Jacobian matrix} and the integral of a matrix is to be understood componentwise.
:::
<!-- % '''Proof.''' Let ''f''<sub>1</sub>, ..., ''f<sub>m</sub>'' denote the components of ''f'' and define: -->

<!-- % :<math>\begin{cases} g_i : [0,1] \to \mathbf{R} \\ g_i(t) = f_i (x +th) \end{cases} </math> -->

<!-- % Then we have -->

<!-- % \begin{align}
% & f_i(x+h)-f_i(x) = g_i(1)-g_i(0) =\int_0^1 g_i'(t)\,dt \\
% = {} & \int_0^1 \left (\sum_{j=1}^n \frac{\partial f_i}{\partial x_j} (x+th)h_j\right)\,dt = \sum_{j=1}^n \left (\int_0^1 \frac{\partial f_i}{\partial x_j}(x+th)\,dt\right)h_j.
% \end{align}
 -->
<!-- % The claim follows since ''Df'' is the matrix consisting of the components <math>\tfrac{\partial f_i}{\partial x_j}.</math> -->

<!-- % :'''Lemma 2.''' Let ''v'' : [''a'', ''b''] → '''R'''<sup>''m''</sup> be a continuous function defined on the interval [''a'', ''b''] ⊂ '''R'''. Then we have -->
<!-- % ::<math>\left \|\int_a^b v(t)\,dt\right\|\leqslant \int_a^b \|v(t)\|\,dt.</math> -->

<!-- % '''Proof.''' Let ''u'' in '''R'''<sup>''m''</sup> denote the value of the integral
% :<math>u:=\int_a^b v(t)\,dt.</math>
% Now we have (using the [[Cauchy–Schwarz inequality]]):
% :<math>\|u\|^2 = \langle u,u \rangle =\left \langle \int_a^b v(t) \, dt, u \right\rangle = \int_a^b \langle v(t),u \rangle \,dt \leqslant \int_a^b \| v(t) \|\cdot \|u \|\,dt = \|u\| \int_a^b \|v(t)\|\,dt</math>
% Now cancelling the norm of ''u'' from both ends gives us the desired inequality.
 -->
<!-- % :'''Mean Value Inequality.''' If the norm of ''Df''(''x'' + ''th'') is bounded by some constant ''M'' for ''t'' in [0, 1], then -->
<!-- % ::<math>\|f(x+h)-f(x)\| \leqslant M\|h\|.</math> -->

<!-- % '''Proof.''' From Lemma 1 and 2  it follows that -->
<!-- % :<math>\|f(x+h)-f(x)\|=\left \|\int_0^1 (Df(x+th)\cdot h)\,dt\right\|  \leqslant \int_0^1 \|Df(x+th)\| \cdot \|h\|\, dt \leqslant M\| h\|.</math> -->


If $f$ is twice continuously differentiable, then

$$f(\mathbf{x}_0 + \mathbf{h}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top\mathbf{h} + \frac{1}{2}\mathbf{h}^\top\nabla^2f(\mathbf{x}_0)\mathbf{h}$$

is a parabolic approximation to $f$ that has the same Gradient and Hessian as $f$ at $\mathbf{x}_0$.

