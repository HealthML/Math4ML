# Taylor's Theorem

Taylor's theorem is a fundamental result in calculus that provides an approximation of a function as a polynomial. It states that if a function $ f $ is sufficiently smooth (i.e., it has derivatives of all orders) at a point $ a $, then the function can be approximated by its Taylor series around that point.
The Taylor series of a function $ f $ at a point $ a $ is given by:

$$f(x) = f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2 + \frac{f'''(a)}{3!}(x - a)^3 + \ldots$$
or more compactly:

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x - a)^n$$

where $ f^{(n)}(a) $ is the $ n $-th derivative of $ f $ evaluated at $ a $, and $ n! $ is the factorial of $ n $.
The remainder term $ R_n(x) $ in the Taylor series can be expressed as:

$$R_n(x) = \frac{f^{(n+1)}(c)}{(n+1)!}(x - a)^{n+1}$$

for some $ c $ between $ a $ and $ x $. This remainder term gives an estimate of the error in the approximation.
The Taylor series converges to the function $ f $ if the limit of the remainder term $ R_n(x) $ approaches zero as $ n $ approaches infinity.
The Taylor series is particularly useful for approximating functions that are difficult to compute directly, and it is widely used in numerical analysis, physics, and engineering.

## Example
Let's consider the function $ f(x) = e^x $ and approximate it around the point $ a = 0 $.
The derivatives of $ f $ at $ a = 0 $ are:
- $ f(0) = e^0 = 1 $
- $ f'(0) = e^0 = 1 $
- $ f''(0) = e^0 = 1 $
- $ f'''(0) = e^0 = 1 $
- ...
- $ f^{(n)}(0) = e^0 = 1 $ for all $ n $

Thus, the Taylor series for $ e^x $ around $ a = 0 $ is:

$$e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \ldots = \sum_{n=0}^{\infty} \frac{x^n}{n!}$$

This series converges to $ e^x $ for all $ x $.


## Conclusion  
Taylor's theorem is a powerful tool in mathematics that allows us to approximate functions using polynomials. It provides a framework for understanding the behavior of functions near a specific point and has wide-ranging applications in various fields. By using Taylor series, we can simplify complex problems and gain insights into the properties of functions.

# Taylor's theorem
Taylor's theorem has natural generalizations to functions of more than one variable.

:::{prf:theorem} Taylors theorem
:label: thm-taylor-theorem-numopt2
:nonumber:
Suppose $f : \mathbb{R}^d \to \mathbb{R}$ is continuously differentiable, and let $\mathbf{h} \in \mathbb{R}^d$.

Then there exists $t \in (0,1)$ such that

$$f(\mathbf{x} + \mathbf{h}) = f(\mathbf{x}) + \nabla f(\mathbf{x} + t\mathbf{h})^\top\mathbf{h}$$

Furthermore, if $f$ is twice continuously differentiable, then

$$\nabla f(\mathbf{x} + \mathbf{h}) = \nabla f(\mathbf{x}) + \int_0^1 \nabla^2 f(\mathbf{x} + t\mathbf{h})\mathbf{h} \mathrm{d}{t}$$

and there exists $t \in (0,1)$ such that

$$f(\mathbf{x} + \mathbf{h}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^\top\mathbf{h} + \frac{1}{2}\mathbf{h}^\top\nabla^2f(\mathbf{x}+t\mathbf{h})\mathbf{h}$$
:::

This theorem is used in proofs about conditions for local minima of unconstrained optimization problems.
It is also used in proofs about the convergence of gradient descent algorithms.
The Taylor series for functions of several variables is given by:

$$f(\mathbf{x}) = \sum_{k=0}^{\infty} \frac{1}{k!}\nabla^k f(\mathbf{x})^\top (\mathbf{x} - \mathbf{a})^k$$

where $ \nabla^k f(\mathbf{x}) $ is the $ k $-th derivative of $ f $ at $ \mathbf{x} $.
The Taylor series converges to the function $ f $ if the limit of the remainder term approaches zero as $ n $ approaches infinity.
The Taylor series is particularly useful for approximating functions that are difficult to compute directly, and it is widely used in numerical analysis, physics, and engineering.
The Taylor series is a powerful tool in mathematics that allows us to approximate functions using polynomials. It provides a framework for understanding the behavior of functions near a specific point and has wide-ranging applications in various fields. By using Taylor series, we can simplify complex problems and gain insights into the properties of functions.
