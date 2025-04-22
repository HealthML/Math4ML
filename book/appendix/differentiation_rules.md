# Basic Differentiation Rules

The following rules are useful for computing derivatives of common functions.
Let $f$ and $g$ be differentiable, $C$ a constant:
- **Constant**: $C$ is a constant, $x$ is a variable.

  $$
    D(C) = 0
  $$
- **Power**: $n$ is a real number.
  
  $$
    D(x^n) = n\,x^{n-1}
  $$
- **Exponential**: $e$ is the base of the natural logarithm.
  
  $$
    D(e^x) = e^x
  $$
- **Logarithm**: $x>0$.
  
  $$
    D(\ln x) = \frac{1}{x}
  $$
- **Trigonometric**: $x$ is a variable.
  
  $$
    D(\sin x) = \cos x,\quad
    D(\cos x) = -\sin x,\quad
    D(\tan x) = \sec^2 x
  $$
- **Inverse trigonometric**: $x>0$.
  
  $$
    D(\operatorname{arcsin} x) = \frac{1}{\sqrt{1-x^2}},\quad
    D(\operatorname{arccos} x) = -\frac{1}{\sqrt{1-x^2}},\quad
    D(\operatorname{arctan} x) = \frac{1}{1+x^2}
  $$
- **Hyperbolic**: $x$ is a variable.

  $$
    D(\sinh x) = \cosh x,\quad
    D(\cosh x) = \sinh x,\quad
    D(\tanh x) = \operatorname{sech}^2 x
  $$
- **Inverse hyperbolic**: $x>0$.

  $$
    D(\operatorname{arcsinh} x) = \frac{1}{\sqrt{1+x^2}},\quad
    D(\operatorname{arccosh} x) = \frac{1}{\sqrt{x^2-1}},\quad
    D(\operatorname{arctanh} x) = \frac{1}{1-x^2}
  $$
- **Absolute value**: $x$ is a variable.

  $$
    D(|x|) = \begin{cases}
      1 & \text{if } x>0 \\
      -1 & \text{if } x<0
    \end{cases}
  $$
- **Step function**: $x$ is a variable.

  $$
    D(\text{sgn}(x)) = \begin{cases}
      1 & \text{if } x>0 \\
      -1 & \text{if } x<0
    \end{cases}
  $$
- **Heaviside step function**: $x$ is a variable.

  $$
    D(H(x)) = \delta(x) = \begin{cases}
      0 & \text{if } x\neq0 \\
      \infty & \text{if } x=0
    \end{cases}
  $$
  where $\delta(x)$ is the Dirac delta function.
- **Dirac delta function**: $x$ is a variable.

  $$
    D(\delta(x)) = \delta'(x) = \begin{cases}
      0 & \text{if } x\neq0 \\
      \infty & \text{if } x=0
    \end{cases}
  $$
    where $\delta'(x)$ is the derivative of the Dirac delta function.

- **Constant multiple**  
  $\displaystyle D\bigl[C\,f(x)\bigr] = C\,f'(x)$.
- **Sum rule**  
  $\displaystyle D\bigl[f(x)+g(x)\bigr] = f'(x)+g'(x)$.
- **Product rule**  
  $\displaystyle D\bigl[f(x)\,g(x)\bigr] = f(x)\,g'(x) + g(x)\,f'(x)$.
- **Quotient rule**  

  $$
    D\!\Bigl[\tfrac{f(x)}{g(x)}\Bigr]
    = \frac{g(x)\,f'(x)\;-\;f(x)\,g'(x)}{[g(x)]^2}.
  $$
