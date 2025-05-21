# Exercise Sheet 2 Solutions


### 1.
#### (a)
Let  
\[
f(x, y) =
\begin{cases}
\dfrac{x \sin y}{x^2 + y^2} & \text{if } (x, y) \neq (0, 0) \\
0 & \text{if } (x, y) = (0, 0)
\end{cases}
\]

We are asked to examine the **continuity of \( f \) in \( \mathbb{R}^2 \)**.

*Remark*
We say that a single-variable function \( f : \mathbb{R} \rightarrow \mathbb{R} \) is **continuous at a point** \( a \in \mathbb{R} \) if

\[
\lim_{x \to a} f(x) = f(a)
\]

*Extension to Two Variables*
Similarly, for a function of two variables \( f : \mathbb{R}^2 \rightarrow \mathbb{R} \), we say that \( f \) is continuous at the point \( (a, b) \in \mathbb{R}^2 \) if

\[
\lim_{(x, y) \to (a, b)} f(x, y) = f(a, b)
\]

So, to study the continuity of \( f \), we need to check whether this limit exists and equals the value of the function at that point.

*Strategy*
To determine the **existence of**  
\[
\lim_{(x, y) \to (0, 0)} f(x, y),
\]
we must examine whether the limit exists and is the same **along all possible directions** towards \( (0, 0) \).

*Direction 1: Along the x-axis*
We approach \( (0, 0) \) along the **x-axis**, i.e., set \( y = 0 \).  
Then:

\[
f(x, 0) = \frac{x \sin(0)}{x^2 + 0^2} = \frac{0}{x^2} = 0 \quad \text{for all } x \neq 0
\]

\[
\Rightarrow \lim_{(x, y) \to (0, 0)} f(x, y) = \lim_{x \to 0} f(x, 0) = 0
\]

*Direction 2: Along the y-axis*
Let \( x = 0 \), then:

\[
f(0, y) = \frac{0 \cdot \sin(y)}{0 + y^2} = 0
\]

\[
\Rightarrow \lim_{(x, y) \to (0, 0)} f(x, y) = \lim_{y \to 0} f(0, y) = 0
\]

*Direction 3: Along \( y = x \)*
Now we approach the origin along a different line, say \( y = x \):

\[
f(x, x) = \frac{x \sin x}{x^2 + x^2} = \frac{x \sin x}{2x^2} = \frac{\sin x}{2x}
\]

\[
\Rightarrow \lim_{(x, y) \to (0, 0)} f(x, y) = \lim_{x \to 0} \frac{\sin x}{2x} = \frac{1}{2}
\]

Since this limit \(\frac{1}{2} \neq 0\), the two-dimensional limit  
\[
\lim_{(x, y) \to (0, 0)} f(x, y)
\]
**does not exist**, and hence \( f(x, y) \) is **not continuous** at the point \( (0, 0) \).

#### (b)
*Partial derivatives of \( f \) at point \( (0, 0) \)*
If we have a function of two variables  
\[
f : \mathbb{R}^2 \rightarrow \mathbb{R}, \quad (x, y) \mapsto f(x, y)
\]
then the partial derivative of \( f \) with respect to \( x \) at \( (a, b) \) is defined as:

\[
f_x(a, b) = \lim_{h \to 0} \frac{f(a + h, b) - f(a, b)}{h}
\]

and similarly, the partial derivative of \( f \) with respect to \( y \) at \( (a, b) \) is:

\[
f_y(a, b) = \lim_{h \to 0} \frac{f(a, b + h) - f(a, b)}{h}
\]

*Compute partial derivatives at \( (0, 0) \)*
-Partial derivative with respect to \( x \):
\[
f_x(0, 0) = \lim_{h \to 0} \frac{f(h, 0) - f(0, 0)}{h}
= \lim_{h \to 0} \frac{0 - 0}{h} = 0
\]

-Partial derivative with respect to \( y \):

\[
f_y(0, 0) = \lim_{h \to 0} \frac{f(0, h) - f(0, 0)}{h}
= \lim_{h \to 0} \frac{0 - 0}{h} = 0
\]

#### (c)
*At which points is \( f \) differentiable?*
To determine where the function \( f \) is differentiable, we use the following **theorem**:

*Remark (Theorem)*
If \( f \) is a continuous function in an open set \( U \),  
and has **continuous partial derivatives** at \( U \),  
then \( f \) is **continuously differentiable** at all points in \( U \).

Let \( U = \mathbb{R}^2 \setminus \{(0, 0)\} \).  
The function \( f(x, y) = \dfrac{x \sin y}{x^2 + y^2} \) is continuous at all points in \( U \).

Now we examine the partial derivatives of \( f \):

*Compute \( \dfrac{\partial f}{\partial x} \) and \( \dfrac{\partial f}{\partial y} \)*
\[
\frac{\partial}{\partial x} \left( \frac{x \sin y}{x^2 + y^2} \right)
= \frac{(x^2 + y^2)\sin y - 2x^2 \sin y}{(x^2 + y^2)^2}
= \frac{(y^2 - x^2)\sin y}{(x^2 + y^2)^2}
\]

\[
\frac{\partial}{\partial y} \left( \frac{x \sin y}{x^2 + y^2} \right)
= \frac{x \cos y (x^2 + y^2) - 2x y \sin y}{(x^2 + y^2)^2}
\]

These are rational functions where the **numerator and denominator** are composed of continuous functions, and the **denominator only vanishes at the origin** \( (0, 0) \).  
Thus, the partial derivatives are continuous **everywhere in \( U \)**.

*Conclusion*
So, based on the theorem, function \( f \) is **differentiable at all points except** the origin, that is, point \( (0, 0) \).


### 2.
#### (a)  
Let the function \( f(z) = \exp\left(-\dfrac{1}{2} z\right) \),  
where \( z = g(y) = y^\top S^{-1} y \),  
and \( y = h(x) = x - u \),  
with:

- \( x, u \in \mathbb{R}^D \)
- \( S \in \mathbb{R}^{D \times D} \)

*Chain Rule*
Based on the chain rule, we have:

\[
\frac{df}{dx} = \frac{df}{dz} \cdot \frac{dz}{dy} \cdot \frac{dy}{dx}
\]

*Step 1: Note the functions and their domains*
- \( y = h(x) = x - u \) → maps \( \mathbb{R}^D \to \mathbb{R}^D \)
- \( z = g(y) = y^\top S^{-1} y \) → maps \( \mathbb{R}^D \to \mathbb{R} \)
- \( f(z) = e^{- \frac{1}{2} z} \) → maps \( \mathbb{R} \to \mathbb{R} \)

So the full composition is:  
\[
x \mapsto y = x - u \mapsto z = y^\top S^{-1} y \mapsto f(z) = e^{- \frac{1}{2} z}
\]

*Step 2: Compute \( \dfrac{dy}{dx} \)*
Since \( y = x - u \), the Jacobian \( \dfrac{dy}{dx} \) is:

\[
\frac{dy}{dx} = I_{D \times D}
\quad \text{(identity matrix)}
\]

*Step 3: Compute \( \dfrac{dz}{dy} \)*
We have \( z = y^\top S^{-1} y \).  
Using gradient rules for quadratic forms:

\[
\frac{d}{dy} (y^\top A y) = y^\top (A + A^\top)
\]

Apply this:

\[
\frac{dz}{dy} = y^\top (S^{-1} + (S^{-1})^\top)
\quad \in \mathbb{R}^{1 \times D}
\]

*Step 4: Compute \( \dfrac{df}{dz} \)*
\[
f(z) = e^{- \frac{1}{2} z}
\quad \Rightarrow \quad
\frac{df}{dz} = -\frac{1}{2} e^{- \frac{1}{2} z}
\quad \in \mathbb{R}
\]

*Final Result*
\[
\frac{df}{dx} = -\frac{1}{2} e^{- \frac{1}{2} z} \cdot y^\top (S^{-1} + (S^{-1})^\top)
\quad \in \mathbb{R}^{1 \times D}
\]

#### (b)
Let

\[
f(z) = \tanh(z), \quad z = Ax + b
\]

where:

- \( x \in \mathbb{R}^N \)
- \( A \in \mathbb{R}^{M \times N} \)
- \( b \in \mathbb{R}^M \)

*Apply Chain Rule*
\[
\frac{df}{dx} = \frac{df}{dz} \cdot \frac{dz}{dx}
\]

*Step 1: Understand \( z = Ax + b \)*
We note:

\[
z = Ax + b \in \mathbb{R}^M
\Rightarrow \frac{dz}{dx} = A \in \mathbb{R}^{M \times N}
\]

*Step 2: Compute \( \dfrac{df}{dz} \)*
We have:

\[
f(z) =
\begin{bmatrix}
\tanh(z_1) \\
\tanh(z_2) \\
\vdots \\
\tanh(z_M)
\end{bmatrix}
\]

So the Jacobian of \( f \) is diagonal:

\[
\frac{df}{dz} =
\begin{bmatrix}
\text{sech}^2(z_1) & & \\
& \ddots & \\
& & \text{sech}^2(z_M)
\end{bmatrix}
\in \mathbb{R}^{M \times M}
\]

*Final Result*
```math
\frac{df}{dx}
= 
\operatorname{diag}\left(
  \text{sech}^2(z_1),\ 
  \text{sech}^2(z_2),\ 
  \dots,\ 
  \text{sech}^2(z_M)
\right)
\cdot A
\quad \in \mathbb{R}^{M \times N}
```

### 3.
#### (a)
Let  
```math
x^{(0)} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \quad \eta = 1
```  
and perform two steps of **gradient descent**.

The update rule for gradient descent is:

```math
x^{(i+1)} = x^{(i)} - \eta \nabla f(x^{(i)})
```

So two steps of the gradient descent algorithm are:

```math
\text{Step 1:} \quad x^{(1)} = x^{(0)} - \eta \nabla f(x^{(0)})  
```

```math
\text{Step 2:} \quad x^{(2)} = x^{(1)} - \eta \nabla f(x^{(1)})
```

Given the gradient:

```math
\nabla f = \begin{bmatrix}
x_1 + 2 \\
2x_2 + 1
\end{bmatrix}
```

We compute:

```math
x^{(0)} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
\quad \Rightarrow \quad
\nabla f(x^{(0)}) = \begin{bmatrix} 2 \\ 1 \end{bmatrix}
```

*Step 1:*
```math
x^{(1)} = x^{(0)} - 1 \cdot \nabla f(x^{(0)}) = 
\begin{bmatrix} 0 \\ 0 \end{bmatrix}
- \begin{bmatrix} 2 \\ 1 \end{bmatrix}
= \begin{bmatrix} -2 \\ -1 \end{bmatrix}
```

```math
\nabla f(x^{(1)}) = 
\begin{bmatrix} -2 + 2 \\ -2 + 1 \end{bmatrix}
= \begin{bmatrix} 0 \\ -1 \end{bmatrix}
```

*Step 2:*
```math
x^{(2)} = x^{(1)} - 1 \cdot \nabla f(x^{(1)}) =
\begin{bmatrix} -2 \\ -1 \end{bmatrix}
- \begin{bmatrix} 0 \\ -1 \end{bmatrix}
= \begin{bmatrix} -2 \\ 0 \end{bmatrix}
```

#### (b)
Will the gradient descent procedure from part (b) converge to the minimizer \( x^* \)? Why or why not? How can we fix it?

Let’s look at the values over iterations:

```math
x^{(0)} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \quad
x^{(1)} = \begin{bmatrix} -2 \\ -1 \end{bmatrix}, \quad
x^{(2)} = \begin{bmatrix} -2 \\ 0 \end{bmatrix}, \quad
x^* = \begin{bmatrix} -2 \\ -0.5 \end{bmatrix}
```

And:

```math
\nabla f(x^{(0)}) = \begin{bmatrix} 2 \\ 1 \end{bmatrix}, \quad
\nabla f(x^{(1)}) = \begin{bmatrix} 0 \\ -1 \end{bmatrix}, \quad
\nabla f(x^{(2)}) = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad
\nabla f(x^*) = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
```

We observe that **gradient descent does not converge** to \( x^* \). Why?

Because the gradients do **not decrease constantly**.  
Let’s examine the **partial derivatives**:

```math
\frac{\partial f}{\partial x_1} \big|_{x^{(0)}} = 2, \quad
\frac{\partial f}{\partial x_1} \big|_{x^{(1)}} = 0, \quad
\frac{\partial f}{\partial x_1} \big|_{x^{(2)}} = 0
```

```math
\frac{\partial f}{\partial x_2} \big|_{x^{(0)}} = 1, \quad
\frac{\partial f}{\partial x_2} \big|_{x^{(1)}} = -1, \quad
\frac{\partial f}{\partial x_2} \big|_{x^{(2)}} = 1
```

Since \( x^* \) is a minimum and \( \nabla f(x^*) = 0 \), we expect the GD algorithm to converge to \( x^* \) **if the partial derivatives reduce toward zero**.

But here, GD **jumps over the minimum** due to a **too high learning rate** \( \eta = 1 \). If we **decrease** the learning rate, convergence improves.

*Trying smaller learning rates:*
Let’s try \( \eta = 0.5 \):

```math
x^{(0)} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \quad 
\nabla f(x^{(0)}) = \begin{bmatrix} 2 \\ 1 \end{bmatrix}
```

*Step 1:*
```math
x^{(1)} = x^{(0)} - 0.5 \cdot \nabla f(x^{(0)}) 
= \begin{bmatrix} 0 \\ 0 \end{bmatrix}
- 0.5 \cdot \begin{bmatrix} 2 \\ 1 \end{bmatrix}
= \begin{bmatrix} -1 \\ -0.5 \end{bmatrix}
```

```math
\nabla f(x^{(1)}) = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
```

*Step 2:*
```math
x^{(2)} = x^{(1)} - 0.5 \cdot \nabla f(x^{(1)}) 
= \begin{bmatrix} -1 \\ -0.5 \end{bmatrix}
- 0.5 \cdot \begin{bmatrix} 1 \\ 0 \end{bmatrix}
= \begin{bmatrix} -1.5 \\ -0.5 \end{bmatrix}
```

Now we see that the GD algorithm converges towards:

```math
x^* = \begin{bmatrix} -2 \\ -0.5 \end{bmatrix}
```

with gradients:

```math
\nabla f(x^{(0)}) = \begin{bmatrix} 2 \\ 1 \end{bmatrix}, \quad
\nabla f(x^{(1)}) = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad
\nabla f(x^{(2)}) = \begin{bmatrix} 0.5 \\ 0 \end{bmatrix}, \quad
\nabla f(x^*) = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
```

✔️ So a smaller \( \eta \) leads to proper convergence!
