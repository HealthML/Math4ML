# Exercise Sheet 1 Solutions


## 1.
### (a)
Take any $v_1=(a,b)$ and $v_2=(c,d)$ in $V$; then $b=3a+1$ and $d=3c+1$.  
Their sum is  

$$
v_1+v_2=(a+c,\;b+d)=(a+c,\;3a+1+3c+1)=\bigl(a+c,\;3(a+c)+2\bigr),
$$  
which **does not** satisfy $b+d=3(a+c)+1$. Hence $V$ is *not* closed under addition ⇒ **not a vector space**.  
(Equivalently, the additive identity $(0,0)\notin V$, violating axiom V1.)

### (b)
All axioms except **distributivity over scalar addition** fail:

Take $v=(a,b)$ and scalars $\alpha,\beta\in\mathbb R$.

$$
(\alpha+\beta)\,v=((\alpha+\beta)a,\;b),
\quad
\alpha v+\beta v=(\alpha a,\;b)+(\beta a,\;b)=((\alpha+\beta)a,\;2b).
$$
Unless $b=0$, the second component differs, so  
$(\alpha+\beta)v\neq\alpha v+\beta v$.  
Therefore $V$ is **not** a vector space.


## 2.
### (a)
*Zero vector:* $(0,0)$ satisfies $0=2\cdot0$.  
*Closure (addition):* if $y_1=2x_1$ and $y_2=2x_2$, then

$$
y_1+y_2 = 2(x_1+x_2).
$$
*Closure (scalar mult.):* for $\alpha\in\mathbb R$,

$$
\alpha(x,y)=(\alpha x,\;2\alpha x).
$$
All three conditions hold ⇒ $W$ **is a subspace**.

### (b)
Pick $(x,y)\in W$ with $x>0$ and any negative scalar $\alpha<0$.  
Then

$$
\alpha(x,y)=(\alpha x,\;\alpha y),
$$
and $\alpha x<0$. Thus $\alpha(x,y)\notin W$.  
Not closed under scalar multiplication ⇒ **not a subspace**.


## 3.
For $x=(a,b)$, $y=(c,d)$ and scalars $\alpha,\beta$:

$$
T(\alpha x+\beta y)=\bigl((\alpha a+\beta c)^{2},\;\alpha b+\beta d\bigr),
$$

$$
\alpha T(x)+\beta T(y)=\bigl(\alpha^{2}a^{2}+\beta^{2}c^{2},\;\alpha b+\beta d\bigr).
$$
The first components differ unless $a c=0$ or $\alpha\beta=0$.  
Hence $T$ **violates additivity/homogeneity ⇒ not linear**.
