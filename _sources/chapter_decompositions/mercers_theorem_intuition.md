# Mercer’s Theorem Intuition

### 1. **Numerical sanity check** (live coding)

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random 2-D points
np.random.seed(42)
n = 60
X = np.random.randn(n, 2)

gamma = 0.5

# Compute squared Euclidean distance matrix
sq_dists = np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)

# RBF kernel (Gram matrix)
K = np.exp(-gamma * sq_dists)

# Eigenvalues of the Gram matrix
eigvals = np.linalg.eigvalsh(K)  # already sorted ascending
eigvals_desc = eigvals[::-1]

print("First 10 eigenvalues (descending):")
print(eigvals_desc[:10])
print(f"\nSmallest eigenvalue: {eigvals[0]:.6f}")

# Plot the eigenvalue spectrum
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, n + 1), eigvals_desc, marker='o')
plt.yscale('log')
plt.xlabel('Eigenvalue index (descending)')
plt.ylabel('Eigenvalue magnitude (log scale)')
plt.title('Eigenvalue spectrum of RBF Gram matrix')
plt.tight_layout()
plt.show()
```

Run the short notebook cell above with your own γ or data.
Students immediately **see** that every eigenvalue of the Gram matrix is ≥ 0 and how the spectrum decays.
*Talking points*:

* Why `eigvalsh` (symmetric) never shows negative values for a Mercer kernel.
* How the spectrum’s tail hints at *effective* dimensionality of the RKHS.
* What changes as you slide γ (broad vs. narrow kernels).

Feel free to add a slider for γ so they can watch eigenvalues move in real time.

---

### 2. **From pixels to features** (finite vs. infinite)

1. Draw two 1-D signals on the board (e.g. two little “mountains”).
2. Show how a *finite* polynomial kernel maps them into a handful of monomials (draw a short arrow diagram).
3. Contrast with the RBF: the “feature arrow” explodes into a *continuum* of Gaussians centered everywhere.
4. Now reveal the Mercer expansion formula

$$
k(x,x')=\sum_{i=1}^\infty \lambda_i \,\phi_i(x)\phi_i(x')
$$

and ask: *“Where are those Gaussians hiding in this series?”*
This opens the door to Bochner’s theorem and random Fourier features if you like.

---

### 3. **Induction proof as a Lego tower**

Print the block-matrix form

$$
K_{n+1}=
\begin{bmatrix}
K_n & \mathbf k\\
\mathbf k^\top & 1
\end{bmatrix}
$$

on a slide, but bring **LEGO® bricks** (or colored sticky notes):

* top-left brick = old PSD block,
* right column / bottom row = new similarities,
* 1 × 1 brick = self-similarity.
  Stack bricks as you add points; the quadratic-form ≥0 argument becomes “no matter how you press on the tower, it won’t collapse.”

---

### 4. **Kernel PCA demo**

* Take 2-D moons data.
* Run kernel PCA with RBF γ you choose.
* Plot first two kernel-PCA components; students see the nonlinear *unfolding*.
  Then point to the eigen-decomposition $K = V\Lambda V^\top$ and say: *“Those eigenvectors are the discrete cousins of Mercer’s eigenfunctions.”*

---

### 5. **Mercer vs. non-Mercer “jump scare”**

Show the cosine kernel $k(x,x')=\cos(x-x')$ on $\mathbb R$.
Sample 30 random points; compute its Gram matrix and eigenvalues — you get negatives!
Flip to RBF: no negatives.
Students grasp *why* the compact-operator assumption (or Bochner’s positive spectral density) matters.

---

### 6. **Fourier-space view (Bochner’s theorem)**

Derive quickly that the RBF’s Fourier transform is another Gaussian:

$$
k(x,x') = \int_{\mathbb R^d} e^{i\omega^\top(x-x')} e^{-\|\omega\|^2/4\gamma}\,d\omega .
$$

Hand-wave: “positive weight everywhere ⇒ PSD.”
Then draw random Fourier features:

$$
\varphi_\omega(x)=\sqrt{2}\cos(\omega^\top x+b),\quad
\omega\sim\mathcal N(0,2\gamma I),\;b\sim\mathcal U(0,2\pi)
$$

and let students approximate the kernel with finitely many such features to *see* convergence.

---

### 7. **Physical analogy**

Compare Mercer’s expansion to a vibrating drum:

* Drumhead ↔ compact operator.
* Normal modes ↔ eigenfunctions $\phi_i$.
* Energy in each mode ↔ eigenvalue $\lambda_i$.
  Just as any strike excites a combination of modes, every data pair’s similarity is a weighted sum of modes.

---

## Cheat-sheet to hand out

| Concept               | 1-Sentence Student Takeaway                                                                                                  |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Mercer kernel**     | Any continuous symmetric PSD kernel on a compact set behaves like an *infinite Gram matrix* with an orthonormal eigen-basis. |
| **Integral operator** | “Multiply-then-integrate” turns your kernel into a self-adjoint *matrix on functions*.                                       |
| **Eigenfunctions**    | The kernel is literally its own feature map: $k=\sum \lambda_i\phi_i\otimes\phi_i$.                                          |
| **RBF PSD**           | Because its Fourier spectrum is non-negative everywhere.                                                                     |
| **Gram-matrix test**  | Sample points → compute $K$ → *all* eigenvalues ≥ 0 ⇔ kernel is Mercer.                                                      |

---

### Where to go next

* **Random Fourier Features** (Rahimi & Recht, 2007) — shows you can *sample* Mercer’s infinite series.
* **Nyström method** — connects kernel sub-sampling to integral-operator approximation.
* **Deep-kernel GP view** — Mercerian story continues in Gaussian-process land.

Use whichever of these slices matches your class’s background and time budget.
Pair two or three and you’ll give students both the rigorous spine **and** the visceral “aha!” Mercer moment.
