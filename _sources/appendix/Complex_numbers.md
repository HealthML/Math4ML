---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Complex numbers in one page 🌌

You can think of a complex number

$$
z \;=\; a + b\,i
$$

as a **point or arrow** in a 2-D plane:

| Coordinate view                     | Meaning                                                                     |
| ----------------------------------- | --------------------------------------------------------------------------- |
| **Cartesian** (rectangular) $(a,b)$ | *a* = distance along the real axis, *b* = distance along the imaginary axis |
| **Polar** $(r,\theta)$              | length $r = \sqrt{a^{2}+b^{2}}$ and angle $\theta = \arctan2(b,a)$          |

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Plot the unit circle and a few complex exponentials
theta = np.linspace(0, 2*np.pi, 400)
unit_x = np.cos(theta)
unit_y = np.sin(theta)

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(unit_x, unit_y)
ax.set_aspect('equal')
ax.set_xlabel('Real axis')
ax.set_ylabel('Imag axis')
ax.set_title('Unit circle: $e^{i\\theta}$')

# Mark three example angles
angles = [0, np.pi/3, 2*np.pi/3]
for a in angles:
    ax.plot([0, np.cos(a)], [0, np.sin(a)], marker='o')
    ax.text(1.05*np.cos(a), 1.05*np.sin(a), f"$e^{{i{a:.2f}}}$")

ax.grid(True)
plt.tight_layout()
plt.show()
```

The little plot above shows a *unit circle*: all complex numbers with $r=1$.
Three arrows illustrate $e^{i\theta}$ for different angles.

---

#### Why the exponential?

Euler’s famous identity links trig functions to exponentials:

$$
e^{i\theta} = \cos\theta + i\sin\theta .
$$

So a **rotating complex exponential** is exactly the same as a **sine & cosine pair**—that’s why the FFT uses $e^{-2\pi i k n/N}$: it’s a compact way to test “how much of a $k$-cycle sinusoid lives in the data.”

---

#### Key operations (no scary algebra)

| Operation                                      | Cartoon version                                      |       |                     |
| ---------------------------------------------- | ---------------------------------------------------- | ----- | ------------------- |
| **Magnitude** (                                | z                                                    | = r ) | length of the arrow |
| **Phase** $\arg z = \theta$                    | which direction the arrow points                     |       |                     |
| **Complex conjugate** $\overline{z} = a - b i$ | mirror across the real axis (flip sign of imag part) |       |                     |
| **Multiplication** $z_1 z_2$                   | multiply lengths, **add** angles (rotations add!)    |       |                     |

That last property is the magic behind **convolution theorem**: in the frequency domain we just *multiply* complex numbers, and that adds their phases—equivalent to convolving signals in the time domain.

---

#### How FFT outputs relate to things you care about

* **Magnitude spectrum** $|\widehat{x}[k]|$ → “strength” of that frequency.
* **Phase spectrum** $\arg\widehat{x}[k]$ → how the sinusoid is *shifted* in time.
* For **real-valued signals** you only get unique information up to $N/2$ (the rest is the complex conjugate mirror).

---

### Small takeaway

1. **Complex numbers = 2-D vectors.**
2. **Euler’s $e^{i\theta}$** wraps cos & sin into one neat term.
3. FFT simply decomposes your signal into these rotating arrows; you look at their **lengths** (magnitudes) and **angles** (phases).
4. Spectral neural layers just *learn to scale & rotate* these arrows per frequency—much cheaper than learning giant spatial kernels.
