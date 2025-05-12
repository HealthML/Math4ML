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
# Determinant

The **determinant** of a square matrix can be defined in several
different ways.

Let's illustrate the determinant geometrically.
The determinant can be considered a factor on the change of volume of a unit square after transformation.

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Define matrices to show area effects
matrices = {
    "Area = 1 (Identity)": np.array([[1, 0], [0, 1]]),
    "Area > 1": np.array([[2, 0.5], [0.5, 1.5]]),
    "Area < 0 (Flip)": np.array([[0, 1], [1, 0]]),
    "Rotation (Area = 1)": np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                     [np.sin(np.pi/4),  np.cos(np.pi/4)]])
}

# Unit square
square = np.array([[0, 0],
                   [1, 0],
                   [1, 1],
                   [0, 1],
                   [0, 0]]).T

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, (title, M) in zip(axes, matrices.items()):
    transformed_square = M @ square
    area = np.abs(np.linalg.det(M))
    det = np.linalg.det(M)

    # Plot original unit square
    ax.plot(square[0], square[1], 'k--', label='Unit square')
    ax.fill(square[0], square[1], facecolor='lightgray', alpha=0.4)

    # Plot transformed shape
    ax.plot(transformed_square[0], transformed_square[1], 'b-', label='Transformed')
    ax.fill(transformed_square[0], transformed_square[1], facecolor='skyblue', alpha=0.6)

    # Add vector arrows for columns of M
    origin = np.array([[0, 0]]).T
    for i in range(2):
        vec = M[:, i]
        ax.quiver(*origin, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color='red')

    ax.set_title(f"{title}\nDet = {det:.2f}, Area = {area:.2f}")
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()

plt.suptitle("Geometric Interpretation of the Determinant (Area Scaling and Orientation)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
```

1. **Identity**: No change — area = 1.
2. **Stretch**: Expands area — determinant > 1.
3. **Flip**: Reflects across the diagonal — determinant < 0.
4. **Rotation**: Rotates without distortion — determinant = 1.

---

The determinant has several important properties:

(i) $\det(\mathbf{I}) = 1$

(ii) $\det(\mathbf{A}^{\!\top\!}) = \det(\mathbf{A})$

(iii) $\det(\mathbf{A}\mathbf{B}) = \det(\mathbf{A})\det(\mathbf{B})$

(iv) $\det(\mathbf{A}^{-1}) = \det(\mathbf{A})^{-1}$

(v) $\det(\alpha\mathbf{A}) = \alpha^n \det(\mathbf{A})$

---


