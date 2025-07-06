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
#  Building an L-BFGS Optimizer

## 🌟 Motivation

Optimization is the heart of machine learning. Every time you fit a neural network, train a regression model, or perform Bayesian inference, you rely on powerful numerical optimization methods to find solutions efficiently.

* **Gradient Descent (GD)** is intuitive but can be painfully slow, especially when the objective has tricky curvature.
* **Newton’s method** is powerful but computationally expensive, requiring the full Hessian matrix (second derivatives).
* **BFGS (Broyden–Fletcher–Goldfarb–Shanno)** methods cleverly approximate the Hessian, speeding convergence dramatically—but they're memory-intensive.

**Limited-memory BFGS (L-BFGS)** elegantly balances these trade-offs:

* It approximates curvature with low memory overhead, storing only a small set of vectors.
* It offers robust and fast convergence for a wide variety of optimization tasks.
* Widely used in practice, it's the backbone of many industrial and academic optimization routines.

By building your own L-BFGS optimizer, you'll deepen your understanding of optimization, numerical stability, and algorithmic efficiency—essential skills for any machine learning practitioner.

---

## 📌 What You Will Do in This Project

Your clear goal is:

> **Implement a limited-memory BFGS (L-BFGS) optimizer from scratch using NumPy**, and test it on classical benchmark problems such as the Rosenbrock function and a simple neural-network regression loss.

Specifically, you'll:

* Implement the **two-loop recursion** algorithm at the core of L-BFGS.
* Integrate a robust **line search method** satisfying Wolfe conditions.
* Compare your custom NumPy implementation with SciPy’s built-in optimizer for performance and convergence behavior.

---

## 🔍 Key Concepts You'll Master

This project introduces you to crucial optimization ideas:

* **Two-loop recursion**:
  Efficiently approximating Hessian information using a small set of stored vectors $(s_k, y_k)$.

* **Line search (Wolfe conditions)**:
  Ensuring stable convergence by selecting suitable step-sizes along the optimization direction.

* **Curvature approximation**:
  Understand clearly how limited-memory methods differ from full BFGS and why they dramatically reduce computational requirements.

---

## 🚧 Core Tasks (Implementation Details)

You'll concretely engage with:

* **Two-loop recursion coding**:
  Clearly implement this key recursion from scratch, storing and managing pairs $(s_k, y_k)$ efficiently.

* **Line-search implementation**:
  Implement a robust Wolfe-condition line-search method (or adapt a provided stub) to ensure stable convergence.

* **Benchmark testing**:
  Test your implementation on:

  * The Rosenbrock function, a classical nonlinear optimization benchmark.
  * A small neural-network regression loss (e.g., a 1-layer MLP with few parameters).

* **Comparison and analysis**:
  Compare your implementation's performance against SciPy's optimized implementation clearly in terms of speed, convergence, and iterations.

---

## 📝 Reporting: Analysis and Insights

In your concise (\~2 pages) report, clearly discuss:

* **Curvature approximation**:
  How well does L-BFGS approximate curvature compared to full BFGS and Gradient Descent (GD)? Illustrate with convergence plots.

* **Speed and iterations**:
  Clearly summarize how your NumPy implementation compares in terms of convergence rate and computational efficiency with SciPy’s version.

* **Insightful observations**:
  Highlight what you've learned about optimization methods, curvature approximation, and practical numerical stability.

---

## 🚀 Stretch Goals (Optional, Advanced)

For further insight, you might:

* Extend your implementation to include **bound-constraints (L-BFGS-B)**, enforcing optimization within predefined intervals, e.g., $[-2, 2]^d$.
* Experiment with different memory sizes, observing how more (or fewer) $(s_k, y_k)$ pairs affect convergence behavior and efficiency.

---

## 📚 Resources and Support

* Starter notebook and visualisation code for Rosenbrock and neural network regression problems will be provided.
* Feel free to utilize external resources, lecture notes, and clearly documented AI-assisted tools.

---

## ✅ Why This Matters

Implementing L-BFGS will elevate your optimization skills significantly:

* **Highly relevant**: Optimization algorithms are central to virtually every machine learning task you'll encounter.
* **Numerical mastery**: By coding from scratch, you'll develop a robust intuition and understanding of the numerical methods used across machine learning, deep learning, and scientific computing.
* **Valuable portfolio item**: A self-built optimization library or algorithm strongly showcases your coding proficiency, mathematical insight, and practical problem-solving skills.

Let’s build and explore your very own L-BFGS optimizer!


## Project Summary: Building an L-BFGS Optimizer

*(NumPy)*

| Item           | Details                                                                                               |
| -------------- | ----------------------------------------------------------------------------------------------------- |
| **Goal**       | Implement limited-memory BFGS and test on Rosenbrock & a small NN regression loss.                    |
| **Key ideas**  | Two-loop recursion, line search (Wolfe), storing $s_k, y_k$ pairs.                                    |
| **Core tasks** | <ul><li>Coding pure NumPy L-BFGS.</li><li>Compare to scipy.optimize for speed / iterations.</li></ul> |
| **Report**     | Discuss curvature approximation vs. full BFGS and GD.                                                 |
| **Stretch**    | Add bound-constraints (L-BFGS-B) for $[-2,2]^d$ box.                                                  |

---