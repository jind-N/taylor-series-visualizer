# Interactive Taylor Series Approximations in Python

An interactive mathematical and computational framework interfacing **Calculus** with **Python** to analyze and visualize Taylor series approximations for single-variable and multivariable functions ($n$-dimensional).

Developed for the **CalSymPy Project (Topic 11)**.

---

## 📌 Project Overview
Taylor polynomials serve as foundational approximation tools across computational physics, engineering, and machine learning. This tool enables users to:
- Compute exact symbolic derivatives and total-degree Taylor polynomials using **SymPy**.
- Analyze 1D, 2D, and $N$-dimensional functions (with 2D/3D slice visualizations).
- Dynamically adjust the expansion center $a$ and degree $n$.
- Quantify error distributions numerically and compare them against theoretical **Lagrange Error Bounds**.
- Study real-world physical models (e.g., 2D steady-state thermal distribution, pendulum linearization).

---

## 🧮 Mathematical Formulation

### 1D Taylor Polynomial & Remainder
For an infinitely differentiable function $f(x)$ centered at $x = a$:
$$P_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!} (x - a)^k$$

The approximation error is bounded by the **Lagrange Remainder Theorem**:
$$|R_n(x)| \le \frac{M}{(n+1)!} |x - a|^{n+1}, \quad M = \max_{\xi \in [a, x]} |f^{(n+1)}(\xi)|$$

### Multivariable Total-Degree Taylor Expansion (2D)
For a scalar field $f(x, y)$ centered at $(a, b)$ up to total degree $n$:
$$P_n(x, y) = \sum_{i+j \le n} \frac{1}{i!\, j!} \frac{\partial^{i+j} f}{\partial x^i \partial y^j}(a, b) (x - a)^i (y - b)^j$$

---

## 🚀 Features
- **Intelligent Variable Detection**: Automatically extracts and sorts symbolic variables from user input.
- **Robust Domain Validation**: Prevents singularities, complex derivatives, and division-by-zero crashes using NaN-masking and automatic fallback recovery.
- **Interactive Web UI**: Built with Streamlit, supporting 1D line comparisons, 2D heatmaps, and 3D wireframe surface plots.
- **Automated Verification**: Integrates unit tests comparing custom symbolic outputs with SymPy's reference `series()` engine.

---

## 💻 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jind-N/taylor-series-visualizer.git
   cd taylor-series-visualizer
