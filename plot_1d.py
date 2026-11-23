import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from taylor import taylor_polynomial


def plot_taylor_1d(function, variable, center, degrees, x_min, x_max):
    """Plot a function, its Taylor approximations, and their errors."""

    # Generate 400 x-values between x_min and x_max.
    x_values = np.linspace(x_min, x_max, 400)

    # lambdify turns a SymPy expression into a function that works
    # with NumPy arrays.
    original_function = sp.lambdify(variable, function, "numpy")
    actual_values = original_function(x_values)

    figure, (function_ax, error_ax) = plt.subplots(
        1, 2, figsize=(12, 5)
    )

    function_ax.plot(
        x_values, actual_values,
        color="black", linewidth=2, label="Original function"
    )

    for degree in degrees:
        polynomial = taylor_polynomial(
            function, (variable,), (center,), degree
        )

        numerical_polynomial = sp.lambdify(
            variable, polynomial, "numpy"
        )

        # A degree-0 polynomial may be a constant. Broadcasting makes
        # its output the same shape as x_values.
        approximate_values = np.broadcast_to(
            numerical_polynomial(x_values), x_values.shape
        )

        absolute_error = np.abs(actual_values - approximate_values)

        function_ax.plot(
            x_values, approximate_values,
            label=f"Degree {degree}"
        )

        error_ax.plot(
            x_values, absolute_error,
            label=f"Degree {degree}"
        )

    function_ax.axvline(
        center, color="gray", linestyle=":", label="Center"
    )
    function_ax.set_title("Function and Taylor approximations")
    function_ax.set_xlabel(str(variable))
    function_ax.set_ylabel("Value")
    function_ax.legend()
    function_ax.grid(alpha=0.3)

    error_ax.axvline(center, color="gray", linestyle=":")
    error_ax.set_title("Absolute error")
    error_ax.set_xlabel(str(variable))
    error_ax.set_ylabel("|Actual − approximation|")
    error_ax.legend()
    error_ax.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig("sine_taylor_plot.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    x = sp.symbols("x")

    plot_taylor_1d(
        function=sp.sin(x),
        variable=x,
        center=0,
        degrees=(1, 3, 5),
        x_min=-3,
        x_max=3,
    )