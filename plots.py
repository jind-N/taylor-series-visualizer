import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def evaluate(expression, variables, coordinates):
    """Evaluate a SymPy expression on NumPy grids.

    Undefined results (log of negatives, division by zero) become NaN,
    so Matplotlib leaves a gap instead of crashing.
    """
    numerical = sp.lambdify(variables, expression, "numpy")

    with np.errstate(all="ignore"):
        values = numerical(*coordinates)

    shape = np.broadcast_shapes(*(np.shape(c) for c in coordinates))
    values = np.broadcast_to(np.asarray(values, dtype=float), shape).copy()

    # Replace +inf / -inf with NaN (treated as "no value")
    values[~np.isfinite(values)] = np.nan
    return values


def safe_limits(values, headroom=0.3):
    """Axis limits that are ALWAYS valid numbers.

    Ignores NaN/inf, avoids overflow, and falls back to (-1, 1).
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0

    # Ignore extreme outliers (e.g. near a vertical asymptote)
    low, high = np.percentile(finite, [1, 99])
    low, high = float(low), float(high)

    # Keep values in a range Matplotlib can always handle
    limit = 1e12
    low = max(min(low, limit), -limit)
    high = max(min(high, limit), -limit)

    margin = headroom * (high - low) if high > low else 1.0
    low, high = low - margin, high + margin

    if not (np.isfinite(low) and np.isfinite(high)) or low >= high:
        return -1.0, 1.0
    return low, high


def plot_1d(function, polynomials, variable, center, x_min, x_max):
    """Line plots for a one-variable function.

    polynomials: dictionary {degree: polynomial}
    """
    x_values = np.linspace(x_min, x_max, 400)
    actual = evaluate(function, (variable,), (x_values,))

    figure, (ax_function, ax_error) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_function.plot(x_values, actual, color="black", linewidth=2, label="Original")

    for degree, polynomial in polynomials.items():
        approximation = evaluate(polynomial, (variable,), (x_values,))
        error = np.abs(actual - approximation)
        ax_function.plot(x_values, approximation, "--", label=f"Degree {degree}")
        ax_error.plot(x_values, error, label=f"Degree {degree}")

    for ax in (ax_function, ax_error):
        ax.axvline(center, color="gray", linestyle=":", label="Center")
        ax.set_xlabel(str(variable))
        ax.grid(alpha=0.3)
        ax.legend()

    ax_function.set_title("Function vs Taylor approximations")
    ax_error.set_title("Absolute error |f − P|")

    # Zoom on the real function, ignoring undefined points
    ax_function.set_ylim(*safe_limits(actual, headroom=1.0))

    figure.tight_layout()
    return figure


def plot_2d(function, polynomial, variables, degree, ranges, fixed_values):
    """Heatmaps for functions of 2 or more variables.

    The first two variables are the plot axes.
    Any remaining variables are held at fixed_values (a 2D slice).
    """
    (x_min, x_max), (y_min, y_max) = ranges

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 120),
        np.linspace(y_min, y_max, 120),
    )
    coordinates = (grid_x, grid_y, *fixed_values)

    actual = evaluate(function, variables, coordinates)
    approximation = evaluate(polynomial, variables, coordinates)
    error = np.abs(actual - approximation)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    panels = [
        (actual, "Original function", "viridis"),
        (approximation, f"Taylor degree {degree}", "viridis"),
        (error, "Absolute error |f − P|", "magma"),
    ]

    for ax, (values, title, cmap) in zip(axes, panels):
        low, high = safe_limits(values, headroom=0.0)
        image = ax.imshow(
            values,
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            cmap=cmap,
            aspect="auto",
            vmin=low,
            vmax=high,
        )
        figure.colorbar(image, ax=ax, shrink=0.85)
        ax.set_title(title)
        ax.set_xlabel(str(variables[0]))
        ax.set_ylabel(str(variables[1]))

    figure.tight_layout()
    return figure


def plot_surface(function, polynomial, variables, degree, ranges, fixed_values):
    """3D surface plots: original, Taylor approximation, and error."""
    (x_min, x_max), (y_min, y_max) = ranges

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 60),
        np.linspace(y_min, y_max, 60),
    )
    coordinates = (grid_x, grid_y, *fixed_values)

    actual = evaluate(function, variables, coordinates)
    approximation = evaluate(polynomial, variables, coordinates)
    error = np.abs(actual - approximation)

    z_low, z_high = safe_limits(actual)

    # Hide mesh parts that shoot outside the 3D box
    clipped_approx = np.where(
        (approximation >= z_low) & (approximation <= z_high),
        approximation,
        np.nan,
    )

    figure = plt.figure(figsize=(15, 5.5))

    ax = figure.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(grid_x, grid_y, actual, cmap="viridis", alpha=0.8)
    ax.plot_wireframe(
        grid_x, grid_y, clipped_approx,
        color="crimson", linewidth=0.8, rstride=3, cstride=3,
    )
    ax.set_zlim(z_low, z_high)
    ax.set_title(f"Original (surface) vs Taylor degree {degree} (red mesh)")
    ax.set_xlabel(str(variables[0]))
    ax.set_ylabel(str(variables[1]))
    ax.set_zlabel("value")

    ax_error = figure.add_subplot(1, 2, 2, projection="3d")
    ax_error.plot_surface(grid_x, grid_y, error, cmap="magma")
    ax_error.set_zlim(*safe_limits(error, headroom=0.1))
    ax_error.set_title("Absolute error |f − P|")
    ax_error.set_xlabel(str(variables[0]))
    ax_error.set_ylabel(str(variables[1]))
    ax_error.set_zlabel("error")

    figure.tight_layout()
    return figure