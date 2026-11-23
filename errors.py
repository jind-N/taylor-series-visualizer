from math import factorial

import numpy as np
import pandas as pd
import sympy as sp

from taylor import taylor_polynomial


def lagrange_bound_1d(function, variable, center_val, point_val, degree):
    """
    Theoretical maximum error for 1-variable Taylor approximation:
        |R_n(x)| <= (M / (n + 1)!) * |x - a|^(n + 1)
    where M is the maximum of |f^(n+1)(t)| between center 'a' and point 'x'.
    """
    next_derivative = sp.diff(function, variable, degree + 1)
    numerical_deriv = sp.lambdify(variable, next_derivative, "numpy")

    a = float(center_val)
    x = float(point_val)
    low, high = min(a, x), max(a, x)
    sample_t = np.linspace(low, high, 400)

    try:
        with np.errstate(all="ignore"):
            vals = np.broadcast_to(
                np.asarray(numerical_deriv(sample_t), dtype=float), sample_t.shape
            )
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return float("nan")
        max_deriv = float(np.max(np.abs(finite)))
    except Exception:
        return float("nan")

    distance = abs(x - a)
    return (max_deriv / factorial(degree + 1)) * (distance ** (degree + 1))


def compare_at_point(function, variables, center, point, max_degree):
    """Compare Taylor approximations of degrees 0..max_degree at one point."""
    if len(variables) != len(point):
        raise ValueError("Provide one point coordinate per variable.")

    substitutions = dict(zip(variables, point))
    actual_expr = function.subs(substitutions, simultaneous=True).evalf()

    if actual_expr.is_finite is not True or actual_expr.is_real is not True:
        raise ValueError("The function is undefined or non-real at this point.")

    actual = float(actual_expr)
    rows = []

    for degree in range(max_degree + 1):
        polynomial = taylor_polynomial(function, variables, center, degree)
        approx_expr = polynomial.subs(substitutions, simultaneous=True).evalf()
        approx = float(approx_expr)
        abs_error = abs(actual - approx)

        row = {
            "degree": degree,
            "actual": actual,
            "approximation": approx,
            "absolute_error": abs_error,
        }

        # Add theoretical upper bound for 1D functions
        if len(variables) == 1:
            row["lagrange_bound"] = lagrange_bound_1d(
                function, variables[0], center[0], point[0], degree
            )

        rows.append(row)

    return pd.DataFrame(rows)