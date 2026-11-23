from itertools import product
from math import factorial

import numpy as np
import sympy as sp


def taylor_polynomial(function, variables, center, degree):
    """Return the Taylor polynomial up to the given total degree."""
    variables = tuple(variables)
    center = tuple(center)

    if len(variables) == 0:
        raise ValueError("Provide at least one variable.")

    if len(variables) != len(center):
        raise ValueError("Each variable needs a center value.")

    if degree < 0:
        raise ValueError("Degree cannot be negative.")

    polynomial = 0

    # Combinations of derivative orders, e.g. (1, 2) -> once in x, twice in y
    for orders in product(range(degree + 1), repeat=len(variables)):
        if sum(orders) > degree:
            continue

        derivative = function
        for variable, order in zip(variables, orders):
            derivative = sp.diff(derivative, variable, order)

        value_at_center = derivative.subs(
            dict(zip(variables, center)), simultaneous=True
        )

        evaluated = value_at_center.evalf()
        if evaluated.is_finite is not True or evaluated.is_real is not True:
            raise ValueError(
                "A required derivative is undefined or non-real at the center."
            )

        term = value_at_center
        for variable, center_value, order in zip(variables, center, orders):
            term *= (variable - center_value) ** order
            term /= factorial(order)

        polynomial += term

    return sp.expand(polynomial)


def matches_sympy_series_1d(function, variable, center_val, degree):
    """Check our 1D Taylor polynomial against SymPy's built-in series()."""
    try:
        ours = taylor_polynomial(function, (variable,), (center_val,), degree)
        reference = sp.series(function, variable, center_val, degree + 1).removeO()

        # Check values numerically across a small window near the center
        sample = np.linspace(float(center_val) - 0.5, float(center_val) + 0.5, 25)
        f_ours = sp.lambdify(variable, ours, "numpy")
        f_ref = sp.lambdify(variable, reference, "numpy")

        vals_ours = np.broadcast_to(np.asarray(f_ours(sample), dtype=float), sample.shape)
        vals_ref = np.broadcast_to(np.asarray(f_ref(sample), dtype=float), sample.shape)
        return bool(np.allclose(vals_ours, vals_ref, atol=1e-7))
    except Exception:
        return False