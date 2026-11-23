from itertools import product
from math import factorial

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

    # Each tuple represents derivative orders.
    # For (x, y), (1, 2) means once in x and twice in y.
    for orders in product(range(degree + 1), repeat=len(variables)):

        if sum(orders) > degree:
            continue

        derivative = function

        for variable, order in zip(variables, orders):
            derivative = sp.diff(derivative, variable, order)

        value_at_center = derivative.subs(
            dict(zip(variables, center))
        )

        if value_at_center.is_finite is not True:
            raise ValueError(
                "A required derivative is undefined at the center."
            )

        term = value_at_center

        for variable, center_value, order in zip(
            variables, center, orders
        ):
            term *= (variable - center_value) ** order
            term /= factorial(order)

        polynomial += term

    return sp.expand(polynomial)


# Examples to run when we execute this file
if __name__ == "__main__":
    x, y = sp.symbols("x y")

    one_variable = taylor_polynomial(
        sp.exp(x), (x,), (0,), 2
    )

    two_variables = taylor_polynomial(
        sp.exp(x + y), (x, y), (0, 0), 2
    )

    print("exp(x), degree 2:")
    print(one_variable)

    print("\nexp(x + y), degree 2:")
    print(two_variables)