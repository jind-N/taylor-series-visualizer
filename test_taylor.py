import unittest

import sympy as sp

from taylor import taylor_polynomial


class TestTaylorPolynomial(unittest.TestCase):

    def check_equal(self, actual, expected):
        """Check whether two expressions are mathematically equal."""
        self.assertEqual(sp.simplify(actual - expected), 0)

    def test_one_variable(self):
        x = sp.symbols("x")

        result = taylor_polynomial(
            sp.exp(x), (x,), (0,), 2
        )

        self.check_equal(result, 1 + x + x**2 / 2)

    def test_two_variables(self):
        x, y = sp.symbols("x y")

        result = taylor_polynomial(
            sp.exp(x + y), (x, y), (0, 0), 2
        )

        expected = 1 + x + y + x**2 / 2 + x*y + y**2 / 2
        self.check_equal(result, expected)

    def test_three_variables(self):
        x, y, z = sp.symbols("x y z")

        result = taylor_polynomial(
            sp.exp(x + y + z), (x, y, z), (0, 0, 0), 1
        )

        self.check_equal(result, 1 + x + y + z)

    def test_nonzero_center(self):
        x = sp.symbols("x")

        result = taylor_polynomial(
            x**2, (x,), (2,), 1
        )

        self.check_equal(result, 4 + 4*(x - 2))

    def test_undefined_center(self):
        x = sp.symbols("x")

        with self.assertRaises(ValueError):
            taylor_polynomial(sp.log(x), (x,), (0,), 2)


if __name__ == "__main__":
    unittest.main()
    