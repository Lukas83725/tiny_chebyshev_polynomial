import math
import unittest

from tiny_chebyshev_polynomial import (
    chebyshev_t,
    chebyshev_u,
    chebyshev_t_array,
    chebyshev_u_array,
)


class TestChebyshevT(unittest.TestCase):
    def test_degree_zero_is_one(self):
        for x in (-2.0, -0.5, 0.0, 0.5, 2.0):
            self.assertEqual(chebyshev_t(0, x), 1.0)

    def test_degree_one_is_x(self):
        for x in (-2.0, -0.5, 0.0, 0.5, 2.0):
            self.assertEqual(chebyshev_t(1, x), float(x))

    def test_known_values_inside_unit_interval(self):
        # T_2(x) = 2x^2 - 1, T_3(x) = 4x^3 - 3x
        self.assertAlmostEqual(chebyshev_t(2, 0.5), -0.5)
        self.assertAlmostEqual(chebyshev_t(3, 0.5), -1.0)
        self.assertAlmostEqual(chebyshev_t(4, 0.5), -0.5)
        self.assertAlmostEqual(chebyshev_t(5, 0.5), 0.5)

    def test_endpoints(self):
        # T_n(1) = 1 and T_n(-1) = (-1)^n
        self.assertAlmostEqual(chebyshev_t(7, 1.0), 1.0)
        self.assertAlmostEqual(chebyshev_t(7, -1.0), -1.0)
        self.assertAlmostEqual(chebyshev_t(8, -1.0), 1.0)

    def test_outside_unit_interval(self):
        # T_2(2) = 2*4 - 1 = 7, T_3(2) = 4*8 - 3*2 = 26
        self.assertAlmostEqual(chebyshev_t(2, 2.0), 7.0)
        self.assertAlmostEqual(chebyshev_t(3, 2.0), 26.0)

    def test_negative_degree_raises(self):
        with self.assertRaises(ValueError):
            chebyshev_t(-1, 0.0)

    def test_non_integer_degree_raises(self):
        with self.assertRaises(TypeError):
            chebyshev_t(1.5, 0.0)
        with self.assertRaises(TypeError):
            chebyshev_t(True, 0.0)


class TestChebyshevU(unittest.TestCase):
    def test_degree_zero_is_one(self):
        for x in (-2.0, -0.5, 0.0, 0.5, 2.0):
            self.assertEqual(chebyshev_u(0, x), 1.0)

    def test_degree_one_is_two_x(self):
        for x in (-2.0, -0.5, 0.0, 0.5, 2.0):
            self.assertEqual(chebyshev_u(1, x), 2.0 * x)

    def test_known_values_inside_unit_interval(self):
        # U_2(x) = 4x^2 - 1, U_3(x) = 8x^3 - 4x
        self.assertAlmostEqual(chebyshev_u(2, 0.5), 0.0)
        self.assertAlmostEqual(chebyshev_u(3, 0.5), -1.0)
        self.assertAlmostEqual(chebyshev_u(4, 0.5), -1.0)

    def test_endpoints(self):
        # U_n(1) = n + 1 and U_n(-1) = (-1)^n (n + 1)
        self.assertAlmostEqual(chebyshev_u(7, 1.0), 8.0)
        self.assertAlmostEqual(chebyshev_u(7, -1.0), -8.0)
        self.assertAlmostEqual(chebyshev_u(8, -1.0), 9.0)

    def test_outside_unit_interval(self):
        # U_2(2) = 4*4 - 1 = 15, U_3(2) = 8*8 - 4*2 = 56
        self.assertAlmostEqual(chebyshev_u(2, 2.0), 15.0)
        self.assertAlmostEqual(chebyshev_u(3, 2.0), 56.0)

    def test_negative_degree_raises(self):
        with self.assertRaises(ValueError):
            chebyshev_u(-1, 0.0)

    def test_non_integer_degree_raises(self):
        with self.assertRaises(TypeError):
            chebyshev_u(1.5, 0.0)
        with self.assertRaises(TypeError):
            chebyshev_u(True, 0.0)


class TestArrayHelpers(unittest.TestCase):
    def test_t_array_matches_scalar_calls(self):
        xs = [-1.0, -0.25, 0.0, 0.25, 1.0]
        got = chebyshev_t_array(4, xs)
        want = [chebyshev_t(4, x) for x in xs]
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w)

    def test_u_array_matches_scalar_calls(self):
        xs = [-1.0, -0.25, 0.0, 0.25, 1.0]
        got = chebyshev_u_array(4, xs)
        want = [chebyshev_u(4, x) for x in xs]
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w)

    def test_array_empty_input(self):
        self.assertEqual(chebyshev_t_array(3, []), [])
        self.assertEqual(chebyshev_u_array(3, []), [])

    def test_array_validates_degree(self):
        with self.assertRaises(ValueError):
            chebyshev_t_array(-1, [0.0])
        with self.assertRaises(TypeError):
            chebyshev_u_array(2.0, [0.0])


class TestRecurrenceConsistency(unittest.TestCase):
    def test_three_term_recurrence_holds(self):
        # For a few x and n, verify 2x T_n - T_{n-1} == T_{n+1}
        for x in (-0.9, -0.1, 0.0, 0.3, 0.8):
            for n in range(1, 8):
                lhs = 2.0 * x * chebyshev_t(n, x) - chebyshev_t(n - 1, x)
                rhs = chebyshev_t(n + 1, x)
                self.assertAlmostEqual(lhs, rhs, places=12)

    def test_second_kind_recurrence_holds(self):
        for x in (-0.9, -0.1, 0.0, 0.3, 0.8):
            for n in range(1, 8):
                lhs = 2.0 * x * chebyshev_u(n, x) - chebyshev_u(n - 1, x)
                rhs = chebyshev_u(n + 1, x)
                self.assertAlmostEqual(lhs, rhs, places=12)

    def test_derivative_relation(self):
        # U_n(x) = (T_{n+1}'(x)) / (n+1).  We approximate the derivative with a
        # central difference and check it agrees to a loose tolerance.  This is
        # deterministic and does not depend on wall-clock time.
        h = 1e-5
        for x in (-0.6, -0.2, 0.1, 0.7):
            for n in range(1, 5):
                t_plus = chebyshev_t(n + 1, x + h)
                t_minus = chebyshev_t(n + 1, x - h)
                deriv = (t_plus - t_minus) / (2.0 * h)
                self.assertAlmostEqual(deriv / (n + 1), chebyshev_u(n, x), places=6)


if __name__ == "__main__":
    unittest.main()
