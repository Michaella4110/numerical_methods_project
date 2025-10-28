# numerical_solver/tests/test_bisection.py
import unittest
import math
from solvers import bisection
from numerical_methods_project.src import utils

class TestBisection(unittest.TestCase):

    def setUp(self):
        # Define some test functions
        self.func1_str = "x**3 - x - 2" # Root near 1.521
        self.func2_str = "math.sin(x) - x/2" # Root at 0, and others
        self.func3_str = "x**2 - 4" # Roots at 2, -2
        
        self.func1 = utils.parse_function(self.func1_str)
        self.func2 = utils.parse_function(self.func2_str)
        self.func3 = utils.parse_function(self.func3_str)


    def test_bisection_converges(self):
        # Test case 1: x^3 - x - 2 = 0 in [1, 2]
        root, history, status = bisection.solve(self.func1, 1, 2, tolerance=1e-6)
        self.assertIsNotNone(root)
        self.assertAlmostEqual(root, 1.521379, places=5)
        self.assertIn("Converged", status)
        self.assertGreater(len(history), 0)

        # Test case 2: sin(x) - x/2 = 0 in [1, 3] (another root is near 1.895)
        root, history, status = bisection.solve(self.func2, 1, 3, tolerance=1e-6)
        self.assertIsNotNone(root)
        self.assertAlmostEqual(root, 1.895494, places=5)
        self.assertIn("Converged", status)

    def test_bisection_invalid_interval(self):
        # Test case where f(a) * f(b) >= 0
        root, history, status = bisection.solve(self.func3, 1, 1.5, tolerance=1e-6)
        self.assertIsNone(root)
        self.assertIn("Failed: Function has same sign at interval endpoints.", status)
        self.assertEqual(len(history), 0)

    def test_bisection_max_iterations(self):
        # Set a very small max_iter to force non-convergence
        root, history, status = bisection.solve(self.func1, 1, 2, tolerance=1e-12, max_iter=5)
        self.assertIn("Max iterations reached", status)
        self.assertEqual(len(history), 5) # Should run exactly max_iter times

    def test_bisection_tolerance_zero_fc(self):
        # Test case where f(c) hits zero exactly or very close
        # For x^2 - 4 = 0, with a=1, b=3, root is 2.
        root, history, status = bisection.solve(self.func3, 1, 3, tolerance=1e-12)
        self.assertIsNotNone(root)
        self.assertAlmostEqual(root, 2.0, places=6)
        self.assertIn("Converged", status)

    def test_bisection_with_different_tolerance(self):
        root_loose, history_loose, status_loose = bisection.solve(self.func1, 1, 2, tolerance=1e-2)
        root_tight, history_tight, status_tight = bisection.solve(self.func1, 1, 2, tolerance=1e-8)
        
        self.assertIn("Converged", status_loose)
        self.assertIn("Converged", status_tight)
        self.assertLess(len(history_loose), len(history_tight)) # More iterations for tighter tolerance