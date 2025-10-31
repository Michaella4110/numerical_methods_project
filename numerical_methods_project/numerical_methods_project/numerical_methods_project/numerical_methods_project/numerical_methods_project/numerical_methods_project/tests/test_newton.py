# numerical_solver/tests/test_newton.py
import unittest
import math
from solvers import newton
from numerical_methods_project.src import utils

class TestNewton(unittest.TestCase):

    def setUp(self):
        # Define some test functions and their derivatives
        self.func1_str = "x**3 - x - 2" # Root near 1.521
        self.func1 = utils.parse_function(self.func1_str)
        self.func1_prime = utils.get_derivative(self.func1_str)

        self.func2_str = "math.sin(x) - x/2" # Root at 0, and others
        self.func2 = utils.parse_function(self.func2_str)
        self.func2_prime = utils.get_derivative(self.func2_str)
        
        self.func3_str = "x**2 - 4" # Roots at 2, -2
        self.func3 = utils.parse_function(self.func3_str)
        self.func3_prime = utils.get_derivative(self.func3_str)

        # Function where derivative can be zero
        self.func_zero_prime_str = "x**2"
        self.func_zero_prime = utils.parse_function(self.func_zero_prime_str)
        self.func_zero_prime_prime = utils.get_derivative(self.func_zero_prime_str)


    def test_newton_converges(self):
        # Test case 1: x^3 - x - 2 = 0 starting x0 = 1.5
        root, history, status = newton.solve(self.func1, self.func1_prime, 1.5, tolerance=1e-6)
        self.assertIsNotNone(root)
        self.assertAlmostEqual(root, 1.5213797068, places=5)
        self.assertIn("Converged", status)
        self.assertGreater(len(history), 0)

        # Test case 2: sin(x) - x/2 = 0 starting x0 = 2.0 (root near 1.895)
        root, history, status = newton.solve(self.func2, self.func2_prime, 2.0, tolerance=1e-6)
        self.assertIsNotNone(root)
        self.assertAlmostEqual(root, 1.8954942670, places=5)
        self.assertIn("Converged", status)

    def test_newton_max_iterations(self):
        # Set a very small max_iter to force non-convergence
        root, history, status = newton.solve(self.func1, self.func1_prime, 0.1, tolerance=1e-12, max_iter=3)
        self.assertIn("Max iterations reached", status)
        self.assertEqual(len(history), 3)

    def test_newton_derivative_near_zero(self):
        # Test func_zero_prime = x^2, with x0 = 0.001 (derivative will quickly approach zero)
        root, history, status = newton.solve(self.func_zero_prime, self.func_zero_prime_prime, 0.001, tolerance=1e-6)
        self.assertIsNotNone(root) # Should still find the root at 0, but status indicates the check was hit
        self.assertAlmostEqual(root, 0.0, places=6)
        self.assertIn("Converged", status) # It converges to 0.0, and f'(0) is 0, so the check will stop it once it hits
        
        # Another case, specifically to test the derivative near zero stopping condition *before* convergence
        # A function like f(x) = x^3 - 1e-10 * x. If x0 is very small, f'(x) will be near 0
        func_flat_str = "x**3 - 1e-10 * x"
        func_flat = utils.parse_function(func_flat_str)
        func_flat_prime = utils.get_derivative(func_flat_str)
        
        root, history, status = newton.solve(func_flat, func_flat_prime, 0.0001, tolerance=1e-6, max_iter=10)
        self.assertIn("Derivative is near zero", status)

    def test_newton_with_different_tolerance(self):
        root_loose, history_loose, status_loose = newton.solve(self.func1, self.func1_prime, 1.5, tolerance=1e-2)
        root_tight, history_tight, status_tight = newton.solve(self.func1, self.func1_prime, 1.5, tolerance=1e-8)
        
        self.assertIn("Converged", status_loose)
        self.assertIn("Converged", status_tight)
        self.assertLess(len(history_loose), len(history_tight)) # More iterations for tighter tolerance