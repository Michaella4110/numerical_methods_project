# numerical_solver/tests/test_gauss_seidel.py
import unittest
import numpy as np
from solvers import gauss_seidel
from numerical_methods_project.src import utils

class TestGaussSeidel(unittest.TestCase):

    def test_gauss_seidel_converges_diagonally_dominant(self):
        # Diagonally dominant matrix - should converge
        A = np.array([[10., -1., 2., 0.],
                      [-1., 11., -1., 3.],
                      [2., -1., 10., -1.],
                      [0., 3., -1., 8.]])
        b = np.array([6., 25., -11., 15.])
        x0 = np.array([0., 0., 0., 0.])
        
        solution, history, status = gauss_seidel.solve(A, b, x0, tolerance=1e-6)
        
        self.assertIsNotNone(solution)
        self.assertIn("Converged", status)
        self.assertGreater(len(history), 0)
        
        # Expected solution (calculated by numpy.linalg.solve)
        expected_solution = np.linalg.solve(A, b)
        np.testing.assert_array_almost_equal(solution, expected_solution, decimal=5)

    def test_gauss_seidel_converges_another_case(self):
        A = np.array([[3., 1.],
                      [1., 2.]])
        b = np.array([5., 5.])
        x0 = np.array([0., 0.])

        solution, history, status = gauss_seidel.solve(A, b, x0, tolerance=1e-6)
        
        self.assertIsNotNone(solution)
        self.assertIn("Converged", status)
        expected_solution = np.linalg.solve(A, b)
        np.testing.assert_array_almost_equal(solution, expected_solution, decimal=5)


    def test_gauss_seidel_max_iterations(self):
        # Diagonally dominant, but with very few iterations
        A = np.array([[10., -1., 2.],
                      [-1., 11., -1.],
                      [2., -1., 10.]])
        b = np.array([6., 25., -11.])
        x0 = np.array([0., 0., 0.])

        solution, history, status = gauss_seidel.solve(A, b, x0, tolerance=1e-12, max_iter=5)
        self.assertIn("Max iterations reached", status)
        self.assertEqual(len(history), 5) # Should run exactly max_iter times

    def test_gauss_seidel_singular_matrix(self):
        # Singular matrix (no unique solution) - should lead to division by zero or large numbers
        A = np.array([[1., 1.],
                      [1., 1.]])
        b = np.array([2., 3.]) # Inconsistent system

        x0 = np.array([0., 0.])
        solution, history, status = gauss_seidel.solve(A, b, x0, tolerance=1e-6)
        
        self.assertIsNotNone(solution) # It will return something, but likely not meaningful or converged
        self.assertIn("Max iterations reached", status) # It won't converge

    def test_gauss_seidel_not_diagonally_dominant_but_converges(self):
        # This matrix is NOT diagonally dominant, but Gauss-Seidel still converges for it
        # This tests that the solver itself works even if the pre-check warns.
        A = np.array([[1., 2.],
                      [3., 4.]])
        b = np.array([5., 6.])
        x0 = np.array([0., 0.])

        solution, history, status = gauss_seidel.solve(A, b, x0, tolerance=1e-6, max_iter=100)
        self.assertIsNotNone(solution)
        self.assertIn("Converged", status)
        
        expected_solution = np.linalg.solve(A, b)
        np.testing.assert_array_almost_equal(solution, expected_solution, decimal=5)

    def test_gauss_seidel_matrix_with_zero_on_diagonal(self):
        # Matrix with a zero on the diagonal (should cause ZeroDivisionError without handling)
        A = np.array([[1., 2.],
                      [0., 4.]])
        b = np.array([5., 6.])
        x0 = np.array([0., 0.])
        
        # This scenario specifically tests the robustness against division by zero during iteration
        # In a real application, A[i,i] == 0 would be caught by initial validation for example.
        # For now, we expect it to either fail or produce inf/nan if not handled
        # The current implementation will raise ZeroDivisionError for A[1,1]=0
        with self.assertRaises(ZeroDivisionError):
             gauss_seidel.solve(A, b, x0, tolerance=1e-6)

    def test_is_diagonally_dominant(self):
        # Diagonally dominant
        A_dd = np.array([[4, 1, 1],
                         [1, 5, 1],
                         [1, 1, 6]])
        self.assertTrue(utils.is_diagonally_dominant(A_dd))

        # Not diagonally dominant
        A_not_dd = np.array([[1, 5, 1],
                             [4, 1, 1],
                             [1, 1, 6]])
        self.assertFalse(utils.is_diagonally_dominant(A_not_dd))

        # Edge case: 2x2 dominant
        A_2x2_dd = np.array([[2, 0.5],
                             [0.5, 2]])
        self.assertTrue(utils.is_diagonally_dominant(A_2x2_dd))

        # Edge case: 2x2 not dominant
        A_2x2_not_dd = np.array([[1, 2],
                                 [0.5, 2]])
        self.assertFalse(utils.is_diagonally_dominant(A_2x2_not_dd))