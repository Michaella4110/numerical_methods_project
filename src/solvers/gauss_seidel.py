import numpy as np
from ..utils import is_diagonally_dominant # Assuming utils is in the parent directory

class GaussSeidelSolver:
    """
    Solves a system of linear equations Ax = b using the Gauss-Seidel method.
    Supports Successive Over-Relaxation (SOR) and attempts matrix reordering
    to achieve diagonal dominance.
    """
    def __init__(self, A: np.ndarray, b: np.ndarray, tol: float = 1e-8,
                 max_iter: int = 100, omega: float = 1.0, try_reorder: bool = True):
        
        if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square 2D numpy array.")
        if not isinstance(b, np.ndarray) or b.ndim != 1 or b.shape[0] != A.shape[0]:
            raise ValueError("b must be a 1D numpy array with length matching A's dimensions.")
        if not (0 < omega < 2):
            raise ValueError("Omega (relaxation factor) must be between 0 and 2.")

        self._original_A = A.copy()
        self._original_b = b.copy()
        self.A = A.copy()
        self.b = b.copy()
        self.n = A.shape[0]
        self.tol = tol
        self.max_iter = max_iter
        self.omega = omega
        self.reordered_indices = None # Stores the permutation from original to current A
        self.reordered = False

        if try_reorder:
            self.attempt_reorder()

        # Check for diagonal elements being zero after potential reordering
        # This is critical for Gauss-Seidel division
        if np.any(np.diag(self.A) == 0):
            raise ValueError("Matrix has zero(s) on the diagonal, even after reordering attempt. "
                             "Gauss-Seidel cannot proceed as division by zero would occur.")

    def attempt_reorder(self):
        """
        Attempts to reorder the matrix A and vector b to achieve strict diagonal dominance.
        This is a heuristic greedy approach.
        If successful, updates self.A, self.b, and self.reordered_indices.
        """
        if is_diagonally_dominant(self.A):
            self.reordered = False # It was already dominant or the current A is from a previous reorder
            return # Already dominant

        n = self.n
        temp_A = self._original_A.copy()
        temp_b = self._original_b.copy()
        
        new_A = np.zeros_like(temp_A)
        new_b = np.zeros_like(temp_b)
        new_indices = np.zeros(n, dtype=int) # Maps new row index to original row index

        available_original_rows = list(range(n))
        
        # Iterate through each row position in the *new* matrix (target matrix)
        for i in range(n):
            best_original_row_idx = -1
            max_dominance_ratio = -1.0 # Metric for selecting the best row

            # Iterate through available original rows to find the best fit for current new row `i`
            for original_row_candidate_idx in available_original_rows:
                # Calculate dominance for this candidate row if it were placed at new_A[i, i] diagonal
                diagonal_val = abs(temp_A[original_row_candidate_idx, i])
                row_sum_off_diagonals = np.sum(abs(temp_A[original_row_candidate_idx, :])) - diagonal_val
                
                # Avoid division by zero if all off-diagonals are zero
                dominance_ratio = diagonal_val / (row_sum_off_diagonals + 1e-12) if row_sum_off_diagonals > 0 else (1e12 if diagonal_val > 0 else 0)

                if dominance_ratio > max_dominance_ratio:
                    max_dominance_ratio = dominance_ratio
                    best_original_row_idx = original_row_candidate_idx
            
            if best_original_row_idx != -1:
                new_A[i, :] = temp_A[best_original_row_idx, :]
                new_b[i] = temp_b[best_original_row_idx]
                new_indices[i] = best_original_row_idx
                available_original_rows.remove(best_original_row_idx)
            else:
                # If no good dominant row was found for this position,
                # fill remaining with arbitrary available rows to maintain solvability
                # (though dominance is not guaranteed)
                print(f"Warning: Could not find a strictly dominant row for new matrix row {i}. "
                      "Attempting to fill with remaining rows.")
                
                # Fill the rest of the new_A with remaining available_original_rows in order
                for k, original_row_idx in enumerate(available_original_rows):
                    new_A[i + k, :] = temp_A[original_row_idx, :]
                    new_b[i + k] = temp_b[original_row_idx]
                    new_indices[i + k] = original_row_idx
                break # All remaining rows placed, exit loop

        self.A = new_A
        self.b = new_b
        self.reordered_indices = new_indices
        self.reordered = True
        
        if not is_diagonally_dominant(self.A):
            print("Warning: Reordering attempt did not achieve strict diagonal dominance. "
                  "Convergence is not guaranteed and might be slow or fail.")
            self.reordered = False # Mark as not successfully reordered for dominance


    def solve(self, x0: np.ndarray = None, progress_callback=None) -> (np.ndarray, list, str):
        """
        Executes the Gauss-Seidel method (with SOR if omega != 1.0) to solve Ax = b.
        Args:
            x0 (np.ndarray, optional): Initial guess vector. If None, uses zeros.
            progress_callback (callable, optional): A function to call with iteration history
                                                    (dict) for GUI updates.
        Returns:
            tuple: (result (np.ndarray), history (list of dicts), status_string (str))
        """
        x = np.zeros(self.n) if x0 is None else np.array(x0, dtype=float)
        history = []
        status_str = "Maximum iterations reached"
        
        prev_err = None # For divergence detection
        
        for k in range(1, self.max_iter + 1):
            x_old = x.copy() # Store previous iteration's x for error calculation
            
            for i in range(self.n):
                # sigma = sum(A[i, j] * x[j]) for j != i
                # Optimized calculation: dot product of row with x, minus diagonal element contribution
                sigma = np.dot(self.A[i, :], x) - self.A[i, i] * x[i]
                
                # Gauss-Seidel update (intermediate value)
                x_i_gs = (self.b[i] - sigma) / self.A[i, i]
                
                # SOR (Successive Over-Relaxation) update
                x[i] = x_old[i] + self.omega * (x_i_gs - x_old[i])
            
            err = np.linalg.norm(x - x_old, np.inf) # Max absolute difference
            
            # Record history
            iteration_data = {
                'Iteration': k,
                'x_vector': x.copy(), # Store a copy to prevent modification later
                'Error': err
            }
            history.append(iteration_data)
            if progress_callback:
                progress_callback(iteration_data)

            if err < self.tol:
                status_str = "Converged"
                break
            
            # Divergence check: if error grows by a huge factor compared to previous, stop
            if prev_err is not None and err > prev_err * 1e6 and k > 2: # Give it a few iterations before checking
                status_str = "Diverging - stopped early"
                break
            prev_err = err # Update previous error for next iteration's check

            if k == self.max_iter:
                status_str = "Maximum iterations reached"
        
        return x, history, status_str