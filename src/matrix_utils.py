import numpy as np
from scipy.linalg import lu_factor, lu_solve # For robust direct solve

def is_diagonally_dominant(A: np.ndarray) -> bool:
    """
    Checks if a square matrix A is strictly diagonally dominant.
    A matrix is strictly diagonally dominant if, for every row,
    the absolute value of the diagonal element is strictly greater
    than the sum of the absolute values of the other elements in that row.
    
    Args:
        A (np.ndarray): The input square matrix.
        
    Returns:
        bool: True if strictly diagonally dominant, False otherwise.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square.")
        
    n = A.shape[0]
    for i in range(n):
        diagonal_element = abs(A[i, i])
        
        # Calculate the sum of absolute values of off-diagonal elements in the current row
        row_sum_of_off_diagonals = np.sum(abs(A[i, :])) - diagonal_element
        
        if diagonal_element <= row_sum_of_off_diagonals:
            return False # Not strictly diagonally dominant
            
    return True # All rows satisfy the condition

def try_make_diagonally_dominant(A: np.ndarray, b: np.ndarray) -> (np.ndarray, np.ndarray, bool):
    """
    Attempts to reorder the rows of matrix A and vector b to achieve strict
    diagonal dominance. This uses a greedy pivoting heuristic.

    Args:
        A (np.ndarray): The input square coefficient matrix.
        b (np.ndarray): The input constant vector.

    Returns:
        tuple: (A_reordered, b_reordered, is_strictly_dominant)
            A_reordered (np.ndarray): The reordered matrix (or original if no reorder helps).
            b_reordered (np.ndarray): The reordered vector.
            is_strictly_dominant (bool): True if the returned (A_reordered, b_reordered)
                                         is strictly diagonally dominant, False otherwise.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix A must be square.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("Input vector b must be 1D and match A's dimension.")

    n = A.shape[0]
    
    # If already dominant, return as is
    if is_diagonally_dominant(A):
        return A.copy(), b.copy(), True

    # Use original copies for reordering
    original_A = A.copy()
    original_b = b.copy()

    # Stores the new order of original row indices
    new_row_order = [-1] * n 
    
    # Keep track of which original rows have been used
    used_original_rows = [False] * n 

    # Iterate through each column to find the best pivot for that column's diagonal position
    for col_idx in range(n):
        best_original_row = -1
        # Metric to find the "most dominant" available row for the current diagonal position
        max_dominance_score = -1.0 

        # Search through all original rows that haven't been placed yet
        for original_row_idx in range(n):
            if not used_original_rows[original_row_idx]:
                # The candidate diagonal element for this new position (col_idx, col_idx)
                candidate_diag_val = abs(original_A[original_row_idx, col_idx])
                
                # Calculate sum of off-diagonal elements in the candidate row (for a fixed column col_idx)
                # This check is more complex than just checking a single row's dominance,
                # as we're trying to build dominance across the *new* diagonal.
                
                # For a true dominance check, we need the sum of other elements in that *row*
                # when placed at this new position.
                
                # A simpler greedy approach: for current column `col_idx`, pick the available
                # row `original_row_idx` that has the largest absolute value at `original_A[original_row_idx, col_idx]`.
                # This is "maximum absolute column pivoting".
                current_pivot_val = abs(original_A[original_row_idx, col_idx])
                if current_pivot_val > max_dominance_score:
                    max_dominance_score = current_pivot_val
                    best_original_row = original_row_idx
        
        if best_original_row != -1:
            new_row_order[col_idx] = best_original_row
            used_original_rows[best_original_row] = True
        else:
            # This should ideally not happen if matrix is non-singular,
            # but as a fallback, fill remaining with any unused rows.
            # This means strict dominance might not be achievable even with best efforts.
            remaining_unplaced_rows = [r for r, used in enumerate(used_original_rows) if not used]
            for i in range(col_idx, n):
                if remaining_unplaced_rows:
                    new_row_order[i] = remaining_unplaced_rows.pop(0)
                else:
                    break # All rows are placed

    # Construct the reordered matrix and vector
    reordered_A = original_A[new_row_order, :]
    reordered_b = original_b[new_row_order]

    # Check if the reordered matrix is strictly diagonally dominant
    is_dominant = is_diagonally_dominant(reordered_A)

    if not is_dominant:
        # If strict dominance wasn't achieved, we've returned the best possible
        # permutation based on the greedy heuristic, but it's not strictly dominant.
        # This is a "best effort" return.
        print("Warning: Attempted reordering did not achieve strict diagonal dominance.")

    return reordered_A, reordered_b, is_dominant


def safe_direct_solve(A: np.ndarray, b: np.ndarray) -> (np.ndarray, str):
    """
    Attempts to solve a system of linear equations Ax = b using a direct method (LU decomposition).
    Handles singular matrices gracefully.

    Args:
        A (np.ndarray): The coefficient matrix.
        b (np.ndarray): The constant vector.

    Returns:
        tuple: (x or None, error_msg or None)
            x (np.ndarray or None): The solution vector if successful, None otherwise.
            error_msg (str or None): An error message if the solve fails, None otherwise.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None, "Input matrix A must be square."
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        return None, "Input vector b must be 1D and match A's dimension."

    try:
        # Perform LU decomposition with pivoting
        lu, piv = lu_factor(A)
        
        # Check for singularity (determinant of U matrix is product of diagonal elements)
        # If any diagonal element of U is zero, matrix is singular.
        if np.any(np.diag(lu) == 0):
            return None, "Matrix is singular (or numerically ill-conditioned), direct solve failed."
            
        # Solve the system using the LU factors
        x = lu_solve((lu, piv), b)
        return x, None
    except np.linalg.LinAlgError as e:
        return None, f"Linear algebra error during direct solve: {e}"
    except Exception as e:
        return None, f"An unexpected error occurred during direct solve: {e}"