import numpy as np

def solve(A, b, x0, tolerance, max_iter=100):
    """
    Solves a system of linear equations Ax = b using the Gauss-Seidel iterative method.

    Args:
        A (numpy.ndarray): The coefficient matrix.
        b (numpy.ndarray): The constant vector.
        x0 (numpy.ndarray): The initial guess vector.
        tolerance (float): The desired accuracy (infinity norm of the difference between successive approximations).
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        tuple: (solution_vector, history_log, status_message)
            solution_vector (numpy.ndarray): The estimated solution vector.
            history_log (list of dict): A log of each iteration's state.
            status_message (str): A message indicating the convergence status.
    """
    history_log = []
    status_message = ""
    n = A.shape[0]
    x = x0.copy().astype(float) # Ensure x is float and a copy

    # Check for diagonal elements being zero
    if np.any(np.diag(A) == 0):
        return None, history_log, "Failed: Diagonal element(s) are zero, cannot proceed with Gauss-Seidel."

    for k in range(1, max_iter + 1):
        x_old = x.copy()
        
        for i in range(n):
            sigma = 0
            for j in range(n):
                if i != j:
                    sigma += A[i, j] * x[j] # Use the most recently updated x values

            x[i] = (b[i] - sigma) / A[i, i]
        
        error = np.linalg.norm(x - x_old, np.inf)

        step_data = {
            'Iteration': k,
            'x_vector': x.copy(), # Store a copy to prevent modification later
            'Error': error
        }
        history_log.append(step_data)

        if error < tolerance:
            status_message = f"Converged in {k} iterations."
            return x, history_log, status_message

    status_message = f"Max iterations ({max_iter}) reached without achieving desired tolerance."
    return x, history_log, status_message