import numpy as np

def solve(func, func_prime, x0, tolerance, max_iter=100):
    """
    Solves for the root of a nonlinear equation f(x) = 0 using the Newton-Raphson method.

    Args:
        func (callable): The function f(x) for which to find the root.
        func_prime (callable): The derivative of the function f'(x).
        x0 (float): The initial guess.
        tolerance (float): The desired accuracy of the root.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        tuple: (root, history_log, status_message)
            root (float): The estimated root.
            history_log (list of dict): A log of each iteration's state.
            status_message (str): A message indicating the convergence status.
    """
    history_log = []
    status_message = ""
    x_n = float(x0) # Ensure x_n is a float

    for k in range(1, max_iter + 1):
        f_x_n = func(x_n)
        f_prime_x_n = func_prime(x_n)

        if abs(f_prime_x_n) < 1e-12: # Check for derivative near zero
            status_message = f"Iteration stopped: Derivative is near zero at x = {x_n:.6f}."
            return x_n, history_log, status_message

        x_n_plus_1 = x_n - f_x_n / f_prime_x_n
        error = abs(x_n_plus_1 - x_n)

        step_data = {
            'Iteration': k,
            'x_n': x_n,
            'f(x_n)': f_x_n,
            "f'(x_n)": f_prime_x_n,
            'x_n+1': x_n_plus_1,
            'Error': error
        }
        history_log.append(step_data)

        if error < tolerance:
            status_message = f"Converged in {k} iterations."
            return x_n_plus_1, history_log, status_message

        x_n = x_n_plus_1

    status_message = f"Max iterations ({max_iter}) reached without achieving desired tolerance."
    return x_n, history_log, status_message