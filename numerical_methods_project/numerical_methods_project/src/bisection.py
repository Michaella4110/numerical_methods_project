import numpy as np

def solve(func, a, b, tolerance, max_iter=100):
    """
    Solves for the root of a nonlinear equation f(x) = 0 using the Bisection method.

    Args:
        func (callable): The function f(x) for which to find the root.
        a (float): The start of the bracketing interval.
        b (float): The end of the bracketing interval.
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

    if func(a) * func(b) >= 0:
        # This case should ideally be caught by utils.validate_bisection
        # but included here for robustness within the solver itself.
        return None, history_log, "Failed: Function has same sign at interval endpoints."

    c = a
    for k in range(1, max_iter + 1):
        c_prev = c
        c = (a + b) / 2
        f_c = func(c)
        error = abs(c - c_prev) if k > 1 else abs(b - a) / 2 # Error based on interval size or successive approximation

        step_data = {
            'Iteration': k,
            'a': a,
            'b': b,
            'c': c,
            'f(c)': f_c,
            'Error': error
        }
        history_log.append(step_data)

        if abs(f_c) < 1e-12 or error < tolerance:
            status_message = f"Converged in {k} iterations."
            return c, history_log, status_message

        if func(a) * f_c < 0:
            b = c
        else:
            a = c

    status_message = f"Max iterations ({max_iter}) reached without achieving desired tolerance."
    return c, history_log, status_message