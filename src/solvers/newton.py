import numpy as np
from ..utils import FunctionParser # Assuming utils is in the parent directory

class NewtonSolver:
    """
    Solves for roots of a single-variable function using Newton's method.
    Supports symbolic and numeric derivatives, and includes damping for stability.
    """
    def __init__(self, func_parser: FunctionParser, tol: float = 1e-8, max_iter: int = 100):
        if not isinstance(func_parser, FunctionParser):
            raise TypeError("func_parser must be an instance of FunctionParser.")
        
        self.func_parser = func_parser
        # Parse the function string provided to the FunctionParser instance
        self.f, _, parse_error = self.func_parser.parse(self.func_parser.original_string)
        if parse_error:
            raise ValueError(f"Function parsing failed for NewtonSolver: {parse_error}")

        self.df, self.deriv_source = self.func_parser.get_derivative()
        if self.df is None:
            raise ValueError("Could not obtain a derivative for Newton's method.")

        self.tol = tol
        self.max_iter = max_iter

    def _compute_derivative(self, x_val):
        """Helper to compute derivative, handling potential errors from the callable."""
        try:
            return self.df(x_val)
        except Exception:
            # Fallback if the derived function itself fails at a point
            h_fallback = 1e-6
            return (self.f(x_val + h_fallback) - self.f(x_val - h_fallback)) / (2 * h_fallback)

    def solve(self, x0: float, progress_callback=None) -> (float, list, str):
        """
        Executes Newton's method to find a root.
        Args:
            x0 (float): Initial guess.
            progress_callback (callable, optional): A function to call with iteration history
                                                    (dict) for GUI updates.
        Returns:
            tuple: (result (float), history (list of dicts), status_string (str))
        """
        x = float(x0)
        history = []
        status_str = "Maximum iterations reached"
        
        for k in range(1, self.max_iter + 1):
            fx = self.f(x)

            # Record history for current x before potential update
            iteration_data = {
                'Iteration': k,
                'x_n': x,
                'f(x_n)': fx,
                'Error': abs(fx) # Error defined as |f(x_n)| for convergence check
            }
            
            if abs(fx) < self.tol:
                status_str = "Converged"
                history.append(iteration_data) # Append final converged state
                if progress_callback:
                    progress_callback(iteration_data)
                break

            fpx = self._compute_derivative(x)
            
            # Handle derivative near zero or undefined
            if fpx is None or abs(fpx) < 1e-12: # Add a threshold for 'near zero'
                # Before failing, try numeric derivative with a slightly larger h if using symbolic
                if self.deriv_source == "sympy" and abs(fpx) < 1e-12:
                    h_try = 1e-6
                    fpx_numeric = (self.f(x + h_try) - self.f(x - h_try)) / (2 * h_try)
                    if abs(fpx_numeric) < 1e-14: # Even larger h didn't help
                        status_str = "Failed: Derivative near zero or undefined."
                        history.append(iteration_data)
                        if progress_callback:
                            progress_callback(iteration_data)
                        break
                    else:
                        fpx = fpx_numeric # Use the numeric one
                else:
                    status_str = "Failed: Derivative near zero or undefined."
                    history.append(iteration_data)
                    if progress_callback:
                        progress_callback(iteration_data)
                    break

            step = fx / fpx
            lam = 1.0 # Damping parameter

            # Damping (backtracking line search)
            x_new_candidate = x - lam * step
            f_x_new_candidate = None # Initialize outside loop

            # Limit the number of backtracking steps (e.g., 8)
            for _ in range(8):
                try:
                    f_x_new_candidate = self.f(x_new_candidate)
                    if abs(f_x_new_candidate) <= abs(fx):
                        break # Found a better point
                except Exception: # Catch cases where f(x_new_candidate) might be undefined
                    pass # Try a smaller lambda
                lam *= 0.5
                x_new_candidate = x - lam * step
            
            x_new = x_new_candidate # This is the accepted new point after damping

            # Clamp huge steps to prevent blowing up (if damping wasn't enough)
            max_step_abs = max(1.0, abs(x) * 10) # max_step should be an absolute value
            if abs(x_new - x) > max_step_abs:
                # Limit step size but maintain direction
                step_direction = np.sign(x_new - x) if (x_new - x) != 0 else 1
                x_new = x + step_direction * max_step_abs


            if np.isnan(x_new) or np.isinf(x_new):
                status_str = "Failed: x became NaN or Inf."
                history.append(iteration_data) # Append state before failure
                if progress_callback:
                    progress_callback(iteration_data)
                break
            
            x = x_new
            
            history.append(iteration_data) # Append iteration data after x is updated
            if progress_callback:
                progress_callback(iteration_data)

            if k == self.max_iter:
                 status_str = "Maximum iterations reached"

        return x, history, status_str