import numpy as np
from ..utils import FunctionParser, validate_bisection # Assuming utils is in the parent directory

class BisectionSolver:
    """
    Solves for roots of a single-variable function using the Bisection method.
    Includes methods for validating brackets and attempting to auto-find brackets.
    """
    def __init__(self, func_parser: FunctionParser, tol: float = 1e-8, max_iter: int = 100):
        if not isinstance(func_parser, FunctionParser):
            raise TypeError("func_parser must be an instance of FunctionParser.")
        
        self.func_parser = func_parser
        self.f, _, parse_error = self.func_parser.parse(self.func_parser.original_string)
        if parse_error:
            raise ValueError(f"Function parsing failed for BisectionSolver: {parse_error}")

        self.tol = tol
        self.max_iter = max_iter

    def validate_bracket(self, a: float, b: float) -> bool:
        """
        Checks if the interval [a, b] is a valid bracket for the Bisection method
        (i.e., f(a) and f(b) have opposite signs).
        """
        try:
            return validate_bisection(self.f, a, b)
        except Exception:
            return False # If function evaluation fails, it's not a valid bracket

    def auto_bracket_search(self, start_a: float, end_b: float, steps: int = 200) -> (float, float):
        """
        Attempts to find a valid bracket [a, b] within a given search range.
        Args:
            start_a (float): The beginning of the search range.
            end_b (float): The end of the search range.
            steps (int): Number of steps to divide the search range into.
        Returns:
            tuple: (a (float), b (float)) if a bracket is found, else (None, None).
        """
        xs = np.linspace(start_a, end_b, steps)
        
        fa = np.nan # Initialize as NaN to handle potential evaluation issues at xs[0]
        try:
            fa = self.f(xs[0])
        except Exception:
            pass # Keep fa as NaN if evaluation fails

        for i in range(1, len(xs)):
            current_x = xs[i]
            fb = np.nan # Initialize as NaN
            try:
                fb = self.f(current_x)
            except Exception:
                pass # Keep fb as NaN if evaluation fails

            # Handle cases where function evaluation might yield NaN
            if np.isnan(fa):
                fa = fb # If previous fa was bad, make current fb the new fa for next iteration
                continue
            if np.isnan(fb):
                continue # If current fb is bad, skip this interval, keep current fa for next iteration

            if fa * fb < 0:
                return xs[i-1], current_x
            
            fa = fb # Move fb to fa for the next iteration

        return None, None

    def solve(self, a: float, b: float, progress_callback=None) -> (float, list, str):
        """
        Executes the Bisection method to find a root within the interval [a, b].
        Args:
            a (float): Lower bound of the initial bracket.
            b (float): Upper bound of the initial bracket.
            progress_callback (callable, optional): A function to call with iteration history
                                                    (dict) for GUI updates.
        Returns:
            tuple: (result (float), history (list of dicts), status_string (str))
        """
        a = float(a)
        b = float(b)
        history = []
        status_str = "Maximum iterations reached"

        if not self.validate_bracket(a, b):
            return None, history, "Initial interval does not bracket a root (f(a) * f(b) >= 0)."

        # Initial values for f(a) and f(b)
        try:
            fa_initial = self.f(a)
            fb_initial = self.f(b)
        except Exception as e:
            return None, history, f"Function evaluation failed at interval endpoints: {e}"

        c = a # Initialize c, will be updated in the loop
        for k in range(1, self.max_iter + 1):
            c_prev = c
            c = (a + b) / 2
            
            try:
                fc = self.f(c)
            except Exception:
                status_str = f"Function evaluation failed at c={c:.8f}. Stopping."
                history.append({'Iteration': k, 'a': a, 'b': b, 'c': c, 'f(c)': np.nan, 'Error': abs(b - a)})
                if progress_callback: progress_callback(history[-1])
                return c, history, status_str

            error = abs(b - a) # Error is the size of the interval

            # Record history
            iteration_data = {
                'Iteration': k,
                'a': a,
                'b': b,
                'c': c,
                'f(c)': fc,
                'Error': error
            }
            history.append(iteration_data)
            if progress_callback:
                progress_callback(iteration_data)

            if abs(fc) < self.tol or error < self.tol:
                status_str = "Converged"
                break

            # Re-evaluate fa and fb inside the loop for robustness if 'a' or 'b' change
            # Also ensures func(a) * fc < 0 check is always valid for the *current* 'a'
            if self.f(a) * fc < 0:
                b = c
            else:
                a = c
            
            if k == self.max_iter:
                status_str = "Maximum iterations reached"

        return c, history, status_str