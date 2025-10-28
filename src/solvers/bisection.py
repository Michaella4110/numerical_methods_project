import numpy as np
import sys
import os
import re

# Add the parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import validate_bisection, parse_function_simple
from typing import Tuple, List, Optional, Callable, Union

class FunctionParser:
    """
    A more robust function parser that handles various function formats
    without requiring sympy simplification.
    """
    def __init__(self):
        self.original_string = ""
        
    def parse(self, func_str: str) -> Tuple[Callable, Optional[str], Optional[str]]:
        """
        Parse a function string into a callable function.
        
        Args:
            func_str: Function string (e.g., "x**3 - x - 1")
            
        Returns:
            tuple: (function, simplified_string, error_message)
        """
        self.original_string = func_str.strip()
        
        try:
            # Clean the function string
            cleaned_func = self._clean_function_string(self.original_string)
            
            # Create a safe function that handles various formats
            func = self._create_function(cleaned_func)
            
            return func, cleaned_func, None
            
        except Exception as e:
            return None, None, f"Failed to parse function: {str(e)}"
    
    def _clean_function_string(self, func_str: str) -> str:
        """
        Clean and normalize the function string.
        """
        # Remove any spaces
        cleaned = func_str.replace(" ", "")
        
        # Handle implicit multiplication (e.g., 3x -> 3*x, x( -> x*(, )x -> )*x)
        cleaned = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', cleaned)  # 3x -> 3*x
        cleaned = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', cleaned)  # x3 -> x*3
        cleaned = re.sub(r'([a-zA-Z0-9)])(\()', r'\1*\2', cleaned)  # x( -> x*(
        cleaned = re.sub(r'(\))([a-zA-Z])', r'\1*\2', cleaned)  # )x -> )*x
        
        # Handle multiple variables (e.g., xy -> x*y)
        cleaned = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', cleaned)
        
        # Replace ^ with ** for exponentiation
        cleaned = cleaned.replace('^', '**')
        
        return cleaned
    
    def _create_function(self, func_str: str) -> Callable:
        """
        Create a callable function from the cleaned string.
        """
        # Extract variable names (single character variable names)
        variables = set(re.findall(r'\b[a-zA-Z]\b', func_str))
        
        if not variables:
            # If no variables found, assume 'x'
            variables = {'x'}
            func_expr = func_str
        else:
            # Use the first variable found (or multiple if needed)
            func_expr = func_str
        
        # Create a safe evaluation function
        def safe_eval(x, **kwargs):
            # Handle both single variable and multiple variables
            local_vars = {'x': x, 'np': np, 'exp': np.exp, 'sin': np.sin, 
                         'cos': np.cos, 'tan': np.tan, 'log': np.log, 
                         'sqrt': np.sqrt, 'pi': np.pi, 'e': np.e}
            
            # Add any additional variables from kwargs
            local_vars.update(kwargs)
            
            try:
                return eval(func_expr, {"__builtins__": {}}, local_vars)
            except Exception as e:
                raise ValueError(f"Error evaluating function at x={x}: {str(e)}")
        
        return safe_eval


class BisectionSolver:
    """
    Solves for roots of a single-variable function using the Bisection method.
    Includes methods for validating brackets and attempting to auto-find brackets.
    """
    def __init__(self, func_string: str, tol: float = 1e-8, max_iter: int = 100):
        # Use the simple function parser that always returns a callable
        self.f = parse_function_simple(func_string)
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
            return False  # If function evaluation fails, it's not a valid bracket

    def auto_bracket_search(self, start_a: float, end_b: float, steps: int = 200) -> Tuple[Optional[float], Optional[float]]:
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
        
        fa = np.nan  # Initialize as NaN to handle potential evaluation issues at xs[0]
        try:
            fa = self.f(xs[0])
        except Exception:
            pass  # Keep fa as NaN if evaluation fails

        for i in range(1, len(xs)):
            current_x = xs[i]
            fb = np.nan  # Initialize as NaN
            try:
                fb = self.f(current_x)
            except Exception:
                pass  # Keep fb as NaN if evaluation fails

            # Handle cases where function evaluation might yield NaN
            if np.isnan(fa):
                fa = fb  # If previous fa was bad, make current fb the new fa for next iteration
                continue
            if np.isnan(fb):
                continue  # If current fb is bad, skip this interval, keep current fa for next iteration

            if fa * fb < 0:
                return xs[i-1], current_x
            
            fa = fb  # Move fb to fa for the next iteration

        return None, None

    def solve(
        self, 
        a: float, 
        b: float, 
        progress_callback: Optional[Callable[[dict], None]] = None
    ) -> Tuple[Optional[float], List[dict], str]:
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
        history: List[dict] = []
        status_str = "Maximum iterations reached"

        if not self.validate_bracket(a, b):
            return None, history, "Initial interval does not bracket a root (f(a) * f(b) >= 0)."

        # Initial values for f(a) and f(b)
        try:
            fa_initial = self.f(a)
            fb_initial = self.f(b)
        except Exception as e:
            return None, history, f"Function evaluation failed at interval endpoints: {e}"

        c = a  # Initialize c, will be updated in the loop
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

            error = abs(b - a)  # Error is the size of the interval

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
            try:
                if self.f(a) * fc < 0:
                    b = c
                else:
                    a = c
            except Exception:
                status_str = f"Function evaluation failed during iteration {k}. Stopping."
                break
            
            if k == self.max_iter:
                status_str = "Maximum iterations reached"

        return c, history, status_str


# =============================================================================
# COMPATIBILITY FUNCTIONS FOR GUI
# =============================================================================

def solve(
    func: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    progress_callback: Optional[Callable[[int, float, float], None]] = None
) -> Tuple[Optional[float], List[dict], str]:
    """
    Compatibility function for GUI - matches the expected signature.
    
    Args:
        func: The function for which to find the root
        a: Left endpoint of interval
        b: Right endpoint of interval
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        progress_callback: Callback with signature (iteration, sample_value, error)
        
    Returns:
        tuple: (root, history, status_message)
    """
    # Create a function string representation for the parser
    # Since we have a callable, we need to create a wrapper
    class TempFunctionParser:
        def __init__(self, func_callable):
            self.original_string = "user_provided_function"
            self._callable_numeric = func_callable
            
        def parse(self, s):
            return self._callable_numeric, None, None
            
    # For compatibility, create a solver with a dummy function string
    # but use the provided callable directly
    solver = BisectionSolver("x", tol, max_iter)
    solver.f = func  # Override with the provided callable
    
    # Adapt progress callback format
    def adapted_progress_callback(iteration_data: dict) -> None:
        if progress_callback:
            iteration = iteration_data['Iteration']
            sample_value = iteration_data['c']  # Use 'c' as the sample value
            error = iteration_data['Error']
            progress_callback(iteration, sample_value, error)
    
    # Call the solver with adapted callback
    return solver.solve(a, b, adapted_progress_callback)


# Alternative simple solver function that accepts function string directly
def solve_from_string(
    func_string: str,
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    progress_callback: Optional[Callable[[int, float, float], None]] = None
) -> Tuple[Optional[float], List[dict], str]:
    """
    Solve using function string directly - avoids FunctionParser object issues.
    """
    try:
        f = parse_function_simple(func_string)
        
        # Simple bisection implementation
        history = []
        a_curr, b_curr = float(a), float(b)
        
        # Check initial bracket
        try:
            fa = f(a_curr)
            fb = f(b_curr)
        except Exception as e:
            return None, [], f"Function evaluation failed at endpoints: {e}"
            
        if fa * fb >= 0:
            return None, [], "Initial interval does not bracket a root (f(a) * f(b) >= 0)."
        
        for k in range(1, max_iter + 1):
            c = (a_curr + b_curr) / 2
            
            try:
                fc = f(c)
            except Exception as e:
                return None, history, f"Function evaluation failed at c={c}: {e}"
                
            error = abs(b_curr - a_curr)
            
            iteration_data = {
                'Iteration': k,
                'a': a_curr,
                'b': b_curr,
                'c': c,
                'f(c)': fc,
                'Error': error
            }
            history.append(iteration_data)
            
            if progress_callback:
                progress_callback(k, c, error)
                
            if abs(fc) < tol or error < tol:
                return c, history, "Converged"
                
            if f(a_curr) * fc < 0:
                b_curr = c
            else:
                a_curr = c
                
        return (a_curr + b_curr) / 2, history, "Maximum iterations reached"
        
    except Exception as e:
        return None, [], f"Error: {str(e)}"


# For backward compatibility
bisection = BisectionSolver