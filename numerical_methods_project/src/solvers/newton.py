import numpy as np
import sys
import os

# Add the parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import parse_function_simple, get_derivative
from typing import List, Callable, Tuple, Optional

class NewtonSolver:
    """
    Solves for roots of a single-variable function using Newton's method.
    Supports symbolic and numeric derivatives, and includes damping for stability.
    """
    def __init__(self, func_string: str, tol: float = 1e-8, max_iter: int = 100):
        # Use the simple function parser that always returns a callable
        self.f = parse_function_simple(func_string)
        self.df = get_derivative(func_string)
        self.tol = tol
        self.max_iter = max_iter

    def _compute_derivative(self, x_val: float) -> float:
        """Helper to compute derivative, handling potential errors from the callable."""
        try:
            return self.df(x_val)
        except Exception:
            # Fallback if the derived function itself fails at a point
            h_fallback = 1e-6
            return (self.f(x_val + h_fallback) - self.f(x_val - h_fallback)) / (2 * h_fallback)

    def solve(
        self, 
        x0: float, 
        progress_callback: Optional[Callable[[dict], None]] = None
    ) -> Tuple[float, List[dict], str]:
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
        history: List[dict] = []
        status_str = "Maximum iterations reached"
        
        for k in range(1, self.max_iter + 1):
            fx = self.f(x)

            # Record history for current x before potential update
            iteration_data = {
                'Iteration': k,
                'x_n': x,
                'f(x_n)': fx,
                'Error': abs(fx)  # Error defined as |f(x_n)| for convergence check
            }
            
            if abs(fx) < self.tol:
                status_str = "Converged"
                history.append(iteration_data)  # Append final converged state
                if progress_callback:
                    progress_callback(iteration_data)
                break

            fpx = self._compute_derivative(x)
            
            # Handle derivative near zero or undefined
            if fpx is None or abs(fpx) < 1e-12:  # Add a threshold for 'near zero'
                # Before failing, try numeric derivative with a slightly larger h if using symbolic
                h_try = 1e-6
                fpx_numeric = (self.f(x + h_try) - self.f(x - h_try)) / (2 * h_try)
                if abs(fpx_numeric) < 1e-14:  # Even larger h didn't help
                    status_str = "Failed: Derivative near zero or undefined."
                    history.append(iteration_data)
                    if progress_callback:
                        progress_callback(iteration_data)
                    break
                else:
                    fpx = fpx_numeric  # Use the numeric one

            step = fx / fpx
            lam = 1.0  # Damping parameter

            # Damping (backtracking line search)
            x_new_candidate = x - lam * step
            f_x_new_candidate = None  # Initialize outside loop

            # Limit the number of backtracking steps (e.g., 8)
            for _ in range(8):
                try:
                    f_x_new_candidate = self.f(x_new_candidate)
                    if abs(f_x_new_candidate) <= abs(fx):
                        break  # Found a better point
                except Exception:  # Catch cases where f(x_new_candidate) might be undefined
                    pass  # Try a smaller lambda
                lam *= 0.5
                x_new_candidate = x - lam * step
            
            x_new = x_new_candidate  # This is the accepted new point after damping

            # Clamp huge steps to prevent blowing up (if damping wasn't enough)
            max_step_abs = max(1.0, abs(x) * 10)  # max_step should be an absolute value
            if abs(x_new - x) > max_step_abs:
                # Limit step size but maintain direction
                step_direction = np.sign(x_new - x) if (x_new - x) != 0 else 1
                x_new = x + step_direction * max_step_abs

            if np.isnan(x_new) or np.isinf(x_new):
                status_str = "Failed: x became NaN or Inf."
                history.append(iteration_data)  # Append state before failure
                if progress_callback:
                    progress_callback(iteration_data)
                break
            
            # Update iteration data with derivative and step info
            iteration_data.update({
                "f'(x_n)": fpx,
                'x_n+1': x_new,
                'Step': step,
                'Damping': lam
            })
            
            history.append(iteration_data)
            if progress_callback:
                progress_callback(iteration_data)

            x = x_new

            if k == self.max_iter:
                status_str = "Maximum iterations reached"

        return x, history, status_str


# =============================================================================
# COMPATIBILITY FUNCTIONS FOR GUI
# =============================================================================

def solve(
    func: Callable[[float], float],
    func_prime: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    progress_callback: Optional[Callable[[int, float, float], None]] = None
) -> Tuple[float, List[dict], str]:
    """
    Compatibility function for GUI - matches the expected signature.
    
    Args:
        func: The function for which to find the root
        func_prime: Derivative of the function
        x0: Initial guess
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        progress_callback: Callback with signature (iteration, sample_value, error)
        
    Returns:
        tuple: (root, history, status_message)
    """
    # For compatibility with existing GUI, create a solver with dummy function string
    # but override with provided callables
    solver = NewtonSolver("x", tol, max_iter)
    solver.f = func
    solver.df = func_prime
    
    # Adapt progress callback format
    def adapted_progress_callback(iteration_data: dict) -> None:
        if progress_callback:
            iteration = iteration_data['Iteration']
            sample_value = iteration_data['x_n']  # Use 'x_n' as the sample value
            error = iteration_data['Error']
            progress_callback(iteration, sample_value, error)
    
    # Call the solver with adapted callback
    return solver.solve(x0, adapted_progress_callback)


# Alternative simple solver function that accepts function string directly
def solve_from_string(
    func_string: str,
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    progress_callback: Optional[Callable[[int, float, float], None]] = None
) -> Tuple[Optional[float], List[dict], str]:
    """
    Solve using function string directly - avoids FunctionParser object issues.
    """
    try:
        f = parse_function_simple(func_string)
        df = get_derivative(func_string)
        
        # Simple Newton implementation
        history = []
        x = float(x0)
        
        for k in range(1, max_iter + 1):
            try:
                fx = f(x)
            except Exception as e:
                return None, history, f"Function evaluation failed at x={x}: {e}"
                
            error = abs(fx)
            
            iteration_data = {
                'Iteration': k,
                'x_n': x,
                'f(x_n)': fx,
                'Error': error
            }
            
            if error < tol:
                history.append(iteration_data)
                if progress_callback:
                    progress_callback(k, x, error)
                return x, history, "Converged"
                
            try:
                fpx = df(x)
            except Exception:
                # Fallback to numeric derivative
                h = 1e-7
                fpx = (f(x + h) - f(x - h)) / (2 * h)
                
            if abs(fpx) < 1e-12:
                return x, history, "Derivative near zero (stopped)"
                
            step = fx / fpx
            x_new = x - step
            
            iteration_data.update({
                "f'(x_n)": fpx,
                'x_n+1': x_new,
                'Step': step
            })
            
            history.append(iteration_data)
            if progress_callback:
                progress_callback(k, x, error)
                
            x = x_new
            
        return x, history, "Maximum iterations reached"
        
    except Exception as e:
        return None, [], f"Error: {str(e)}"


# For backward compatibility
newton_method = NewtonSolver