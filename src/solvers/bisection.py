import numpy as np
import sys
import os
import re
from typing import Tuple, List, Optional, Callable, Union, Dict, Any

# Add the parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import validate_bisection, parse_function_simple


class FunctionParser:
    """
    A robust function parser that handles various mathematical function formats
    with enhanced error handling and support for common mathematical operations.
    """
    
    def __init__(self):
        self.original_string = ""
        self._supported_functions = {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
            'exp': np.exp, 'log': np.log, 'log10': np.log10,
            'sqrt': np.sqrt, 'abs': np.abs, 'pi': np.pi,
            'e': np.e, 'inf': np.inf
        }
        
    def parse(self, func_str: str) -> Tuple[Callable, Optional[str], Optional[str]]:
        """
        Parse a function string into a callable function with comprehensive validation.
        """
        self.original_string = func_str.strip()
        
        if not self.original_string:
            return None, None, "Function string cannot be empty"
            
        try:
            # Clean and validate the function string
            cleaned_func = self._clean_function_string(self.original_string)
            
            # Validate mathematical expression
            validation_error = self._validate_expression(cleaned_func)
            if validation_error:
                return None, None, validation_error
            
            # Create a safe, callable function
            func = self._create_function(cleaned_func)
            
            return func, cleaned_func, None
            
        except Exception as e:
            return None, None, f"Failed to parse function: {str(e)}"
    
    def _clean_function_string(self, func_str: str) -> str:
        """Clean and normalize the function string with enhanced pattern matching."""
        # Remove any spaces and normalize case for certain functions
        cleaned = func_str.replace(" ", "").lower()
        
        # Handle implicit multiplication patterns
        patterns = [
            (r'(\d)([a-zA-Z])', r'\1*\2'),        # 3x -> 3*x
            (r'([a-zA-Z])(\d)', r'\1*\2'),        # x3 -> x*3
            (r'([a-zA-Z0-9)])(\()', r'\1*\2'),    # x( -> x*(
            (r'(\))([a-zA-Z])', r'\1*\2'),        # )x -> )*x
            (r'([a-zA-Z])([a-zA-Z])', r'\1*\2'),  # xy -> x*y
        ]
        
        for pattern, replacement in patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Replace ^ with ** for exponentiation
        cleaned = cleaned.replace('^', '**')
        
        # Handle special cases like e^x, pi*x, etc.
        cleaned = re.sub(r'\b(e|pi)([a-zA-Z(])', r'\1*\2', cleaned)
        
        return cleaned
    
    def _validate_expression(self, func_str: str) -> Optional[str]:
        """Validate the mathematical expression for safety and correctness."""
        # Check for potentially dangerous operations
        dangerous_patterns = [
            r'__', r'import', r'eval', r'exec', r'compile', r'open',
            r'file', r'os\.', r'sys\.', r'__builtins__'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, func_str, re.IGNORECASE):
                return f"Potentially dangerous operation detected: {pattern}"
        
        # Check for balanced parentheses
        stack = []
        for char in func_str:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if not stack:
                    return "Unbalanced parentheses"
                stack.pop()
        
        if stack:
            return "Unbalanced parentheses"
            
        return None
    
    def _create_function(self, func_str: str) -> Callable:
        """Create a safe, callable function from the cleaned string."""
        # Extract variable names (single character variables)
        variables = set(re.findall(r'\b[a-zA-Z]\b', func_str))
        
        if not variables:
            # Default to 'x' if no variables found
            variables = {'x'}
            func_expr = func_str
        else:
            func_expr = func_str
        
        # Create safe evaluation environment
        def safe_eval(x: float, **kwargs) -> float:
            local_vars = {'x': x, 'np': np}
            local_vars.update(self._supported_functions)
            local_vars.update(kwargs)
            
            try:
                result = eval(func_expr, {"__builtins__": {}}, local_vars)
                # Ensure result is numeric
                if not isinstance(result, (int, float, complex)):
                    raise ValueError(f"Function returned non-numeric value: {type(result)}")
                return float(result)
            except ZeroDivisionError:
                return float('inf') if x > 0 else float('-inf')
            except ValueError as e:
                if "math domain error" in str(e).lower():
                    return float('nan')
                raise
            except Exception as e:
                raise ValueError(f"Error evaluating function at x={x}: {str(e)}")
        
        return safe_eval


class BisectionSolver:
    """
    Enhanced Bisection method solver with comprehensive error handling,
    auto-bracket detection, and detailed progress tracking.
    """
    
    def __init__(self, func_string: str, tol: float = 1e-8, max_iter: int = 100):
        """
        Initialize the bisection solver.
        """
        self.parser = FunctionParser()
        self.func_string = func_string
        
        # Parse the function
        self.f, self.cleaned_func, self.parse_error = self.parser.parse(func_string)
        if self.parse_error:
            raise ValueError(f"Function parsing failed: {self.parse_error}")
            
        self.tol = tol
        self.max_iter = max_iter
        self._last_bracket = None

    def validate_bracket(self, a: float, b: float) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Enhanced bracket validation with detailed error reporting and endpoint root detection.
        
        Returns:
            tuple: (is_valid, error_message, root_at_endpoint)
        """
        if a >= b:
            return False, "Invalid interval: a must be less than b", None
            
        try:
            fa = self.f(a)
            fb = self.f(b)
        except Exception as e:
            return False, f"Function evaluation failed: {e}", None
        
        # Handle NaN and infinity
        if np.isnan(fa) or np.isnan(fb):
            return False, "Function evaluation resulted in NaN", None
            
        if np.isinf(fa) or np.isinf(fb):
            return False, "Function evaluation resulted in infinity", None
        
        # ✅ FIXED: Handle zero endpoints - check if root is exactly at a or b
        if abs(fa) < self.tol:
            self._last_bracket = (a, a)
            return True, f"Root found exactly at a = {a}", a
        if abs(fb) < self.tol:
            self._last_bracket = (b, b)
            return True, f"Root found exactly at b = {b}", b

        # ✅ Normal sign check
        if fa * fb < 0:
            self._last_bracket = (a, b)
            return True, None, None
        else:
            return False, "Function values at endpoints have same sign (no root bracketed)", None

    def auto_bracket_search(
        self,
        start_a: float,
        end_b: float,
        steps: int = 200,
        progress_callback: Optional[Callable[[dict], None]] = None,
        abort_flag: Optional[Callable[[], bool]] = None
    ) -> Tuple[Optional[float], Optional[float], str]:
        """
        Enhanced auto-bracket search with progress tracking and abort support.
        """
        xs = np.linspace(start_a, end_b, steps)
        last_f = None
        
        try:
            last_f = self.f(xs[0])
            # Check first point for root
            if abs(last_f) < self.tol:
                return xs[0], xs[0], "Root found at start of search range"
        except Exception:
            last_f = np.nan

        total = len(xs) - 1
        for i in range(1, len(xs)):
            if abort_flag and abort_flag():
                return None, None, "Aborted by user"
                
            x0 = xs[i-1]
            x1 = xs[i]
            try:
                f1 = self.f(x1)
                # Check current point for root
                if abs(f1) < self.tol:
                    return x1, x1, "Root found during bracket search"
            except Exception:
                f1 = np.nan

            pct = int((i / total) * 100)
            
            # Report progress
            if progress_callback:
                progress_callback({
                    'phase': 'bracket_search',
                    'step': i,
                    'x0': x0,
                    'x1': x1,
                    'f0': last_f,
                    'f1': f1,
                    'progress': pct,
                    'status': 'searching'
                })

            if np.isnan(last_f):
                last_f = f1
                continue
            if np.isnan(f1):
                continue

            if last_f * f1 < 0:
                # Found bracket
                if progress_callback:
                    progress_callback({
                        'phase': 'bracket_search',
                        'step': i,
                        'x0': x0,
                        'x1': x1,
                        'f0': last_f,
                        'f1': f1,
                        'progress': 100,
                        'status': 'found'
                    })
                return x0, x1, "Found valid bracket"

            last_f = f1

        if progress_callback:
            progress_callback({
                'phase': 'bracket_search',
                'step': total,
                'x0': xs[-2],
                'x1': xs[-1],
                'f0': last_f,
                'f1': np.nan,
                'progress': 100,
                'status': 'not_found'
            })
        return None, None, "No valid bracket found in range"

    def solve(
        self, 
        a: float, 
        b: float, 
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Tuple[Optional[float], List[Dict[str, Any]], str]:
        """
        Enhanced bisection method implementation with comprehensive progress tracking
        and endpoint root detection.
        """
        a = float(a)
        b = float(b)
        history: List[Dict[str, Any]] = []
        status_str = "Maximum iterations reached"

        # ✅ FIXED: Enhanced bracket validation with endpoint root detection
        is_valid, error_msg, endpoint_root = self.validate_bracket(a, b)
        
        if not is_valid:
            return None, history, error_msg
        
        # ✅ If root is found exactly at an endpoint, return immediately
        if endpoint_root is not None:
            # Create a single history entry for the endpoint root
            history.append({
                'Iteration': 0,
                'a': endpoint_root,
                'b': endpoint_root,
                'c': endpoint_root,
                'f(c)': 0.0,
                'Error': 0.0,
                'Interval_Size': 0.0,
                'phase': 'bisection',
                'endpoint_root': True
            })
            
            if progress_callback:
                progress_callback(history[-1])
                
            return endpoint_root, history, f"Root found exactly at endpoint: {endpoint_root}"

        # Initialize variables for normal bisection
        a_curr, b_curr = a, b
        c_prev = None
        
        try:
            fa = self.f(a_curr)
            fb = self.f(b_curr)
        except Exception as e:
            return None, history, f"Function evaluation failed at endpoints: {e}"

        for k in range(1, self.max_iter + 1):
            c = (a_curr + b_curr) / 2
            
            try:
                fc = self.f(c)
            except Exception as e:
                status_str = f"Function evaluation failed at iteration {k}: {e}"
                history.append(self._create_iteration_data(k, a_curr, b_curr, c, np.nan, abs(b_curr - a_curr)))
                if progress_callback: 
                    progress_callback(history[-1])
                return c, history, status_str

            error = abs(b_curr - a_curr)
            
            # Create iteration data
            iteration_data = self._create_iteration_data(k, a_curr, b_curr, c, fc, error, c_prev)
            history.append(iteration_data)
            
            if progress_callback:
                progress_callback(iteration_data)

            # Check convergence
            if abs(fc) < self.tol or error < self.tol:
                status_str = f"Converged after {k} iterations"
                break

            # Update bracket
            try:
                fa_current = self.f(a_curr)
                if fa_current * fc < 0:
                    b_curr = c
                else:
                    a_curr = c
            except Exception as e:
                status_str = f"Function evaluation failed during bracket update: {e}"
                break
            
            c_prev = c
            
            if k == self.max_iter:
                status_str = f"Maximum iterations ({self.max_iter}) reached"

        root = (a_curr + b_curr) / 2
        return root, history, status_str

    def solve_with_auto_bracket(
        self,
        a: float,
        b: float,
        auto_bracket_enabled: bool = False,
        search_start: float = None,
        search_end: float = None,
        search_steps: int = 200,
        progress_callback: Optional[Callable[[dict], None]] = None,
        abort_flag: Optional[Callable[[], bool]] = None
    ) -> Tuple[Optional[float], List[dict], str]:
        """
        Enhanced solver with auto-bracket capability.
        """
        # 1) Validate initial bracket with endpoint detection
        is_valid, error_msg, endpoint_root = self.validate_bracket(a, b)
        
        if is_valid and endpoint_root is not None:
            # Root found at endpoint - return immediately
            return endpoint_root, [{
                'Iteration': 0,
                'a': endpoint_root,
                'b': endpoint_root,
                'c': endpoint_root,
                'f(c)': 0.0,
                'Error': 0.0,
                'Interval_Size': 0.0,
                'phase': 'bisection',
                'endpoint_root': True
            }], f"Root found exactly at endpoint: {endpoint_root}"
            
        if is_valid:
            # Normal flow - use existing solve method
            return self.solve(a, b, progress_callback)

        # 2) If bracket invalid and auto-bracket disabled, inform caller
        if not auto_bracket_enabled:
            return None, [], error_msg

        # Determine search bounds: prefer provided values otherwise expand around given a,b
        if search_start is None: 
            search_start = min(a, b) - abs(b - a) * 5
        if search_end is None:   
            search_end = max(a, b) + abs(b - a) * 5

        # 3) Run auto-bracket search
        found_a, found_b, bracket_status = self.auto_bracket_search(
            search_start, search_end, steps=search_steps,
            progress_callback=progress_callback,
            abort_flag=abort_flag
        )

        if not found_a or not found_b:
            return None, [], f"Bracket search failed: {bracket_status}"

        # 4) Continue with normal bisection using the found bracket
        return self.solve(found_a, found_b, progress_callback)

    def _create_iteration_data(
        self, 
        iteration: int, 
        a: float, 
        b: float, 
        c: float, 
        fc: float, 
        error: float,
        c_prev: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create standardized iteration data dictionary."""
        data = {
            'Iteration': iteration,
            'a': a,
            'b': b,
            'c': c,
            'f(c)': fc,
            'Error': error,
            'Interval_Size': abs(b - a),
            'phase': 'bisection'  # Mark as bisection phase
        }
        
        if c_prev is not None:
            data['Relative_Error'] = abs(c - c_prev) / abs(c) if c != 0 else float('inf')
        
        return data


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
    Enhanced compatibility function for GUI with improved error handling.
    """
    # Create a solver with dummy function string but use provided callable
    solver = BisectionSolver("x", tol, max_iter)
    solver.f = func  # Override with provided callable
    
    # Adapt progress callback format
    def adapted_progress_callback(iteration_data: Dict[str, Any]) -> None:
        if progress_callback:
            iteration = iteration_data['Iteration']
            sample_value = iteration_data['c']
            error = iteration_data['Error']
            progress_callback(iteration, sample_value, error)
    
    # Call the solver with adapted callback
    return solver.solve(a, b, adapted_progress_callback)


def solve_with_auto_bracket(
    func: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    auto_bracket_enabled: bool = False,
    search_start: float = None,
    search_end: float = None,
    search_steps: int = 200,
    progress_callback: Optional[Callable[[dict], None]] = None,
    abort_flag: Optional[Callable[[], bool]] = None
) -> Tuple[Optional[float], List[dict], str]:
    """
    Enhanced solver with auto-bracket capability for GUI integration.
    """
    # Create a solver with dummy function string but use provided callable
    solver = BisectionSolver("x", tol, max_iter)
    solver.f = func  # Override with provided callable
    
    return solver.solve_with_auto_bracket(
        a, b, auto_bracket_enabled, search_start, search_end, 
        search_steps, progress_callback, abort_flag
    )


def solve_from_string(
    func_string: str,
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    progress_callback: Optional[Callable[[int, float, float], None]] = None
) -> Tuple[Optional[float], List[dict], str]:
    """
    Enhanced solver function that accepts function string directly.
    """
    try:
        # Use the enhanced FunctionParser
        parser = FunctionParser()
        f, cleaned_func, error = parser.parse(func_string)
        
        if error:
            return None, [], f"Function parsing error: {error}"
        
        # Create solver instance
        solver = BisectionSolver(func_string, tol, max_iter)
        solver.f = f  # Use the parsed function
        
        # Adapt progress callback
        def adapted_progress_callback(iteration_data: Dict[str, Any]) -> None:
            if progress_callback:
                iteration = iteration_data['Iteration']
                sample_value = iteration_data['c']
                error = iteration_data['Error']
                progress_callback(iteration, sample_value, error)
        
        return solver.solve(a, b, adapted_progress_callback)
        
    except Exception as e:
        return None, [], f"Error initializing solver: {str(e)}"


# Alternative simple implementation for backward compatibility
def simple_bisection(
    func_string: str,
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Tuple[Optional[float], List[dict], str]:
    """Simple bisection implementation for quick use cases."""
    return solve_from_string(func_string, a, b, tol, max_iter)


# For backward compatibility
bisection = BisectionSolver


# Test the fix
if __name__ == "__main__":
    def test_endpoint_fix():
        print("Testing endpoint root detection fix...")
        
        # Test case 1: cos(x) on [0, π/2] - root at b
        solver = BisectionSolver("cos(x)")
        root, history, status = solver.solve(0, np.pi/2)
        print(f"cos(x) on [0, π/2]: Root = {root}, Status = {status}")
        
        # Test case 2: (x-2)^2 on [2, 4] - root at a
        solver2 = BisectionSolver("(x-2)**2")
        root2, history2, status2 = solver2.solve(2, 4)
        print(f"(x-2)^2 on [2, 4]: Root = {root2}, Status = {status2}")
        
        # Test case 3: Normal case - x^2 - 4 on [1, 3]
        solver3 = BisectionSolver("x**2 - 4")
        root3, history3, status3 = solver3.solve(1, 3)
        print(f"x^2-4 on [1, 3]: Root = {root3}, Status = {status3}, Iterations = {len(history3)}")
        
        # Test case 4: Invalid bracket - x^2 + 1 on [1, 2]
        solver4 = BisectionSolver("x**2 + 1")
        root4, history4, status4 = solver4.solve(1, 2)
        print(f"x^2+1 on [1, 2]: Root = {root4}, Status = {status4}")
    
    test_endpoint_fix()
