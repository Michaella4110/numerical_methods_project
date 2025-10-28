import numpy as np
import sympy
from sympy.abc import x  # Import x from sympy.abc for symbolic operations
from sympy import sympify, symbols, lambdify
from sympy.core.sympify import SympifyError
import math
import os
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from colorama import Fore, Style, init
import re
from typing import Callable, Optional, Tuple, List, Dict, Any

# Initialize colorama
init(autoreset=True)

SAFE_MATH = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
SAFE_MATH.update({'np': np})


def sanitize(s: str) -> str:
    """Sanitizes a function string by replacing common syntax variations."""
    s = s.strip()
    s = s.replace('^', '**')
    s = s.replace('ln(', 'log(')
    return s


def try_sympy_callable(s: str) -> Tuple[Optional[Callable], Optional[sympy.Expr], Optional[Exception]]:
    """
    Attempts to parse a string into a sympy expression and then a callable function.
    Returns (callable f, sympy_expr, error_or_None).
    """
    try:
        expr = sympify(s)
        # Check if the expression contains only 'x' as a free symbol or is a constant
        if not expr.free_symbols or (len(expr.free_symbols) == 1 and list(expr.free_symbols)[0] == x):
            f = lambdify(x, expr, modules=["numpy", "math"])
            _ = f(1.0)  # quick smoke test
            return f, expr, None
        else:
            return None, None, ValueError("Function must be in terms of 'x' only for symbolic parsing.")
    except Exception as e:
        return None, None, e


def try_eval_callable(s: str) -> Tuple[Optional[Callable], None, Optional[Exception]]:
    """
    Attempts to create a callable function using eval, with a restricted namespace.
    Returns (callable f, None, error_or_None).
    """
    try:
        def f_numeric(xval):
            local = {k: SAFE_MATH[k] for k in SAFE_MATH}
            local['x'] = xval
            return eval(s, {"__builtins__": {}}, local)
        # smoke test outside
        _ = f_numeric(1.0)
        return f_numeric, None, None
    except Exception as e:
        return None, None, e


class RobustFunctionParser:
    """
    A robust function parser that handles various mathematical expressions
    without requiring complex symbolic computation as primary method.
    """
    def __init__(self):
        self.original_string = ""
        
    def parse(self, func_str: str) -> Tuple[Callable, Optional[str], Optional[str]]:
        """
        Parse a function string into a callable function.
        
        Args:
            func_str: Function string (e.g., "x**3 - x - 1", "3x + 2", etc.)
            
        Returns:
            tuple: (function, cleaned_string, error_message)
        """
        self.original_string = func_str.strip()
        
        try:
            # First try the original sympy-based approach
            f_sympy, expr_sympy, error_sympy = try_sympy_callable(self.original_string)
            if f_sympy:
                return f_sympy, str(expr_sympy), None
                
            # If sympy fails, try the robust numeric approach
            cleaned_func = self._clean_function_string(self.original_string)
            func = self._create_safe_function(cleaned_func)
            
            return func, cleaned_func, None
            
        except Exception as e:
            return None, None, f"Failed to parse function: {str(e)}"
    
    def _clean_function_string(self, func_str: str) -> str:
        """
        Clean and normalize the function string to handle various formats.
        """
        if not func_str:
            return "x"  # Default to identity function
            
        # Remove all spaces
        cleaned = func_str.replace(" ", "")
        
        # Handle implicit multiplication
        cleaned = self._handle_implicit_multiplication(cleaned)
        
        # Replace ^ with ** for exponentiation
        cleaned = cleaned.replace('^', '**')
        
        # Ensure proper syntax for common functions
        cleaned = self._normalize_functions(cleaned)
        
        return cleaned
    
    def _handle_implicit_multiplication(self, expr: str) -> str:
        """
        Handle implicit multiplication in expressions.
        """
        # Patterns for implicit multiplication
        patterns = [
            (r'(\d)([a-zA-Z])', r'\1*\2'),        # 3x -> 3*x
            (r'([a-zA-Z])(\d)', r'\1*\2'),        # x3 -> x*3
            (r'([a-zA-Z0-9)])(\()', r'\1*\2'),    # x( -> x*(
            (r'(\))([a-zA-Z])', r'\1*\2'),        # )x -> )*x
            (r'([a-zA-Z])([a-zA-Z])', r'\1*\2'),  # xy -> x*y
        ]
        
        result = expr
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)
            
        return result
    
    def _normalize_functions(self, expr: str) -> str:
        """
        Normalize function names to numpy equivalents.
        """
        replacements = {
            'sin': 'np.sin',
            'cos': 'np.cos', 
            'tan': 'np.tan',
            'exp': 'np.exp',
            'log': 'np.log',
            'ln': 'np.log',
            'sqrt': 'np.sqrt',
            'arcsin': 'np.arcsin',
            'arccos': 'np.arccos',
            'arctan': 'np.arctan',
        }
        
        result = expr
        for old, new in replacements.items():
            result = re.sub(r'\b' + old + r'\b', new, result)
            
        return result
    
    def _create_safe_function(self, expr: str) -> Callable:
        """
        Create a safe callable function from the expression.
        """
        # Default to using 'x' as variable
        variables = set(re.findall(r'\b[a-zA-Z]\b', expr))
        
        if not variables:
            # If no variables found, assume 'x'
            func_expr = expr
            variables = {'x'}
        else:
            func_expr = expr
        
        def safe_eval(x_val, **kwargs):
            # Create safe environment for evaluation
            local_vars = {
                'x': x_val,
                'np': np,
                'pi': np.pi,
                'e': np.e,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'exp': np.exp,
                'log': np.log,
                'sqrt': np.sqrt,
                'arcsin': np.arcsin,
                'arccos': np.arccos,
                'arctan': np.arctan
            }
            
            # Add any additional variables
            local_vars.update(kwargs)
            
            try:
                return eval(func_expr, {"__builtins__": {}}, local_vars)
            except Exception as e:
                raise ValueError(f"Error evaluating '{expr}' at x={x_val}: {str(e)}")
        
        return safe_eval


class FunctionParser:
    """
    A robust parser for mathematical functions, supporting symbolic and numeric evaluation
    and derivative calculation.
    """

    def __init__(self):
        self._original_string = None
        self._sanitized_string = None
        self._sympy_expr = None
        self._callable_sympy = None
        self._callable_numeric = None
        self._last_error = None
        self._derivative_source = None  # "sympy" or "numeric"
        self._robust_parser = RobustFunctionParser()

    def parse(self, s: str) -> Tuple[Optional[Callable], Optional[sympy.Expr], Optional[str]]:
        """
        Parses a function string, attempting symbolic parsing first, then numeric eval fallback.
        Returns (callable f, sympy_expr_or_None, error_message_or_None).
        The callable 'f' is the best available function (sympy or numeric).
        """
        self._original_string = s
        self._sanitized_string = sanitize(s)
        self._last_error = None

        # Try SymPy parsing first
        f_sympy, expr_sympy, error_sympy = try_sympy_callable(self._sanitized_string)
        if f_sympy:
            self._callable_sympy = f_sympy
            self._sympy_expr = expr_sympy
            self._callable_numeric = None  # Prefer sympy if successful
            return self._callable_sympy, self._sympy_expr, None
        else:
            # Fallback to robust numeric parser
            try:
                f_numeric, cleaned_str, error_numeric = self._robust_parser.parse(s)
                if f_numeric:
                    self._callable_numeric = f_numeric
                    self._callable_sympy = None
                    self._sympy_expr = None
                    return self._callable_numeric, None, None
                else:
                    self._last_error = f"SymPy Error: {error_sympy}\nRobust Parser Error: {error_numeric}"
                    self._callable_sympy = None
                    self._callable_numeric = None
                    self._sympy_expr = None
                    return None, None, self._last_error
            except Exception as e:
                self._last_error = f"SymPy Error: {error_sympy}\nRobust Parser Exception: {str(e)}"
                return None, None, self._last_error

    def get_derivative(self) -> Tuple[Optional[Callable], Optional[str]]:
        """
        Returns a callable for the derivative of the parsed function.
        Prioritizes symbolic differentiation, falls back to numeric central-difference.
        Returns (callable d, "sympy"|"numeric"|None)
        """
        if self._sympy_expr:
            try:
                deriv_expr = sympy.diff(self._sympy_expr, x)
                # Derivative of a constant is 0
                if not deriv_expr.free_symbols and self._sympy_expr.free_symbols:  # e.g. diff(5,x) = 0
                    self._derivative_source = "sympy"
                    return lambda val: 0.0, "sympy"
                elif deriv_expr.free_symbols:  # Ensure it's not a constant function that somehow got derivative.
                    f_deriv = lambdify(x, deriv_expr, modules=["numpy", "math"])
                    _ = f_deriv(1.0)  # Smoke test
                    self._derivative_source = "sympy"
                    return f_deriv, "sympy"
                elif not deriv_expr.free_symbols and not self._sympy_expr.free_symbols:  # 5 -> 0, both constant
                    self._derivative_source = "sympy"
                    return lambda val: 0.0, "sympy"
            except Exception as e:
                # If symbolic differentiation fails, try numeric
                print(f"Warning: Symbolic derivative failed ({e}). Falling back to numeric.")
                pass

        # Numeric derivative fallback
        if self._callable_sympy or self._callable_numeric:
            h = 1e-7  # Small step for central difference
            base_func = self._callable_sympy if self._callable_sympy else self._callable_numeric

            def f_prime_numeric(x_val):
                return (base_func(x_val + h) - base_func(x_val - h)) / (2 * h)

            self._derivative_source = "numeric"
            return f_prime_numeric, "numeric"

        self._derivative_source = None
        return None, None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def original_string(self) -> Optional[str]:
        return self._original_string

    @property
    def sanitized_string(self) -> Optional[str]:
        return self._sanitized_string

    @property
    def sympy_expression(self) -> Optional[sympy.Expr]:
        return self._sympy_expr

    @property
    def derivative_source(self) -> Optional[str]:
        return self._derivative_source


# --- SIMPLIFIED COMPATIBILITY FUNCTIONS ---
# These are the key functions your solvers should use

def parse_function(func_string: str) -> Callable:
    """
    Parse a function string into a callable function.
    This is the MAIN function your solvers should use.
    """
    # First try the robust parser
    robust_parser = RobustFunctionParser()
    func, _, robust_error = robust_parser.parse(func_string)
    
    if func is not None:
        return func
    
    # If robust parser fails, try the full FunctionParser
    parser = FunctionParser()
    func, _, error = parser.parse(func_string)
    
    if func is not None:
        return func
    else:
        raise ValueError(f"Failed to parse function '{func_string}': {error}")


def get_derivative(func_string: str) -> Callable:
    """
    Get the derivative of a function string as a callable function.
    Uses numeric differentiation as fallback.
    """
    # First try to get symbolic derivative
    parser = FunctionParser()
    func, _, error = parser.parse(func_string)
    
    if func is not None:
        deriv_func, source = parser.get_derivative()
        if deriv_func is not None:
            return deriv_func
    
    # Fallback to numeric differentiation
    base_func = parse_function(func_string)
    h = 1e-7
    
    def numeric_derivative(x_val):
        return (base_func(x_val + h) - base_func(x_val - h)) / (2 * h)
    
    return numeric_derivative


def parse_function_simple(func_string: str) -> Callable:
    """
    Simple wrapper that always returns a callable function.
    Use this in your solvers to avoid FunctionParser object issues.
    """
    return parse_function(func_string)


# --- ORIGINAL HELPER FUNCTIONS (unchanged) ---

def validate_bisection(func: Callable, a: float, b: float) -> bool:
    """
    Returns True if f(a) * f(b) < 0, else False.
    """
    try:
        if func(a) * func(b) < 0:
            return True
        else:
            return False
    except Exception:  # Catch potential errors during function evaluation
        return False


def is_diagonally_dominant(A: np.ndarray) -> bool:
    """
    Checks if a matrix A is strictly diagonally dominant.
    Returns True or False.
    """
    n = A.shape[0]
    for i in range(n):
        diagonal_element = abs(A[i, i])
        row_sum_of_off_diagonals = np.sum(abs(A[i, :])) - diagonal_element
        if diagonal_element <= row_sum_of_off_diagonals:
            return False
    return True


def expression_to_latex(func_string: str) -> str:
    """
    Converts a function string into its LaTeX representation.
    """
    try:
        expr = sympy.sympify(func_string)
        return sympy.latex(expr)
    except (sympy.SympifyError, ValueError) as e:
        return f"Error converting to LaTeX: {e}"


def print_table_from_log(history_log: List[Dict[str, Any]]) -> None:
    """
    Takes a list of dictionaries (history_log) and prints a clean, aligned table to the console.
    Handles different keys for Bisection, Newton, and Gauss-Seidel.
    Includes color-coding for the 'Error' column.
    """
    if not history_log:
        print("No iteration history to display.")
        return

    # Determine columns based on the first entry (assuming consistent structure)
    headers = list(history_log[0].keys())

    # Custom formatting for 'x_vector' or similar array-like columns
    def format_value(key: str, value: Any) -> str:
        if isinstance(value, np.ndarray):
            return np.array2string(value, precision=6, separator=', ', suppress_small=True)
        elif isinstance(value, (float, np.float64)):
            return f"{value:.8f}"
        return str(value)

    # Calculate maximum column widths
    column_widths = {header: len(header) for header in headers}
    for entry in history_log:
        for header in headers:
            column_widths[header] = max(column_widths[header], len(format_value(header, entry.get(header, ''))))

    # Print header
    header_line = " | ".join(f"{header:<{column_widths[header]}}" for header in headers)
    print(header_line)
    print("-" * len(header_line))

    # Print rows with color-coding for 'Error'
    for entry in history_log:
        row_values = []
        for header in headers:
            value = entry.get(header, '')
            formatted_val = format_value(header, value)

            # Apply color to the Error column
            if header == 'Error' and isinstance(value, (float, np.float64)):
                error = value
                if error > 1:
                    color = Fore.RED   # diverging
                elif error < 1e-3:
                    color = Fore.GREEN  # converging nicely
                else:
                    color = Fore.YELLOW
                row_values.append(color + f"{formatted_val:<{column_widths[header]}}" + Style.RESET_ALL)
            else:
                row_values.append(f"{formatted_val:<{column_widths[header]}}")
        print(" | ".join(row_values))
    print("-" * len(header_line))  # End table with a line


def save_log_to_csv(history_log: List[Dict[str, Any]], filename_prefix: str = "results") -> Optional[str]:
    """
    Saves the iteration history log to a timestamped CSV file in the /data/ folder.
    """
    if not history_log:
        print("No history log to save.")
        return None

    os.makedirs('data', exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = os.path.join('data', f"{filename_prefix}_{timestamp}.csv")

    with open(filename, 'w', newline='') as csvfile:
        fieldnames = list(history_log[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for entry in history_log:
            # Convert numpy arrays in the log to string for CSV
            for key, value in entry.items():
                if isinstance(value, np.ndarray):
                    entry[key] = np.array2string(value, separator=';', precision=8, suppress_small=True)
                elif isinstance(value, (float, np.float64)):
                    entry[key] = f"{value:.10f}"  # Ensure high precision for floats
            writer.writerow(entry)
    print(f"Iteration log saved to {filename}")
    return filename


def save_plot_from_log(history_log: List[Dict[str, Any]], filename_prefix: str = "plot") -> Optional[str]:
    """
    Generates and saves a convergence plot (Error vs. Iteration) to the /plots/ folder.
    """
    if not history_log or 'Error' not in history_log[0]:
        print("No error data in history log to plot.")
        return None

    iterations = [entry['Iteration'] for entry in history_log]
    errors = [entry['Error'] for entry in history_log]

    os.makedirs('plots', exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = os.path.join('plots', f"{filename_prefix}_convergence_{timestamp}.png")

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, errors, marker='o', linestyle='-', color='blue')
    plt.yscale('log')  # Use log scale for error for better visualization of convergence
    plt.title(f"Convergence Plot for {filename_prefix}")
    plt.xlabel("Iteration")
    plt.ylabel("Error (log scale)")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Convergence plot saved to {filename}")
    return filename


def save_log_as_latex(history_log: List[Dict[str, Any]], filename_prefix: str = "iterations") -> Optional[str]:
    """
    Exports the iteration history log to a LaTeX table in a timestamped .tex file.
    Handles different keys for Bisection, Newton, and Gauss-Seidel.
    """
    if not history_log:
        print("No iteration history to save as LaTeX.")
        return None

    os.makedirs('data', exist_ok=True)  # Assuming LaTeX files go into 'data' or a dedicated 'latex' folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = os.path.join('data', f"{filename_prefix}_latex_{timestamp}.tex")

    headers = list(history_log[0].keys())
    # Format headers for LaTeX, replacing underscores and using math mode
    latex_headers = [f"\\textbf{{{h.replace('_', ' ').title()}}}" for h in headers]

    # Determine column alignment based on content type, typically 'c' for centered
    # If a column contains arrays (like x_vector), adjust for multi-line output or specific formatting
    num_columns = len(headers)
    column_format = ' '.join(['c'] * num_columns)  # All columns centered for simplicity

    with open(filename, "w") as f:
        # Preamble for a standalone table
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{amsmath}\n")
        f.write("\\usepackage{amsfonts}\n")
        f.write("\\usepackage{amssymb}\n")
        f.write("\\usepackage{booktabs}\n")  # For better table rules
        f.write("\\begin{document}\n")
        f.write(f"\\section*{{Iteration History ({filename_prefix.replace('_', ' ').title()})}}\n")
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Iteration Log for {filename_prefix.replace('_', ' ').title()}}}\n")
        f.write(f"\\label{{tab:iteration_log_{timestamp}}}\n")
        f.write(f"\\begin{{tabular}}{{{column_format}}}\n")
        f.write("\\toprule\n")  # Top rule from booktabs
        f.write(" & ".join(latex_headers) + " \\\\\n")
        f.write("\\midrule\n")  # Mid rule from booktabs

        for step in history_log:
            row_values = []
            for header in headers:
                value = step.get(header, '')
                if isinstance(value, np.ndarray):
                    # Convert numpy array to a LaTeX-friendly string (e.g., [1.0, 2.0])
                    # For more complex vectors, consider bmatrix or pmatrix
                    formatted_value = '[' + ', '.join(f"{v:.8f}" for v in value) + ']'
                elif isinstance(value, (float, np.float64)):
                    formatted_value = f"{value:.8e}"  # Use scientific notation for floats
                else:
                    formatted_value = str(value)

                # Escape special LaTeX characters and wrap in math mode if numeric/formula-like
                # This is a simplification; a full LaTeX parser would be more robust
                if header in ['x', 'x_n', 'c', 'Error']:  # Common numerical columns
                    formatted_value = f"${formatted_value}$"
                else:
                    formatted_value = formatted_value.replace('_', '\\_')  # Escape underscore if not math

                row_values.append(formatted_value)
            f.write(" & ".join(row_values) + " \\\\\n")

        f.write("\\bottomrule\n")  # Bottom rule from booktabs
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
        f.write("\\end{document}\n")
    print(f"Iteration log saved as LaTeX to {filename}")
    return filename


# Alternative function names for backward compatibility
parse_function_string = parse_function
compute_numerical_derivative = get_derivative
check_diagonal_dominance = is_diagonally_dominant