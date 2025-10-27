import numpy as np
import sympy
from sympy.abc import x
import os
from datetime import datetime
import csv
import matplotlib.pyplot as plt

def parse_function(func_string):
    """
    Takes a string like "x**2 - 3" and returns a callable function f(x).
    Uses sympy for safe parsing and evaluation.
    """
    try:
        expr = sympy.sympify(func_string)
        # Check if the expression contains only 'x' as a free symbol
        if expr.free_symbols and len(expr.free_symbols) == 1 and list(expr.free_symbols)[0] == x:
            return sympy.lambdify(x, expr, 'numpy')
        elif not expr.free_symbols: # Constant function
             return sympy.lambdify(x, expr, 'numpy')
        else:
            raise ValueError("Function must be in terms of 'x' only.")
    except (sympy.SympifyError, ValueError) as e:
        raise ValueError(f"Invalid function string: {e}")

def get_derivative(func_string):
    """
    Takes the function string and returns a callable for its derivative, f'(x).
    """
    try:
        expr = sympy.sympify(func_string)
        if expr.free_symbols and len(expr.free_symbols) == 1 and list(expr.free_symbols)[0] == x:
            deriv_expr = sympy.diff(expr, x)
            return sympy.lambdify(x, deriv_expr, 'numpy')
        elif not expr.free_symbols: # Derivative of a constant is 0
             return lambda val: 0.0
        else:
            raise ValueError("Function must be in terms of 'x' only to get derivative.")
    except (sympy.SympifyError, ValueError) as e:
        raise ValueError(f"Could not compute derivative: {e}")

def validate_bisection(func, a, b):
    """
    Returns True if f(a) * f(b) < 0, else False.
    """
    try:
        if func(a) * func(b) < 0:
            return True
        else:
            return False
    except Exception: # Catch potential errors during function evaluation
        return False

def is_diagonally_dominant(A):
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

def print_table_from_log(history_log):
    """
    Takes a list of dictionaries (history_log) and prints a clean, aligned table to the console.
    Handles different keys for Bisection, Newton, and Gauss-Seidel.
    """
    if not history_log:
        print("No iteration history to display.")
        return

    # Determine columns based on the first entry (assuming consistent structure)
    headers = list(history_log[0].keys())
    
    # Custom formatting for 'x_vector' or similar array-like columns
    def format_value(key, value):
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

    # Print rows
    for entry in history_log:
        row_values = []
        for header in headers:
            value = entry.get(header, '')
            row_values.append(f"{format_value(header, value):<{column_widths[header]}}")
        print(" | ".join(row_values))
    print("-" * len(header_line)) # End table with a line


def save_log_to_csv(history_log, filename_prefix="results"):
    """
    Saves the iteration history log to a timestamped CSV file in the /data/ folder.
    """
    if not history_log:
        print("No history log to save.")
        return

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
                     entry[key] = f"{value:.10f}" # Ensure high precision for floats
            writer.writerow(entry)
    print(f"Iteration log saved to {filename}")
    return filename

def save_plot_from_log(history_log, filename_prefix="plot"):
    """
    Generates and saves a convergence plot (Error vs. Iteration) to the /plots/ folder.
    """
    if not history_log or 'Error' not in history_log[0]:
        print("No error data in history log to plot.")
        return

    iterations = [entry['Iteration'] for entry in history_log]
    errors = [entry['Error'] for entry in history_log]

    os.makedirs('plots', exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = os.path.join('plots', f"{filename_prefix}_convergence_{timestamp}.png")

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, errors, marker='o', linestyle='-', color='blue')
    plt.yscale('log') # Use log scale for error for better visualization of convergence
    plt.title(f"Convergence Plot for {filename_prefix}")
    plt.xlabel("Iteration")
    plt.ylabel("Error (log scale)")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Convergence plot saved to {filename}")
    return filename