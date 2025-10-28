import tkinter as tk
from tkinter import messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import sys
import os
import re # For parsing equations

# Ensure src directory is in path for imports
sys.path.append(os.path.dirname(__file__))

import bisection 
import newton
import utils # For parse_function, get_derivative, is_diagonally_dominant
import gauss_seidel # Assuming gauss_seidel.py contains the solver function

class NumericalMethodsGUI:
    def __init__(self, master):
        self.master = master
        master.title("Numerical Methods Solver")

        self.function_str = tk.StringVar(master)
        self.a_val = tk.StringVar(master)
        self.b_val = tk.StringVar(master)
        self.x0_val = tk.StringVar(master)
        self.tolerance = tk.StringVar(master, value="1e-6") # Default tolerance

        self.create_widgets()

    def create_widgets(self):
        # --- Input Frame ---
        input_frame = tk.LabelFrame(self.master, text="Inputs (Nonlinear Equations)", padx=10, pady=10)
        input_frame.pack(padx=10, pady=5, fill="x")

        tk.Label(input_frame, text="Function f(x):").grid(row=0, column=0, sticky="w", pady=2)
        tk.Entry(input_frame, textvariable=self.function_str, width=40).grid(row=0, column=1, sticky="ew", pady=2, padx=5)
        self.function_str.set("x**3 - x - 1") # Default for example

        tk.Label(input_frame, text="Interval 'a' (Bisection):").grid(row=1, column=0, sticky="w", pady=2)
        tk.Entry(input_frame, textvariable=self.a_val, width=20).grid(row=1, column=1, sticky="w", pady=2, padx=5)
        self.a_val.set("1.0") # Default for example

        tk.Label(input_frame, text="Interval 'b' (Bisection):").grid(row=2, column=0, sticky="w", pady=2)
        tk.Entry(input_frame, textvariable=self.b_val, width=20).grid(row=2, column=1, sticky="w", pady=2, padx=5)
        self.b_val.set("2.0") # Default for example

        tk.Label(input_frame, text="Initial Guess x0 (Newton):").grid(row=3, column=0, sticky="w", pady=2)
        tk.Entry(input_frame, textvariable=self.x0_val, width=20).grid(row=3, column=1, sticky="w", pady=2, padx=5)
        self.x0_val.set("1.0") # Default for example

        tk.Label(input_frame, text="Tolerance:").grid(row=4, column=0, sticky="w", pady=2)
        tk.Entry(input_frame, textvariable=self.tolerance, width=20).grid(row=4, column=1, sticky="w", pady=2, padx=5)

        input_frame.grid_columnconfigure(1, weight=1) # Allow entry fields to expand

        # --- Buttons Frame ---
        button_frame = tk.Frame(self.master, padx=10, pady=5)
        button_frame.pack(padx=10, pady=5, fill="x")

        tk.Button(button_frame, text="Run Bisection", command=self.run_bisection).pack(side="left", padx=5, expand=True)
        tk.Button(button_frame, text="Run Newton-Raphson", command=self.run_newton_raphson).pack(side="left", padx=5, expand=True)
        
        # Add Gauss-Seidel Button
        tk.Button(button_frame, text="Gauss-Seidel Method", command=self.open_gauss_window).pack(side="left", padx=5, expand=True)
        tk.Button(button_frame, text="Clear Results", command=self.clear_results).pack(side="right", padx=5, expand=True)

        # --- Results Frame ---
        results_frame = tk.LabelFrame(self.master, text="Results", padx=10, pady=10)
        results_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.result_text = scrolledtext.ScrolledText(results_frame, height=10, wrap=tk.WORD)
        self.result_text.pack(fill="both", expand=True)

        # --- Plot Frame ---
        plot_frame = tk.LabelFrame(self.master, text="Convergence Plot", padx=10, pady=10)
        plot_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def clear_results(self):
        self.result_text.delete(1.0, tk.END)
        self.ax.clear()
        self.ax.set_title("Convergence Plot")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Error (log scale)")
        self.ax.grid(True, which="both", ls="--")
        self.canvas.draw()

    # --- Bisection Solver ---
    def run_bisection(self):
        try:
            func_str = self.function_str.get()
            a = float(self.a_val.get())
            b = float(self.b_val.get())
            tol = float(self.tolerance.get())

            if a >= b:
                messagebox.showerror("Input Error", "'a' must be less than 'b'.")
                return
            if tol <= 0:
                messagebox.showerror("Input Error", "Tolerance must be a positive number.")
                return

            func = utils.parse_function(func_str)
            
            if not utils.validate_bisection(func, a, b):
                messagebox.showerror("Bisection Error", 
                                     f"f({a}) and f({b}) have the same sign ({func(a):.4f} and {func(b):.4f}). "
                                     "Bisection method requires opposite signs.")
                return

            solution, history, status = bisection.solve(func, a, b, tol)

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"--- Bisection Method Results ---\n")
            self.result_text.insert(tk.END, f"Function: {func_str}\n")
            self.result_text.insert(tk.END, f"Interval: [{a}, {b}]\n")
            self.result_text.insert(tk.END, f"Tolerance: {tol}\n")
            self.result_text.insert(tk.END, f"Status: {status}\n")
            if solution is not None:
                self.result_text.insert(tk.END, f"Final Approximate Root: {solution:.8f}\n")
            self.result_text.insert(tk.END, "\nIteration Log:\n")
            
            self._display_history_in_text(history)
            self.plot_convergence(history, "Bisection Method")

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid numerical input: {e}")
        except Exception as e:
            messagebox.showerror("Solver Error", f"An unexpected error occurred: {e}")

    # --- Newton-Raphson Solver ---
    def run_newton_raphson(self):
        try:
            func_str = self.function_str.get()
            x0 = float(self.x0_val.get())
            tol = float(self.tolerance.get())

            if tol <= 0:
                messagebox.showerror("Input Error", "Tolerance must be a positive number.")
                return

            func = utils.parse_function(func_str)
            func_prime = utils.get_derivative(func_str)

            solution, history, status = newton.solve(func, func_prime, x0, tol)

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"--- Newton-Raphson Method Results ---\n")
            self.result_text.insert(tk.END, f"Function: {func_str}\n")
            self.result_text.insert(tk.END, f"Initial Guess x0: {x0}\n")
            self.result_text.insert(tk.END, f"Tolerance: {tol}\n")
            self.result_text.insert(tk.END, f"Status: {status}\n")
            if solution is not None:
                self.result_text.insert(tk.END, f"Final Approximate Root: {solution:.8f}\n")
            self.result_text.insert(tk.END, "\nIteration Log:\n")
            
            self._display_history_in_text(history)
            self.plot_convergence(history, "Newton-Raphson Method")

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid numerical input: {e}")
        except Exception as e:
            messagebox.showerror("Solver Error", f"An unexpected error occurred: {e}")

    # Helper to display history in the text widget
    def _display_history_in_text(self, history):
        if history:
            headers = list(history[0].keys())
            header_line = " | ".join(f"{h:<12}" for h in headers)
            self.result_text.insert(tk.END, header_line + "\n")
            self.result_text.insert(tk.END, "-" * len(header_line) + "\n")
            for entry in history:
                row_values = []
                for header in headers:
                    value = entry.get(header, '')
                    if isinstance(value, np.ndarray):
                        row_values.append(f"{np.array2string(value, precision=4, separator=', ', suppress_small=True):<12}")
                    elif isinstance(value, (float, np.float64)):
                        row_values.append(f"{value:<12.6g}")
                    else:
                        row_values.append(f"{str(value):<12}")
                self.result_text.insert(tk.END, " | ".join(row_values) + "\n")
            self.result_text.insert(tk.END, "-" * len(header_line) + "\n")
        else:
            self.result_text.insert(tk.END, "No iteration history to display.\n")

    # --- Plotting Function ---
    def plot_convergence(self, history_log, title_prefix):
        self.ax.clear()
        if history_log and 'Error' in history_log[0]:
            iterations = [entry['Iteration'] for entry in history_log]
            errors = [entry['Error'] for entry in history_log]
            
            if errors:
                self.ax.plot(iterations, errors, marker='o', linestyle='-', color='blue')
                self.ax.set_yscale('log')
                self.ax.set_title(f"Convergence Plot ({title_prefix})")
                self.ax.set_xlabel("Iteration")
                self.ax.set_ylabel("Error (log scale)")
                self.ax.grid(True, which="both", ls="--")
            else:
                self.ax.text(0.5, 0.5, "No Error data to plot", horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes, fontsize=12)
                self.ax.set_title(f"Convergence Plot ({title_prefix})")
                
        else:
            self.ax.text(0.5, 0.5, "No history or error data to plot", horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes, fontsize=12)
            self.ax.set_title(f"Convergence Plot ({title_prefix})")

        self.canvas.draw()

    # --- Gauss-Seidel Method Window ---
    def open_gauss_window(self):
        gs_window = tk.Toplevel(self.master)
        gs_window.title("Gauss-Seidel Method")
        gs_window.geometry("600x600") # Set a default size

        # Input Frame for Gauss-Seidel
        gs_input_frame = tk.LabelFrame(gs_window, text="System of Equations (Ax = b)", padx=10, pady=10)
        gs_input_frame.pack(padx=10, pady=5, fill="x")

        tk.Label(gs_input_frame, text="Enter equations (line by line, e.g., '3x + y = 5'):").pack(anchor="w")
        self.eq_input = scrolledtext.ScrolledText(gs_input_frame, height=6, width=60)
        self.eq_input.pack(fill="x", expand=True)
        # Default example equations
        self.eq_input.insert(tk.END, "3x + y = 5\n")
        self.eq_input.insert(tk.END, "x + 4y = 6\n")

        tk.Label(gs_input_frame, text="Initial Guess (comma separated, e.g., '0,0'):").pack(anchor="w")
        self.initial_entry = tk.Entry(gs_input_frame)
        self.initial_entry.pack(fill="x", expand=True)
        self.initial_entry.insert(0, "0,0") # Default for example

        tk.Label(gs_input_frame, text="Tolerance:").pack(anchor="w")
        self.gs_tol_entry = tk.Entry(gs_input_frame)
        self.gs_tol_entry.insert(0, "1e-6")
        self.gs_tol_entry.pack(fill="x", expand=True)

        tk.Label(gs_input_frame, text="Max Iterations:").pack(anchor="w")
        self.iter_entry = tk.Entry(gs_input_frame)
        self.iter_entry.insert(0, "50")
        self.iter_entry.pack(fill="x", expand=True)

        tk.Button(gs_window, text="Solve Gauss-Seidel", command=self._solve_gs).pack(pady=10)

        # Output Frame for Gauss-Seidel
        gs_output_frame = tk.LabelFrame(gs_window, text="Gauss-Seidel Results", padx=10, pady=10)
        gs_output_frame.pack(padx=10, pady=5, fill="both", expand=True)
        self.gs_output_text = scrolledtext.ScrolledText(gs_output_frame, height=10, width=50, wrap=tk.WORD)
        self.gs_output_text.pack(fill="both", expand=True)

        # Plot Frame for Gauss-Seidel convergence
        gs_plot_frame = tk.LabelFrame(gs_window, text="Convergence Plot", padx=10, pady=10)
        gs_plot_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.gs_fig, self.gs_ax = plt.subplots(figsize=(5, 3))
        self.gs_canvas = FigureCanvasTkAgg(self.gs_fig, master=gs_plot_frame)
        self.gs_canvas_widget = self.gs_canvas.get_tk_widget()
        self.gs_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.gs_toolbar = NavigationToolbar2Tk(self.gs_canvas, gs_plot_frame)
        self.gs_toolbar.update()
        self.gs_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _parse_gauss_seidel_equations(self, equations_str_list):
        """
        Parses a list of equation strings into matrix A and vector b.
        Assumes equations are linear in x, y, z... and RHS is a constant.
        e.g., "3x + 2y - z = 10"
        This is a simplified parser and might not handle all complex cases.
        """
        
        # Determine the number of variables (n) from the first equation
        # This is a basic approach; a more robust parser would be needed for complex cases.
        first_eq = equations_str_list[0]
        # Regex to find variable names (e.g., x, y, z, x1, x2, etc.)
        variables_found = sorted(list(set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', first_eq.split('=')[0]))))
        n = len(variables_found)
        if n == 0:
            raise ValueError("Could not determine variables from equations.")

        # Map variables to indices (e.g., {'x': 0, 'y': 1})
        var_to_idx = {var: i for i, var in enumerate(variables_found)}

        A = np.zeros((n, n))
        b = np.zeros(n)

        for i, eq_str in enumerate(equations_str_list):
            if '=' not in eq_str:
                raise ValueError(f"Equation '{eq_str}' is missing an '=' sign.")
            
            lhs, rhs = eq_str.split('=')
            rhs = float(rhs.strip())
            b[i] = rhs

            # Parse LHS for coefficients
            # This regex splits by '+' or '-' while keeping the operator with the next term
            terms = re.findall(r'[+\-]?\s*\d*\.?\d*\s*[a-zA-Z_][a-zA-Z0-9_]*', lhs.strip())
            
            for term in terms:
                term = term.strip()
                if not term: continue

                # Extract coefficient and variable
                match = re.match(r'([+\-]?\s*\d*\.?\d*)\s*([a-zA-Z_][a-zA-Z0-9_]*)', term)
                if match:
                    coeff_str = match.group(1).strip()
                    var_name = match.group(2).strip()

                    # Handle implied 1 or -1 coefficients
                    if not coeff_str: coeff_str = '1'
                    if coeff_str == '+': coeff_str = '1'
                    if coeff_str == '-': coeff_str = '-1'

                    coeff = float(coeff_str)
                    
                    if var_name not in var_to_idx:
                        raise ValueError(f"Variable '{var_name}' in equation {i+1} not found in first equation's variables.")
                    
                    A[i, var_to_idx[var_name]] = coeff
                else:
                    raise ValueError(f"Could not parse term '{term}' in equation {i+1}.")
        return A, b, variables_found

    def _solve_gs(self):
        try:
            equations_raw = self.eq_input.get("1.0", tk.END).strip()
            equations_list = [eq.strip() for eq in equations_raw.split("\n") if eq.strip()]
            
            if not equations_list:
                messagebox.showerror("Input Error", "Please enter at least one equation.")
                return

            A, b, variables = self._parse_gauss_seidel_equations(equations_list)
            
            x0_str = self.initial_entry.get()
            x0 = np.array(list(map(float, x0_str.split(","))))
            
            tol = float(self.gs_tol_entry.get())
            max_iter = int(self.iter_entry.get())

            if len(x0) != A.shape[0]:
                messagebox.showerror("Input Error", f"Initial guess vector size ({len(x0)}) must match matrix size ({A.shape[0]}).")
                return
            if tol <= 0:
                messagebox.showerror("Input Error", "Tolerance must be a positive number.")
                return
            if max_iter <= 0:
                messagebox.showerror("Input Error", "Max Iterations must be a positive integer.")
                return

            # Optional: Check for diagonal dominance
            if not utils.is_diagonally_dominant(A):
                self.gs_output_text.insert(tk.END, "Warning: The matrix is NOT strictly diagonally dominant. Convergence is not guaranteed.\n\n")

            solution, history, status = gauss_seidel.solve(A, b, x0, tol, max_iter)

            self.gs_output_text.delete(1.0, tk.END)
            self.gs_output_text.insert(tk.END, f"--- Gauss-Seidel Method Results ---\n")
            self.gs_output_text.insert(tk.END, f"Status: {status}\n")
            if solution is not None:
                solution_str = ", ".join([f"{var}={val:.8f}" for var, val in zip(variables, solution)])
                self.gs_output_text.insert(tk.END, f"Final Approximate Solution: [{solution_str}]\n")
            self.gs_output_text.insert(tk.END, "\nIteration Log:\n")
            
            # Display iteration log
            if history:
                headers = list(history[0].keys())
                header_line = " | ".join(f"{h:<12}" for h in headers)
                self.gs_output_text.insert(tk.END, header_line + "\n")
                self.gs_output_text.insert(tk.END, "-" * len(header_line) + "\n")
                for entry in history:
                    row_values = []
                    for header in headers:
                        value = entry.get(header, '')
                        if isinstance(value, np.ndarray):
                            # Display x_n as vector for Gauss-Seidel
                            formatted_vec = '[' + ', '.join(f"{v:.4f}" for v in value) + ']'
                            row_values.append(f"{formatted_vec:<12}")
                        elif isinstance(value, (float, np.float64)):
                            row_values.append(f"{value:<12.6g}")
                        else:
                            row_values.append(f"{str(value):<12}")
                    self.gs_output_text.insert(tk.END, " | ".join(row_values) + "\n")
                self.gs_output_text.insert(tk.END, "-" * len(header_line) + "\n")
            else:
                self.gs_output_text.insert(tk.END, "No iteration history to display.\n")

            self._plot_gs_convergence(history, "Gauss-Seidel Method")

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input for Gauss-Seidel: {e}")
        except Exception as e:
            messagebox.showerror("Solver Error", f"An unexpected error occurred during Gauss-Seidel: {e}")

    # Plotting for Gauss-Seidel (uses its own figure/axes)
    def _plot_gs_convergence(self, history_log, title_prefix):
        self.gs_ax.clear()
        if history_log and 'Error' in history_log[0]:
            iterations = [entry['Iteration'] for entry in history_log]
            errors = [entry['Error'] for entry in history_log]
            
            if errors:
                self.gs_ax.plot(iterations, errors, marker='o', linestyle='-', color='red') # Different color
                self.gs_ax.set_yscale('log')
                self.gs_ax.set_title(f"Convergence Plot ({title_prefix})")
                self.gs_ax.set_xlabel("Iteration")
                self.gs_ax.set_ylabel("Error (log scale)")
                self.gs_ax.grid(True, which="both", ls="--")
            else:
                self.gs_ax.text(0.5, 0.5, "No Error data to plot", horizontalalignment='center', verticalalignment='center', transform=self.gs_ax.transAxes, fontsize=12)
                self.gs_ax.set_title(f"Convergence Plot ({title_prefix})")
                
        else:
            self.gs_ax.text(0.5, 0.5, "No history or error data to plot", horizontalalignment='center', verticalalignment='center', transform=self.gs_ax.transAxes, fontsize=12)
            self.gs_ax.set_title(f"Convergence Plot ({title_prefix})")

        self.gs_canvas.draw()


def main():
    root = tk.Tk()
    app = NumericalMethodsGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()