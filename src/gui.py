import tkinter as tk
from tkinter import messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import sys
import os
import re
import threading
import concurrent.futures

# Ensure src directory is in path for imports
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import from the same directory
from solvers.bisection import solve as bisection_solve
from solvers.newton import solve as newton_solve
from solvers.gauss_seidel import solve as gauss_seidel_solve

# Import from utils in the same directory
from utils import (
    parse_function, 
    get_derivative, 
    validate_bisection, 
    is_diagonally_dominant
)

# Import the Theme Manager
from theme import theme

class NumericalMethodsGUI:
    def __init__(self, master):
        self.master = master
        master.title("Numerical Methods Solver")
        
        # Apply main theme to the root window
        theme.apply_main_theme(master)

        self.function_str = tk.StringVar(master)
        self.a_val = tk.StringVar(master)
        self.b_val = tk.StringVar(master)
        self.x0_val = tk.StringVar(master)
        self.tolerance = tk.StringVar(master, value="1e-6") # Default tolerance
        self.max_iter_nonlinear = tk.StringVar(master, value="100") # Max iterations for nonlinear solvers

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) # For background tasks
        self.current_solver_future = None # To keep track of the running solver

        # Attributes for Gauss-Seidel window, initialized to None
        self.gs_window = None
        self.gs_output_text = None
        self.gs_fig = None
        self.gs_ax = None
        self.gs_canvas = None

        self.create_widgets()

    def create_widgets(self):
        # --- Input Frame (Nonlinear Equations) ---
        input_frame = tk.LabelFrame(self.master, text="Inputs (Nonlinear Equations)", padx=10, pady=10)
        theme.style_frame(input_frame, 'input')
        input_frame.pack(padx=10, pady=5, fill="x")

        # Function input
        func_label = tk.Label(input_frame, text="Function f(x):")
        theme.style_label(func_label, 'input')
        func_label.grid(row=0, column=0, sticky="w", pady=2)
        
        func_entry = tk.Entry(input_frame, textvariable=self.function_str, width=40)
        theme.style_entry(func_entry)
        func_entry.grid(row=0, column=1, sticky="ew", pady=2, padx=5)
        self.function_str.set("x**3 - x - 1") # Default for example

        # Interval a
        a_label = tk.Label(input_frame, text="Interval 'a' (Bisection):")
        theme.style_label(a_label, 'input')
        a_label.grid(row=1, column=0, sticky="w", pady=2)
        
        a_entry = tk.Entry(input_frame, textvariable=self.a_val, width=20)
        theme.style_entry(a_entry)
        a_entry.grid(row=1, column=1, sticky="w", pady=2, padx=5)
        self.a_val.set("1.0") # Default for example

        # Interval b
        b_label = tk.Label(input_frame, text="Interval 'b' (Bisection):")
        theme.style_label(b_label, 'input')
        b_label.grid(row=2, column=0, sticky="w", pady=2)
        
        b_entry = tk.Entry(input_frame, textvariable=self.b_val, width=20)
        theme.style_entry(b_entry)
        b_entry.grid(row=2, column=1, sticky="w", pady=2, padx=5)
        self.b_val.set("2.0") # Default for example

        # Initial guess
        x0_label = tk.Label(input_frame, text="Initial Guess x0 (Newton):")
        theme.style_label(x0_label, 'input')
        x0_label.grid(row=3, column=0, sticky="w", pady=2)
        
        x0_entry = tk.Entry(input_frame, textvariable=self.x0_val, width=20)
        theme.style_entry(x0_entry)
        x0_entry.grid(row=3, column=1, sticky="w", pady=2, padx=5)
        self.x0_val.set("1.0") # Default for example

        # Tolerance
        tol_label = tk.Label(input_frame, text="Tolerance:")
        theme.style_label(tol_label, 'input')
        tol_label.grid(row=4, column=0, sticky="w", pady=2)
        
        tol_entry = tk.Entry(input_frame, textvariable=self.tolerance, width=20)
        theme.style_entry(tol_entry)
        tol_entry.grid(row=4, column=1, sticky="w", pady=2, padx=5)
        
        # Max iterations
        iter_label = tk.Label(input_frame, text="Max Iterations:")
        theme.style_label(iter_label, 'input')
        iter_label.grid(row=5, column=0, sticky="w", pady=2)
        
        iter_entry = tk.Entry(input_frame, textvariable=self.max_iter_nonlinear, width=20)
        theme.style_entry(iter_entry)
        iter_entry.grid(row=5, column=1, sticky="w", pady=2, padx=5)

        input_frame.grid_columnconfigure(1, weight=1) # Allow entry fields to expand

        # --- Buttons Frame ---
        button_frame = tk.Frame(self.master, padx=10, pady=5)
        theme.style_frame(button_frame, 'main')
        button_frame.pack(padx=10, pady=5, fill="x")

        # Method buttons with method-specific colors
        bisection_btn = tk.Button(button_frame, text="Run Bisection", command=self.run_bisection)
        theme.style_button(bisection_btn, 'bisection')
        bisection_btn.pack(side="left", padx=5, expand=True)

        newton_btn = tk.Button(button_frame, text="Run Newton-Raphson", command=self.run_newton_raphson)
        theme.style_button(newton_btn, 'newton')
        newton_btn.pack(side="left", padx=5, expand=True)
        
        # Gauss-Seidel Button
        gs_btn = tk.Button(button_frame, text="Gauss-Seidel Method", command=self.open_gauss_window)
        theme.style_button(gs_btn, 'gauss_seidel')
        gs_btn.pack(side="left", padx=5, expand=True)
        
        # Action buttons
        clear_btn = tk.Button(button_frame, text="Clear Results", command=self.clear_results)
        theme.style_button(clear_btn, 'warning')
        clear_btn.pack(side="right", padx=5, expand=True)
        
        stop_btn = tk.Button(button_frame, text="Stop Solver", command=self.stop_solver)
        theme.style_button(stop_btn, 'danger')
        stop_btn.pack(side="right", padx=5, expand=True)

        # --- Results Frame ---
        results_frame = tk.LabelFrame(self.master, text="Results", padx=10, pady=10)
        theme.style_frame(results_frame, 'results')
        results_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.result_text = scrolledtext.ScrolledText(results_frame, height=10, wrap=tk.WORD)
        theme.style_text(self.result_text, 'results')
        self.result_text.pack(fill="both", expand=True)

        # --- Plot Frame ---
        plot_frame = tk.LabelFrame(self.master, text="Convergence Plot", padx=10, pady=10)
        theme.style_frame(plot_frame, 'plot')
        plot_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        # Set plot background to match theme
        self.fig.patch.set_facecolor(theme.colors['dark_lighter'])
        self.ax.set_facecolor(theme.colors['dark_lighter'])
        self.ax.tick_params(colors=theme.colors['text_light'])
        self.ax.xaxis.label.set_color(theme.colors['text_light'])
        self.ax.yaxis.label.set_color(theme.colors['text_light'])
        self.ax.title.set_color(theme.colors['text_light'])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        # Style the toolbar to match theme
        self.toolbar.config(background=theme.colors['dark_light'])
        for child in self.toolbar.winfo_children():
            if isinstance(child, tk.Button):
                child.configure(
                    bg=theme.colors['primary'],
                    fg=theme.colors['text_light'],
                    relief='raised',
                    bd=1
                )
        self.toolbar.update()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.clear_results() # Initialize plot

    def stop_solver(self):
        if self.current_solver_future and self.current_solver_future.running():
            messagebox.showinfo("Stop Solver", "Attempting to stop the solver. It will complete its current iteration.")
        else:
            messagebox.showinfo("Stop Solver", "No solver is currently running.")

    def clear_results(self):
        self.result_text.delete(1.0, tk.END)
        self.ax.clear()
        self.ax.set_title("Convergence Plot", color=theme.colors['text_light'])
        self.ax.set_xlabel("Iteration", color=theme.colors['text_light'])
        self.ax.set_ylabel("Error (log scale)", color=theme.colors['text_light'])
        self.ax.grid(True, which="both", ls="--", alpha=0.3)
        self.ax.tick_params(colors=theme.colors['text_light'])
        self.canvas.draw()

    # --- Progress Callback for Solvers ---
    def _progress_callback(self, iteration, sample_values, error, solver_name):
        self.master.after(0, lambda: self._update_progress_ui(iteration, sample_values, error, solver_name))

    def _update_progress_ui(self, iteration, sample_values, error, solver_name):
        if iteration == 0:
            self.result_text.insert(tk.END, f"\n--- {solver_name} Progress ---\n")
            self.result_text.insert(tk.END, f"{'Iter':<5} | {'Sample Val':<15} | {'Error':<15}\n")
            self.result_text.insert(tk.END, "-" * 40 + "\n")
        
        # Format sample_values: first, middle, last
        sample_str = ""
        if isinstance(sample_values, (list, np.ndarray)):
            if len(sample_values) == 1:
                sample_str = f"{sample_values[0]:.4g}"
            elif len(sample_values) > 1:
                mid_idx = len(sample_values) // 2
                sample_str = f"{sample_values[0]:.4g}, ..., {sample_values[-1]:.4g}"
        elif isinstance(sample_values, (float, np.float64)):
            sample_str = f"{sample_values:.4g}"

        self.result_text.insert(tk.END, f"{iteration:<5} | {sample_str:<15} | {error:<15.6g}\n")
        self.result_text.see(tk.END) # Scroll to bottom

    def _run_solver_async(self, solver_func, args, on_done_callback, on_progress_callback=None):
        """
        Runs a solver function in a background thread.
        """
        self.result_text.delete(1.0, tk.END) # Clear previous results
        self.result_text.insert(tk.END, "Solver started... please wait.\n")
        
        def worker():
            try:
                # Pass the progress_callback to the solver function
                sol, hist, status = solver_func(*args, progress_callback=on_progress_callback)
                self.master.after(0, lambda: on_done_callback(sol, hist, status))
            except Exception as e:
                self.master.after(0, lambda: messagebox.showerror("Solver Error (Background)", f"An error occurred: {e}"))
                self.master.after(0, lambda: on_done_callback(None, [], f"Error: {e}"))
        
        self.current_solver_future = self.executor.submit(worker)

    # --- Bisection Solver ---
    def run_bisection(self):
        try:
            func_str = self.function_str.get()
            a = float(self.a_val.get())
            b = float(self.b_val.get())
            tol = float(self.tolerance.get())
            max_iter = int(self.max_iter_nonlinear.get())

            if a >= b:
                messagebox.showerror("Input Error", "'a' must be less than 'b'.")
                return
            if tol <= 0:
                messagebox.showerror("Input Error", "Tolerance must be a positive number.")
                return
            if max_iter <= 0:
                messagebox.showerror("Input Error", "Max Iterations must be a positive integer.")
                return

            func = parse_function(func_str)
            
            if not validate_bisection(func, a, b):
                messagebox.showerror("Bisection Error", 
                                     f"f({a}) and f({b}) have the same sign ({func(a):.4f} and {func(b):.4f}). "
                                     "Bisection method requires opposite signs.")
                return

            def on_bisection_done(solution, history, status):
                self.result_text.delete(1.0, tk.END) # Clear progress messages
                self.result_text.insert(tk.END, f"--- Bisection Method Results ---\n")
                self.result_text.insert(tk.END, f"Function: {func_str}\n")
                self.result_text.insert(tk.END, f"Interval: [{a}, {b}]\n")
                self.result_text.insert(tk.END, f"Tolerance: {tol}\n")
                self.result_text.insert(tk.END, f"Max Iterations: {max_iter}\n")
                self.result_text.insert(tk.END, f"Status: {status}\n")
                if solution is not None:
                    self.result_text.insert(tk.END, f"Final Approximate Root: {solution:.8f}\n")
                self.result_text.insert(tk.END, "\nIteration Log:\n")
                
                self._display_history_in_text(history)
                self.plot_convergence(history, "Bisection Method")
                
                if "Diverging" in status or "Derivative near zero" in status:
                    messagebox.showwarning("Solver Warning", f"Bisection Method: {status}")

            # FIXED: Use the imported alias bisection_solve instead of bisection.solve
            self._run_solver_async(
                bisection_solve,  # CHANGED: Use the imported alias
                (func, a, b, tol, max_iter),
                on_bisection_done, 
                lambda iter, sample, err: self._progress_callback(iter, sample, err, "Bisection")
            )

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
            max_iter = int(self.max_iter_nonlinear.get())

            if tol <= 0:
                messagebox.showerror("Input Error", "Tolerance must be a positive number.")
                return
            if max_iter <= 0:
                messagebox.showerror("Input Error", "Max Iterations must be a positive integer.")
                return

            func = parse_function(func_str)
            func_prime = get_derivative(func_str)

            def on_newton_done(solution, history, status):
                self.result_text.delete(1.0, tk.END) # Clear progress messages
                self.result_text.insert(tk.END, f"--- Newton-Raphson Method Results ---\n")
                self.result_text.insert(tk.END, f"Function: {func_str}\n")
                self.result_text.insert(tk.END, f"Initial Guess x0: {x0}\n")
                self.result_text.insert(tk.END, f"Tolerance: {tol}\n")
                self.result_text.insert(tk.END, f"Max Iterations: {max_iter}\n")
                self.result_text.insert(tk.END, f"Status: {status}\n")
                if solution is not None:
                    self.result_text.insert(tk.END, f"Final Approximate Root: {solution:.8f}\n")
                self.result_text.insert(tk.END, "\nIteration Log:\n")
                
                self._display_history_in_text(history)
                self.plot_convergence(history, "Newton-Raphson Method")

                if "Diverging" in status or "Derivative near zero" in status:
                    messagebox.showwarning("Solver Warning", f"Newton-Raphson Method: {status}")

            # FIXED: Use the imported alias newton_solve instead of newton_method.solve
            self._run_solver_async(
                newton_solve,  # CHANGED: Use the imported alias
                (func, func_prime, x0, tol, max_iter),
                on_newton_done, 
                lambda iter, sample, err: self._progress_callback(iter, sample, err, "Newton-Raphson")
            )

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid numerical input: {e}")
        except Exception as e:
            messagebox.showerror("Solver Error", f"An unexpected error occurred: {e}")

    # Helper to display history in the text widget
    def _display_history_in_text(self, history):
        if history:
            # Ensure 'Iteration' and 'Error' are always first, if present
            headers = []
            if 'Iteration' in history[0]: headers.append('Iteration')
            if 'Error' in history[0]: headers.append('Error')
            
            # Add other headers, ensuring no duplicates
            for key in history[0].keys():
                if key not in headers:
                    headers.append(key)

            header_line = " | ".join(f"{h:<12}" for h in headers)
            self.result_text.insert(tk.END, header_line + "\n")
            self.result_text.insert(tk.END, "-" * len(header_line) + "\n")
            for entry in history:
                row_values = []
                for header in headers:
                    value = entry.get(header, '')
                    if isinstance(value, np.ndarray):
                        # For vector values (like x_n in Gauss-Seidel)
                        formatted_vec = '[' + ', '.join(f"{v:.4g}" for v in value) + ']'
                        row_values.append(f"{formatted_vec:<12}")
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
            
            if errors and iterations:
                # Filter out non-positive errors for log scale
                positive_errors_indices = [i for i, err in enumerate(errors) if err > 0]
                if positive_errors_indices:
                    filtered_iterations = [iterations[i] for i in positive_errors_indices]
                    filtered_errors = [errors[i] for i in positive_errors_indices]
                    
                    # Use method-specific colors
                    if "Bisection" in title_prefix:
                        color = theme.colors['bisection']
                    elif "Newton" in title_prefix:
                        color = theme.colors['newton']
                    elif "Gauss" in title_prefix:
                        color = theme.colors['gauss_seidel']
                    else:
                        color = theme.colors['primary']
                    
                    self.ax.plot(filtered_iterations, filtered_errors, marker='o', linestyle='-', color=color)
                    self.ax.set_yscale('log')
                    self.ax.set_title(f"Convergence Plot ({title_prefix})", color=theme.colors['text_light'])
                    self.ax.set_xlabel("Iteration", color=theme.colors['text_light'])
                    self.ax.set_ylabel("Error (log scale)", color=theme.colors['text_light'])
                    self.ax.grid(True, which="both", ls="--", alpha=0.3)
                    self.ax.tick_params(colors=theme.colors['text_light'])
                else:
                    self.ax.text(0.5, 0.5, "No positive error data to plot on log scale", 
                                horizontalalignment='center', verticalalignment='center', 
                                transform=self.ax.transAxes, fontsize=10, color=theme.colors['text_light'])
                    self.ax.set_title(f"Convergence Plot ({title_prefix})", color=theme.colors['text_light'])
            else:
                self.ax.text(0.5, 0.5, "No Error data to plot", 
                            horizontalalignment='center', verticalalignment='center', 
                            transform=self.ax.transAxes, fontsize=12, color=theme.colors['text_light'])
                self.ax.set_title(f"Convergence Plot ({title_prefix})", color=theme.colors['text_light'])
                
        else:
            self.ax.text(0.5, 0.5, "No history or error data to plot", 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=self.ax.transAxes, fontsize=12, color=theme.colors['text_light'])
            self.ax.set_title(f"Convergence Plot ({title_prefix})", color=theme.colors['text_light'])

        self.canvas.draw()

    # --- Gauss-Seidel Method Window ---
    def open_gauss_window(self):
        if self.gs_window and self.gs_window.winfo_exists():
            self.gs_window.lift() # Bring to front if already open
            return

        self.gs_window = tk.Toplevel(self.master)
        self.gs_window.title("Gauss-Seidel Method")
        self.gs_window.geometry("600x650") # Set a default size
        theme.apply_main_theme(self.gs_window)
        self.gs_window.protocol("WM_DELETE_WINDOW", self._on_gs_window_close) # Handle close event

        # Input Frame for Gauss-Seidel
        gs_input_frame = tk.LabelFrame(self.gs_window, text="System of Equations (Ax = b)", padx=10, pady=10)
        theme.style_frame(gs_input_frame, 'input')
        gs_input_frame.pack(padx=10, pady=5, fill="x")

        eq_label = tk.Label(gs_input_frame, text="Enter equations (line by line, e.g., '3x + y = 5'):")
        theme.style_label(eq_label, 'input')
        eq_label.pack(anchor="w")
        
        self.eq_input = scrolledtext.ScrolledText(gs_input_frame, height=6, width=60)
        theme.style_text(self.eq_input, 'results')
        self.eq_input.pack(fill="x", expand=True)
        # Default example equations
        self.eq_input.insert(tk.END, "10x + 2y - z = 27\n")
        self.eq_input.insert(tk.END, "-3x - 6y + 2z = -61.5\n")
        self.eq_input.insert(tk.END, "x + y + 5z = -21.5\n")

        init_label = tk.Label(gs_input_frame, text="Initial Guess (comma separated, e.g., '0,0,0'):")
        theme.style_label(init_label, 'input')
        init_label.pack(anchor="w")
        
        self.initial_entry = tk.Entry(gs_input_frame)
        theme.style_entry(self.initial_entry)
        self.initial_entry.pack(fill="x", expand=True)
        self.initial_entry.insert(0, "0,0,0") # Default for example

        tol_label = tk.Label(gs_input_frame, text="Tolerance:")
        theme.style_label(tol_label, 'input')
        tol_label.pack(anchor="w")
        
        self.gs_tol_entry = tk.Entry(gs_input_frame)
        theme.style_entry(self.gs_tol_entry)
        self.gs_tol_entry.insert(0, "1e-6")
        self.gs_tol_entry.pack(fill="x", expand=True)

        iter_label = tk.Label(gs_input_frame, text="Max Iterations:")
        theme.style_label(iter_label, 'input')
        iter_label.pack(anchor="w")
        
        self.iter_entry = tk.Entry(gs_input_frame)
        theme.style_entry(self.iter_entry)
        self.iter_entry.insert(0, "50")
        self.iter_entry.pack(fill="x", expand=True)

        solve_btn = tk.Button(self.gs_window, text="Solve Gauss-Seidel", command=self._solve_gs)
        theme.style_button(solve_btn, 'gauss_seidel')
        solve_btn.pack(pady=10)

        # Output Frame for Gauss-Seidel
        gs_output_frame = tk.LabelFrame(self.gs_window, text="Gauss-Seidel Results", padx=10, pady=10)
        theme.style_frame(gs_output_frame, 'results')
        gs_output_frame.pack(padx=10, pady=5, fill="both", expand=True)
        
        self.gs_output_text = scrolledtext.ScrolledText(gs_output_frame, height=10, width=50, wrap=tk.WORD)
        theme.style_text(self.gs_output_text, 'results')
        self.gs_output_text.pack(fill="both", expand=True)

        # Plot Frame for Gauss-Seidel convergence
        gs_plot_frame = tk.LabelFrame(self.gs_window, text="Convergence Plot", padx=10, pady=10)
        theme.style_frame(gs_plot_frame, 'plot')
        gs_plot_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.gs_fig, self.gs_ax = plt.subplots(figsize=(5, 3))
        # Style the Gauss-Seidel plot
        self.gs_fig.patch.set_facecolor(theme.colors['dark_lighter'])
        self.gs_ax.set_facecolor(theme.colors['dark_lighter'])
        self.gs_ax.tick_params(colors=theme.colors['text_light'])
        self.gs_ax.xaxis.label.set_color(theme.colors['text_light'])
        self.gs_ax.yaxis.label.set_color(theme.colors['text_light'])
        self.gs_ax.title.set_color(theme.colors['text_light'])
        
        self.gs_canvas = FigureCanvasTkAgg(self.gs_fig, master=gs_plot_frame)
        self.gs_canvas_widget = self.gs_canvas.get_tk_widget()
        self.gs_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.gs_toolbar = NavigationToolbar2Tk(self.gs_canvas, gs_plot_frame)
        # Style the toolbar
        self.gs_toolbar.config(background=theme.colors['dark_light'])
        for child in self.gs_toolbar.winfo_children():
            if isinstance(child, tk.Button):
                child.configure(
                    bg=theme.colors['gauss_seidel'],
                    fg=theme.colors['text_light'],
                    relief='raised',
                    bd=1
                )
        self.gs_toolbar.update()
        self.gs_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.gs_ax.set_title("Gauss-Seidel Convergence Plot", color=theme.colors['text_light'])
        self.gs_ax.set_xlabel("Iteration", color=theme.colors['text_light'])
        self.gs_ax.set_ylabel("Error (log scale)", color=theme.colors['text_light'])
        self.gs_ax.grid(True, which="both", ls="--", alpha=0.3)
        self.gs_ax.tick_params(colors=theme.colors['text_light'])
        self.gs_canvas.draw()
    
    def _on_gs_window_close(self):
        # Clear references when the window is closed
        if self.gs_window:
            self.gs_window.destroy()
            self.gs_window = None
            self.gs_output_text = None
            if self.gs_fig:
                plt.close(self.gs_fig)
            self.gs_fig = None
            self.gs_ax = None
            self.gs_canvas = None

    def _parse_gauss_seidel_equations(self, equations_str_list):
        """
        Parses a list of equation strings into matrix A and vector b.
        Ensures consistent variable ordering using logical order.
        """
        
        # Determine all unique variables across all equations
        all_variables = set()
        for eq_str in equations_str_list:
            if '=' not in eq_str:
                raise ValueError(f"Equation '{eq_str}' is missing an '=' sign.")
            lhs = eq_str.split('=')[0]
            all_variables.update(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', lhs))
        
        # IMPORTANT: Use logical variable ordering instead of alphabetical
        preferred_order = ['x', 'y', 'z', 'w', 'u', 'v', 'a', 'b', 'c', 'd']
        
        # Sort variables: preferred ones first, then alphabetical for others
        variables_found = sorted(
            list(all_variables), 
            key=lambda var: (preferred_order.index(var) if var in preferred_order else len(preferred_order), var)
        )
        
        n = len(variables_found)
        
        if n == 0:
            raise ValueError("Could not determine variables from equations.")

        var_to_idx = {var: i for i, var in enumerate(variables_found)}
        
        print(f"DEBUG: Variables detected in order: {variables_found}")  # For debugging

        A = np.zeros((n, n))
        b = np.zeros(n)

        for i, eq_str in enumerate(equations_str_list):
            if '=' not in eq_str:
                raise ValueError(f"Equation '{eq_str}' is missing an '=' sign.")
            
            lhs, rhs = eq_str.split('=')
            try:
                b[i] = float(rhs.strip())
            except ValueError:
                raise ValueError(f"Could not parse right-hand side of equation '{eq_str}'.")

            # Parse LHS for coefficients
            terms = re.findall(r'([+\-]?)?\s*(\d*\.?\d*)?\s*([a-zA-Z_][a-zA-Z0-9_]*)', lhs.strip())
            
            # Reset row A[i, :] before processing terms
            A[i, :] = 0.0 

            for sign_str, coeff_str, var_name in terms:
                if not var_name: 
                    continue 

                coeff_val = 1.0  # Default for 'x' or 'y'
                if coeff_str:  # if a number is explicitly given
                    try:
                        coeff_val = float(coeff_str)
                    except ValueError:  # Case where coeff_str is empty but var_name exists
                        coeff_val = 1.0
                
                # Apply sign
                if sign_str == '-':
                    coeff_val *= -1

                if var_name not in var_to_idx:
                    raise ValueError(f"Variable '{var_name}' in equation {i+1} is not consistent with other equations.")
                
                # Add to the matrix
                A[i, var_to_idx[var_name]] += coeff_val
        
        print(f"DEBUG: Matrix A:\n{A}")  # For debugging
        print(f"DEBUG: Vector b: {b}")   # For debugging
        
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

            # Check for diagonal dominance
            if not is_diagonally_dominant(A):
                response = messagebox.askyesno("Warning", 
                                                "The matrix is NOT strictly diagonally dominant. Continue anyway?")
                if not response:
                    return

            def on_gs_progress(iteration, sample_values, error):
                # Marshal progress updates to GUI thread
                self.master.after(0, lambda: self._update_gs_progress_ui(iteration, sample_values, error))

            def on_gs_done(solution, history, status):
                self.gs_output_text.insert(tk.END, f"\n--- Gauss-Seidel Method Results ---\n")
                self.gs_output_text.insert(tk.END, f"Matrix Size: {A.shape[0]}x{A.shape[1]}\n")
                self.gs_output_text.insert(tk.END, f"Variables: {', '.join(variables)}\n")
                self.gs_output_text.insert(tk.END, f"Initial Guess: {x0}\n")
                self.gs_output_text.insert(tk.END, f"Tolerance: {tol}\n")
                self.gs_output_text.insert(tk.END, f"Max Iterations: {max_iter}\n")
                self.gs_output_text.insert(tk.END, f"Status: {status}\n")
                
                if solution is not None:
                    self.gs_output_text.insert(tk.END, f"\nFinal Solution:\n")
                    for i, var in enumerate(variables):
                        self.gs_output_text.insert(tk.END, f"{var} = {solution[i]:.8f}\n")
                
                self.gs_output_text.insert(tk.END, "\nDetailed Iteration History:\n")
                self._display_gs_history_in_text(history, variables)
                
                # Plot convergence
                self._plot_gs_convergence(history, "Gauss-Seidel Method")
                
                # Show warnings for problematic status
                if "Diverging" in status or "Derivative near zero" in status:
                    messagebox.showwarning("Solver Warning", f"Gauss-Seidel Method: {status}")

            # Clear previous results
            self.gs_output_text.delete(1.0, tk.END)
            self.gs_output_text.insert(tk.END, "Gauss-Seidel solver started...\n")
            
            # FIXED: Use the imported alias gauss_seidel_solve instead of gauss_seidel.solve
            self._run_gs_solver_async(
                gauss_seidel_solve,  # CHANGED: Use the imported alias
                (A, b, x0, tol, max_iter),
                on_gs_done,
                on_gs_progress
            )

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid numerical input: {e}")
        except Exception as e:
            messagebox.showerror("Solver Error", f"An unexpected error occurred: {e}")

    def _run_gs_solver_async(self, solver_func, args, on_done_callback, on_progress_callback=None):
        """Run Gauss-Seidel solver in background thread"""
        def worker():
            try:
                sol, hist, status = solver_func(*args, progress_callback=on_progress_callback)
                self.master.after(0, lambda: on_done_callback(sol, hist, status))
            except Exception as e:
                self.master.after(0, lambda: messagebox.showerror("Gauss-Seidel Solver Error", f"An error occurred: {e}"))
                self.master.after(0, lambda: on_done_callback(None, [], f"Error: {e}"))
        
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _update_gs_progress_ui(self, iteration, sample_values, error):
        """Update Gauss-Seidel progress in the GUI thread"""
        if iteration == 0:
            self.gs_output_text.insert(tk.END, f"\n--- Gauss-Seidel Progress ---\n")
            self.gs_output_text.insert(tk.END, f"{'Iter':<5} | {'Sample Values':<30} | {'Error':<15}\n")
            self.gs_output_text.insert(tk.END, "-" * 60 + "\n")
        
        # Format sample values for display
        if isinstance(sample_values, (list, np.ndarray)):
            if len(sample_values) <= 3:
                sample_str = ", ".join(f"{val:.4g}" for val in sample_values)
            else:
                # Show first, middle, last for large vectors
                mid_idx = len(sample_values) // 2
                sample_str = f"[{sample_values[0]:.4g}, ..., {sample_values[mid_idx]:.4g}, ..., {sample_values[-1]:.4g}]"
        else:
            sample_str = f"{sample_values:.4g}"

        self.gs_output_text.insert(tk.END, f"{iteration:<5} | {sample_str:<30} | {error:<15.6g}\n")
        self.gs_output_text.see(tk.END)  # Scroll to bottom

    def _display_gs_history_in_text(self, history, variables):
        """Display Gauss-Seidel iteration history in text widget"""
        if not history:
            self.gs_output_text.insert(tk.END, "No iteration history available.\n")
            return

        # Create headers
        headers = ['Iteration', 'Error']
        headers.extend(variables)
        
        # Create header line
        header_line = " | ".join(f"{h:<12}" for h in headers)
        self.gs_output_text.insert(tk.END, header_line + "\n")
        self.gs_output_text.insert(tk.END, "-" * len(header_line) + "\n")
        
        # Add each iteration
        for entry in history:
            row_values = []
            for header in headers:
                value = entry.get(header, '')
                if isinstance(value, np.ndarray):
                    # Handle vector values
                    if len(value) == 1:
                        row_values.append(f"{value[0]:<12.6g}")
                    else:
                        row_values.append(f"[{', '.join(f'{v:.4g}' for v in value[:3])}{', ...' if len(value) > 3 else ''}]")
                elif isinstance(value, (float, np.float64)):
                    row_values.append(f"{value:<12.6g}")
                else:
                    row_values.append(f"{str(value):<12}")
            
            self.gs_output_text.insert(tk.END, " | ".join(row_values) + "\n")
        
        self.gs_output_text.insert(tk.END, "-" * len(header_line) + "\n")

    def _plot_gs_convergence(self, history_log, title_prefix):
        """Plot convergence for Gauss-Seidel method"""
        self.gs_ax.clear()
        
        if history_log and 'Error' in history_log[0]:
            iterations = [entry['Iteration'] for entry in history_log]
            errors = [entry['Error'] for entry in history_log]
            
            if errors and iterations:
                # Filter out non-positive errors for log scale
                positive_errors_indices = [i for i, err in enumerate(errors) if err > 0]
                if positive_errors_indices:
                    filtered_iterations = [iterations[i] for i in positive_errors_indices]
                    filtered_errors = [errors[i] for i in positive_errors_indices]
                    self.gs_ax.plot(filtered_iterations, filtered_errors, marker='o', linestyle='-', color=theme.colors['gauss_seidel'])
                    self.gs_ax.set_yscale('log')
                    self.gs_ax.set_title(f"Convergence Plot ({title_prefix})", color=theme.colors['text_light'])
                    self.gs_ax.set_xlabel("Iteration", color=theme.colors['text_light'])
                    self.gs_ax.set_ylabel("Error (log scale)", color=theme.colors['text_light'])
                    self.gs_ax.grid(True, which="both", ls="--", alpha=0.3)
                    self.gs_ax.tick_params(colors=theme.colors['text_light'])
                else:
                    self.gs_ax.text(0.5, 0.5, "No positive error data to plot on log scale", 
                                   horizontalalignment='center', verticalalignment='center', 
                                   transform=self.gs_ax.transAxes, fontsize=10, color=theme.colors['text_light'])
                    self.gs_ax.set_title(f"Convergence Plot ({title_prefix})", color=theme.colors['text_light'])
            else:
                self.gs_ax.text(0.5, 0.5, "No Error data to plot", 
                               horizontalalignment='center', verticalalignment='center', 
                               transform=self.gs_ax.transAxes, fontsize=12, color=theme.colors['text_light'])
                self.gs_ax.set_title(f"Convergence Plot ({title_prefix})", color=theme.colors['text_light'])
        else:
            self.gs_ax.text(0.5, 0.5, "No history or error data to plot", 
                           horizontalalignment='center', verticalalignment='center', 
                           transform=self.gs_ax.transAxes, fontsize=12, color=theme.colors['text_light'])
            self.gs_ax.set_title(f"Convergence Plot ({title_prefix})", color=theme.colors['text_light'])

        self.gs_canvas.draw()


# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = NumericalMethodsGUI(root)
    root.mainloop()