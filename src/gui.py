import tkinter as tk
from tkinter import ttk
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
        
        # Remove fixed window size - let it be resizable
        master.minsize(700, 800)  # Reasonable minimum size
        
        # Apply main theme to the root window
        theme.apply_main_theme(master)

        # Configure modern ttk styles
        self.configure_modern_styles()
        
        # Set global font - FIXED: Use proper tkinter font syntax
        master.option_add("*Font", ("Segoe UI", 10))

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
        self.gs_scrollable_frame = None
        self.gs_toolbar = None

        self.create_widgets()

    def configure_modern_styles(self):
        """Configure modern ttk styles for the application"""
        style = ttk.Style()
        
        # Use a modern theme
        style.theme_use("clam")
        
        # Configure modern button style
        style.configure(
            "Modern.TButton",
            padding=(10, 5),
            relief="flat",
            background="#4CAF50",
            foreground="white",
            focuscolor="none"
        )
        
        # Method-specific button styles
        style.configure(
            "Bisection.TButton",
            padding=(10, 5),
            relief="flat", 
            background="#2196F3",
            foreground="white",
            focuscolor="none"
        )
        
        style.configure(
            "Newton.TButton",
            padding=(10, 5),
            relief="flat",
            background="#FF9800", 
            foreground="white",
            focuscolor="none"
        )
        
        style.configure(
            "GaussSeidel.TButton",
            padding=(10, 5),
            relief="flat",
            background="#9C27B0",
            foreground="white",
            focuscolor="none"
        )
        
        style.configure(
            "Warning.TButton",
            padding=(10, 5),
            relief="flat",
            background="#FFC107",
            foreground="black",
            focuscolor="none"
        )
        
        style.configure(
            "Danger.TButton",
            padding=(10, 5),
            relief="flat",
            background="#F44336",
            foreground="white",
            focuscolor="none"
        )
        
        # Configure label styles for better appearance
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Bold.TLabel", font=("Segoe UI", 10, "bold"))

    def create_widgets(self):
        # Create main frame with scrollbar - NO SPACE BETWEEN CONTENT AND SCROLLBAR
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Create canvas and scrollbar - NO SPACE
        self.canvas_main = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas_main.yview)
        self.scrollable_frame = ttk.Frame(self.canvas_main)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas_main.configure(scrollregion=self.canvas_main.bbox("all"))
        )
        
        self.canvas_main.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_main.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar - NO SPACE
        self.canvas_main.pack(side="left", fill="both", expand=True, padx=0, pady=0)
        scrollbar.pack(side="right", fill="y", padx=0, pady=0)
        
        # Bind mousewheel to scroll
        self.canvas_main.bind("<MouseWheel>", self._on_mousewheel)
        self.scrollable_frame.bind("<MouseWheel>", self._on_mousewheel)

        # Add header title - FIXED: Use tk.Label for custom fonts
        title = tk.Label(self.scrollable_frame, text="Numerical Solver", font=("Segoe UI", 16, "bold"))
        theme.style_label(title, 'main')
        title.pack(pady=12)

        # --- Input Frame (Nonlinear Equations) ---
        input_frame = ttk.LabelFrame(self.scrollable_frame, text="Inputs (Nonlinear Equations)", padding=12)
        input_frame.pack(padx=8, pady=6, fill="x")

        # Function input
        func_label = ttk.Label(input_frame, text="Function f(x):")
        func_label.grid(row=0, column=0, sticky="w", pady=5)
        
        func_entry = ttk.Entry(input_frame, textvariable=self.function_str, width=30)
        func_entry.grid(row=0, column=1, sticky="ew", pady=5, padx=4)
        self.function_str.set("x**3 - x - 1") # Default for example

        # Interval a
        a_label = ttk.Label(input_frame, text="Interval 'a' (Bisection):")
        a_label.grid(row=1, column=0, sticky="w", pady=5)
        
        a_entry = ttk.Entry(input_frame, textvariable=self.a_val, width=30)
        a_entry.grid(row=1, column=1, sticky="ew", pady=5, padx=4)
        self.a_val.set("1.0") # Default for example

        # Interval b
        b_label = ttk.Label(input_frame, text="Interval 'b' (Bisection):")
        b_label.grid(row=2, column=0, sticky="w", pady=5)
        
        b_entry = ttk.Entry(input_frame, textvariable=self.b_val, width=30)
        b_entry.grid(row=2, column=1, sticky="ew", pady=5, padx=4)
        self.b_val.set("2.0") # Default for example

        # Initial guess
        x0_label = ttk.Label(input_frame, text="Initial Guess x0 (Newton):")
        x0_label.grid(row=3, column=0, sticky="w", pady=5)
        
        x0_entry = ttk.Entry(input_frame, textvariable=self.x0_val, width=30)
        x0_entry.grid(row=3, column=1, sticky="ew", pady=5, padx=4)
        self.x0_val.set("1.0") # Default for example

        # Tolerance
        tol_label = ttk.Label(input_frame, text="Tolerance:")
        tol_label.grid(row=4, column=0, sticky="w", pady=5)
        
        tol_entry = ttk.Entry(input_frame, textvariable=self.tolerance, width=30)
        tol_entry.grid(row=4, column=1, sticky="ew", pady=5, padx=4)
        
        # Max iterations
        iter_label = ttk.Label(input_frame, text="Max Iterations:")
        iter_label.grid(row=5, column=0, sticky="w", pady=5)
        
        iter_entry = ttk.Entry(input_frame, textvariable=self.max_iter_nonlinear, width=30)
        iter_entry.grid(row=5, column=1, sticky="ew", pady=5, padx=4)

        input_frame.grid_columnconfigure(1, weight=1) # Allow entry fields to expand

        # --- Buttons Frame ---
        button_frame = ttk.Frame(self.scrollable_frame, padding=8)
        button_frame.pack(padx=8, pady=6, fill="x")

        # Method buttons with modern styles
        bisection_btn = ttk.Button(button_frame, text="Run Bisection", command=self.run_bisection, style="Bisection.TButton")
        bisection_btn.pack(side="left", padx=4, pady=4, expand=True)

        newton_btn = ttk.Button(button_frame, text="Run Newton-Raphson", command=self.run_newton_raphson, style="Newton.TButton")
        newton_btn.pack(side="left", padx=4, pady=4, expand=True)
        
        # Gauss-Seidel Button
        gs_btn = ttk.Button(button_frame, text="Gauss-Seidel Method", command=self.open_gauss_window, style="GaussSeidel.TButton")
        gs_btn.pack(side="left", padx=4, pady=4, expand=True)
        
        # Action buttons
        clear_btn = ttk.Button(button_frame, text="Clear Results", command=self.clear_results, style="Warning.TButton")
        clear_btn.pack(side="right", padx=4, pady=4, expand=True)
        
        stop_btn = ttk.Button(button_frame, text="Stop Solver", command=self.stop_solver, style="Danger.TButton")
        stop_btn.pack(side="right", padx=4, pady=4, expand=True)

        # --- Results Frame ---
        results_frame = ttk.LabelFrame(self.scrollable_frame, text="Results", padding=12)
        results_frame.pack(padx=8, pady=6, fill="both", expand=True)

        self.result_text = scrolledtext.ScrolledText(results_frame, height=10, wrap=tk.WORD)
        theme.style_text(self.result_text, 'results')
        self.result_text.pack(fill="both", expand=True)

        # --- Plot Frame ---
        plot_frame = ttk.LabelFrame(self.scrollable_frame, text="Convergence Plot", padding=12)
        plot_frame.pack(padx=8, pady=6, fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        # Set plot background to match theme
        self.fig.patch.set_facecolor(theme.colors['dark_lighter'])
        self.ax.set_facecolor(theme.colors['dark_lighter'])
        self.ax.tick_params(colors=theme.colors['text_light'])
        self.ax.xaxis.label.set_color(theme.colors['text_light'])
        self.ax.yaxis.label.set_color(theme.colors['text_light'])
        self.ax.title.set_color(theme.colors['text_light'])
        
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.plot_canvas_widget = self.plot_canvas.get_tk_widget()
        self.plot_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Navigation toolbar with save button like main window
        self.toolbar = NavigationToolbar2Tk(self.plot_canvas, plot_frame)
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
        self.plot_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.clear_results() # Initialize plot

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.canvas_main.yview_scroll(int(-1*(event.delta/120)), "units")

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
        self.plot_canvas.draw()

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

        self.plot_canvas.draw()

    # --- Gauss-Seidel Method Window ---
    def open_gauss_window(self):
        if self.gs_window and self.gs_window.winfo_exists():
            self.gs_window.lift() # Bring to front if already open
            return

        self.gs_window = tk.Toplevel(self.master)
        self.gs_window.title("Gauss-Seidel Method")
        # Remove fixed window size for Gauss-Seidel window - let it be resizable
        self.gs_window.minsize(700, 800)  # Reasonable minimum size
        
        theme.apply_main_theme(self.gs_window)
        
        # Create main frame with scrollbar for Gauss-Seidel window - EXACTLY LIKE MAIN WINDOW
        gs_main_frame = tk.Frame(self.gs_window)
        gs_main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Create canvas and scrollbar for Gauss-Seidel - EXACTLY LIKE MAIN WINDOW, NO SPACE
        gs_canvas = tk.Canvas(gs_main_frame, highlightthickness=0)
        gs_scrollbar = ttk.Scrollbar(gs_main_frame, orient="vertical", command=gs_canvas.yview)
        self.gs_scrollable_frame = ttk.Frame(gs_canvas)
        
        self.gs_scrollable_frame.bind(
            "<Configure>",
            lambda e: gs_canvas.configure(scrollregion=gs_canvas.bbox("all"))
        )
        
        gs_canvas.create_window((0, 0), window=self.gs_scrollable_frame, anchor="nw")
        gs_canvas.configure(yscrollcommand=gs_scrollbar.set)
        
        # Pack canvas and scrollbar - EXACTLY LIKE MAIN WINDOW, NO SPACE
        gs_canvas.pack(side="left", fill="both", expand=True, padx=0, pady=0)
        gs_scrollbar.pack(side="right", fill="y", padx=0, pady=0)
        
        # Bind mousewheel to scroll - EXACTLY LIKE MAIN WINDOW
        gs_canvas.bind("<MouseWheel>", lambda e: gs_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.gs_scrollable_frame.bind("<MouseWheel>", lambda e: gs_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # Configure modern styles for the Gauss-Seidel window
        self.gs_window.option_add("*Font", ("Segoe UI", 10))
        
        # Add header to Gauss-Seidel window
        gs_title = tk.Label(self.gs_scrollable_frame, text="Gauss-Seidel Solver", font=("Segoe UI", 14, "bold"))
        theme.style_label(gs_title, 'main')
        gs_title.pack(pady=10)

        # Input Frame for Gauss-Seidel
        gs_input_frame = ttk.LabelFrame(self.gs_scrollable_frame, text="System of Equations (Ax = b)", padding=12)
        gs_input_frame.pack(padx=8, pady=6, fill="x")

        eq_label = ttk.Label(gs_input_frame, text="Enter equations (line by line, e.g., '3x + y = 5'):")
        eq_label.pack(anchor="w", pady=5)
        
        self.eq_input = scrolledtext.ScrolledText(gs_input_frame, height=4, width=65)
        theme.style_text(self.eq_input, 'results')
        self.eq_input.pack(fill="x", expand=True, pady=5)
        # Default example equations
        self.eq_input.insert(tk.END, "10x + 2y - z = 27\n")
        self.eq_input.insert(tk.END, "-3x - 6y + 2z = -61.5\n")
        self.eq_input.insert(tk.END, "x + y + 5z = -21.5\n")

        init_label = ttk.Label(gs_input_frame, text="Initial Guess (comma separated, e.g., '0,0,0'):")
        init_label.pack(anchor="w", pady=5)
        
        self.initial_entry = ttk.Entry(gs_input_frame)
        self.initial_entry.pack(fill="x", expand=True, pady=5)
        self.initial_entry.insert(0, "0,0,0") # Default for example

        tol_label = ttk.Label(gs_input_frame, text="Tolerance:")
        tol_label.pack(anchor="w", pady=5)
        
        self.gs_tol_entry = ttk.Entry(gs_input_frame)
        self.gs_tol_entry.insert(0, "1e-6")
        self.gs_tol_entry.pack(fill="x", expand=True, pady=5)

        iter_label = ttk.Label(gs_input_frame, text="Max Iterations:")
        iter_label.pack(anchor="w", pady=5)
        
        self.iter_entry = ttk.Entry(gs_input_frame)
        self.iter_entry.insert(0, "50")
        self.iter_entry.pack(fill="x", expand=True, pady=5)

        # Action buttons for Gauss-Seidel - EXACTLY LIKE MAIN WINDOW
        gs_button_frame = ttk.Frame(self.gs_scrollable_frame, padding=8)
        gs_button_frame.pack(padx=8, pady=6, fill="x")

        solve_btn = ttk.Button(gs_button_frame, text="Solve Gauss-Seidel", command=self._solve_gs, style="GaussSeidel.TButton")
        solve_btn.pack(side="left", padx=4, pady=4, expand=True)

        clear_gs_btn = ttk.Button(gs_button_frame, text="Clear Results", command=self._clear_gs_results, style="Warning.TButton")
        clear_gs_btn.pack(side="right", padx=4, pady=4, expand=True)

        # Output Frame for Gauss-Seidel
        gs_output_frame = ttk.LabelFrame(self.gs_scrollable_frame, text="Gauss-Seidel Results", padding=12)
        gs_output_frame.pack(padx=8, pady=6, fill="both", expand=True)
        
        self.gs_output_text = scrolledtext.ScrolledText(gs_output_frame, height=12, width=65, wrap=tk.WORD)
        theme.style_text(self.gs_output_text, 'results')
        self.gs_output_text.pack(fill="both", expand=True)

        # Plot Frame for Gauss-Seidel convergence - EXACTLY LIKE MAIN WINDOW
        gs_plot_frame = ttk.LabelFrame(self.gs_scrollable_frame, text="Convergence Plot", padding=12)
        gs_plot_frame.pack(padx=8, pady=6, fill="both", expand=True)

        self.gs_fig, self.gs_ax = plt.subplots(figsize=(6, 3.5))
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

        # Navigation toolbar with save button like main window - EXACTLY LIKE MAIN WINDOW
        self.gs_toolbar = NavigationToolbar2Tk(self.gs_canvas, gs_plot_frame)
        # Style the toolbar to match theme - EXACTLY LIKE MAIN WINDOW
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
        
        self._clear_gs_plot() # Initialize plot
        
        self.gs_window.protocol("WM_DELETE_WINDOW", self._on_gs_window_close)

    def _clear_gs_results(self):
        """Clear Gauss-Seidel results"""
        if self.gs_output_text:
            self.gs_output_text.delete(1.0, tk.END)
        self._clear_gs_plot()

    def _clear_gs_plot(self):
        """Clear Gauss-Seidel plot"""
        if self.gs_ax:
            self.gs_ax.clear()
            self.gs_ax.set_title("Gauss-Seidel Convergence Plot", color=theme.colors['text_light'])
            self.gs_ax.set_xlabel("Iteration", color=theme.colors['text_light'])
            self.gs_ax.set_ylabel("Error (log scale)", color=theme.colors['text_light'])
            self.gs_ax.grid(True, which="both", ls="--", alpha=0.3)
            self.gs_ax.tick_params(colors=theme.colors['text_light'])
        if self.gs_canvas:
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
            self.gs_scrollable_frame = None
            self.gs_toolbar = None

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