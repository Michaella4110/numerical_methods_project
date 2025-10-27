import time
import os
import sys

# Ensure src directory is in path for imports
sys.path.append(os.path.dirname(__file__))

import bisection
import newton
import gauss_seidel
import utils
import numpy as np

class NumericalMethodsSolver:
    def __init__(self):
        # Create data and plots directories if they don't exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('plots', exist_ok=True)

    def display_main_menu(self):
        """Prints the main options and returns the user's choice."""
        print("\n--- Numerical Methods Solver ---")
        print("1) Solve Nonlinear Equation (f(x) = 0)")
        print("2) Solve System of Linear Equations (Ax = b)")
        print("3) Try Example Problems")
        print("4) Exit")
        choice = input("Enter your choice: ")
        return choice

    def handle_nonlinear_menu(self):
        """Handles the nonlinear equation menu (Bisection/Newton)."""
        while True:
            print("\n--- Solve Nonlinear Equation ---")
            print("1) Bisection Method")
            print("2) Newton-Raphson Method")
            print("3) Back to Main Menu")
            choice = input("Enter your choice: ")

            if choice == '1':
                self.solve_bisection_interactive()
            elif choice == '2':
                self.solve_newton_interactive()
            elif choice == '3':
                break
            else:
                print("Invalid choice. Please try again.")

    def solve_bisection_interactive(self, predefined_func_str=None, predefined_a=None, predefined_b=None, predefined_tol=None, is_example=False):
        """Prompts user for Bisection inputs and runs the solver."""
        print("\n--- Bisection Method ---")
        func_str = predefined_func_str
        a_val = predefined_a
        b_val = predefined_b
        tolerance = predefined_tol

        if not is_example:
            func_str = input("Enter the function f(x) (e.g., 'x**3 - cos(x)'): ")
            while True:
                try:
                    a_val = float(input("Enter the start of the interval 'a': "))
                    b_val = float(input("Enter the end of the interval 'b': "))
                    if a_val >= b_val:
                        print("Error: 'a' must be less than 'b'. Please re-enter.")
                    else:
                        break
                except ValueError:
                    print("Invalid input. Please enter a numerical value for 'a' and 'b'.")
            while True:
                try:
                    tolerance = float(input("Enter the desired tolerance (e.g., 1e-6): "))
                    if tolerance <= 0:
                        print("Tolerance must be a positive number.")
                    else:
                        break
                except ValueError:
                    print("Invalid input. Please enter a numerical value for tolerance.")
        
        try:
            func = utils.parse_function(func_str)
        except ValueError as e:
            print(f"Error parsing function: {e}")
            return

        # Validate interval
        while not utils.validate_bisection(func, a_val, b_val):
            print(f"Invalid interval: f({a_val}) and f({b_val}) have the same sign ({func(a_val):.4f} and {func(b_val):.4f}). Please re-enter.")
            try:
                a_val = float(input("Enter the start of the interval 'a': "))
                b_val = float(input("Enter the end of the interval 'b': "))
                if a_val >= b_val:
                    print("Error: 'a' must be less than 'b'. Please re-enter.")
            except ValueError:
                print("Invalid input. Please enter a numerical value for 'a' and 'b'.")
                continue # Skip the validation for this iteration

        start_time = time.perf_counter()
        solution, history, status = bisection.solve(func, a_val, b_val, tolerance)
        end_time = time.perf_counter()

        print(f"\n--- Results (Bisection Method) ---")
        print(f"Status: {status}")
        utils.print_table_from_log(history)
        if solution is not None:
            print(f"Final Approximate Root: {solution:.8f}")
        print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")

        if not is_example:
            self.ask_to_save_results(history, "bisection_results", "bisection_plot")

    def solve_newton_interactive(self, predefined_func_str=None, predefined_x0=None, predefined_tol=None, is_example=False):
        """Prompts user for Newton-Raphson inputs and runs the solver."""
        print("\n--- Newton-Raphson Method ---")
        func_str = predefined_func_str
        x0_val = predefined_x0
        tolerance = predefined_tol

        if not is_example:
            func_str = input("Enter the function f(x) (e.g., 'x**3 - cos(x)'): ")
            while True:
                try:
                    x0_val = float(input("Enter the initial guess x0: "))
                    break
                except ValueError:
                    print("Invalid input. Please enter a numerical value for x0.")
            while True:
                try:
                    tolerance = float(input("Enter the desired tolerance (e.g., 1e-6): "))
                    if tolerance <= 0:
                        print("Tolerance must be a positive number.")
                    else:
                        break
                except ValueError:
                    print("Invalid input. Please enter a numerical value for tolerance.")

        try:
            func = utils.parse_function(func_str)
            func_prime = utils.get_derivative(func_str)
        except ValueError as e:
            print(f"Error parsing function or derivative: {e}")
            return

        start_time = time.perf_counter()
        solution, history, status = newton.solve(func, func_prime, x0_val, tolerance)
        end_time = time.perf_counter()

        print(f"\n--- Results (Newton-Raphson Method) ---")
        print(f"Status: {status}")
        utils.print_table_from_log(history)
        if solution is not None:
            print(f"Final Approximate Root: {solution:.8f}")
        print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")

        if not is_example:
            self.ask_to_save_results(history, "newton_results", "newton_plot")

    def solve_gauss_seidel_interactive(self, predefined_A=None, predefined_b=None, predefined_x0=None, predefined_tol=None, is_example=False):
        """Prompts user for Gauss-Seidel inputs and runs the solver."""
        print("\n--- Gauss-Seidel Method ---")
        matrix_A = predefined_A
        vector_b = predefined_b
        initial_guess_x0 = predefined_x0
        tolerance = predefined_tol

        if not is_example:
            while True:
                try:
                    n = int(input("Enter the size of the system (n for n x n matrix): "))
                    if n <= 0:
                        print("Size must be a positive integer.")
                        continue
                    break
                except ValueError:
                    print("Invalid input. Please enter an integer.")

            print("Enter the matrix A row by row, with comma-separated values (e.g., '4, 1, -1'):")
            matrix_A_list = []
            for i in range(n):
                while True:
                    row_str = input(f"Enter row {i+1}: ")
                    try:
                        row = [float(val.strip()) for val in row_str.split(',')]
                        if len(row) != n:
                            print(f"Error: Row must contain {n} values. Please re-enter.")
                        else:
                            matrix_A_list.append(row)
                            break
                    except ValueError:
                        print("Invalid input. Please enter comma-separated numerical values.")
            matrix_A = np.array(matrix_A_list)

            print("Enter the vector b with comma-separated values (e.g., '1, 2, 3'):")
            while True:
                b_str = input("Enter vector b: ")
                try:
                    vector_b_list = [float(val.strip()) for val in b_str.split(',')]
                    if len(vector_b_list) != n:
                        print(f"Error: Vector b must contain {n} values. Please re-enter.")
                    else:
                        vector_b = np.array(vector_b_list)
                        break
                except ValueError:
                    print("Invalid input. Please enter comma-separated numerical values.")
            
            print("Enter the initial guess vector x0 with comma-separated values (e.g., '0, 0, 0'):")
            while True:
                x0_str = input("Enter initial guess x0: ")
                try:
                    initial_guess_x0_list = [float(val.strip()) for val in x0_str.split(',')]
                    if len(initial_guess_x0_list) != n:
                        print(f"Error: Initial guess x0 must contain {n} values. Please re-enter.")
                    else:
                        initial_guess_x0 = np.array(initial_guess_x0_list)
                        break
                except ValueError:
                    print("Invalid input. Please enter comma-separated numerical values.")
            
            while True:
                try:
                    tolerance = float(input("Enter the desired tolerance (e.g., 1e-6): "))
                    if tolerance <= 0:
                        print("Tolerance must be a positive number.")
                    else:
                        break
                except ValueError:
                    print("Invalid input. Please enter a numerical value for tolerance.")

        # Validate Gauss-Seidel specific conditions
        if not utils.is_diagonally_dominant(matrix_A):
            print("\nWarning: The matrix is NOT strictly diagonally dominant. Convergence is not guaranteed.")
        
        start_time = time.perf_counter()
        solution, history, status = gauss_seidel.solve(matrix_A, vector_b, initial_guess_x0, tolerance)
        end_time = time.perf_counter()

        print(f"\n--- Results (Gauss-Seidel Method) ---")
        print(f"Status: {status}")
        utils.print_table_from_log(history)
        if solution is not None:
            print(f"Final Approximate Solution Vector: {np.array2string(solution, precision=8, separator=', ')}")
        print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")

        if not is_example:
            self.ask_to_save_results(history, "gauss_seidel_results", "gauss_seidel_plot")

    def handle_examples(self):
        """Provides a sub-menu for built-in test examples."""
        while True:
            print("\n--- Try Example Problems ---")
            print("1) Bisection Example 1: x^3 - x - 1 = 0, in [1, 2]")
            print("2) Newton-Raphson Example 1: x^2 - 2 = 0, x0 = 1")
            print("3) Gauss-Seidel Example 1: 3x+y=5, x+4y=6")
            print("4) Back to Main Menu")
            choice = input("Enter your choice: ")

            if choice == '1':
                print("\nRunning Bisection Example 1...")
                self.solve_bisection_interactive(
                    predefined_func_str="x**3 - x - 1",
                    predefined_a=1.0,
                    predefined_b=2.0,
                    predefined_tol=1e-6,
                    is_example=True
                )
            elif choice == '2':
                print("\nRunning Newton-Raphson Example 1...")
                self.solve_newton_interactive(
                    predefined_func_str="x**2 - 2",
                    predefined_x0=1.0,
                    predefined_tol=1e-6,
                    is_example=True
                )
            elif choice == '3':
                print("\nRunning Gauss-Seidel Example 1...")
                self.solve_gauss_seidel_interactive(
                    predefined_A=np.array([[3.0, 1.0], [1.0, 4.0]]),
                    predefined_b=np.array([5.0, 6.0]),
                    predefined_x0=np.array([0.0, 0.0]),
                    predefined_tol=1e-6,
                    is_example=True
                )
            elif choice == '4':
                break
            else:
                print("Invalid choice. Please try again.")

    def ask_to_save_results(self, history, csv_prefix, plot_prefix):
        """Asks the user if they want to save results and calls utility functions."""
        if history:
            save_choice = input("\nSave results to /data/ and /plots/ (y/n)? ").lower()
            if save_choice == 'y':
                utils.save_log_to_csv(history, csv_prefix)
                if 'Error' in history[0]: # Only try to plot if error data exists
                    utils.save_plot_from_log(history, plot_prefix)
                else:
                    print("Skipping plot generation: No 'Error' key found in history log.")
            else:
                print("Results not saved.")

    def run(self):
        """The main application loop that shows the menu and calls the correct handler."""
        while True:
            choice = self.display_main_menu()
            if choice == '1':
                self.handle_nonlinear_menu()
            elif choice == '2':
                self.solve_gauss_seidel_interactive()
            elif choice == '3':
                self.handle_examples()
            elif choice == '4':
                print("Exiting Numerical Methods Solver. Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    solver = NumericalMethodsSolver()
    solver.run()