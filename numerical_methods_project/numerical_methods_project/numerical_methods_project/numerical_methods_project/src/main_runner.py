# src/main_runner.py
from solvers.bisection import solve as bisection_solve
from solvers.newton import solve as newton_solve
import pprint

def test_bisection(func, a, b):
    """
    Test bisection method with a function string and interval
    """
    print(f"Bisection: f(x) = {func} on [{a}, {b}]")
    root, history, status = bisection_solve(func, a, b, tol=1e-10, max_iter=200)
    print("Status:", status)
    print("Root:", root)
    print(f"Iterations: {len(history)}")
    if history:
        print("Last 3 iterations:")
        for entry in history[-3:]:
            print(f"  Iter {entry['Iteration']}: c={entry['c']:.8f}, f(c)={entry['f(c)']:.2e}, Error={entry['Error']:.2e}")
    else:
        print("No iteration history available")
    print("-" * 60)

def test_newton(func, x0):
    """
    Test Newton's method with a function string and initial guess
    """
    print(f"Newton: f(x) = {func} with x0 = {x0}")
    root, history, status = newton_solve(func, x0, tol=1e-10, max_iter=200)
    print("Status:", status)
    print("Root:", root)
    print(f"Iterations: {len(history)}")
    if history:
        print("Last 3 iterations:")
        for entry in history[-3:]:
            print(f"  Iter {entry['Iteration']}: x={entry['x_n']:.8f}, f(x)={entry['f(x_n)']:.2e}, Error={entry['Error']:.2e}")
    else:
        print("No iteration history available")
    print("-" * 60)

def test_function_parsing():
    """
    Test the robust function parser with various formats
    """
    print("TESTING FUNCTION PARSING")
    print("=" * 60)
    
    test_functions = [
        "x**3 - x - 1",      # Standard notation
        "3x + 2",            # Implicit multiplication
        "x^2 + 2x + 1",      # Using ^ for exponentiation
        "sin(x) + cos(x)",   # Trigonometric functions
        "exp(-x) - x",       # Exponential function
        "2sin(x) - 1",       # Implicit multiplication with functions
        "x(x+1)",            # Implicit multiplication with parentheses
    ]
    
    from utils import parse_function_simple
    
    for func_str in test_functions:
        print(f"Testing: {func_str}")
        try:
            f = parse_function_simple(func_str)
            # Test evaluation
            test_val = 1.0
            result = f(test_val)
            print(f"  ✓ Success: f({test_val}) = {result:.6f}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    print("=" * 60)
    print()

def run_comprehensive_tests():
    """
    Run comprehensive tests for both methods
    """
    print("COMPREHENSIVE NUMERICAL METHODS TEST SUITE")
    print("=" * 60)
    
    # Test function parsing first
    test_function_parsing()
    
    print("BISECTION METHOD TESTS")
    print("=" * 60)
    
    tests_bis = [
        ("x**3 - x - 1", 1, 2),           # Expected: ~1.3247
        ("x**2 - 4", 1, 3),               # Expected: 2.0
        ("3x - cos(x) - 1", 0, 1),        # Expected: ~0.6071
        ("exp(-x) - x", 0, 1),            # Expected: ~0.5671
        ("x**3 - 2x - 5", 2, 3),          # Expected: ~2.0946
        ("x**4 - 3x**2 - 3", 1, 2),       # Expected: ~1.8229
        ("sin(x) - x/2", 1, 2),           # Expected: ~1.8955
    ]
    
    for i, (f, a, b) in enumerate(tests_bis, 1):
        print(f"Test {i}:")
        test_bisection(f, a, b)
    
    print("NEWTON-RAPHSON METHOD TESTS")
    print("=" * 60)
    
    tests_newt = [
        ("x**2 - 2", 1.5),                # Expected: √2 ≈ 1.4142
        ("exp(x) - 2", 0.5),              # Expected: ln(2) ≈ 0.6931
        ("x**3 - 2x - 5", 2.0),           # Expected: ~2.0946
        ("cos(x) - x", 0.5),              # Expected: ~0.7391
        ("x**3 - 3x**2 + 3x - 1", 0.5),   # Multiple root at x=1
        ("x**5 - x - 1", 1.0),            # Expected: ~1.1673
    ]
    
    for i, (f, x0) in enumerate(tests_newt, 1):
        print(f"Test {i}:")
        test_newton(f, x0)

def run_quick_validation():
    """
    Quick validation with essential test cases
    """
    print("QUICK VALIDATION TESTS")
    print("=" * 60)
    
    # Essential test cases
    quick_tests = [
        ("Bisection", "x**3 - x - 1", 1, 2, None),
        ("Newton", "x**2 - 2", None, None, 1.5),
        ("Bisection", "3x + sin(x) - 2", 0, 1, None),  # Test implicit multiplication
    ]
    
    for method, func, a, b, x0 in quick_tests:
        if method == "Bisection":
            print(f"{method}: f(x) = {func} on [{a}, {b}]")
            root, history, status = bisection_solve(func, a, b, tol=1e-8, max_iter=100)
        else:
            print(f"{method}: f(x) = {func} with x0 = {x0}")
            root, history, status = newton_solve(func, x0, tol=1e-8, max_iter=100)
        
        print(f"  Result: {root}")
        print(f"  Status: {status}")
        print(f"  Iterations: {len(history)}")
        print()

def test_error_cases():
    """
    Test error handling and edge cases
    """
    print("ERROR CASE TESTS")
    print("=" * 60)
    
    # Test cases that should produce errors
    error_tests = [
        ("x**2 + 1", 0, 1, None, "bisection"),  # No real roots in interval
        ("1/x", -1, 1, None, "bisection"),      # Function undefined at 0
        ("x**2 + 1", None, None, 1, "newton"),  # No real roots
        ("log(x)", None, None, -1, "newton"),   # Function undefined at initial guess
    ]
    
    for func, a, b, x0, method in error_tests:
        print(f"Testing {method}: f(x) = {func}")
        try:
            if method == "bisection":
                root, history, status = bisection_solve(func, a, b)
            else:
                root, history, status = newton_solve(func, x0)
            print(f"  Result: {root}")
            print(f"  Status: {status}")
        except Exception as e:
            print(f"  Exception: {e}")
        print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            run_quick_validation()
        elif sys.argv[1] == "errors":
            test_error_cases()
        elif sys.argv[1] == "parsing":
            test_function_parsing()
        else:
            run_comprehensive_tests()
    else:
        # Run comprehensive tests by default
        run_comprehensive_tests()
    
    print("TESTING COMPLETE")