# Numerical Methods Project

A Python-based desktop application for solving linear systems and nonlinear equations using numerical methods. The application features a modern graphical user interface built with PySide6, interactive visualizations, convergence analysis, and structured numerical results.

## Features

### 1. Nonlinear Equation Solvers

Implements numerical methods for finding approximate roots of nonlinear equations:

- **Bisection Method:** Finds roots by repeatedly reducing an interval that contains a sign change.
- **Newton-Raphson Method:** Uses function values and derivatives to iteratively approximate a root.

### 2. Linear System Solver

- **Gauss-Seidel Method:** Iteratively solves systems of linear equations using successive approximations.

### 3. Graphical User Interface

- Modern desktop interface built with PySide6.
- Professional dark-themed design.
- Real-time progress tracking.
- Structured presentation of numerical results.
- Interactive convergence plots.

### 4. Visualization and Numerical Analysis

- Method-specific graph visualizations.
- Interactive graph navigation controls.
- Error analysis.
- Convergence rate analysis.
- Visual representation of iterative calculations.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PySide6 | Desktop graphical user interface |
| NumPy | Numerical computations and matrix operations |
| Matplotlib | Data visualization and convergence plots |
| Pytest | Automated testing |

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Michaella4110/numerical_methods_project.git
cd numerical_methods_project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the main application using:

```bash
python src/gui.py
```

## Example Problems

### 1. Nonlinear Equations

**Example Function:**

```python
f(x) = x**3 - x - 1
```

**Bisection Method:**

- Initial interval: `[1.0, 2.0]`
- Tolerance: `1e-6`

**Newton-Raphson Method:**

- Initial guess: `1.0`
- Tolerance: `1e-6`

### 2. Linear Systems

The Gauss-Seidel method can be used to solve the following system:

```text
10x + 2y - z = 27
-3x - 6y + 2z = -61.5
x + y + 5z = -21.5
```

**Initial Guess:**

```python
[0, 0, 0]
```

## Running Tests

The project includes automated unit tests for the implemented numerical methods.

Run the test suite using Pytest:

```bash
pytest tests/
```

The tests cover:

- Bisection Method
- Newton-Raphson Method
- Gauss-Seidel Method

## Project Structure

```text
numerical_methods_project/
│
├── .vscode/
├── data/
├── docs/
├── plots/
│
├── src/
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── bisection.py
│   │   ├── gauss_seidel.py
│   │   └── newton.py
│   │
│   ├── __init__.py
│   ├── gui.py
│   ├── main_runner.py
│   ├── matrix_utils.py
│   ├── theme.py
│   └── utils.py
│
├── tests/
│   ├── test_bisection.py
│   ├── test_gauss_seidel.py
│   └── test_newton.py
│
├── README.md
├── requirements.txt
└── LICENSE
```

## Dependencies

- Python 3.8+
- PySide6
- NumPy
- Matplotlib
- Pytest (for testing)

All required Python packages are listed in `requirements.txt`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

**Developed using Python, PySide6, NumPy, and Matplotlib.**
