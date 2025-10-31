# Numerical Methods Project

A comprehensive Python application implementing common numerical methods for solving linear and nonlinear equations. Features a modern PySide6 GUI with real-time visualization and professional results display.

## 🚀 Features

- **Nonlinear Equation Solvers**
  - Bisection Method
  - Newton-Raphson Method
- **Linear System Solver**
  - Gauss-Seidel Method
- **Modern GUI**
  - Professional dark theme interface
  - Real-time progress tracking
  - Interactive convergence plots
  - Structured table results
- **Advanced Visualization**
  - Method-specific plot styling
  - Navigation controls for graphs
  - Error analysis and convergence rates

## 📦 Installation

1. **Clone the repository**
  
   git clone https://github.com/Michaella4110/numerical_methods_project.git
   cd numerical_methods_project

2. **Install dependencies**

   pip install -r requirements.txt

## 🎯 Usage

**Run the main application:**

python src/gui.py


### Example Problems

**Nonlinear Equations:**
- Function: `x**3 - x - 1`
- Interval: `[1.0, 2.0]` (Bisection)
- Initial Guess: `1.0` (Newton)
- Tolerance: `1e-6`

**Linear Systems:**

10x + 2y - z = 27
-3x - 6y + 2z = -61.5
x + y + 5z = -21.5
Initial Guess: [0, 0, 0]


## 🧪 Running Tests

Execute the test suite using pytest:

pytest tests/


## 📁 Project Structure


numerical_methods_project/
├── .vscode/
├── data/
├── docs/
├── plots/
├── src/
│   ├── __pycache__/
│   ├── solvers/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── bisection.py
│   │   ├── gauss_seidel.py
│   │   └── newton.py
│   ├── __init__.py
│   ├── gui.py
│   ├── main_runner.py
│   ├── matrix_utils.py
│   ├── theme.py
│   └── utils.py
├── tests/
│   ├── test_bisection.py
│   ├── test_gauss_seidel.py
│   └── test_newton.py
├── README.md
└── requirements.txt


## 🛠 Dependencies

- Python 3.8+
- PySide6
- NumPy
- Matplotlib
- Pytest (for testing)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


**Built with ❤️ using PySide6, NumPy, and Matplotlib**

This version is:
- **Professional**: Clean formatting with emojis and clear sections
- **Complete**: Includes all essential information for users
- **Ready to copy-paste**: Properly formatted for GitHub
- **Comprehensive**: Covers installation, usage, examples, and structure
- **Visually appealing**: Uses markdown formatting for better readability
