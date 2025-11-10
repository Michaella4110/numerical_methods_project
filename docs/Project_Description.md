# Numerical Methods Solver - Project Description

## Project Overview

The **Numerical Methods Solver** is a modern, professional desktop application built with PySide6 that provides an intuitive interface for solving mathematical problems using various numerical methods. This application combines the power of Python's scientific computing libraries with a sleek, user-friendly GUI to make numerical analysis accessible and efficient.

## Key Features

### Nonlinear Equation Solvers
- **Bisection Method**: Root-finding algorithm for continuous functions
- **Newton-Raphson Method**: Fast-converging root-finding using derivatives
- **Real-time Progress Tracking**: Live iteration updates during computation
- **Convergence Analysis**: Visual representation of method performance

### Linear System Solver
- **Gauss-Seidel Method**: Iterative solver for linear systems (Ax = b)
- **Matrix Parsing**: Automatic conversion of equations to matrix form
- **Diagonal Dominance Check**: Automatic validation for convergence conditions
- **Multi-variable Support**: Handles systems with multiple variables

### Modern User Interface
- **Dark Theme**: Professional dark color scheme with blue accents
- **Tabbed Interface**: Organized workflow with separate tabs for different solver types
- **Responsive Design**: Scrollable interface with optimized layouts
- **Professional Styling**: Custom CSS styling for all UI components

### Advanced Visualization
- **Interactive Plots**: Large, detailed convergence graphs with navigation controls
- **Method-Specific Colors**: Distinct visual styles for each numerical method
- **Real-time Updates**: Live plotting during computation
- **Export Capabilities**: Save plots in multiple formats

### Data Management
- **Structured Results**: Professional table displays with sorting capabilities
- **Iteration History**: Complete record of all computation steps
- **Error Tracking**: Comprehensive error analysis and convergence rates
- **Export Ready**: Copy-paste friendly results formatting

## Technical Architecture

### Frontend
- **PySide6**: Modern Qt6-based GUI framework
- **Matplotlib**: Professional plotting and visualization
- **Custom Widgets**: Tailored UI components for numerical analysis

### Backend
- **NumPy**: Efficient numerical computations
- **Custom Solvers**: Optimized implementations of numerical algorithms
- **Multi-threading**: Non-blocking UI with background computation

### Algorithm Implementation
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │ -> │  Parser &        │ -> │   Numerical     │
│                 │    │  Validator       │    │   Solvers       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         v                       v                       v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Real-time     │    │   Convergence    │    │   Results       │
│   UI Updates    │    │   Analysis       │    │   Display       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Supported Methods

### 1. Bisection Method
- **Purpose**: Finding roots of continuous functions
- **Requirements**: Function must change sign over interval [a,b]
- **Convergence**: Linear convergence guaranteed
- **Best For**: Reliable but slower convergence

### 2. Newton-Raphson Method
- **Purpose**: Fast root finding with derivatives
- **Requirements**: Function must be differentiable
- **Convergence**: Quadratic convergence (when it converges)
- **Best For**: Fast convergence when initial guess is good

### 3. Gauss-Seidel Method
- **Purpose**: Solving linear systems Ax = b
- **Requirements**: Matrix should be diagonally dominant
- **Convergence**: Iterative refinement
- **Best For**: Large sparse systems

## Installation & Setup

### Prerequisites
- Python 3.8+
- PySide6
- NumPy
- Matplotlib

### Installation Steps
1. Clone the repository
2. Install dependencies: `pip install pyside6 numpy matplotlib`
3. Run the application: `python src/gui.py`

### Project Structure
```
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
└── requirements.txt          # Documentation and user guides


## Usage Examples

### Nonlinear Equations
```python
# Example: Find root of x³ - x - 1 = 0
Function: x**3 - x - 1
Interval: [1.0, 2.0] (Bisection)
Initial Guess: 1.0 (Newton)
Tolerance: 1e-6
```

### Linear Systems
```python
# Example: Solve 3x3 system
10x + 2y - z = 27
-3x - 6y + 2z = -61.5
x + y + 5z = -21.5
Initial Guess: [0, 0, 0]
```

## UI/UX Features

### Input Validation
- Real-time equation syntax checking
- Numerical range validation
- Convergence condition warnings
- User-friendly error messages

### Performance Optimization
- Background thread computation
- Progress indicators for long-running operations
- Cancelable operations
- Memory-efficient data handling

### Accessibility
- Keyboard navigation support
- High contrast color schemes
- Responsive layout for different screen sizes
- Clear visual hierarchy

## Educational Value

This project serves as an excellent educational tool for:
- **Mathematics Students**: Understanding numerical methods in practice
- **Engineering Students**: Applying computational methods to real problems
- **Researchers**: Quick prototyping and analysis of numerical algorithms
- **Educators**: Demonstrating numerical methods in classroom settings

## Performance Metrics

- **Computation Speed**: Optimized algorithms with O(n) to O(n²) complexity
- **Memory Usage**: Efficient data structures for large iteration histories
- **Accuracy**: Double precision floating-point arithmetic
- **Scalability**: Handles systems with up to 100+ variables

## Future Enhancements

### Planned Features
- [ ] Additional numerical methods (Secant, Fixed Point, etc.)
- [ ] Matrix visualization tools
- [ ] Export functionality for results and plots
- [ ] Batch processing for multiple problems
- [ ] Plugin system for custom solvers
- [ ] Cloud integration for distributed computing

### Technical Improvements
- [ ] Performance profiling and optimization
- [ ] Unit test coverage expansion
- [ ] Documentation automation
- [ ] Multi-language support
- [ ] Mobile-responsive web version

## Contributing

We welcome contributions in the following areas:
- New numerical method implementations
- UI/UX improvements
- Performance optimizations
- Documentation enhancements
- Bug reports and feature requests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **PySide6 Team**: For the excellent Qt6 Python bindings
- **NumPy & Matplotlib Communities**: For robust scientific computing tools
- **Numerical Analysis Researchers**: For the mathematical foundations
- **Open Source Community**: For inspiration and best practices

---

**Project Maintainer**: [Tesfamichael Assefa]  
**Version**: 2025.1  
**Last Updated**: 2025  

*Built with ❤️ using PySide6, NumPy, and Matplotlib*

