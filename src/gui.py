# gui.py (simplified version that works with current imports)
import sys
import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Set the Qt backend before importing matplotlib backends
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QScrollArea, QGroupBox, QLabel, QLineEdit, QPushButton, 
    QTextEdit, QTabWidget, QGridLayout, QMessageBox, QSizePolicy,
    QSplitter, QFrame, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QPalette, QColor, QTextCursor

# Ensure src directory is in path for imports
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import from the same directory - use only available imports
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


class SolverWorker(QThread):
    """Worker thread for running solvers in background"""
    progress_signal = Signal(int, object, float, str)  # iteration, sample_values, error, solver_name
    finished_signal = Signal(object, list, str)  # solution, history, status
    error_signal = Signal(str)

    def __init__(self, solver_func, args, solver_name):
        super().__init__()
        self.solver_func = solver_func
        self.args = args
        self.solver_name = solver_name
        self._is_running = True

    def run(self):
        try:
            def progress_callback(iter, sample, error):
                if self._is_running:
                    self.progress_signal.emit(iter, sample, error, self.solver_name)

            # Add progress_callback to args if function accepts it
            args_list = list(self.args)
            # Check if the solver function accepts progress_callback parameter
            import inspect
            sig = inspect.signature(self.solver_func)
            if 'progress_callback' in sig.parameters:
                args_list.append(progress_callback)

            sol, hist, status = self.solver_func(*args_list)
            if self._is_running:
                self.finished_signal.emit(sol, hist, status)
        except Exception as e:
            if self._is_running:
                self.error_signal.emit(str(e))

    def stop(self):
        self._is_running = False


class ModernMatplotlibCanvas(FigureCanvasQTAgg):
    """Modern styled matplotlib canvas with optimized size"""
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.update_tight_layout()
        
        # Apply modern styling
        self.apply_modern_style()

    def apply_modern_style(self):
        """Apply modern styling to the plot"""
        # Set dark theme compatible colors
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        
        # Style the axes
        self.ax.tick_params(colors='#e0e0e0', which='both', labelsize=10)
        self.ax.xaxis.label.set_color('#e0e0e0')
        self.ax.yaxis.label.set_color('#e0e0e0')
        self.ax.title.set_color('#e0e0e0')
        self.ax.title.set_fontsize(13)
        self.ax.xaxis.label.set_fontsize(11)
        self.ax.yaxis.label.set_fontsize(11)
        
        # Style the spines
        for spine in self.ax.spines.values():
            spine.set_color('#e0e0e0')
            spine.set_linewidth(1.0)
            
        # Grid styling
        self.ax.grid(True, alpha=0.3, color='#e0e0e0', linestyle='--')

    def update_tight_layout(self):
        """Update layout to be tight"""
        self.fig.tight_layout(pad=3.0)


class CustomNavigationToolbar(NavigationToolbar2QT):
    """Custom navigation toolbar with colorful buttons"""
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.setStyleSheet("""
            QToolBar {
                background-color: #353535;
                border: 1px solid #555;
                border-radius: 6px;
                spacing: 3px;
                padding: 5px;
            }
            QToolButton {
                background-color: #42a2d8;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 10px;
                font-weight: bold;
                min-width: 30px;
                min-height: 25px;
            }
            QToolButton:hover {
                background-color: #5bb8e8;
            }
            QToolButton:pressed {
                background-color: #2187c6;
            }
            QToolButton:disabled {
                background-color: #666666;
                color: #999999;
            }
        """)


class ResultsTableWidget(QTableWidget):
    """Custom table widget for displaying solver results"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 4px;
                gridline-color: #555;
                color: #e0e0e0;
            }
            QTableWidget::item {
                padding: 6px;
                border-bottom: 1px solid #444;
            }
            QTableWidget::item:selected {
                background-color: #42a2d8;
                color: white;
            }
            QHeaderView::section {
                background-color: #353535;
                color: #42a2d8;
                font-weight: bold;
                padding: 8px;
                border: 1px solid #555;
            }
        """)
        
        # Configure table behavior
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        
    def setup_table(self, headers):
        """Setup table with headers"""
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        
        # Configure header
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        
    def add_row(self, row_data):
        """Add a row to the table"""
        row = self.rowCount()
        self.insertRow(row)
        
        for col, data in enumerate(row_data):
            item = QTableWidgetItem(str(data))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(row, col, item)


class NumericalMethodsGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Numerical Methods Solver")
        self.setMinimumSize(1100, 850)
        
        # Apply modern styling
        self.apply_modern_theme()
        
        # Initialize variables
        self.function_str = "x**3 - x - 1"
        self.a_val = "1.0"
        self.b_val = "2.0"
        self.x0_val = "1.0"
        self.tolerance = "1e-6"
        self.max_iter_nonlinear = "100"
        
        self.current_solver_thread = None
        
        # Initialize UI
        self.init_ui()
        
    def apply_modern_theme(self):
        """Apply modern dark theme"""
        app = QApplication.instance()
        app.setStyle('Fusion')
        
        # Modern dark palette
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.Text, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        
        app.setPalette(palette)
        
        # Set modern font
        font = QFont("Segoe UI", 10)
        app.setFont(font)

    def init_ui(self):
        """Initialize the main UI with scroll area"""
        # Create scroll area for the entire window
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #2b2b2b;
            }
            QScrollBar:vertical {
                background-color: #353535;
                width: 15px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #42a2d8;
                border-radius: 7px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #5bb8e8;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        # Create central widget that will be scrollable
        central_widget = QWidget()
        scroll_area.setWidget(central_widget)
        self.setCentralWidget(scroll_area)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Title
        title = QLabel("Numerical Methods Solver")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont("Segoe UI", 18, QFont.Weight.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: #42a2d8; margin: 10px;")
        main_layout.addWidget(title)
        
        # Create tab widget for different solvers
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #2b2b2b;
                border-radius: 6px;
            }
            QTabBar::tab {
                background-color: #353535;
                color: #e0e0e0;
                padding: 10px 18px;
                border: 1px solid #555;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-weight: bold;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background-color: #42a2d8;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #505050;
            }
        """)
        
        # Create tabs
        self.create_nonlinear_tab()
        self.create_linear_tab()
        
        main_layout.addWidget(self.tab_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                background-color: #353535;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #42a2d8;
                border-radius: 3px;
            }
        """)
        main_layout.addWidget(self.progress_bar)

    def create_nonlinear_tab(self):
        """Create tab for nonlinear equation solvers"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        
        # Input section
        input_group = QGroupBox("Equation Parameters")
        input_group.setStyleSheet(self.get_groupbox_style())
        input_layout = QGridLayout(input_group)
        
        # Function input
        input_layout.addWidget(QLabel("Function f(x):"), 0, 0)
        self.function_edit = QLineEdit(self.function_str)
        self.function_edit.setStyleSheet(self.get_line_edit_style())
        input_layout.addWidget(self.function_edit, 0, 1)
        
        # Interval inputs
        input_layout.addWidget(QLabel("Interval a (Bisection):"), 1, 0)
        self.a_edit = QLineEdit(self.a_val)
        self.a_edit.setStyleSheet(self.get_line_edit_style())
        input_layout.addWidget(self.a_edit, 1, 1)
        
        input_layout.addWidget(QLabel("Interval b (Bisection):"), 2, 0)
        self.b_edit = QLineEdit(self.b_val)
        self.b_edit.setStyleSheet(self.get_line_edit_style())
        input_layout.addWidget(self.b_edit, 2, 1)
        
        # Auto-bracket checkbox (simple version)
        self.auto_bracket_checkbox = QCheckBox("Enable Auto-Bracket (Manual Search)")
        self.auto_bracket_checkbox.setChecked(False)
        self.auto_bracket_checkbox.setStyleSheet("color: #e0e0e0; font-weight: bold;")
        input_layout.addWidget(self.auto_bracket_checkbox, 3, 0, 1, 2)
        
        # Initial guess
        input_layout.addWidget(QLabel("Initial Guess x₀ (Newton):"), 4, 0)
        self.x0_edit = QLineEdit(self.x0_val)
        self.x0_edit.setStyleSheet(self.get_line_edit_style())
        input_layout.addWidget(self.x0_edit, 4, 1)
        
        # Tolerance and iterations
        input_layout.addWidget(QLabel("Tolerance:"), 5, 0)
        self.tol_edit = QLineEdit(self.tolerance)
        self.tol_edit.setStyleSheet(self.get_line_edit_style())
        input_layout.addWidget(self.tol_edit, 5, 1)
        
        input_layout.addWidget(QLabel("Max Iterations:"), 6, 0)
        self.iter_edit = QLineEdit(self.max_iter_nonlinear)
        self.iter_edit.setStyleSheet(self.get_line_edit_style())
        input_layout.addWidget(self.iter_edit, 6, 1)
        
        layout.addWidget(input_group)
        
        # Solver buttons
        button_layout = QHBoxLayout()
        
        self.bisection_btn = QPushButton("Bisection Method")
        self.bisection_btn.setStyleSheet(self.get_button_style("#2196F3"))
        self.bisection_btn.clicked.connect(self.run_bisection)
        button_layout.addWidget(self.bisection_btn)
        
        self.newton_btn = QPushButton("Newton-Raphson")
        self.newton_btn.setStyleSheet(self.get_button_style("#FF9800"))
        self.newton_btn.clicked.connect(self.run_newton_raphson)
        button_layout.addWidget(self.newton_btn)
        
        self.stop_btn = QPushButton("Stop Solver")
        self.stop_btn.setStyleSheet(self.get_button_style("#f44336"))
        self.stop_btn.clicked.connect(self.stop_solver)
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        
        # Results and plot splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #555; }")
        
        # Results area with table
        results_group = QGroupBox("Results")
        results_group.setStyleSheet(self.get_groupbox_style())
        results_layout = QVBoxLayout(results_group)
        
        # Summary text
        self.results_summary = QTextEdit()
        self.results_summary.setStyleSheet(self.get_text_edit_style())
        self.results_summary.setMaximumHeight(120)
        self.results_summary.setReadOnly(True)
        results_layout.addWidget(self.results_summary)
        
        # Results table
        self.results_table = ResultsTableWidget()
        results_layout.addWidget(self.results_table)
        
        # Plot area
        plot_group = QGroupBox("Convergence Analysis")
        plot_group.setStyleSheet(self.get_groupbox_style())
        plot_layout = QVBoxLayout(plot_group)
        
        plot_container = QWidget()
        plot_container_layout = QVBoxLayout(plot_container)
        plot_container_layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_canvas = ModernMatplotlibCanvas(self, width=10, height=6)
        self.plot_toolbar = CustomNavigationToolbar(self.plot_canvas, self)
        plot_container_layout.addWidget(self.plot_toolbar)
        plot_container_layout.addWidget(self.plot_canvas)
        
        plot_layout.addWidget(plot_container)
        
        splitter.addWidget(results_group)
        splitter.addWidget(plot_group)
        splitter.setSizes([450, 650])
        
        layout.addWidget(splitter, 1)
        
        self.tab_widget.addTab(tab, "Nonlinear Equations")

    def create_linear_tab(self):
        """Create tab for linear system solver (Gauss-Seidel)"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        
        # Input section
        input_group = QGroupBox("Linear System (Ax = b)")
        input_group.setStyleSheet(self.get_groupbox_style())
        input_layout = QVBoxLayout(input_group)
        
        # Equations input
        input_layout.addWidget(QLabel("Equations (one per line):"))
        self.equations_edit = QTextEdit()
        self.equations_edit.setStyleSheet(self.get_text_edit_style())
        self.equations_edit.setMaximumHeight(100)
        self.equations_edit.setPlainText(
            "10x + 2y - z = 27\n"
            "-3x - 6y + 2z = -61.5\n"
            "x + y + 5z = -21.5"
        )
        input_layout.addWidget(self.equations_edit)
        
        # Parameters grid
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel("Initial Guess:"), 0, 0)
        self.gs_x0_edit = QLineEdit("0,0,0")
        self.gs_x0_edit.setStyleSheet(self.get_line_edit_style())
        params_layout.addWidget(self.gs_x0_edit, 0, 1)
        
        params_layout.addWidget(QLabel("Tolerance:"), 1, 0)
        self.gs_tol_edit = QLineEdit("1e-6")
        self.gs_tol_edit.setStyleSheet(self.get_line_edit_style())
        params_layout.addWidget(self.gs_tol_edit, 1, 1)
        
        params_layout.addWidget(QLabel("Max Iterations:"), 2, 0)
        self.gs_iter_edit = QLineEdit("50")
        self.gs_iter_edit.setStyleSheet(self.get_line_edit_style())
        params_layout.addWidget(self.gs_iter_edit, 2, 1)
        
        input_layout.addLayout(params_layout)
        layout.addWidget(input_group)
        
        # Solver button
        self.gs_btn = QPushButton("Solve Gauss-Seidel")
        self.gs_btn.setStyleSheet(self.get_button_style("#9C27B0"))
        self.gs_btn.clicked.connect(self.run_gauss_seidel)
        layout.addWidget(self.gs_btn)
        
        # Results and plot splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #555; }")
        
        # Results area with table
        gs_results_group = QGroupBox("Gauss-Seidel Results")
        gs_results_group.setStyleSheet(self.get_groupbox_style())
        gs_results_layout = QVBoxLayout(gs_results_group)
        
        # Summary text
        self.gs_results_summary = QTextEdit()
        self.gs_results_summary.setStyleSheet(self.get_text_edit_style())
        self.gs_results_summary.setMaximumHeight(120)
        self.gs_results_summary.setReadOnly(True)
        gs_results_layout.addWidget(self.gs_results_summary)
        
        # Results table
        self.gs_results_table = ResultsTableWidget()
        gs_results_layout.addWidget(self.gs_results_table)
        
        # Plot area
        gs_plot_group = QGroupBox("Convergence Analysis")
        gs_plot_group.setStyleSheet(self.get_groupbox_style())
        gs_plot_layout = QVBoxLayout(gs_plot_group)
        
        gs_plot_container = QWidget()
        gs_plot_container_layout = QVBoxLayout(gs_plot_container)
        gs_plot_container_layout.setContentsMargins(0, 0, 0, 0)
        
        self.gs_plot_canvas = ModernMatplotlibCanvas(self, width=10, height=6)
        self.gs_plot_toolbar = CustomNavigationToolbar(self.gs_plot_canvas, self)
        gs_plot_container_layout.addWidget(self.gs_plot_toolbar)
        gs_plot_container_layout.addWidget(self.gs_plot_canvas)
        
        gs_plot_layout.addWidget(gs_plot_container)
        
        splitter.addWidget(gs_results_group)
        splitter.addWidget(gs_plot_group)
        splitter.setSizes([450, 650])
        
        layout.addWidget(splitter, 1)
        
        self.tab_widget.addTab(tab, "Linear Systems")

    def get_groupbox_style(self):
        return """
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #353535;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #42a2d8;
                font-size: 12px;
            }
        """

    def get_line_edit_style(self):
        return """
            QLineEdit {
                background-color: #2b2b2b;
                border: 2px solid #555;
                border-radius: 6px;
                padding: 8px;
                color: #e0e0e0;
                font-size: 11px;
                selection-background-color: #42a2d8;
            }
            QLineEdit:focus {
                border: 2px solid #42a2d8;
            }
        """

    def get_text_edit_style(self):
        return """
            QTextEdit {
                background-color: #2b2b2b;
                border: 2px solid #555;
                border-radius: 6px;
                color: #e0e0e0;
                font-size: 11px;
                selection-background-color: #42a2d8;
                padding: 6px;
            }
        """

    def get_button_style(self, color):
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {self.adjust_color(color, 1.2)};
            }}
            QPushButton:pressed {{
                background-color: {self.adjust_color(color, 0.8)};
            }}
            QPushButton:disabled {{
                background-color: #666;
                color: #999;
            }}
        """

    def adjust_color(self, color, factor):
        """Adjust color brightness by factor"""
        import re
        match = re.search(r'#(..)(..)(..)', color)
        if match:
            r = int(match.group(1), 16)
            g = int(match.group(2), 16)
            b = int(match.group(3), 16)
            r = min(255, int(r * factor))
            g = min(255, int(g * factor))
            b = min(255, int(b * factor))
            return f"#{r:02x}{g:02x}{b:02x}"
        return color

    def run_bisection(self):
        """Run bisection method with basic auto-bracket suggestion"""
        try:
            func_str = self.function_edit.text()
            a = float(self.a_edit.text())
            b = float(self.b_edit.text())
            tol = float(self.tol_edit.text())
            max_iter = int(self.iter_edit.text())

            if a >= b:
                QMessageBox.critical(self, "Input Error", "'a' must be less than 'b'.")
                return
            if tol <= 0:
                QMessageBox.critical(self, "Input Error", "Tolerance must be positive.")
                return

            func = parse_function(func_str)
            
            if not validate_bisection(func, a, b):
                reply = QMessageBox.question(self, "Invalid Bracket", 
                                           f"f({a}) and f({b}) have same sign. Would you like to try a wider search range?",
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    # Suggest wider range
                    wider_a = a - 5
                    wider_b = b + 5
                    self.a_edit.setText(str(wider_a))
                    self.b_edit.setText(str(wider_b))
                    QMessageBox.information(self, "Range Updated", 
                                          f"Search range updated to [{wider_a}, {wider_b}]. Click Solve again.")
                    return
                else:
                    return

            self.start_solver_thread(bisection_solve, (func, a, b, tol, max_iter), "Bisection")

        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Invalid input: {e}")

    def run_newton_raphson(self):
        """Run Newton-Raphson method"""
        try:
            func_str = self.function_edit.text()
            x0 = float(self.x0_edit.text())
            tol = float(self.tol_edit.text())
            max_iter = int(self.iter_edit.text())

            if tol <= 0:
                QMessageBox.critical(self, "Input Error", "Tolerance must be positive.")
                return

            func = parse_function(func_str)
            func_prime = get_derivative(func_str)

            self.start_solver_thread(newton_solve, (func, func_prime, x0, tol, max_iter), "Newton-Raphson")

        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Invalid input: {e}")

    def run_gauss_seidel(self):
        """Run Gauss-Seidel method"""
        try:
            equations_raw = self.equations_edit.toPlainText().strip()
            equations_list = [eq.strip() for eq in equations_raw.split("\n") if eq.strip()]
            
            if not equations_list:
                QMessageBox.critical(self, "Input Error", "Please enter equations.")
                return

            A, b, variables = self._parse_gauss_seidel_equations(equations_list)
            
            x0_str = self.gs_x0_edit.text()
            x0 = np.array(list(map(float, x0_str.split(","))))
            
            tol = float(self.gs_tol_edit.text())
            max_iter = int(self.gs_iter_edit.text())

            if len(x0) != A.shape[0]:
                QMessageBox.critical(self, "Input Error", 
                                   f"Initial guess size ({len(x0)}) must match matrix size ({A.shape[0]}).")
                return

            if not is_diagonally_dominant(A):
                reply = QMessageBox.question(self, "Warning", 
                                           "Matrix is not strictly diagonally dominant. Continue?",
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply != QMessageBox.StandardButton.Yes:
                    return

            self.start_solver_thread(gauss_seidel_solve, (A, b, x0, tol, max_iter), "Gauss-Seidel")

        except Exception as e:
            QMessageBox.critical(self, "Input Error", f"Invalid input: {e}")

    def start_solver_thread(self, solver_func, args, solver_name):
        """Start solver in background thread"""
        if self.current_solver_thread and self.current_solver_thread.isRunning():
            QMessageBox.information(self, "Solver Busy", "Please wait for current solver to finish.")
            return

        # Clear previous results
        if solver_name in ["Bisection", "Newton-Raphson"]:
            self.results_summary.clear()
            self.results_summary.append(f"<h3>Running {solver_name}...</h3>")
            self.results_table.setRowCount(0)
        else:
            self.gs_results_summary.clear()
            self.gs_results_summary.append(f"<h3>Running {solver_name}...</h3>")
            self.gs_results_table.setRowCount(0)

        # Disable buttons and show progress
        self.set_solver_buttons_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        # Create and start worker thread
        self.current_solver_thread = SolverWorker(solver_func, args, solver_name)
        self.current_solver_thread.progress_signal.connect(self.on_solver_progress)
        self.current_solver_thread.finished_signal.connect(self.on_solver_finished)
        self.current_solver_thread.error_signal.connect(self.on_solver_error)
        self.current_solver_thread.start()

    def on_solver_progress(self, iteration, sample_values, error, solver_name):
        """Update progress from solver"""
        progress_text = f"Iteration {iteration}: Error = {error:.6g}"
        
        if solver_name in ["Bisection", "Newton-Raphson"]:
            self.results_summary.append(progress_text)
            self.results_summary.moveCursor(QTextCursor.MoveOperation.End)
        else:
            self.gs_results_summary.append(progress_text)
            self.gs_results_summary.moveCursor(QTextCursor.MoveOperation.End)

    def on_solver_finished(self, solution, history, status):
        """Handle solver completion"""
        self.progress_bar.setVisible(False)
        self.set_solver_buttons_enabled(True)
        
        # Update results display
        if self.current_solver_thread.solver_name in ["Bisection", "Newton-Raphson"]:
            self.display_results(solution, history, status)
            self.plot_convergence(history, self.current_solver_thread.solver_name, self.plot_canvas)
        else:
            self.display_gs_results(solution, history, status)
            self.plot_convergence(history, "Gauss-Seidel", self.gs_plot_canvas)

        if "Diverging" in status or "Derivative near zero" in status:
            QMessageBox.warning(self, "Solver Warning", f"{self.current_solver_thread.solver_name}: {status}")

    def on_solver_error(self, error_msg):
        """Handle solver errors"""
        self.progress_bar.setVisible(False)
        self.set_solver_buttons_enabled(True)
        QMessageBox.critical(self, "Solver Error", f"An error occurred: {error_msg}")

    def stop_solver(self):
        """Stop current solver"""
        if self.current_solver_thread and self.current_solver_thread.isRunning():
            self.current_solver_thread.stop()
            self.current_solver_thread.wait(1000)
            self.progress_bar.setVisible(False)
            self.set_solver_buttons_enabled(True)
            QMessageBox.information(self, "Solver Stopped", "Solver has been stopped.")
        else:
            QMessageBox.information(self, "No Solver", "No solver is currently running.")

    def set_solver_buttons_enabled(self, enabled):
        """Enable or disable solver buttons"""
        self.bisection_btn.setEnabled(enabled)
        self.newton_btn.setEnabled(enabled)
        self.gs_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(not enabled)

    def display_results(self, solution, history, status):
        """Display solver results in structured format"""
        self.results_summary.clear()
        self.results_summary.append(f"<h3>{self.current_solver_thread.solver_name} Results</h3>")
        self.results_summary.append(f"<b>Status:</b> {status}<br>")
        
        if solution is not None:
            self.results_summary.append(f"<b>Solution:</b> {solution:.10f}<br>")
        
        if history:
            self.results_summary.append(f"<b>Iterations:</b> {len(history)}<br>")
            if len(history) > 0:
                final_error = history[-1].get('Error', 'N/A')
                self.results_summary.append(f"<b>Final Error:</b> {final_error:.2e}")

        # Update table
        if history:
            headers = list(history[0].keys())
            self.results_table.setup_table(headers)
            
            for entry in history:
                row_data = []
                for header in headers:
                    value = entry.get(header, '')
                    if isinstance(value, (int, float)):
                        if header == 'Error':
                            row_data.append(f"{value:.6e}")
                        else:
                            row_data.append(f"{value:.6f}")
                    else:
                        row_data.append(str(value))
                self.results_table.add_row(row_data)

    def display_gs_results(self, solution, history, status):
        """Display Gauss-Seidel results in structured format"""
        self.gs_results_summary.clear()
        self.gs_results_summary.append("<h3>Gauss-Seidel Results</h3>")
        self.gs_results_summary.append(f"<b>Status:</b> {status}<br>")
        
        if solution is not None:
            self.gs_results_summary.append("<b>Solution:</b><br>")
            for i, val in enumerate(solution):
                self.gs_results_summary.append(f"x{i+1} = {val:.10f}<br>")
        
        if history:
            self.gs_results_summary.append(f"<b>Iterations:</b> {len(history)}<br>")
            if len(history) > 0:
                final_error = history[-1].get('Error', 'N/A')
                self.gs_results_summary.append(f"<b>Final Error:</b> {final_error:.2e}")

        # Update table
        if history:
            headers = list(history[0].keys())
            self.gs_results_table.setup_table(headers)
            
            for entry in history:
                row_data = []
                for header in headers:
                    value = entry.get(header, '')
                    if isinstance(value, np.ndarray):
                        formatted = '[' + ', '.join(f"{v:.4f}" for v in value[:3])
                        if len(value) > 3:
                            formatted += ', ...'
                        formatted += ']'
                        row_data.append(formatted)
                    elif isinstance(value, (int, float)):
                        if header == 'Error':
                            row_data.append(f"{value:.6e}")
                        else:
                            row_data.append(f"{value:.6f}")
                    else:
                        row_data.append(str(value))
                self.gs_results_table.add_row(row_data)

    def plot_convergence(self, history, title, canvas):
        """Plot convergence history with optimized styling"""
        canvas.ax.clear()
        
        if history and 'Error' in history[0]:
            iterations = [entry['Iteration'] for entry in history]
            errors = [entry['Error'] for entry in history]
            
            if errors and iterations:
                # Filter positive errors for log scale
                positive_indices = [i for i, err in enumerate(errors) if err > 0]
                if positive_indices:
                    filtered_iter = [iterations[i] for i in positive_indices]
                    filtered_errors = [errors[i] for i in positive_indices]
                    
                    # Choose color based on method
                    if "Bisection" in title:
                        color = "#2196F3"
                        marker = 's'
                        linestyle = '-'
                    elif "Newton" in title:
                        color = "#FF9800"
                        marker = '^'
                        linestyle = '--'
                    else:
                        color = "#9C27B0"
                        marker = 'o'
                        linestyle = '-.'
                    
                    canvas.ax.plot(filtered_iter, filtered_errors, marker=marker, linestyle=linestyle, 
                                 color=color, markersize=6, linewidth=2.5, alpha=0.9,
                                 markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.2)
                    
                    canvas.ax.set_yscale('log')
                    canvas.ax.set_title(f"{title} Convergence", color='#e0e0e0', pad=15, fontsize=14)
                    canvas.ax.set_xlabel("Iteration", color='#e0e0e0', labelpad=12, fontsize=12)
                    canvas.ax.set_ylabel("Error (log scale)", color='#e0e0e0', labelpad=12, fontsize=12)
                    
                    canvas.ax.grid(True, alpha=0.3, color='#e0e0e0', linestyle='--', linewidth=0.8)
                    
                    if len(filtered_errors) > 1:
                        convergence_rate = filtered_errors[-1] / filtered_errors[-2] if filtered_errors[-2] != 0 else 0
                        stats_text = f'Final Error: {filtered_errors[-1]:.2e}\nIterations: {len(iterations)}'
                        if len(filtered_errors) > 2:
                            stats_text += f'\nRate: {convergence_rate:.4f}'
                        canvas.ax.text(0.02, 0.98, stats_text, 
                                     transform=canvas.ax.transAxes, color='#e0e0e0', fontsize=10,
                                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#353535', alpha=0.8))
                    
                    canvas.update_tight_layout()
                    canvas.draw()
                    return
        
        # No data to plot
        canvas.ax.text(0.5, 0.5, "No convergence data to display", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=canvas.ax.transAxes, color='#e0e0e0', fontsize=12)
        canvas.ax.set_title(f"{title} Convergence", color='#e0e0e0', fontsize=14)
        canvas.update_tight_layout()
        canvas.draw()

    def _parse_gauss_seidel_equations(self, equations_str_list):
        """Parse equations into matrix A and vector b"""
        all_variables = set()
        for eq_str in equations_str_list:
            if '=' not in eq_str:
                raise ValueError(f"Equation '{eq_str}' missing '='")
            lhs = eq_str.split('=')[0]
            all_variables.update(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', lhs))
        
        # Logical variable ordering
        preferred_order = ['x', 'y', 'z', 'w', 'u', 'v', 'a', 'b', 'c', 'd']
        variables_found = sorted(
            list(all_variables), 
            key=lambda var: (preferred_order.index(var) if var in preferred_order else len(preferred_order), var)
        )
        
        n = len(variables_found)
        var_to_idx = {var: i for i, var in enumerate(variables_found)}
        
        A = np.zeros((n, n))
        b = np.zeros(n)

        for i, eq_str in enumerate(equations_str_list):
            lhs, rhs = eq_str.split('=')
            try:
                b[i] = float(rhs.strip())
            except ValueError:
                raise ValueError(f"Invalid RHS in equation: {eq_str}")

            # Parse coefficients from LHS
            terms = re.findall(r'([+\-]?)?\s*(\d*\.?\d*)?\s*([a-zA-Z_][a-zA-Z0-9_]*)', lhs.strip())
            A[i, :] = 0.0

            for sign_str, coeff_str, var_name in terms:
                if not var_name:
                    continue
                
                coeff_val = 1.0
                if coeff_str:
                    try:
                        coeff_val = float(coeff_str) if coeff_str else 1.0
                    except ValueError:
                        coeff_val = 1.0
                
                if sign_str == '-':
                    coeff_val *= -1

                if var_name not in var_to_idx:
                    raise ValueError(f"Unknown variable: {var_name}")
                
                A[i, var_to_idx[var_name]] += coeff_val
        
        return A, b, variables_found


def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Numerical Methods Solver")
    app.setApplicationVersion("2025.1")
    app.setOrganizationName("MathLab")
    
    window = NumericalMethodsGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
