import tkinter as tk
from typing import Dict, Any

class ThemeManager:
    """
    A comprehensive theme manager for the Numerical Methods GUI
    Provides consistent coloring and styling across the application
    """
    
    def __init__(self):
        # Main color palette - Scientific Professional Theme
        self.colors = {
            # Primary colors
            'primary': '#2E86AB',      # Professional blue (Bisection)
            'primary_light': '#5FA8D3', # Lighter blue
            'primary_dark': '#1B5E7B',  # Darker blue
            
            # Secondary colors  
            'secondary': '#A23B72',    # Purple (Newton)
            'secondary_light': '#C45B95',
            'secondary_dark': '#7A2A55',
            
            # Success colors
            'success': '#18A999',      # Teal (Gauss-Seidel)
            'success_light': '#2CD4B3',
            'success_dark': '#128A7A',
            
            # Status colors
            'warning': '#F18F01',      # Orange (warnings/running)
            'danger': '#C73E1D',       # Red (errors/stop)
            'info': '#6C757D',         # Gray (info)
            
            # Background colors
            'dark': '#2F2D2E',         # Dark gray (main background)
            'dark_light': '#3A3839',   # Lighter dark
            'dark_lighter': '#454344', # Even lighter dark
            
            # Light colors
            'light': '#F8F9FA',        # Light gray (input areas)
            'light_dark': '#E9ECEF',   # Darker light
            'light_darker': '#DEE2E6', # Even darker light
            
            # Accent colors
            'accent': '#FF6B6B',       # Coral (important elements)
            'accent_light': '#FF8E8E',
            'accent_dark': '#E55A5A',
            
            # Text colors
            'text_dark': '#212529',    # Almost black (text on light)
            'text_light': '#FFFFFF',   # White (text on dark)
            'text_muted': '#6C757D',   # Muted text
            
            # Border colors
            'border_light': '#CED4DA',
            'border_dark': '#495057',
            
            # Special method colors
            'bisection': '#2E86AB',    # Blue
            'newton': '#A23B72',       # Purple  
            'gauss_seidel': '#18A999', # Teal
        }
        
        # Font definitions
        self.fonts = {
            'title': ('Arial', 16, 'bold'),
            'heading': ('Arial', 12, 'bold'),
            'subheading': ('Arial', 11, 'bold'),
            'normal': ('Arial', 10),
            'small': ('Arial', 9),
            'monospace': ('Courier New', 10),
        }
        
        # Method-specific styling
        self.method_styles = {
            'bisection': {
                'button_bg': self.colors['bisection'],
                'button_fg': self.colors['text_light'],
                'frame_bg': self.colors['primary_light'],
                'text_bg': self.colors['light'],
            },
            'newton': {
                'button_bg': self.colors['newton'],
                'button_fg': self.colors['text_light'],
                'frame_bg': self.colors['secondary_light'],
                'text_bg': self.colors['light'],
            },
            'gauss_seidel': {
                'button_bg': self.colors['gauss_seidel'],
                'button_fg': self.colors['text_light'],
                'frame_bg': self.colors['success_light'],
                'text_bg': self.colors['light'],
            }
        }
    
    def apply_main_theme(self, master: tk.Tk) -> None:
        """Apply the main theme to the root window"""
        master.configure(bg=self.colors['dark'])
        master.option_add('*Font', self.fonts['normal'])
    
    def style_frame(self, frame: tk.Frame, frame_type: str = 'default') -> None:
        """Style frames based on type"""
        if frame_type == 'main':
            frame.configure(bg=self.colors['dark'])
        elif frame_type == 'input':
            frame.configure(bg=self.colors['light'], relief='raised', bd=1)
        elif frame_type == 'results':
            frame.configure(bg=self.colors['dark_light'], relief='sunken', bd=1)
        elif frame_type == 'plot':
            frame.configure(bg=self.colors['dark_lighter'], relief='sunken', bd=1)
        else:
            frame.configure(bg=self.colors['light'])
    
    def style_label(self, label: tk.Label, label_type: str = 'normal') -> None:
        """Style labels based on type"""
        if label_type == 'title':
            label.configure(font=self.fonts['title'], bg=self.colors['dark'], 
                          fg=self.colors['text_light'])
        elif label_type == 'heading':
            label.configure(font=self.fonts['heading'], bg=self.colors['dark'], 
                          fg=self.colors['text_light'])
        elif label_type == 'subheading':
            label.configure(font=self.fonts['subheading'], bg=self.colors['light'],
                          fg=self.colors['text_dark'])
        elif label_type == 'input':
            label.configure(font=self.fonts['normal'], bg=self.colors['light'],
                          fg=self.colors['text_dark'], anchor='w')
        else:
            label.configure(font=self.fonts['normal'], bg=self.colors['dark'],
                          fg=self.colors['text_light'])
    
    def style_button(self, button: tk.Button, button_type: str = 'primary') -> None:
        """Style buttons based on type"""
        styles = {
            'primary': {'bg': self.colors['primary'], 'fg': self.colors['text_light']},
            'secondary': {'bg': self.colors['secondary'], 'fg': self.colors['text_light']},
            'success': {'bg': self.colors['success'], 'fg': self.colors['text_light']},
            'warning': {'bg': self.colors['warning'], 'fg': self.colors['text_dark']},
            'danger': {'bg': self.colors['danger'], 'fg': self.colors['text_light']},
            'bisection': {'bg': self.colors['bisection'], 'fg': self.colors['text_light']},
            'newton': {'bg': self.colors['newton'], 'fg': self.colors['text_light']},
            'gauss_seidel': {'bg': self.colors['gauss_seidel'], 'fg': self.colors['text_light']},
        }
        
        style = styles.get(button_type, styles['primary'])
        button.configure(
            bg=style['bg'],
            fg=style['fg'],
            font=self.fonts['normal'],
            relief='raised',
            bd=2,
            padx=10,
            pady=5,
            activebackground=style['bg'],
            activeforeground=style['fg']
        )
    
    def style_entry(self, entry: tk.Entry) -> None:
        """Style entry widgets"""
        entry.configure(
            bg=self.colors['light_darker'],
            fg=self.colors['text_dark'],
            relief='sunken',
            bd=2,
            font=self.fonts['normal'],
            insertbackground=self.colors['text_dark']
        )
    
    def style_text(self, text_widget: tk.Text, text_type: str = 'results') -> None:
        """Style text widgets"""
        if text_type == 'results':
            text_widget.configure(
                bg=self.colors['dark_lighter'],
                fg=self.colors['text_light'],
                font=self.fonts['monospace'],
                relief='sunken',
                bd=1,
                padx=5,
                pady=5,
                selectbackground=self.colors['primary'],
                selectforeground=self.colors['text_light']
            )
        else:
            text_widget.configure(
                bg=self.colors['light'],
                fg=self.colors['text_dark'],
                font=self.fonts['normal'],
                relief='sunken',
                bd=1
            )
    
    def get_method_style(self, method: str) -> Dict[str, Any]:
        """Get styling for specific numerical methods"""
        return self.method_styles.get(method, self.method_styles['bisection'])
    
    def create_gradient_bg(self, canvas: tk.Canvas, width: int, height: int) -> None:
        """Create a subtle gradient background (optional enhancement)"""
        color1 = self.colors['dark']
        color2 = self.colors['dark_light']
        
        for i in range(height):
            ratio = i / height
            r = int(int(color1[1:3], 16) * (1 - ratio) + int(color2[1:3], 16) * ratio)
            g = int(int(color1[3:5], 16) * (1 - ratio) + int(color2[3:5], 16) * ratio)
            b = int(int(color1[5:7], 16) * (1 - ratio) + int(color2[5:7], 16) * ratio)
            color = f'#{r:02x}{g:02x}{b:02x}'
            canvas.create_line(0, i, width, i, fill=color)


# Global theme instance for easy access
theme = ThemeManager()