# main.py
import tkinter as tk
from app import EggDetectionApp

if __name__ == "__main__":
    root = tk.Tk()
    app = EggDetectionApp(root)
    root.mainloop()