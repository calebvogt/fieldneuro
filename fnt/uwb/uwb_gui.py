# uwb/uwb_gui.py

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import simpledialog
import sqlite3
import matplotlib.pyplot as plt
from preview_database import preview_database
from plot_raw_paths import plot_raw_xy
from uwb_plot_paths import uwb_plot_paths
from uwb_io import select_tags_and_display_ids

def open_gui():
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    # Prompt user to select SQLite file
    file_path = filedialog.askopenfilename(title="Select SQLite File", filetypes=[("SQLite Files", "*.sqlite"), ("All Files", "*.*")])

    if not file_path:
        print("No file selected. Exiting.")
        return

    # Prompt user if they want to preview the database
    preview_response = tk.messagebox.askyesno("Preview Database", "Do you want to preview the database?")

    if preview_response:
        preview_database(file_path)

    # Ask user if they want to plot tag paths
    plot_response = tk.messagebox.askyesno("Plot Tag Paths", "Do you want to plot the paths of the selected tags?")

    if plot_response:
        # Use the IO module to select tags and display IDs
        selected_tags = select_tags_and_display_ids(file_path)

        # Prompt user for smoothing preference
        smoothing = simpledialog.askstring("Smoothing", "Enter smoothing method (savitzky-golay, rolling-average, or leave blank for none):")

        # Call uwb_plot_paths with selected tags and smoothing
        uwb_plot_paths(file_path, list(selected_tags.keys()), smoothing)


if __name__ == "__main__":
    print("Running uwb_gui.py...")
    open_gui()
