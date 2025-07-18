import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import sqlite3

def uwb_plot_paths():
    """
    Standalone function to plot UWB tag paths with optional downsampling and smoothing.
    """
    print("Starting UWB Tag Path Plotting...")

    # ===== COLLECT ALL USER CHOICES UPFRONT =====
    print("Collecting all user preferences...")
    
    # 1. Prompt user to select SQLite file
    root = tk.Tk()
    root.withdraw()
    print("Prompting user to select SQLite file...")
    file_path = filedialog.askopenfilename(title="Select SQLite File", filetypes=[("SQLite Files", "*.sqlite")])

    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"Selected file: {file_path}")

    # 2. Ask user if they want to downsample to 1Hz
    print("Prompting user for downsampling option...")
    downsample = messagebox.askyesno("Downsample Data", "Do you want to downsample the data to 1Hz?")
    print(f"Downsampling option selected: {'Yes' if downsample else 'No'}")

    # 3. Create a new window for smoothing method selection
    print("Prompting user for smoothing method...")
    smoothing_window = tk.Tk()
    smoothing_window.title("Select Smoothing Method")

    smoothing_choice = tk.StringVar(value="none")

    def set_smoothing(choice):
        print(f"Button clicked: {choice}")
        smoothing_choice.set(choice)
        print("Attempting to quit and destroy smoothing window...")
        try:
            smoothing_window.quit()  # Exit the mainloop
            smoothing_window.destroy()  # Destroy the window
            print("Smoothing window successfully quit and destroyed.")
        except Exception as e:
            print(f"Error quitting/destroying smoothing window: {e}")

    tk.Label(smoothing_window, text="Choose a smoothing method:").pack(pady=10)
    tk.Button(smoothing_window, text="Savitzky-Golay", command=lambda: set_smoothing("savitzky-golay")).pack(pady=5)
    tk.Button(smoothing_window, text="Rolling-Average", command=lambda: set_smoothing("rolling-average")).pack(pady=5)
    tk.Button(smoothing_window, text="None", command=lambda: set_smoothing("none")).pack(pady=5)

    print("Starting smoothing window mainloop...")
    smoothing_window.mainloop()
    print("Exited smoothing window mainloop.")

    smoothing = smoothing_choice.get()
    print(f"Smoothing method selected: {smoothing}")

    print("All user preferences collected! Now processing data...")
    print("=" * 50)

    # ===== DATABASE OPERATIONS AND DATA PROCESSING =====
    
    # Connect to SQLite database and query data
    print("Connecting to the SQLite database...")
    conn = sqlite3.connect(file_path)
    print("Querying data from the database...")
    query = "SELECT * FROM data"
    data = pd.read_sql_query(query, conn)
    conn.close()
    print("Data successfully imported.")

    print("Processing data...")
    data['Timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', origin='unix', utc=True)
    data['location_x'] *= 0.0254
    data['location_y'] *= 0.0254
    data = data.sort_values(by=['shortid', 'Timestamp'])
    print("Data sorted by shortid and timestamp.")

    if downsample:
        print("Downsampling data to 1Hz...")
        data['time_sec'] = (data['Timestamp'].astype(np.int64) // 1_000_000_000).astype(int)
        data = data.groupby(['shortid', 'time_sec']).first().reset_index()
        print("Data successfully downsampled.")

    if smoothing == 'savitzky-golay':
        print("Applying Savitzky-Golay smoothing...")
        def apply_savgol_filter(group):
            window_length = min(31, len(group))
            if window_length % 2 == 0:
                window_length -= 1
            polyorder = min(2, window_length - 1)
            return savgol_filter(group, window_length=window_length, polyorder=polyorder)

        data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(apply_savgol_filter)
        data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(apply_savgol_filter)
        print("Savitzky-Golay smoothing applied.")

    elif smoothing == 'rolling-average':
        print("Applying rolling-average smoothing...")
        data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(lambda x: x.rolling(30, min_periods=1).mean())
        data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(lambda x: x.rolling(30, min_periods=1).mean())
        print("Rolling-average smoothing applied.")

    print("Plotting data...")
    
    # Plot combined view with all tags
    print("Creating combined plot with all tags...")
    plt.figure(figsize=(10, 8))
    for tag in data['shortid'].unique():
        tag_data = data[data['shortid'] == tag]
        x_col = 'smoothed_x' if 'smoothed_x' in tag_data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in tag_data.columns else 'location_y'

        plt.plot(tag_data[x_col], tag_data[y_col], label=f'Tag {tag}')

    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.title('UWB Tag Paths - Combined View')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Combined plot complete.")
    
    # Plot individual views for each tag
    print("Creating individual plots for each tag...")
    for tag in data['shortid'].unique():
        tag_data = data[data['shortid'] == tag]
        x_col = 'smoothed_x' if 'smoothed_x' in tag_data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in tag_data.columns else 'location_y'

        plt.figure(figsize=(10, 8))
        plt.plot(tag_data[x_col], tag_data[y_col], label=f'Tag {tag}', linewidth=2)
        plt.xlabel('X Coordinate (meters)')
        plt.ylabel('Y Coordinate (meters)')
        plt.title(f'UWB Tag {tag} - Individual Path')
        plt.legend()
        plt.grid(True)
        plt.show()
        print(f"Individual plot for Tag {tag} complete.")
    
    print("All plotting complete.")

if __name__ == "__main__":
    uwb_plot_paths()
