import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import sqlite3
import os
import subprocess
import shutil
from tqdm import tqdm
import gc

def uwb_animate_paths():
    """
    Standalone function to create animated videos of UWB tag paths with optional downsampling and smoothing.
    """
    print("Starting UWB Tag Path Animation...")

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

    # 3. Smoothing method selection
    print("Prompting user for smoothing method...")
    smoothing_window = tk.Tk()
    smoothing_window.title("Select Smoothing Method")

    smoothing_choice = tk.StringVar(value="none")

    def set_smoothing(choice):
        print(f"Button clicked: {choice}")
        smoothing_choice.set(choice)
        print("Attempting to quit and destroy smoothing window...")
        try:
            smoothing_window.quit()
            smoothing_window.destroy()
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

    # 4. Timezone selection
    print("Prompting user for timezone selection...")
    timezone_window = tk.Tk()
    timezone_window.title("Select US Timezone")
    
    timezone_choice = tk.StringVar(value="UTC")
    
    def set_timezone(tz):
        timezone_choice.set(tz)
        timezone_window.quit()
        timezone_window.destroy()
    
    tk.Label(timezone_window, text="Select your timezone for timestamp conversion:").pack(pady=10)
    tk.Button(timezone_window, text="Eastern (ET)", command=lambda: set_timezone("US/Eastern")).pack(pady=3)
    tk.Button(timezone_window, text="Central (CT)", command=lambda: set_timezone("US/Central")).pack(pady=3)
    tk.Button(timezone_window, text="Mountain (MT)", command=lambda: set_timezone("US/Mountain")).pack(pady=3)
    tk.Button(timezone_window, text="Pacific (PT)", command=lambda: set_timezone("US/Pacific")).pack(pady=3)
    tk.Button(timezone_window, text="Alaska (AKT)", command=lambda: set_timezone("US/Alaska")).pack(pady=3)
    tk.Button(timezone_window, text="Hawaii (HST)", command=lambda: set_timezone("US/Hawaii")).pack(pady=3)
    tk.Button(timezone_window, text="Keep UTC", command=lambda: set_timezone("UTC")).pack(pady=3)
    
    timezone_window.mainloop()
    
    selected_timezone = timezone_choice.get()
    print(f"Timezone selected: {selected_timezone}")

    # 5. Playback speed selection
    print("Prompting user for playback speed...")
    speed_window = tk.Tk()
    speed_window.title("Select Playback Speed")
    
    speed_choice = tk.StringVar(value="1")
    
    def set_speed(speed):
        speed_choice.set(speed)
        speed_window.quit()
        speed_window.destroy()
    
    tk.Label(speed_window, text="Choose playback speed:").pack(pady=10)
    tk.Button(speed_window, text="1x (Real-time)", command=lambda: set_speed("1")).pack(pady=5)
    tk.Button(speed_window, text="2x", command=lambda: set_speed("2")).pack(pady=5)
    tk.Button(speed_window, text="5x", command=lambda: set_speed("5")).pack(pady=5)
    tk.Button(speed_window, text="10x", command=lambda: set_speed("10")).pack(pady=5)
    
    speed_window.mainloop()
    
    playback_speed = int(speed_choice.get())
    print(f"Playback speed selected: {playback_speed}x")

    print("All user preferences collected! Now checking available dates...")
    print("=" * 50)

    # Quick database query to get available dates without loading all data
    print("Connecting to database to check available dates...")
    conn = sqlite3.connect(file_path)
    
    # Query just timestamps to determine available dates
    date_query = "SELECT DISTINCT date(datetime(timestamp/1000, 'unixepoch')) as date FROM data ORDER BY date"
    available_dates = pd.read_sql_query(date_query, conn)
    conn.close()
    
    if available_dates.empty:
        print("No data found in database. Exiting.")
        return
        
    print(f"Found {len(available_dates)} unique dates in database")

    # 6. Day selection window (using quickly queried dates)
    print("Prompting user for day selection...")
    unique_dates = [pd.to_datetime(date).date() for date in available_dates['date']]
    
    day_window = tk.Tk()
    day_window.title("Select Days to Animate")
    
    selected_days = {}
    day_results = {}  # Store final boolean results
    
    tk.Label(day_window, text="Select days to create videos for:").pack(pady=10)
    
    for date in unique_dates:
        var = tk.BooleanVar()
        var.set(True)  # Set to True after creating the variable
        selected_days[date] = var
        tk.Checkbutton(day_window, text=str(date), variable=var).pack(anchor="w")
    
    def on_day_submit():
        print("Submit button clicked - collecting results before closing")
        # Capture the values before destroying the window
        for date, var in selected_days.items():
            day_results[date] = var.get()
            print(f"  Captured {date}: {day_results[date]}")
        print("About to close window")
        day_window.quit()
        day_window.destroy()
        print("Window closed successfully")
    
    tk.Button(day_window, text="Continue", command=on_day_submit).pack(pady=10)
    day_window.mainloop()
    
    # Use the captured results
    print("Using captured results...")
    selected_date_list = [date for date, selected in day_results.items() if selected]
    print(f"User selected days: {selected_date_list}")
    print(f"Final selected days: {selected_date_list}")

    if not selected_date_list:
        print("No days selected. Exiting.")
        return

    print("All choices finalized! Beginning full data processing...")
    print("=" * 50)

    # ===== FULL DATABASE OPERATIONS AND DATA PROCESSING =====
    
    # Connect to SQLite database and query data
    print("Connecting to the SQLite database for full data load...")
    conn = sqlite3.connect(file_path)
    print("Querying all data from the database...")
    query = "SELECT * FROM data"
    data = pd.read_sql_query(query, conn)
    conn.close()
    print("Data successfully imported.")

    print("Processing data...")
    data['Timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', origin='unix', utc=True)
    
    # Convert timezone if not UTC
    if selected_timezone != "UTC":
        print(f"Converting timestamps from UTC to {selected_timezone}...")
        data['Timestamp'] = data['Timestamp'].dt.tz_convert(selected_timezone)
    
    data['location_x'] *= 0.0254
    data['location_y'] *= 0.0254
    data = data.sort_values(by=['shortid', 'Timestamp'])
    data['Date'] = data['Timestamp'].dt.date
    print("Data sorted by shortid and timestamp.")

    # Filter data to only selected dates
    print(f"Filtering data to only selected dates: {selected_date_list}")
    data = data[data['Date'].isin(selected_date_list)]
    print(f"Filtered data contains {len(data)} records")

    print("All data processing complete! Beginning animation creation...")
    print("=" * 50)

    # Apply downsampling if selected
    if downsample:
        print("Downsampling data to 1Hz...")
        data['time_sec'] = (data['Timestamp'].astype(np.int64) // 1_000_000_000).astype(int)
        data = data.groupby(['shortid', 'time_sec']).first().reset_index()
        print("Data successfully downsampled.")

    # Apply smoothing if selected
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

    # Set column names for plotting
    x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
    y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'

    # Set up output directory - with better desktop detection
    # Try multiple possible desktop locations
    possible_desktops = [
        os.path.join(os.path.expanduser("~"), "Desktop"),  # Standard user desktop
        os.path.join(os.environ.get('USERPROFILE', ''), "Desktop"),  # Windows user profile desktop
        os.path.join(os.environ.get('USERPROFILE', ''), "OneDrive", "Desktop"),  # OneDrive desktop
        "C:\\Users\\caleb\\Desktop",  # Direct path to your desktop
    ]
    
    print("Checking possible desktop locations:")
    for i, desktop in enumerate(possible_desktops):
        exists = os.path.exists(desktop)
        print(f"  {i+1}. {desktop} - {'EXISTS' if exists else 'NOT FOUND'}")
    
    # Use the first existing desktop, or fall back to the SQL file directory
    desktop_path = None
    for desktop in possible_desktops:
        if os.path.exists(desktop):
            desktop_path = desktop
            break
    
    if desktop_path is None:
        print("No desktop folder found, using SQL file directory for temp frames")
        desktop_path = os.path.dirname(file_path)
    
    output_dir = os.path.join(desktop_path, "uwb_animation_frames")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set final video destination to same folder as SQLite file
    sql_file_dir = os.path.dirname(file_path)
    
    print(f"Using desktop path: {desktop_path}")
    print(f"Temp frames directory: {output_dir}")
    print(f"SQLite file directory: {sql_file_dir}")
    print(f"Videos will be saved to: {sql_file_dir}")

    # Calculate global axis limits
    x_min, x_max = data[x_col].min(), data[x_col].max()
    y_min, y_max = data[y_col].min(), data[y_col].max()

    # Create color map for tags
    unique_tags = data['shortid'].unique()
    colors = plt.get_cmap('tab20b', len(unique_tags))
    tag_color_map = {tag: colors(i) for i, tag in enumerate(unique_tags)}

    def create_daily_video(day_data, date_str):
        """Creates a video for a single day."""
        print(f"Creating video for {date_str}...")
        
        # Clean the directory
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Calculate time parameters
        start_time = day_data['Timestamp'].min()
        end_time = day_data['Timestamp'].max()
        time_window = 30  # seconds per frame
        trailing_window = 300  # 5 minutes of trailing data
        
        # Adjust frame rate based on playback speed
        effective_fps = 30 * playback_speed
        
        # Create time intervals
        time_starts = pd.date_range(start=start_time, end=end_time, freq=f'{time_window}s')

        # Create frames
        with tqdm(total=len(time_starts), desc=f"Creating frames for {date_str}") as pbar:
            for i, frame_start in enumerate(time_starts):
                frame_end = frame_start + pd.Timedelta(seconds=trailing_window)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.grid(True, alpha=0.3)

                # Plot each tag
                for tag in unique_tags:
                    tag_data = day_data[(day_data['shortid'] == tag) & 
                                       (day_data['Timestamp'] >= frame_start) & 
                                       (day_data['Timestamp'] <= frame_end)]
                    
                    if tag_data.empty:
                        continue

                    color = tag_color_map[tag]
                    
                    # Plot trailing path
                    ax.plot(tag_data[x_col], tag_data[y_col], color=color, alpha=0.6, linewidth=1.5)
                    
                    # Plot current position (last point)
                    if not tag_data.empty:
                        current_x = tag_data[x_col].iloc[-1]
                        current_y = tag_data[y_col].iloc[-1]
                        ax.plot(current_x, current_y, 'o', color=color, markersize=8)
                        ax.text(current_x, current_y + 0.2, f'Tag {tag}', fontsize=8, ha='center', color=color)

                # Set plot properties
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel('X Coordinate (meters)')
                ax.set_ylabel('Y Coordinate (meters)')
                ax.set_title(f'UWB Tag Paths - {date_str}\nTime: {frame_start.strftime("%H:%M:%S")}')

                # Save frame
                filename = os.path.join(output_dir, f"frame_{i:04d}.png")
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close(fig)
                gc.collect()

                pbar.update(1)

        # Create video using ffmpeg
        video_name = f"uwb_animation_{date_str}_{playback_speed}x.mp4"
        video_output_path = os.path.join(output_dir, video_name)
        
        print(f"Creating video with ffmpeg...")
        subprocess.call([
            'ffmpeg', '-framerate', str(effective_fps), '-i', os.path.join(output_dir, 'frame_%04d.png'),
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p', 
            video_output_path, '-y'
        ])

        # Move video to SQLite file directory
        final_video_path = os.path.join(sql_file_dir, video_name)
        shutil.move(video_output_path, final_video_path)
        
        print(f"Video saved as {final_video_path}")
        print(f"Video file exists: {os.path.exists(final_video_path)}")
        print(f"Video file size: {os.path.getsize(final_video_path) if os.path.exists(final_video_path) else 'N/A'} bytes")

        # Clean up PNG files
        for filename in os.listdir(output_dir):
            if filename.endswith(".png"):
                os.remove(os.path.join(output_dir, filename))

    # Create videos for selected days
    for date in selected_date_list:
        day_data = data[data['Date'] == date]
        if not day_data.empty:
            create_daily_video(day_data, str(date))

    print("All animations complete!")

if __name__ == "__main__":
    uwb_animate_paths()
