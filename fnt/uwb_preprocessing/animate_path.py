import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
import shutil
import numpy as np
from matplotlib.patches import Polygon, Wedge
from tqdm import tqdm
import gc

def animate_path(data, filter_subject="all", filter_day="all", filter_hours="all", filter_minutes="all", time_window=30, trailing_window=1000, fps=20, color_by="ID", arena_coordinates=None, viewing_angle=45):
    """
    Creates an animation of the smoothed trajectories of animals based on the specified filter.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data. Must include columns 'Timestamp', 'smoothed_x', 'smoothed_y', 'ID', 'sex', 'Day', and 'trial'.
    filter_subject (str): The filter criteria for the data. Options are "all", "males", "females", or a specific animal ID. Default is "all".
    filter_day (str): The days to include in the animation, specified as a comma-separated string of day numbers (e.g., "1,2,3") or "all" for all days. Default is "all".
    filter_hours (str): The hours to include in the animation, specified as a comma-separated string of hour numbers (e.g., "0,1,2,3,4,5,7,10,11,12") or "all" for all hours. Default is "all".
    filter_minutes (str): The minutes to include in the animation, specified as a range of values (e.g., "0,20") or "all" for all minutes. Default is "all".
    time_window (int): The time window in seconds for each frame of the animation. Default is 30 seconds.
    trailing_window (int): The trailing window in seconds for the data to be included in each frame. Default is 1000 seconds.
    fps (int): The frames per second for the output video. Default is 20 fps.
    color_by (str): The criteria for coloring the trajectories. Options are "ID" or "sex". Default is "ID".
    arena_coordinates (pd.DataFrame): DataFrame containing the arena coordinates with columns 'zone', 'x', 'y'. Default is None.
    viewing_angle (int): The viewing angle in degrees for the shaded area in front of the animal. Default is 45 degrees.

    Raises:
    ValueError: If the 'Timestamp' column contains NaT values or if 'output_fp' is not provided.
    """
    # Ensure Timestamp is in datetime format
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

    # Check for NaT values in Timestamp
    if data['Timestamp'].isna().sum() > 0:
        raise ValueError("Timestamp column contains NaT values")

    # Calculate heading direction and velocity
    data = data.sort_values(by=['ID', 'Timestamp'])
    
    # Calculate the heading direction for each individual
    data['heading'] = data.groupby('ID', group_keys=False).apply(
        lambda group: np.arctan2(group['smoothed_y'].diff(1), group['smoothed_x'].diff(1))
    )
    
    # Smooth the heading direction using a rolling window for each individual
    data['heading'] = data.groupby('ID', group_keys=False)['heading'].apply(
        lambda group: group.rolling(window=5, min_periods=1).mean()
    )
    
    # Calculate the velocity for each individual
    data['velocity'] = data.groupby('ID', group_keys=False).apply(
        lambda group: np.sqrt(group['smoothed_x'].diff()**2 + group['smoothed_y'].diff()**2) / group['Timestamp'].diff().dt.total_seconds()
    )

    # Filter the DataFrame based on the filter_hours argument
    if filter_hours.lower() != "all":
        hours = [int(hour) for hour in filter_hours.split(',')]
        data = data[data['Timestamp'].dt.hour.isin(hours)]

    # Filter the DataFrame based on the filter_minutes argument
    if filter_minutes.lower() != "all":
        start_minute, end_minute = [int(minute) for minute in filter_minutes.split(',')]
        data = data[(data['Timestamp'].dt.minute >= start_minute) & (data['Timestamp'].dt.minute <= end_minute)]

    # Calculate the global min and max for smoothed_x and smoothed_y
    x_min, x_max = data['smoothed_x'].min(), data['smoothed_x'].max()
    y_min, y_max = data['smoothed_y'].min(), data['smoothed_y'].max()

    def create_video(df_filtered, output_dir, video_name, day):
        """Creates a video using the filtered DataFrame."""
        # Clean the directory by deleting any existing files
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Calculate the minimum and maximum Timestamp for the filtered data
        start = df_filtered['Timestamp'].min()
        end = df_filtered['Timestamp'].max()

        # Create frames for each time window interval
        time_starts = pd.date_range(start=start, end=end, freq=f'{time_window}s')

        # Define a color palette for individual IDs if color_by is "ID"
        if color_by == "ID":
            unique_ids = df_filtered['ID'].unique()
            colors = plt.get_cmap('tab20b', len(unique_ids))
            id_color_map = {ID: colors(i) for i, ID in enumerate(unique_ids)}

        # Create a progress bar
        with tqdm(total=len(time_starts), desc="Creating frames") as pbar:
            for i, frame_start in enumerate(time_starts):
                frame_end = frame_start + pd.Timedelta(seconds=trailing_window)  # Define the end frame for the trailing data
                fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size if needed

                # Turn off the grid
                ax.grid(False)

                # Plot the arena coordinates if provided
                if arena_coordinates is not None:
                    for zone in arena_coordinates['zone'].unique():
                        zone_coords = arena_coordinates[arena_coordinates['zone'] == zone][['x', 'y']].values
                        polygon = Polygon(zone_coords, closed=True, edgecolor='black', facecolor='none', linewidth=1.5)
                        ax.add_patch(polygon)

                # Plot the current positions and the previous trailing_window seconds of data
                for ID in df_filtered['ID'].unique():
                    # Get the data for the trailing line (previous trailing_window seconds)
                    trailing_data = df_filtered[(df_filtered['ID'] == ID) & 
                                                (df_filtered['Timestamp'] >= frame_start) & 
                                                (df_filtered['Timestamp'] <= frame_end)]

                    # Skip if there's no data for this period
                    if trailing_data.empty:
                        continue

                    # Determine color based on the color_by parameter
                    if color_by == "sex":
                        color = 'blue' if trailing_data['sex'].values[0] == 'M' else 'red'
                    elif color_by == "ID":
                        color = id_color_map[ID]

                    # Plot the trailing line
                    ax.plot(trailing_data['smoothed_x'], trailing_data['smoothed_y'], color=color, alpha=0.5, linewidth=1)

                    # Plot the current position (last point in the trailing data)
                    current_x = trailing_data['smoothed_x'].values[-1]
                    current_y = trailing_data['smoothed_y'].values[-1]
                    ax.plot(current_x, current_y, 'o', color=color)

                    # Add the ID label above the dot
                    ax.text(current_x, current_y + 0.2, ID, fontsize=8, ha='center', color=color)

                    # Plot the heading direction if the animal is moving above a velocity threshold
                    if trailing_data['velocity'].values[-1] > 0.01:  ## set the velocity threshold here
                        heading = trailing_data['heading'].values[-1]
                        wedge = Wedge((current_x, current_y), 0.3, np.degrees(heading) - viewing_angle / 2, np.degrees(heading) + viewing_angle / 2, color=color, alpha=0.3)
                        ax.add_patch(wedge)

                # Add a dynamic title to the plot, Day and Date-Time
                ax.set_title(f"{day}\nTime: {frame_start.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=14)

                # Add X and Y axis labels
                ax.set_xlabel("X Coordinate (meters)")
                ax.set_ylabel("Y Coordinate (meters)")

                # Set the axis limits
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

                # Save the frame as a PNG file
                filename = os.path.join(output_dir, f"frame_{i:04d}.png")
                plt.savefig(filename, dpi=300)
                plt.close(fig)  # Close the figure to suppress output in Jupyter Notebook

                # Explicitly close the figure to free up memory
                plt.close(fig)
                gc.collect()  # Run garbage collection to free up memory

                # Update the progress bar
                pbar.update(1)

        # Use ffmpeg to create the video from the saved frames
        video_output_path = os.path.join(output_dir, f'{video_name}.mp4')
        subprocess.call([
            'ffmpeg', '-framerate', str(fps), '-i', os.path.join(output_dir, 'frame_%04d.png'),
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18', '-pix_fmt', 'yuv420p', video_output_path
        ])

        print(f"Animation saved as {video_output_path}")

        # Move the video to the specified output path
        shutil.move(video_output_path, os.path.join(output_fp, f'{video_name}.mp4'))

        # Delete all the PNG files after creating the video
        for filename in os.listdir(output_dir):
            if filename.endswith(".png"):
                os.remove(os.path.join(output_dir, filename))

        print(f"All PNG files have been deleted.")

    # Main function starts here
    # Determine the desktop path
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    output_fp = desktop_path

    # Define the output directory for temporary files
    output_dir = os.path.join(output_fp, "animation_frames")
    os.makedirs(output_dir, exist_ok=True)

    # Filter the DataFrame based on the filter_subject argument
    if filter_subject.lower() == "males":
        df_filtered = data[data['sex'] == 'M']
    elif filter_subject.lower() == "females":
        df_filtered = data[data['sex'] == 'F']
    elif filter_subject.lower() != "all":
        df_filtered = data[data['ID'] == filter_subject]
    else:
        df_filtered = data  # No filtering, use the entire DataFrame

    # Get the trial information
    trial = df_filtered['trial'].iloc[0]

    # Process the filter_day argument
    if filter_day.lower() == "all":
        # Create a single video for all days
        create_video(df_filtered, output_dir, f"{trial}_All_Days_Animation", "All Days")
    else:
        # Process each day individually
        day_range = [int(day) for day in str(filter_day).split(',')]
        for day in day_range:
            df_day_filtered = df_filtered[df_filtered['Day'] == day]
            create_video(df_day_filtered, output_dir, f"{trial}_Day{day}_animation", f"Day {day}")