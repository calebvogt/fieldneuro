from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
import sqlite3
import os
import re
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, Point
import glob

def preprocess_uwb(downsample_rate='off', smoothing='savitzky-golay'):
    """
    Preprocess UWB data from an SQLite database and save the processed data to a CSV file in the same directory

    Parameters:
    downsample_rate (str or int): Downsampling rate in Hz (1-30) or 'off'. Default is 'off'.
    smoothing (str): Smoothing method ('off', 'savitzky-golay', 'rolling-average'). Default is 'savitzky-golay'.
    """
    # Hide the root Tkinter window
    Tk().withdraw()

    # Open a file dialog to select the SQLite database file
    db_file = askopenfilename(title="Select SQLite database file", filetypes=[("SQLite files", "*.sqlite")])
    print("Selected SQLite database file path:", db_file)
    conn = sqlite3.connect(db_file)

    # Derive the output directory from the database file path
    output_dir = os.path.dirname(db_file)
    print("Output directory:", output_dir)

    # Extract the trial ID from the parent folder in the path
    trial_id_match = re.search(r'T\d{3}', db_file)
    if trial_id_match:
        trial_id = trial_id_match.group(0)
        print("Extracted trial ID:", trial_id)
    else:
        raise ValueError("Could not extract trial ID from the database file path")

    # Define a chunk size (number of rows to fetch at a time)
    chunk_size = 10000000  # Adjust based on your memory capacity

    # Initialize variables for iteration
    offset = 0
    chunk_index = 0
    has_more_data = True

    # Process data in chunks
    while has_more_data:
        # Fetch a chunk of data from the database
        query = f"""
        SELECT *
        FROM data
        LIMIT {chunk_size} OFFSET {offset}
        """
        
        uwb = pd.read_sql_query(query, conn)
        
        if uwb.empty:
            has_more_data = False
        else:
            print(f"\nProcessing chunk {chunk_index} starting at offset {offset} with {len(uwb)} rows")

            # Initial row count
            initial_rows = len(uwb)

            # Create trial column
            uwb['trial'] = trial_id

            # Create field time with ms
            uwb['Timestamp'] = pd.to_datetime(uwb['timestamp'], unit='ms', origin='unix', utc=True)

            # Adjust for timezone (e.g., UTC-4)
            # uwb['Timestamp'] = uwb['Timestamp'] - pd.Timedelta(hours=4)
            uwb['Timestamp'] = uwb['Timestamp'].dt.tz_convert('America/New_York')

            # Remove timezone information after adjusting for -4 hours; keep field_time timezone naive
            uwb['Timestamp'] = uwb['Timestamp'].dt.tz_localize(None)

            # Convert location coordinates to meters from inches (from the wiser software)
            uwb['location_x'] *= 0.0254
            uwb['location_y'] *= 0.0254

            # Convert shortid to hex_id
            uwb['HexID'] = uwb['shortid'].apply(lambda x: format(x, 'x'))
            
            # Ensure HexID is of type string
            uwb['HexID'] = uwb['HexID'].astype(str)
        
            # Print unique HexID values from the DataFrame
            print("Unique HexID values in uwb DataFrame:", uwb['HexID'].unique())

            # Downsampling
            if downsample_rate != 'off':
                downsample_rate = int(downsample_rate)
                uwb['Timestamp_s'] = uwb['Timestamp'].dt.floor(f'{int(1/downsample_rate)}S')
                uwb = uwb.groupby(['HexID', 'Timestamp_s']).first().reset_index()

            # Sort the data by HexID and time to ensure proper calculation of velocity
            uwb = uwb.sort_values(by=['HexID', 'Timestamp'])

            # Calculate the time difference in seconds
            uwb['time_diff'] = uwb.groupby('HexID')['Timestamp'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()

            # Calculate distances and velocities between consecutive points within each HexID
            uwb['distance'] = np.sqrt((uwb['location_x'] - uwb.groupby('HexID')['location_x'].shift())**2 + 
                                      (uwb['location_y'] - uwb.groupby('HexID')['location_y'].shift())**2)
            uwb['velocity'] = uwb['distance'] / uwb['time_diff']

            # Filtering based on velocity, acceleration, and large jumps
            # Apply the velocity filter: remove any rows where the velocity exceeds a set threshold (e.g., 2 meters/second)
            velocity_threshold = 2  # meters/second
            uwb_before_velocity_filter = len(uwb)
            uwb = uwb[(uwb['velocity'] <= velocity_threshold) | (uwb['velocity'].isna())]
            after_velocity_filter = len(uwb)
            print(f"Rows after velocity filtering: {after_velocity_filter} (removed {uwb_before_velocity_filter - after_velocity_filter})")

            # Detect sudden jumps and filter out
            jump_threshold = 2  # in meters
            uwb['is_jump'] = (uwb['distance'] > jump_threshold)
            single_jumps = uwb['is_jump'].sum()
            uwb_before_jump_filter = len(uwb)
            uwb = uwb[~uwb['is_jump']]
            after_jump_filter = len(uwb)
            print(f"Number of single jumps detected and removed: {single_jumps}")
            # print(f"Rows after jump filtering: {after_jump_filter} (removed {uwb_before_jump_filter - after_jump_filter})")

            # Group consecutive points that fall within a set time window (in seconds)
            uwb['time_diff_s'] = np.ceil(uwb['time_diff']).astype(int)
            uwb['tw_group'] = uwb.groupby('HexID')['time_diff_s'].apply(lambda x: (x > 30).cumsum()).reset_index(level=0, drop=True)

            # Smoothing
            if smoothing == 'savitzky-golay':
                def apply_savgol_filter(group):
                    window_length = min(31, len(group))  # Ensure window_length is not greater than the group size
                    if window_length % 2 == 0:
                        window_length -= 1  # Ensure window_length is odd
                    polyorder = min(2, window_length - 1)  # Ensure polyorder is less than window_length
                    return savgol_filter(group, window_length=window_length, polyorder=polyorder)
                
                uwb['smoothed_x'] = uwb.groupby(['HexID', 'tw_group'])['location_x'].transform(apply_savgol_filter)
                uwb['smoothed_y'] = uwb.groupby(['HexID', 'tw_group'])['location_y'].transform(apply_savgol_filter)
                print(f"Smoothed xy trajectories using Savitzky-Golay filter")
            elif smoothing == 'rolling-average':
                uwb['smoothed_x'] = uwb.groupby(['HexID', 'tw_group'])['location_x'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
                uwb['smoothed_y'] = uwb.groupby(['HexID', 'tw_group'])['location_y'].transform(lambda y: y.rolling(window=30, min_periods=1).mean())
                print(f"Smoothed xy trajectories using rolling average")
            else:
                uwb['smoothed_x'] = uwb['location_x']
                uwb['smoothed_y'] = uwb['location_y']
                print(f"No smoothing applied")

            final_rows = len(uwb)
            print(f"Final number of rows in this chunk: {final_rows}")
            
            # print(uwb.head())

            # Select final minimum required columns for export
            uwb = uwb[['trial', 'HexID', 'Timestamp', 'smoothed_x', 'smoothed_y', 'location_x', 'location_y',
                       'shortid', 'arenaid', 'arenaname', 'calculation_error', 'anchors_used', 'timestamp']]
            
            # print(uwb.head())

            # Write the processed chunk to a separate CSV file
            chunk_output_file = os.path.join(output_dir, f'output_{chunk_index}.csv')
            uwb.to_csv(chunk_output_file, index=False)
            
            # Increment the chunk index
            chunk_index += 1
        
        # Update offset to fetch the next chunk in the next iteration
        offset += chunk_size

    # Close the connection
    conn.close()

    # List all chunk files
    chunk_files = glob.glob(os.path.join(output_dir, 'output_*.csv'))

    # Read and concatenate all chunk files
    output_df = pd.concat([pd.read_csv(f) for f in chunk_files])

    # Write the concatenated data to the final output file
    # final_output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(db_file))[0] + '_processed.csv')
    final_output_file = os.path.join(output_dir, f"{trial_id}_" + os.path.splitext(os.path.basename(db_file))[0] + '_processed.csv')
    output_df.to_csv(final_output_file, index=False)

    # Optionally, remove the chunk files to free up space
    for f in chunk_files:
        os.remove(f)

    print(f"All chunks have been concatenated and saved to {final_output_file}. Chunk files have been deleted.")
    print(f"Preprocessed UWB data stored in output_df")
    print(len(output_df))
    print(output_df.head())