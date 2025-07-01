import os
import sqlite3
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from tkinter import Tk, messagebox
from tkinter.filedialog import askopenfilename
import glob

def uwb_smoothing(smoothing='savitzky-golay'):
    # Initialize Tkinter and prompt file
    root = Tk()
    root.withdraw()

    db_file = askopenfilename(title="Select SQLite database file", filetypes=[("SQLite files", "*.sqlite")])
    if not db_file:
        print("No file selected. Exiting.")
        return

    # Ask user what to export
    export_raw = messagebox.askyesno("Export Raw?",
                                     "Do you want to export the full raw database (no smoothing or downsampling)?")
    export_smoothed = messagebox.askyesno("Export Smoothed?",
                                          "Do you want to export smoothed full-resolution trajectories?")
    export_1hz = messagebox.askyesno("Export 1Hz Smoothed?",
                                     "Do you want to export smoothed trajectories downsampled to 1Hz?")

    if not any([export_raw, export_smoothed, export_1hz]):
        print("No export options selected. Exiting.")
        return

    print("Selected:", db_file)
    base_filename = os.path.splitext(os.path.basename(db_file))[0]
    output_dir = os.path.dirname(db_file)
    chunk_size = 10000000

    # Prepare output collectors
    raw_chunks = []
    smoothed_chunks = []
    downsampled_chunks = []

    # Read + process chunks
    conn = sqlite3.connect(db_file)
    offset = 0
    chunk_index = 0

    while True:
        query = f"SELECT * FROM data LIMIT {chunk_size} OFFSET {offset}"
        uwb = pd.read_sql_query(query, conn)
        if uwb.empty:
            break

        print(f"\nProcessing chunk {chunk_index} starting at offset {offset} with {len(uwb)} rows")
        uwb['Timestamp'] = pd.to_datetime(uwb['timestamp'], unit='ms', origin='unix', utc=True)
        uwb['location_x'] *= 0.0254
        uwb['location_y'] *= 0.0254
        uwb['HexID'] = uwb['shortid'].apply(lambda x: format(x, 'x')).astype(str)
        uwb = uwb.sort_values(by=['HexID', 'Timestamp'])

        # Save raw if requested
        if export_raw:
            raw_chunks.append(uwb.copy())

        # Compute shared fields for processed versions
        uwb['time_diff'] = uwb.groupby('HexID')['Timestamp'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
        uwb['distance'] = np.sqrt((uwb['location_x'] - uwb.groupby('HexID')['location_x'].shift())**2 +
                                  (uwb['location_y'] - uwb.groupby('HexID')['location_y'].shift())**2)
        uwb['velocity'] = uwb['distance'] / uwb['time_diff']
        uwb = uwb[(uwb['velocity'] <= 2) | (uwb['velocity'].isna())]
        uwb['is_jump'] = (uwb['distance'] > 2)
        uwb = uwb[~uwb['is_jump']]
        uwb['time_diff_s'] = np.ceil(uwb['time_diff']).astype(int)
        uwb['tw_group'] = uwb.groupby('HexID')['time_diff_s'].apply(lambda x: (x > 30).cumsum()).reset_index(level=0, drop=True)

        # Smoothing function
        def apply_savgol_filter(group):
            window_length = min(31, len(group))
            if window_length % 2 == 0:
                window_length -= 1
            polyorder = min(2, window_length - 1)
            return savgol_filter(group, window_length=window_length, polyorder=polyorder)

        # Smoothed full resolution
        if export_smoothed:
            smoothed = uwb.copy()
            if smoothing == 'savitzky-golay':
                smoothed['smoothed_x'] = smoothed.groupby(['HexID', 'tw_group'])['location_x'].transform(apply_savgol_filter)
                smoothed['smoothed_y'] = smoothed.groupby(['HexID', 'tw_group'])['location_y'].transform(apply_savgol_filter)
            elif smoothing == 'rolling-average':
                smoothed['smoothed_x'] = smoothed.groupby(['HexID', 'tw_group'])['location_x'].transform(lambda x: x.rolling(30, min_periods=1).mean())
                smoothed['smoothed_y'] = smoothed.groupby(['HexID', 'tw_group'])['location_y'].transform(lambda x: x.rolling(30, min_periods=1).mean())
            else:
                smoothed['smoothed_x'] = smoothed['location_x']
                smoothed['smoothed_y'] = smoothed['location_y']
            smoothed_chunks.append(smoothed)

        # Smoothed + 1Hz downsampled
        if export_1hz:
            down = uwb.copy()
            down['time_sec'] = (down['Timestamp'].astype(np.int64) // 1_000_000_000).astype(int)
            down = down.groupby(['HexID', 'time_sec']).first().reset_index()

            down['time_diff'] = down.groupby('HexID')['Timestamp'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
            down['distance'] = np.sqrt((down['location_x'] - down.groupby('HexID')['location_x'].shift())**2 +
                                       (down['location_y'] - down.groupby('HexID')['location_y'].shift())**2)
            down['velocity'] = down['distance'] / down['time_diff']
            down = down[(down['velocity'] <= 2) | (down['velocity'].isna())]
            down['is_jump'] = (down['distance'] > 2)
            down = down[~down['is_jump']]
            down['time_diff_s'] = np.ceil(down['time_diff']).astype(int)
            down['tw_group'] = down.groupby('HexID')['time_diff_s'].apply(lambda x: (x > 30).cumsum()).reset_index(level=0, drop=True)

            if smoothing == 'savitzky-golay':
                down['smoothed_x'] = down.groupby(['HexID', 'tw_group'])['location_x'].transform(apply_savgol_filter)
                down['smoothed_y'] = down.groupby(['HexID', 'tw_group'])['location_y'].transform(apply_savgol_filter)
            elif smoothing == 'rolling-average':
                down['smoothed_x'] = down.groupby(['HexID', 'tw_group'])['location_x'].transform(lambda x: x.rolling(30, min_periods=1).mean())
                down['smoothed_y'] = down.groupby(['HexID', 'tw_group'])['location_y'].transform(lambda x: x.rolling(30, min_periods=1).mean())
            else:
                down['smoothed_x'] = down['location_x']
                down['smoothed_y'] = down['location_y']
            downsampled_chunks.append(down)

        offset += chunk_size
        chunk_index += 1

    conn.close()

    # Write final outputs
    if export_raw:
        raw_df = pd.concat(raw_chunks)
        raw_path = os.path.join(output_dir, f"{base_filename}_raw.csv")
        raw_df.to_csv(raw_path, index=False)
        print(f"✅ Saved raw data to {raw_path}")

    if export_smoothed:
        smooth_df = pd.concat(smoothed_chunks)
        smooth_path = os.path.join(output_dir, f"{base_filename}_smoothed.csv")
        smooth_df.to_csv(smooth_path, index=False)
        print(f"✅ Saved full-resolution smoothed data to {smooth_path}")

    if export_1hz:
        down_df = pd.concat(downsampled_chunks)
        down_path = os.path.join(output_dir, f"{base_filename}_smoothed_1hz.csv")
        down_df.to_csv(down_path, index=False)
        print(f"✅ Saved 1Hz smoothed data to {down_path}")

    print("🎉 Done!")

if __name__ == "__main__":
    uwb_smoothing()
