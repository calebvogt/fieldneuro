import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from datetime import datetime, timedelta
import re
from pathlib import Path

def summarize_annotations(file):
    try:
        df = pd.read_csv(file)
    except Exception:
        return pd.DataFrame({'filename': [os.path.basename(file)]})

    if df.empty or 'name' not in df.columns or df['name'].dropna().empty:
        return pd.DataFrame({'filename': [os.path.basename(file)]})

    df['duration'] = df['stop_seconds'] - df['start_seconds']
    summary = df.groupby('name').agg(
        call_count=('name', 'count'),
        mean_duration=('duration', 'mean')
    ).reset_index()

    if summary.empty or 'name' not in summary.columns:
        return pd.DataFrame({'filename': [os.path.basename(file)]})

    try:
        summary['filename'] = os.path.basename(file)
        summary_wide = summary.pivot(index='filename', columns='name')
        summary_wide.columns = [f'{col[1]}_{col[0]}' for col in summary_wide.columns]
        summary_wide = summary_wide.reset_index()
    except Exception as e:
        print(f"Skipping pivot for file {file}: {e}")
        return pd.DataFrame({'filename': [os.path.basename(file)]})

    return summary_wide


def parse_log_file(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()

    pattern = r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2}\.\d+)\s+.*?(\d{4}_\d{2}-\d{2}-\d{2}_\d{6}).*?\.wav"
    data = []

    for line in lines:
        match = re.search(pattern, line)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            wav_id = match.group(3)
            start_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S.%f")
            data.append({'wav_id': wav_id, 'file_start_datetime': start_dt})
    
    return pd.DataFrame(data)

def add_realtime_to_annotations(folder):
    annotation_files = list(Path(folder).glob("*_annotations.csv"))
    log_files = list(Path(folder).glob("ch*.log"))
    all_logs = []

    for log_file in log_files:
        channel = log_file.stem
        log_df = parse_log_file(log_file)
        log_df['channel'] = channel
        all_logs.append(log_df)

    logs_df = pd.concat(all_logs, ignore_index=True)

    full_df = []

    for annotation_file in annotation_files:
        base = os.path.basename(annotation_file)
        match = re.search(r"(ch\d+)_.*?(\d{4}_\d{2}-\d{2}-\d{2}_\d{6})", base)
        if not match:
            continue

        channel = match.group(1)
        wav_id = match.group(2)

        file_start_time = logs_df.query("channel == @channel and wav_id == @wav_id")["file_start_datetime"]
        if file_start_time.empty:
            continue
        file_start_time = file_start_time.iloc[0]

        df = pd.read_csv(annotation_file)
        if df.empty or 'name' not in df.columns:
            continue

        df['channel'] = channel
        df['file_name'] = base
        df['file_start_datetime'] = file_start_time
        df['call_start'] = df['start_seconds']
        df['call_stop'] = df['stop_seconds']
        df['call_start_realtime'] = df['start_seconds'].apply(lambda x: file_start_time + timedelta(seconds=x))
        df['call_stop_realtime'] = df['stop_seconds'].apply(lambda x: file_start_time + timedelta(seconds=x))

        full_df.append(df[['channel', 'file_name', 'file_start_datetime',
                           'name', 'call_start', 'call_start_realtime',
                           'call_stop', 'call_stop_realtime']])

    return pd.concat(full_df, ignore_index=True)

def das_summary():
    root = tk.Tk()
    root.withdraw()
    usv_folder = filedialog.askdirectory(title="Select folder with DAS annotation CSVs and log files")
    if not usv_folder:
        print("No folder selected.")
        return

    print("Summarizing annotation files...")
    annotation_files = list(Path(usv_folder).glob("*_annotations.csv"))
    summaries = [summarize_annotations(f) for f in annotation_files]
    summary_df = pd.concat(summaries, ignore_index=True)

    # Replace NA with 0 for call counts only
    call_count_cols = [col for col in summary_df.columns if col.endswith("_call_count")]
    for col in call_count_cols:
        summary_df[col] = summary_df[col].fillna(0)

    # Save summary CSV
    summary_out = Path(usv_folder) / "DAS_annotation_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print("Summary saved to:", summary_out)

# Call if run directly
if __name__ == "__main__":
    das_summary()
