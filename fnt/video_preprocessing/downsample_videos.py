# TODO: if there are smi timestamps from the security cameras in the folder, prompt the user if the timestamps should be hardcoded into the videos, potentially even in a specific location. 
# Solution for now: use this only for videos with .smi files. Harvested from old CCV scripts. getting this in a python script is more hassle than its worth at this time. 
## open powershell window as administrator or Shift + Right Clickopen powershell window. 
## rename all files if there are spaces in the filenames. 
# Dir | Rename-Item –NewName { $_.name –replace “ “,”_” }

## add pre-fix to all files in a folder in powershell (optional)
#(Get-ChildItem -File) | Rename-Item -NewName { "T004_OB_" + $_.Name }

## start command line within powershell
# cmd

## Functional for downsampling and adding subtitles from iDVR-pro videos that come with .smi files. However, this creates thousands of duplicate frames which is DUE TO THE SMI FILE. 
# for %i in (*.avi) do ffmpeg -i "%i" -vcodec libx265 -preset ultrafast -crf 18 -pix_fmt yuv420p -r 30 -vsync cfr -an -max_muxing_queue_size 10000000 -vf "scale=1920:1080:force_original_aspect_ratio=decrease:eval=frame,pad=1920:1080:-1:-1:color=black,format=gray,subtitles="%~ni.smi":force_style='FontSize=10,Alignment=1,BorderStyle=3,Outline=1,Shadow=0,MarginV=20'" "%~ni.mp4"


import os
import subprocess
import glob
import re
import tkinter as tk
from tkinter import filedialog

def downsample_videos():
    """Allows the user to select a folder, then downsamples all videos in that folder using FFmpeg."""
    
    # Open file explorer to select folder
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(title="Select Folder with Videos")

    if not folder_path:
        print("No folder selected. Exiting...")
        return

    print(f"Selected folder: {folder_path}")

    # Directory to store converted videos
    out_dir = os.path.join(folder_path, "proc")
    os.makedirs(out_dir, exist_ok=True)

    # List of video file extensions
    video_extensions = ["*.avi", "*.mp4", "*.mov"]

    # Find all videos in the folder
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not video_files:
        print("No video files found in the selected folder.")
        return

    # Process each video
    for video_file in video_files:
        video_filename = os.path.basename(video_file)
        video_filename_no_ext = re.sub(r'\.avi|\.mp4|\.mov', '', video_filename)
        output_file = os.path.join(out_dir, f"{video_filename_no_ext}.mp4")

        # FFmpeg command
        cmd = [
            "ffmpeg", "-i", video_file,
            "-vcodec", "libx265", "-preset", "ultrafast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease:eval=frame,pad=1920:1080:-1:-1:color=black,format=gray",
            "-r", "30", # force constant frame rate at 30fps
            "-vsync", "cfr", # CFR mode
            "-an", # disable audio
            "-max_muxing_queue_size", "10000000",
            output_file
        ]

        print(f"Processing: {video_filename} -> {output_file}")

        # Execute FFmpeg and print output in real-time
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line.strip())

        process.wait()

        print(f"Done: {output_file}")

    print("All videos processed and saved in 'proc' folder!")

# Run function if executed directly
if __name__ == "__main__":
    downsample_videos()
