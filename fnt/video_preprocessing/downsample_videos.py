# to do: 
# if there are smi timestamps from the security cameras in the folder, prompt the user if the timestamps should be hardcoded into the videos, potentially even in a specific location. 


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
            "-vcodec", "libx265", "-preset", "ultrafast", "-crf", "15",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease:eval=frame,pad=1920:1080:-1:-1:color=black,format=gray",
            "-an", "-r", "30", "-max_muxing_queue_size", "10000000",
            output_file
        ]

        print(f"Processing: {video_filename} -> {output_file}")

        # Execute FFmpeg and print output in real-time
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line.strip())

        process.wait()

        print(f"✅ Done: {output_file}")

    print("🎉 All videos processed and saved in 'proc' folder!")

# Run function if executed directly
if __name__ == "__main__":
    downsample_videos()
