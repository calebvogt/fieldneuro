import os
import subprocess
import glob
import re
import tkinter as tk
from tkinter import filedialog

def sanitize_filename(file_path):
    """Replaces spaces in filenames with underscores and returns the new path."""
    if not os.path.exists(file_path):
        return file_path  # If file doesn't exist, return original path

    dir_name, file_name = os.path.split(file_path)
    sanitized_name = file_name.replace(" ", "_")
    new_path = os.path.join(dir_name, sanitized_name)

    if file_path != new_path:
        os.rename(file_path, new_path)
        print(f"Renamed: {file_path} -> {new_path}")

    return new_path

def ensure_utf8_encoding(srt_path):
    """Ensures the .srt file is saved in UTF-8 encoding to prevent FFmpeg errors."""
    try:
        with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Re-encoded {srt_path} as UTF-8")
    except Exception as e:
        print(f"Failed to re-encode {srt_path}: {e}")

def convert_smi_to_srt(smi_path):
    """Converts a .smi subtitle file to .srt format using FFmpeg and ensures UTF-8 encoding."""
    srt_path = smi_path.replace(".smi", ".srt")
    cmd = ["ffmpeg", "-i", smi_path, srt_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if os.path.exists(srt_path):
        ensure_utf8_encoding(srt_path)  # Ensures correct encoding
        print(f"Converted {smi_path} -> {srt_path}")
        return srt_path
    else:
        print(f"Failed to convert {smi_path} to .srt")
        return None

def downsample_videos():
    """Downsamples all videos in a selected folder using FFmpeg. Embeds subtitles if .smi exists."""
    # GUI folder picker
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder with Videos")
    if not folder_path:
        print("No folder selected. Exiting...")
        return

    print(f"Selected folder: {folder_path}")

    # Output directory
    out_dir = os.path.join(folder_path, "proc")
    os.makedirs(out_dir, exist_ok=True)

    # Supported input formats
    video_extensions = ["*.avi", "*.mp4", "*.mov"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not video_files:
        print("No video files found in the selected folder.")
        return

    for video_file in video_files:
        video_file = sanitize_filename(video_file)
        video_filename = os.path.basename(video_file)
        video_filename_no_ext = os.path.splitext(video_filename)[0]
        output_file = os.path.join(out_dir, f"{video_filename_no_ext}.mp4")

        # Check for subtitle file
        smi_path = sanitize_filename(os.path.join(folder_path, f"{video_filename_no_ext}.smi"))
        srt_path = None
        if os.path.exists(smi_path):
            srt_path = convert_smi_to_srt(smi_path)

        # Build the filter chain
        base_filters = (
            "scale=1920:1080:force_original_aspect_ratio=decrease:eval=frame,"
            "pad=1920:1080:-1:-1:color=black,format=gray"
        )

        if srt_path:
            subtitle_filter = (
                f"subtitles='{srt_path.replace('\\', '/')}'"
                ":force_style='FontSize=10,Alignment=1,BorderStyle=3,Outline=1,Shadow=0,MarginV=20'"
            )
            vf_filter = f"{base_filters},{subtitle_filter}"
        else:
            vf_filter = base_filters

        cmd = [
            "ffmpeg", "-i", video_file,
            "-vcodec", "libx265", "-preset", "ultrafast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-r", "30", "-vsync", "cfr",
            "-an",  # disable audio
            "-max_muxing_queue_size", "10000000",
            "-vf", vf_filter,
            output_file
        ]

        print("\nRunning FFmpeg command:")
        print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
        print("Processing...")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line.strip())
        process.wait()

        print(f"Done: {output_file}")

    print("All videos processed and saved in 'proc' folder.")

if __name__ == "__main__":
    downsample_videos()
