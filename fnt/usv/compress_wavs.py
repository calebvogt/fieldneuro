import os
import subprocess
import platform
import tkinter as tk
from tkinter import filedialog

def compress_wavs():
    # Select the folder
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select folder containing .wav files")

    if not folder_path:
        print("No folder selected.")
        return

    # Make output subfolder
    output_folder = os.path.join(folder_path, "proc")
    os.makedirs(output_folder, exist_ok=True)

    # List all .wav files
    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]

    if not wav_files:
        print("No .wav files found in the selected folder.")
        return

    print(f"Compressing {len(wav_files)} file(s) in:\n{folder_path}\n")

    for idx, wav_file in enumerate(wav_files, 1):
        input_path = os.path.join(folder_path, wav_file)
        output_path = os.path.join(output_folder, os.path.splitext(wav_file)[0] + ".wav")

        print(f"Processing file {idx} of {len(wav_files)}: {wav_file}")

        # ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-ar", "250000",
            "-ac", "1",
            "-c:a", "adpcm_ima_wav",
            output_path
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error compressing {wav_file}: {e}")

    print("\n✅ Compression completed successfully!")

# Run if script is executed
if __name__ == "__main__":
    compress_wavs()
