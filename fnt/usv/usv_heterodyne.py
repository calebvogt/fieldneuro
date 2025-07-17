import numpy as np
import soundfile as sf
import os
import tkinter as tk
from tkinter import filedialog
from scipy.signal import butter, filtfilt
import glob

def heterodyne_signal(data, rate, carrier_freq=40000, lowcut=1000, highcut=50000):
    t = np.arange(len(data)) / rate
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    mixed = data * carrier

    # Band-pass filter to isolate difference frequencies
    b, a = butter(4, [lowcut / (rate / 2), highcut / (rate / 2)], btype='band')
    filtered = filtfilt(b, a, mixed)

    return filtered

def usv_batch_heterodyne(carrier_freq=40000):
    # File dialog for selecting directory
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select folder containing .wav files")

    if not folder_path:
        print("No folder selected.")
        return

    wav_files = sorted(glob.glob(os.path.join(folder_path, "*.wav")))

    if not wav_files:
        print("No .wav files found in the selected folder.")
        return

    print(f"Found {len(wav_files)} .wav files in:\n{folder_path}\n")

    for idx, wav_path in enumerate(wav_files, 1):
        print(f"Processing file {idx} of {len(wav_files)}: {os.path.basename(wav_path)}")

        # Load and convert to float32
        data, rate = sf.read(wav_path, dtype='float32')

        # Use only first channel if multi-channel
        if data.ndim > 1:
            data = data[:, 0]

        output = heterodyne_signal(data, rate, carrier_freq)

        # Construct output file path
        base, ext = os.path.splitext(wav_path)
        out_path = base + "_heterodyned.wav"

        sf.write(out_path, output, rate)

    print("\n All files processed.")

# Run the batch processor
if __name__ == "__main__":
    usv_batch_heterodyne()
