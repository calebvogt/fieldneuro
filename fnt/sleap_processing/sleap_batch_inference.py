import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

def prompt_model_type():
    return messagebox.askyesno("Model Type", "Are you using a TOP-DOWN model?\n(Click 'No' for bottom-up)")

def select_model_folder(title="Select model folder"):
    return filedialog.askdirectory(title=title)

def select_video_folder():
    return filedialog.askdirectory(title="Select folder containing video files for inference")

def get_output_path(video_path):
    base = os.path.basename(video_path)
    parent = os.path.dirname(video_path)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    filename = f"{base}.{timestamp}.predictions.slp"
    predictions_dir = os.path.join(parent, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    return os.path.join(predictions_dir, filename)

def run_inference_on_video(video_file, model_paths):
    cmd = ["sleap-track", video_file]

    for model_path in model_paths:
        cmd += ["-m", os.path.join(model_path, "training_config.json")]

    output_file = get_output_path(video_file)
    cmd += [
        "--only-suggested-frames",
        "--no-empty-frames",
        "--verbosity", "json",
        "--video.input_format", "channels_last",
        "--gpu", "auto",
        "--batch_size", "4",
        "--peak_threshold", "0.2",
        "--tracking.tracker", "none",
        "--controller_port", "9000",
        "--publish_port", "9001",
        "-o", output_file
    ]

    print(f"\n🔁 Running inference on: {os.path.basename(video_file)}")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd)

def main():
    root = tk.Tk()
    root.withdraw()

    # Select folder containing video files
    video_folder = select_video_folder()
    if not video_folder:
        print("No folder selected.")
        return

    # Get list of video files
    video_files = [f for f in os.listdir(video_folder)
                   if f.lower().endswith((".mp4", ".avi", ".mov"))]

    if not video_files:
        print("No video files found in folder.")
        return

    # Prompt for model type
    is_top_down = prompt_model_type()
    model_paths = []

    if is_top_down:
        model_paths.append(select_model_folder("Select CENTROID model folder"))
        model_paths.append(select_model_folder("Select CENTERED INSTANCE model folder"))
    else:
        model_paths.append(select_model_folder("Select BOTTOM-UP model folder"))

    # Run inference on each video
    for video_file in video_files:
        full_path = os.path.join(video_folder, video_file)
        run_inference_on_video(full_path, model_paths)

    print("\n✅ Inference complete for all videos.")

if __name__ == "__main__":
    main()
