## modified from: https://colab.research.google.com/drive/13Mt5we4keXRbMwJPXSjEogVucrqh81GF?usp=sharing
## in progress, not functional at the moment. 
import os
import subprocess
import glob
import re
import numpy as np
import pandas as pd
import h5py

# get directory where the script is located (copy and paste into folder with .h5 files)
script_dir = os.path.dirname(os.path.realpath(__file__))

# directory to store converted videos
out_dir = os.path.join(script_dir, "sleap_csvs")

# create directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

# list of video file extensions
video_extensions = ["*.h5"]

# loop over each video extension type
for video_extension in video_extensions:
    # list all video files
    video_files = glob.glob(os.path.join(script_dir, video_extension))

    # loop over each video file
    for video_file in video_files:
        # get the video filename without extension
        video_filename = os.path.basename(video_file) ## might not include the full filepath name. 
        video_filename_no_ext = re.sub(r'\.avi|\.mp4|\.mov', '', video_filename)

        # Open the HDF5 file using h5py.
        with h5py.File(video_filename, "r") as f:

        # Print a list of the keys available.
        print("Keys in the HDF5 file:", list(f.keys()))

        # Load all the datasets into a dictionary.
        data = {k: v[:] for k, v in f.items()}

        # Here we're just converting string arrays into regular Python strings.
        data["node_names"] = [s.decode() for s in data["node_names"].tolist()]
        data["track_names"] = [s.decode() for s in data["track_names"].tolist()]

        # And we just flip the order of the tracks axes for convenience.
        data["tracks"] = np.transpose(data["tracks"])

        # And finally convert the data type of the track occupancy array to boolean.
        # We'll see what this array is used for further down.
        data["track_occupancy"] = data["track_occupancy"].astype(bool)


        # Describe the values in the data dictionary we just created.
        for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.dtype} array of shape {value.shape}")
        else:
            print(f"{key}: {value}")

        valid_frame_idxs = np.argwhere(data["track_occupancy"].any(axis=1)).flatten()
        valid_frame_idxs

        tracks = []
        for frame_idx in valid_frame_idxs:
        # Get the tracking data for the current frame.
        frame_tracks = data["tracks"][frame_idx]

        # Loop over the animals in the current frame.
        for i in range(frame_tracks.shape[-1]):
            pts = frame_tracks[..., i]
            
            if np.isnan(pts).all():
            # Skip this animal if all of its points are missing (i.e., it wasn't
            # detected in the current frame).
            continue
            
            # Let's initialize our row with some metadata.
            detection = {"track": data["track_names"][i], "frame_idx": frame_idx}

            # Now let's fill in the coordinates for each body part.
            for node_name, (x, y) in zip(data["node_names"], pts):
            detection[f"{node_name}.x"] = x
            detection[f"{node_name}.y"] = y

            # Add the row to the list and move on to the next detection.
            tracks.append(detection)

        # Once we're done, we can convert this list of rows into a table using Pandas.
        tracks = pd.DataFrame(tracks)
        tracks.to_csv("tracks.csv", index=False)
        files.download("tracks.csv")






        