import os
import subprocess
import glob
import re

# Get directory where the script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Directory to store converted videos
out_dir = os.path.join(script_dir, "proc")

# Create directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

# List of video file extensions
video_extensions = ["*.avi", "*.mp4", "*.mov"]

# Loop over each video extension type
for video_extension in video_extensions:
    # List all video files
    video_files = glob.glob(os.path.join(script_dir, video_extension))

    # Loop over each video file
    for video_file in video_files:
        # Get the video filename without extension
        video_filename = os.path.basename(video_file)
        video_filename_no_ext = re.sub(r'\.avi|\.mp4|\.mov', '', video_filename)

        # Paths for intermediate and output files
        stab_vector_file = os.path.join(out_dir, video_filename_no_ext + '_stabilization.trf')
        stabilized_video_file = os.path.join(out_dir, video_filename_no_ext + '_stabilized.mp4')
        output_file = os.path.join(out_dir, video_filename_no_ext + '.mp4')

        # Step 1: Analyze the video for stabilization
        cmd_analyze = f'ffmpeg -i "{video_file}" -vf vidstabdetect=shakiness=10:accuracy=15:result="{stab_vector_file}" -f null -'
        subprocess.run(cmd_analyze, shell=True)

        # Step 2: Stabilize the video
        cmd_stabilize = f'ffmpeg -i "{video_file}" -vf vidstabtransform=input="{stab_vector_file}" -c:v libx264 -preset fast -crf 18 -an "{stabilized_video_file}"'
        subprocess.run(cmd_stabilize, shell=True)

        # Step 3: Downsample and compress the stabilized video
        cmd_compress = f'ffmpeg -i "{stabilized_video_file}" -vcodec libx265 -preset ultrafast -crf 15 -pix_fmt yuv420p -vf "scale=1920:1080:force_original_aspect_ratio=decrease:eval=frame,pad=1920:1080:-1:-1:color=black" -an -r 30 -max_muxing_queue_size 10000000 "{output_file}"'
        subprocess.run(cmd_compress, shell=True)
