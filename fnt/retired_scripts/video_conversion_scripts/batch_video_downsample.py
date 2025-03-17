import os
import subprocess
import glob
import re

# get directory where the script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# directory to store converted videos
out_dir = os.path.join(script_dir, "proc")

# create directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

# list of video file extensions
video_extensions = ["*.avi", "*.mp4", "*.mov"]

# loop over each video extension type
for video_extension in video_extensions:
    # list all video files
    video_files = glob.glob(os.path.join(script_dir, video_extension))

    # loop over each video file
    for video_file in video_files:
        # get the video filename without extension
        video_filename = os.path.basename(video_file)
        video_filename_no_ext = re.sub(r'\.avi|\.mp4|\.mov', '', video_filename)

        # output file path
        output_file = os.path.join(out_dir, video_filename_no_ext + '.mp4')

        # ffmpeg command. crf default is 23. I cannot tell the difference between original and -crf 15. hyper efficient compression on b/w videos.
	# videos are 30fps, grayscale, forced to 1920x1080p, and frame seekable. 
        cmd = f'ffmpeg -i "{video_file}" -vcodec libx265 -preset ultrafast -crf 15 -pix_fmt yuv420p -vf "scale=1920:1080:force_original_aspect_ratio=decrease:eval=frame,pad=1920:1080:-1:-1:color=black,format=gray" -an -r 30 -max_muxing_queue_size 10000000 "{output_file}"'

        # execute the command
        subprocess.run(cmd, shell=True)
