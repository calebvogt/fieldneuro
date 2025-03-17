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

# rename all files in the directory, replacing white spaces with underscores
for filename in os.listdir(script_dir):
    new_filename = filename.replace(" ", "_")
    os.rename(os.path.join(script_dir, filename), os.path.join(script_dir, new_filename))

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

        # subtitle file path
        subtitle_file = os.path.join(script_dir, video_filename_no_ext + '.smi')

        # output file path
        output_file = os.path.join(out_dir, video_filename_no_ext + '.mp4')

        # ffmpeg command. crf default is 23. Add subtitle processing with the desired style.
        cmd = f'ffmpeg -y -i "{video_file}" -vcodec libx265 -preset ultrafast -crf 15 -pix_fmt yuv420p -an -r 30 -max_muxing_queue_size 10000000 -vf "scale=1920:1080:force_original_aspect_ratio=decrease:eval=frame,pad=1920:1080:-1:-1:color=black,format=gray,subtitles="{subtitle_file}":force_style='FontSize=10,Alignment=1,BorderStyle=3,Outline=1,Shadow=0,MarginV=20'" "{output_file}"'

	##for %i in (*.avi) do ffmpeg -i "%i" -vcodec libx265 -preset ultrafast -crf 15 -pix_fmt yuv420p -an -r 30 -max_muxing_queue_size 10000000 -vf "scale=1920:1080:force_original_aspect_ratio=decrease:eval=frame,pad=1920:1080:-1:-1:color=black,format=gray,subtitles="%~ni.smi":force_style='FontSize=10,Alignment=1,BorderStyle=3,Outline=1,Shadow=0,MarginV=20'" "%~ni.mp4"

        # execute the command
        subprocess.run(cmd, shell=True)
