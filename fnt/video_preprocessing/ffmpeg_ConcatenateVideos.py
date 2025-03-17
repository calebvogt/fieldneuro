import os
import subprocess

def ffmpeg_ConcatenateVideos(input_directory="."):
    """Concatenates all video files in a given directory using FFmpeg."""
    os.chdir(input_directory)  # Move into the working directory

    # List of supported video file extensions
    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")

    # Normalize filenames (removing spaces)
    for f in os.listdir("."):
        if f.endswith(VIDEO_EXTENSIONS):
            new_name = f.replace(" ", "")  # Removes spaces
            if new_name != f:
                os.rename(f, new_name)

    # Generate a list of video files in sorted order
    video_files = sorted([f for f in os.listdir(".") if f.endswith(VIDEO_EXTENSIONS)])

    # Create a text file with the video filenames for FFmpeg
    with open("mylist.txt", "w") as fp:
        for video in video_files:
            fp.write(f"file '{video}'\n")  # Correctly format filenames

    # Run FFmpeg to concatenate videos
    subprocess.call('ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.mp4', shell=True)

    # Delete mylist.txt after processing
    os.remove("mylist.txt")
    print("Concatenation complete! Output file: output.mp4")
    print("Temporary file mylist.txt deleted.")

# Only run if the script is executed directly
if __name__ == "__main__":
    ffmpeg_ConcatenateVideos()
