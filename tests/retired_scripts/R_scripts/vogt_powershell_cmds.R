
# Windows Powershell Video Conversion Code --------------------------------

system("cd f:")
shell("cd f:")


ffmpeg -i "CAM15 2018-07-19 13 02 48 600.avi" -vf "subtitles=CAM15 2018-07-19 13 02 48 600.smi:force_style='Fontsize=16,Alignment=1'" -q 1 -r 10 -max_muxing_queue_size 100000 output.avi


#spaces are the biggest problem with getting the code to run, including the spaces in the file names. Therefore, putting the .avi file name in quotes allows the code to run.

# Alterantive method, changing all .avi and .smi file names into numerics, thereby eliminating the spaces. This allows you to remove the double quotes. However, requires code that cycles through a folder and changing the file names to numbers. 

ffmpeg -i 1.avi -vf "subtitles=1.smi:force_style='Fontsize=16,Alignment=1'" -q 1 -r 10 -max_muxing_queue_size 100000 output.avi


# powershell bash loop temkplate. 
find . -type f -name "*.mp4" -exec bash -c 'FILE="$1"; ffmpeg -i "${FILE}" -s 1280x720 -acodec copy -y "${FILE%.mp4}.shrink.mp4";' _ '{}' \;


find . -type f -name "*.avi" -exec bash -c 'FILE="$1"; ffmpeg -i "${FILE}" -vf "subtitles="

-s 1280x720 -acodec copy -y "${FILE%.mp4}.shrink.mp4";' _ '{}' \;

