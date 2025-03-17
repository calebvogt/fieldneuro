import os
import subprocess
import platform

# Check the operating system
os_name = platform.system()

if os_name == "Windows":
    # Command for Windows
    cmd = (
        'mkdir proc && for %f in (*.wav) do ffmpeg -i "%f" -ar 250000 -ac 1 -c:a adpcm_ima_wav "proc\\%~nf.wav"'
    )
elif os_name == "Darwin":  # macOS
    # Command for macOS (bash shell)
    cmd = (
        'mkdir -p proc && for file in *.wav; do ffmpeg -i "$file" -ar 250000 -ac 1 -c:a adpcm_ima_wav "proc/${file}"; done'
    )
else:
    print(f"Unsupported OS: {os_name}")
    exit()

# Run the appropriate command in a subprocess
try:
    print(f"Detected OS: {os_name}. Running the compression command...")
    subprocess.run(cmd, shell=True, check=True)
    print("Compression completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while running the command: {e}")
