## batch_sleap_inference.py

## > cd /d working_directory_with_videos   # RUN IN FOLDER WITH MODELS AND PREDICTIONS FOLDER ALREADY. 
## > conda activate sleap_env
## > spyder
## copy and paste batch_sleap_processing.py into a new python file. run individually. 
## or in cmd prompt > python batch_sleap_inference.py

# https://sleap.ai/notebooks/Training_and_inference_using_Google_Drive.html 
# add in the tracker information for single animal.... need to connect predictions. 
# how to modify this for multi animal. 
## sleap-track --help

import os
import subprocess


### FOR SINGLE ANIMAL MODEL, CHANGE 
# subprocess.call('(for %i in (*.mp4) do sleap-track ''%i'' -m "SLEAP_Batch_1/models/210311_105936.single_instance.n=928" --tracking.tracker flow)', shell=True)   

### FOR MULTI-ANIMAL MODEL, CHANGE 
# subprocess.call('(for %i in (*.mp4) do sleap-track ''%i'' -m "SLEAP_4_wasps_v3/models/210324_093810.centroid.n=238" -m "SLEAP_4_wasps_v3/models/210324_100652.centered_instance.n=238" --tracking.tracker simple)', shell=True)   
subprocess.call('(for %i in (*.mp4) do sleap-track ''%i'' -m "SLEAP_v2/models/210420_212809.multi_instance.n=207" --no-empty-frames --batch_size 16 --tracking.tracker simple --tracking.target_instance_count 2 --tracking.pre_cull_to_target 2 --tracking.pre_cull_iou_threshold 0.9 --tracking.post_connect_single_breaks 2 --tracking.clean_instance_count 2 --tracking.similarity centroid --tracking.match greedy --tracking.track_window 60)', shell=True) 

