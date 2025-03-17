## Drop this file into the folder with videos already with hardsubs
#that need to be batch appended, then run it by double clicking the
##.py file or hitting F5

import os

import subprocess

for f in os.listdir("."):
    r = f.replace(" ","")  ## As written, removes spaces. swap "." to add period.
    if( r != f):
        os.rename(f,r)

subprocess.call('(for %i in (*.avi) do @echo file ''%i'') > mylist.txt', shell=True)

subprocess.call('ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.avi', shell=True)

