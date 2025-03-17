## https://sleap.ai/api/sleap.io.convert.html?highlight=io%20convert#module-sleap.io.convert
## https://sleap.ai/notebooks/Analysis_examples.html

import os
import subprocess

### FOR MULTI-ANIMAL TOP DOWN MODEL, #4. Does not open gui after tracking. 
subprocess.call('(for %i in (*.slp) do sleap-convert ''%i'' --format analysis)', shell=True)   


