## Open this document as an R.Markdown file. 

# install.packages("reticulate")
library(reticulate)

#list anaconda environments. Choose one with python installed. 
reticulate::conda_list()
use_condaenv("python36", required = TRUE)

# Loads Python Shell
repl_python()

# Check the current Python version
reticulate::py_config()


```{r setup, include = FALSE}
knitr::root.dir("Z:/18_AL_Wasps_Nest_Posture/AL_wasp_v2_SLEAP")

```




```{python}
import numpy as np
import pandas as pd
np.arange(1,10)

df = pd.DataFrame(data = {"sequence":np.arange(1,20,.01)})
df = df.assign(value=np.sin(df["sequence"]))
import matplotlib.pyplot as plt
df.plot(x="sequence", y = "value", title = "Matplotlib")
plt.show()
x = 4

```

```{r}
# call python variables using py$ etc. 
py$x
```

```{python}
import os
os.getcwd()

os.chdir(Z:\18_AL_Wasps_Nest_Posture\AL_wasp_v2_SLEAP)


```


```{r}
wd <- setwd("Z:/18_AL_Wasps_Nest_Posture/AL_wasp_v2_SLEAP")

```

```{r}
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

getwd()
wd <- setwd("Z:/18_AL_Wasps_Nest_Posture/AL_wasp_v2_SLEAP")
BiocManager::install("rhdf5")
library(rhdf5)
h5ls("DSC_8565_1.avi.predictions.cleaned.analysis.h5")

# PULL OUT GROUPS FROM H5 FILE
frames <- h5read("C57-1008_Preference_D1_Estrus_030820_H2O-C57_MB2_CAM6_12_10_58_465_Boris_bandicut.mp4.predictions.slp", "frames")

instances <- h5read("C57-1008_Preference_D1_Estrus_030820_H2O-C57_MB2_CAM6_12_10_58_465_Boris_bandicut.mp4.predictions.slp", "instances")

meta <- h5read("C57-1008_Preference_D1_Estrus_030820_H2O-C57_MB2_CAM6_12_10_58_465_Boris_bandicut.mp4.predictions.slp", "metadata/H5I_GROUP")


library(hdf5r)
?h5read()
```


wd <- setwd("Z:/18_AL_Wasps_Nest_Posture/AL_wasp_v2_SLEAP")
getwd()



