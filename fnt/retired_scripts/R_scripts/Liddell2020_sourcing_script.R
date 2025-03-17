## Liddell2020_sourcing_script.R
## Caleb C. Vogt, PhD Student Cornell University
## Updated on 7.21.2020
## TO DO LIST
#

#DOWNLOAD METADATA AS EXCEL FILE INTO FOLDER AGAIN FROM GOOGLE DRIVE IF THE METADATA NEEDS UPDATING
wd <- setwd("G:/My Drive/Data/7_Liddell_2020")

metadata <- read_excel("Liddell.2020.xlsx", sheet = 1, skip = 1)

parent.folder <- "Z:/7_Liddell_2020/TRIALS"
sub.folders <- list.dirs(parent.folder, recursive=TRUE)[-1]
sub.folders <- sub.folders[-c(1:10)]
sub.folders
r.scripts <- file.path(wd, "Liddell2020Script_v6.R")
# Run scripts in sub-folders 
for(i in sub.folders) {
  setwd(i)
  source(r.scripts)
}




