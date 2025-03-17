## Liddell2020Script.R
## Caleb C. Vogt, PhD Student Cornell University
## Updated on 7.21.2020
## TO DO LIST
# 1. Generate social networks from rfid data
# 2. Have code be blind to which tags are supposed to be in a given paddock. 



library(tidyverse)
library(googlesheets)
library(readxl)
library(dplyr)
library(plyr)
library(ggplot2)
library(data.table)
library(readr)
library(reshape)
library(lubridate)
library(scales)
library(gganimate)
library(plot.matrix)
library(asnipe)
library(igraph)


# IMPORT METADATA ---------------------------------------------------------

#DOWNLOAD METADATA AS EXCEL FILE INTO FOLDER AGAIN FROM GOOGLE DRIVE IF THE METADATA NEEDS UPDATING
setwd("G:/My Drive/Data/7_Liddell_2020")

metadata <- read_excel("Liddell.2020.xlsx", sheet = 1, skip = 1)

# CHOOSE TRIAL, IMPORT RFID DATA, AND CLEAN -----------------------------------------------------

setwd("G:/My Drive/Data/7_Liddell_2020/TRIALS/T001")

# IMPORT ALL RFID DATA SAVED AS TXT FILES
txt <- list.files(pattern = "*.txt")

# IMPORT ALL TEXT FILES. BLIND TO TEXT FILE NAME. 
dfall <-   do.call(rbind, lapply(txt, function(x) read_table(file = x, col_names = TRUE, col_types = NULL,
                   locale = default_locale(), na = "NA", skip = 4, n_max = Inf,
                   progress = show_progress(), comment = "")))

# SELECT COLUMNS COLUMNS
dfall <- dfall[ , c("Scan Date", "Scan Time", "Reader ID", "Antenna ID", "DEC Tag ID")] 

# RENAME COLUMNS
colnames(dfall) <- c("scan.date", "scan.time", "reader.id", "antenna.id", "dec.tag.id")

# DROP ROWS WITH NO DEC.TAG.ID (BASICALLY ALL TRIALS WHERE ANIMALS HAVENT BEEN INJECTED YET)
dfall <- dfall[!is.na(dfall$dec.tag.id), ]

# DROP TRAILING 0'S FROM TAG DEC IDS. METADATA DOESNT SAVE THE TRAILING ZERO ON IMPORT
dfall$dec.tag.id <- sub("0*$", "", dfall$dec.tag.id)

#REMOVE DUPLICATED ROWS FROM TXT WITH REPEATED OBSERVATIONS ACROSS TIME.  
dfall <- unique(dfall)

# MERGE METADATA AND DFNEW BY DEC.TAG.ID. THIS ONLY KEEPS MATCHES. 
dfnew <- merge(dfall, metadata, by.x ="dec.tag.id", by.y = "dec.tag.id")

dfnew1 <- merge(dfall, metadata, by.x = "dec.tag.id", by.y = "dec.tag.id_2")

dfdata <- rbind.fill(dfnew, dfnew1)

# CHOOSE COLUMNS TO KEEP
dfdata <- dfdata[ , c("trial", "reader.id", "paddock", "antenna.id", "scan.date", "scan.time", "strain", "sex", "ID", "field.age", "dec.tag.id")] 

# CREATE FIELD.TIME COLUMN. NOTE THAT MILISECONDS ARE PRESENT, BUT NOT PRINTED. 
dfdata$Field.Time <- as.POSIXct(paste(dfdata$scan.date, dfdata$scan.time), format="%m/%d/%Y %H:%M:%OS")

# SORT DATAFRAME BY DATE AND TIME 
data <- dfdata[order(dfdata$Field.Time), ]



# PLOT 1: INDIVIDUAL MOVEMENT SCATTERPLOTS --------------------------------------------------------

# CHANGE ANTENNA IDS/ZONES TO NUMERICS
data$antenna.id <- as.numeric(data$antenna.id)
data$ID <- as.numeric(data$ID)

# CREATE FULL_IDS 
data$full_ids <- paste0(data$strain,"-", data$sex,"-", data$ID)

# GET UNIQUE IDS OF INDIVIDUALS PRESENT IN THIS TRIAL. 
ids <- unique(data$full_ids)

# THE CODE FOR CREATING GRAPHS ARE WITHIN THE LOOP. CYCLES THROUGH IDS 

for(i in ids[1:length(ids)]) {
  # CREATE MOVE_DF
  move_df <- data
  
  # SAVE CURRENT MOUSE AS VARIABLE
  current_mouse <- print(i)
  
  # ONLY KEEP OBSERVATIONS FROM CURRENT_MOUSE, REMOVE OTHER MICE
  move_df <- subset(move_df, full_ids == current_mouse) 
  
  # SAVE CURRENT SUBJECT AS VARIABLE
  current_mouse_info <- paste(unique(move_df$trial), unique(move_df$strain), unique(move_df$sex), unique(move_df$ID), sep = "_")

  ### wANT TO START GRAPHING? PUTTING THE GRAPHING CODE IN THE LOOP ALLOWS YOU TO GET ALL SUBJECT GRAPHS AT ONCE. 
  
  ## PLOT #1: INDIVIDUAL MOVEMENT ACROSS TUBS FOR ENTIRE TIME PERIOD. 
  
   plot1 <- ggplot(data=move_df, aes(Field.Time, antenna.id)) +
    geom_point(na.rm=TRUE, color="red", size=1) +
    ggtitle(paste0(current_mouse_info, " Zone Movement")) +
    xlab("Date") + ylab("Zone") +
    scale_x_datetime(breaks = "1 day", labels=date_format("%m-%d")) +
    #scale_y_discrete(limits=c(1","2","3","4","5","6","7","8"), drop = FALSE) +m  
    scale_y_continuous(breaks = seq(1,8,1), limits=c(1,8)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          axis.line = element_line(color = "black", size = 1, linetype = "solid"))
  
  plot(plot1)
  
  ggsave(filename=paste0(current_mouse_info, "_Zone_Movement_Scatter.png"), plot=plot1, width = 5, height = 4, dpi = 300, units = "in", device='png')
  
} ## END OF FOR LOOP




# IN PROGRESS -------------------------------------------------------------





