## Liddell2020Script.R
## Caleb C. Vogt, PhD Student Cornell University
## Updated on 7.6.2020
## TO DO LIST
# 1. Generate social networks from rfid data


library(tidyverse)
library(googlesheets)
library(dplyr)
library(ggplot2)
library(data.table)
library(readr)
library(reshape)
library(xlsx)
library(lubridate)
library(scales)
library(gganimate)
library(plot.matrix)
library(asnipe)
library(igraph)


# IMPORT METADATA ---------------------------------------------------------

#DOWNLOAD CSV INTO FOLDER AGAIN FROM GOOGLE DRIVE IF THE METADATA NEEDS UPDATING
setwd("G:/My Drive/Data/7_Liddell_2020")
metadata <- read.csv("7.Liddell.2020.csv", header = TRUE, skip=1)


# CLEAN THE RFID DATA -----------------------------------------------------

setwd("G:/My Drive/Data/7_Liddell_2020/T003/RFID")

df1 <- read_table("T003_RFID_DL_1.txt", col_names = TRUE, col_types = NULL,
           locale = default_locale(), na = "NA", skip = 4, n_max = Inf,
           progress = show_progress(), comment = "")

df2 <- read_table("T003_RFID_DL_2.txt", col_names = TRUE, col_types = NULL,
                    locale = default_locale(), na = "NA", skip = 4, n_max = Inf,
                    progress = show_progress(), comment = "")

df3 <- read_table("T003_RFID_DL_3.txt", col_names = TRUE, col_types = NULL,
                    locale = default_locale(), na = "NA", skip = 4, n_max = Inf,
                    progress = show_progress(), comment = "")

df4 <- read_table("T003_RFID_DL_4.txt", col_names = TRUE, col_types = NULL,
                   locale = default_locale(), na = "NA", skip = 4, n_max = Inf,
                   progress = show_progress(), comment = "")
 
 
df5 <- read_table("T003_RFID_DL_5.txt", col_names = TRUE, col_types = NULL,
                   locale = default_locale(), na = "NA", skip = 4, n_max = Inf,
                   progress = show_progress(), comment = "")
 
df6 <- read_table("T003_RFID_DL_6.txt", col_names = TRUE, col_types = NULL,
                   locale = default_locale(), na = "NA", skip = 4, n_max = Inf,
                   progress = show_progress(), comment = "")
 
# df7 <- read_table("T003_RFID_DL_7.txt", col_names = TRUE, col_types = NULL,
#                   locale = default_locale(), na = "NA", skip = 4, n_max = Inf,
#                   progress = show_progress(), comment = "")
# 
# 
# df8 <- read_table("T003_RFID_DL_8.txt", col_names = TRUE, col_types = NULL,
#                   locale = default_locale(), na = "NA", skip = 4, n_max = Inf,
#                   progress = show_progress(), comment = "")
# 
# df9 <- read_table("T003_RFID_DL_9.txt", col_names = TRUE, col_types = NULL,
#                   locale = default_locale(), na = "NA", skip = 4, n_max = Inf,
#                   progress = show_progress(), comment = "")
# 
# df10 <- read_table("T003_RFID_DL_10.txt", col_names = TRUE, col_types = NULL,
#                   locale = default_locale(), na = "NA", skip = 4, n_max = Inf,
#                   progress = show_progress(), comment = "")


dfall <- rbind(df1,df2,df3,df4,df5,df6)
# dfall <- rbind(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10)

# DROP UNNECESSARY COLUMNS
dfalldropped <- dfall[ , c("Scan Date", "Scan Time", "Reader ID", "Antenna ID", "DEC Tag ID")] 

# RENAME COLUMNS
colnames(dfalldropped) <- c("scan.date", "scan.time", "reader.id", "antenna.id", "dec.tag.id")

# REMOVE ALL IDENTIFIED "TEST" TAGS OR
dfnew <- dfalldropped[(dfalldropped$dec.tag.id != "989.002011236362"), ] # FACTORY TAG?
dfnew <- dfnew[(dfnew$dec.tag.id != "982.091062980706"), ] # KEYCHAIN TAG
dfnew <- dfnew[(dfnew$dec.tag.id != "----------------"), ] # ELIMINATES FIRST ROW OF TEXT FILE
dfnew <- dfnew[(dfnew$dec.tag.id != "006.009407925053"), ] # UNKNOWN TAG #1
dfnew <- dfnew[(dfnew$dec.tag.id != "982.092200090093"), ] # UNKNOWN TAG #2
dfnew <- dfnew[(dfnew$dec.tag.id != "990.228498193554"), ] # UNKNOWN TAG #3

# DROP TRAILING 0'S FROM TAG DEC IDS. METADATA.CSV DOESNT SAVE THE TRAILING ZERO ON IMPORT
dfnew$dec.tag.id <- sub("0*$", "", dfnew$dec.tag.id)

#REMOVE DUPLICATED ROWS FROM TXT WITH REPEATED OBSERVATIONS ACROSS TIME.  
dfnew <- unique(dfnew)

# MERGE METADATA AND DFNEW BY DEC.TAG.ID
dfnew <- merge(dfnew, metadata, by="dec.tag.id", all.x = TRUE)

# ONLY KEEP OBSERVATIONS FROM CURRENT TRIAL
dfnew <- subset(dfnew, dfnew$trial == 'T003') 

# DROP UNNECESSARY COLUMNS
dfnew1 <- dfnew[ , c("trial", "paddock", "antenna.id", "scan.date", "scan.time", "strain", "sex", "ID", "field.age", "dec.tag.id", "reader.id")] 

# CREATE FIELD.TIME COLUMN. NOTE THAT MILISECONDS ARE PRESENT, BUT NOT PRINTED. 
dfnew1$Field.Time <- as.POSIXct(paste(dfnew1$scan.date, dfnew1$scan.time), format="%m/%d/%Y %H:%M:%OS")

# SORT DATAFRAME BY DATE AND TIME 
data <- dfnew1[order(dfnew1$Field.Time), ]



# PLOT 1: INDIVIDUAL MOVEMENT SCATTERPLOTS --------------------------------------------------------

# CHANGE ANTENNA IDS/ZONES TO NUMERICS
data$antenna.id <- as.numeric(data$antenna.id)

ids <- unique(data$dec.tag.id)

# THE CODE FOR CREATING GRAPHS ARE WITHIN THE LOOP. CYCLES THROUGH IDS 

for(i in ids[1:20]){
  # CREATE MOVE_DF
  move_df <- data
  
  # SAVE CURRENT MOUSE AS VARIABLE
  current_mouse <- print(i)
  
  # ONLY KEEP OBSERVATIONS FROM CURRENT_MOUSE, REMOVE OTHER MICE
  move_df <- subset(move_df, dec.tag.id == current_mouse) 
  
  # SAVE CURRENT SUBJECT AS VARIABLE
  current_mouse_info <- paste(unique(move_df$trial), unique(move_df$strain), unique(move_df$sex), unique(move_df$ID), sep = "_")

  ### wANT TO START GRAPHING? PUTTING THE GRAPHING CODE IN THE LOOP ALLOWS YOU TO GET ALL SUBJECT GRAPHS AT ONCE. 
  
  ## PLOT #2: INDIVIDUAL MOVEMENT
  
  # CREATE 4X5 PLOT MATRIX FOR 20 ANIMALS
  #par(mfrow = c(4, 5))
  
  plot2 <- ggplot(data=move_df, aes(Field.Time, antenna.id)) +
    geom_point(na.rm=TRUE, color="red", size=1) +
    ggtitle(paste0(current_mouse_info, " Zone Movement")) +
    xlab("Date") + ylab("Zone") +
    scale_x_datetime(breaks = "1 day", labels=date_format("%m-%d")) +
    #scale_y_discrete(limits=c(1","2","3","4","5","6","7","8"), drop = FALSE) +m  
    scale_y_continuous(breaks = seq(1,8,1), limits=c(1,8)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          axis.line = element_line(color = "black", size = 1, linetype = "solid"))
  
  plot(plot2)
  
  ggsave(filename=paste0(current_mouse_info, "_Zone_Movement_Scatter.png"), plot=plot2, width = 5, height = 4, dpi = 300, units = "in", device='png')
  
} ## END OF FOR LOOP



















# IN PROGRESS -------------------------------------------------------------





