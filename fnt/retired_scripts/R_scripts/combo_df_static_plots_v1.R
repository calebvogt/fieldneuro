# combo_df_static_plots.R
# Caleb Clifton Vogt, PhD Cornell University
# Updated 4.24.2020

# Required Packages ###########

library(googlesheets)
library(ggplot2)
library(data.table)
library(tidyverse)
library(readr)
library(reshape)
library(xlsx)
library(lubridate)
library(scales)
library(lubridate)
library(gridExtra)
library(viridis)
library(hrbrthemes)
library(gganimate)
library(plot.matrix)


## SET WORKING DIRECTORY AND IMPORT COMBO_DF
wd <- setwd("C:/Users/caleb/Desktop/data")
data <- fread("T003_W1_RCNN_COMBO_DF.csv", stringsAsFactors = TRUE) # nrows = , header=, etc... 
data$Zone <- as.numeric(data$Zone)
# CONVERT FIELD TIME TO POSIXT FORMAT. THIS WILL CHANGE OCR MISREADS TO NAS
data$Field.Time <- ymd_hms(data$Field.Time) 
# DROP ANY OCR MISREAD NAS FROM THE DATAFRAME. 
data <- na.omit(data, cols = c("Field.Time"))

i=16
for(i in 10:30){
  # SUBSET BY SUBJECT COLUMNS
  move_df <- data[,c(1:9, ..i)]
  # SAVE SUBJECT AS VARIABLE 
  current_mouse <- names(move_df[,10])
  # CHANGE COLUMN 10 NAME TO "MOUSE"
  names(move_df)[10] <- "Mouse"
  # REMOVE NAS FROM SELECTED SUBJECT COLUMN
  move_df <- na.omit(move_df, cols = 10)
  
  if(nrow(move_df) == 0){
    next 
    }
  else{
  
  
  # REMOVE ALL POINT EVENTS EXCEPT WHEN STATUS IS NOT NA
  move_df <- subset(move_df, Status != "POINT" | is.na(Status)) 
  # DROP COLUMNS
  move_df <- subset(move_df, select = c(Zone, Field.Time, Subject, Behavior, Status, Mouse)) #subset columns
  # ONLY KEEP OBSERVATIONS FROM CURRENT_MOUSE, REMOVE OTHER MICE
  move_df <- subset(move_df, Subject == current_mouse | is.na(Status)) 
  
  
  move_df$Field.Time <- as.POSIXct(move_df$Field.Time)
  move_df <- as.data.frame(move_df)
  
  # THIS LINE IS LIKELY NO LONGER NECESSARY SINCE TRIALS NO LONGER INCLUDED 
  move_df <- move_df[!duplicated(move_df[c("Field.Time", "Zone")]),] # 
  # ORDER ROWS BY FIELD TIME, THEN BY ZONE
  move_df <- move_df[order(move_df$Zone, move_df$Field.Time), ]
  # SPLIT THE ZONES INTO A LIST. ADD BOUT STATUS COLUMN. PLACE START IN FIRST ROW, AND STOP IN LAST ROW. 
  split <- split(move_df, move_df$Zone)
  datalist = list()
  for (aa in (1:length(split))) {
    temp <- split[[aa]]
    temp$bout_status <- NA
    temp$bout_status[1] <- "START"
    n <- nrow(temp)
    temp$bout_status[n] <- "STOP"
    datalist[[aa]] <- temp
  }
  move_df <- do.call(rbind, datalist)
  as.data.table(move_df)
  
  
  # REMOVE OBSERVATIONS THAT LAST ONLY A SINGLE SECOND 
  for (bb in 2:nrow(move_df)-1) {
    tryCatch({
      
      if((difftime(move_df$Field.Time[bb+1], move_df$Field.Time[bb], units="secs") > 30) && (difftime(move_df$Field.Time[bb], move_df$Field.Time[bb-1], units="secs") > 30)) {
        move_df$bout_status[bb] <- "DELETE"
      } else {
      }
    
    }, error=function(e){})
    
  }
  x <- grep('DELETE',move_df$bout_status)
  
  if (length(x) != 0) {
    move_df <- move_df[-c(x[1:length(x)]),] #eliminates issue of NA
  } else {}
  
  
  # ADD STARTS AND STOPS TO BOUT.STATUS. BOUT IS DEFINED AS ENDING IF THE NEXT SIGHTING IS GREATER THAN 30 SECONDS IN THE FUTURE
  for (cc in 2:nrow(move_df)-1) {
    if(difftime(move_df$Field.Time[cc+1], move_df$Field.Time[cc], units="secs") > 30) { # NEGATIVE NUMBERS HERE ARE IGNORED!!! HUZZAH!
      move_df$bout_status[cc] <- "STOP"
      move_df$bout_status[cc+1] <- "START"
    } else {
    }
  }
  
  START_rows <- subset(move_df, (move_df$bout_status == "START")) 
  STOP_rows <- subset(move_df, (move_df$bout_status == 'STOP'))
  
  if(nrow(START_rows) != nrow(STOP_rows)){
    next 
  }
  
  move_new <- cbind(START_rows, STOP_rows)
  names(move_new)[2] <- "RWT_START" 
  names(move_new)[9] <- "RWT_STOP"
  move_new$RWT_START <- ymd_hms(move_new$RWT_START)  
  move_new$RWT_STOP <- ymd_hms(move_new$RWT_STOP)
  move_new$duration <- move_new$RWT_STOP - move_new$RWT_START
  
  # CHECK THAT THERE ARE NO NEGATIVE DURATIONS, POSSIBLE RESULT OF POOR OCR READING.  
  move_bouts <- move_new[move_new$duration >= 0, ]
  as.data.table(move_bouts)
  move_bouts$duration <- as.integer(move_bouts$duration)
  move_bouts$duration <- move_bouts$duration/60 # CHANGES DURATION TO MINUTES 
 
  
  #### MOVE BOUTS DATA FRAME CREATED
  
  ## CREATE STATS_DF DATAFRAME FOR PLOTTING AN INDIVIDUAL'S DURATION BUBBLE PLOT. FIND AND REPLACE SUBJECT
  s1 <- data.frame(matrix(NA, nrow = 8, ncol = 3))
  colnames(s1) <- c("Zone", "x", "y")
  s1$Zone <- c(1,2,3,4,5,6,7,8)
  s1$x <- c(1,2,1,2,1,2,1,2)
  s1$y <- c(1,1,2,2,3,3,4,4)
  s1$Resource <- c("H20 + Lab Chow", "H20", "H20 + Bird Suet", "H20 + Sunflower Seeds", 
                   "H20 + Sunflower Seeds", "H20 + Bird Suet", "H20","H20 + Lab Chow")
  # SUMS ALL 1S IN THE MOUSE COLUMN BY ZONE
  s2 <- aggregate(move_df$Mouse, list(move_df$Zone), sum, na.rm=TRUE)
  colnames(s2) <- c("Zone", "Duration")
  stats_df <- merge(s1, s2, by = "Zone", all = TRUE)
  stats_df[is.na(stats_df)] <- 0 
  stats_df$Duration <- stats_df$Duration/60 # CHANGES DURATION TO MINUTES                       ...AND ## HERE##
  

  ## CREATE STATIC PLOTS
  
  ## PLOT #2: INDIVIDUAL MOVEMENT
  plot2 <- ggplot(data=move_df, aes(Field.Time, Zone)) +
    geom_point(na.rm=TRUE, color="red", size=1) +
    ggtitle(paste0(current_mouse, " Zone Movement")) +
    xlab("Date") + ylab("Zone") +
    scale_x_datetime(breaks = "1 day", labels=date_format("%m-%d")) +
    #scale_y_discrete(limits=c(1","2","3","4","5","6","7","8"), drop = FALSE) +m  
    scale_y_continuous(breaks = seq(1,8,1), limits=c(1,8)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          axis.line = element_line(color = "black", size = 1, linetype = "solid"))
  
  plot(plot2)
  
  ggsave(filename=paste0(current_mouse, "_Zone_Movement_Scatter.png"), plot=plot2, width = 5, height = 4, dpi = 300, units = "in", device='png')
  
  
  
  } ## END OF ELSE STATEMENT

} ## END OF FOR LOOP





