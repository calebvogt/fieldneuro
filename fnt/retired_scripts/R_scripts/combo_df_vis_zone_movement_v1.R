# combo_df_vis_zone_movement.R
# Caleb Clifton Vogt, PhD Cornell University
# Updated 4.22.2020


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
library(moveVis) #not having much luck with this package
library(move)

## SET WORKING DIRECTORY AND IMPORT COMBO_DF
wd <- setwd("Z:/3_Liddell_2018/3_Liddell_2018_Field/T003/T003_W1_RCNN_10Htz")
data <- fread("T003_W1_RCNN_COMBO_DF.csv", stringsAsFactors = TRUE) # nrows = , header=, etc... 
summary(data)
data$Zone <- as.numeric(data$Zone)
# CONVERT FIELD TIME TO POSIXT FORMAT. THIS WILL CHANGE OCR MISREADS TO NAS
data$Field.Time <- ymd_hms(data$Field.Time) 
# DROP ANY OCR MISREAD NAS FROM THE DATAFRAME. 
data <- na.omit(data, cols = c("Field.Time"))

i=25
for(i in 10:30){
  # SUBSET BY SUBJECT COLUMNS
  move_df <- data[,c(1:9, ..i)]
  # SAVE SUBJECT AS VARIABLE 
  current_mouse <- names(move_df[,10])
  # CHANGE COLUMN 10 NAME TO "MOUSE"
  names(move_df)[10] <- "Mouse"
  # REMOVE NAS FROM SELECTED SUBJECT COLUMN
  move_df <- na.omit(move_df, cols = 10)
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
    if((difftime(move_df$Field.Time[bb+1], move_df$Field.Time[bb], units="secs") > 30) && (difftime(move_df$Field.Time[bb], move_df$Field.Time[bb-1], units="secs") > 30)) {
      move_df$bout_status[bb] <- "DELETE"
    } else {
    }
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
  
  # CREATE TRACKS DF 
  tracks <- move_bouts[,1:7]
  
  #add xy coordinate columns to tracks dataframe. would be good to add some jitter to that shit. 
  tracks$x <- ifelse(grepl("1", tracks$Zone), "20", 
                     ifelse(grepl("2", tracks$Zone), "50",
                            ifelse(grepl("3", tracks$Zone), "20",
                                   ifelse(grepl("4", tracks$Zone), "50",
                                          ifelse(grepl("5", tracks$Zone), "20",
                                                 ifelse(grepl("6", tracks$Zone), "50",
                                                        ifelse(grepl("7", tracks$Zone), "20",
                                                               ifelse(grepl("8", tracks$Zone), "50",
                                                                      "none"))))))))
  
  
  tracks$y <- ifelse(grepl("1", tracks$Zone), "25", 
                     ifelse(grepl("2", tracks$Zone), "25",
                            ifelse(grepl("3", tracks$Zone), "65",
                                   ifelse(grepl("4", tracks$Zone), "65",
                                          ifelse(grepl("5", tracks$Zone), "105",
                                                 ifelse(grepl("6", tracks$Zone), "105",
                                                        ifelse(grepl("7", tracks$Zone), "145",
                                                               ifelse(grepl("8", tracks$Zone), "145",
                                                                      "none"))))))))
  
  ## SAMPLE VISUALIZATION 
  Time <- tracks$RWT_START
  X <- tracks$x
  Y <- tracks$y
  Mouse <- tracks$Mouse
  
  move_vis = data.frame(Time, X, Y)
  move_vis2 <- move_vis
  move_vis2$Time <- as.numeric(move_vis2$Time)
  move_vis3 <- move_vis2[c(1,206,465),]
  
  
  anim <- ggplot(move_vis3, aes(X, Y)) +
    geom_point()+
    theme_minimal() +
    transition_states(Time) + 
    ease_aes('cubic-in-out')
    ##fuck didnt work 
  animate(anim, fps=30)
  
  #anim_save("filenamehere.gif", anim)
  

}




## CREATE STATS_DF DATAFRAME FOR PLOTTING AN INDIVIDUAL'S DURATION BUBBLE PLOT. FIND AND REPLACE SUBJECT
s1 <- data.frame(matrix(NA, nrow = 8, ncol = 3))
colnames(s1) <- c("Zone", "x", "y")
s1$Zone <- c(1,2,3,4,5,6,7,8)
s1$x <- c(1,2,1,2,1,2,1,2)
s1$y <- c(1,1,2,2,3,3,4,4)
s1$Resource <- c("H20 + Lab Chow", "H20", "H20 + Bird Suet", "H20 + Sunflower Seeds", 
                 "H20 + Sunflower Seeds", "H20 + Bird Suet", "H20","H20 + Lab Chow")
s2 <- aggregate(move_df$M6, list(move_df$Zone), sum, na.rm=TRUE)
colnames(s2) <- c("Zone", "Duration")
stats_df <- merge(s1, s2, by = "Zone", all = TRUE)
stats_df[is.na(stats_df)] <- 0 
stats_df$Duration <- stats_df$Duration/60 # CHANGES DURATION TO MINUTES                       ...AND ## HERE##



# 11. VIDEO DATA PIPELINE: LIDDELL MOVEMENT ANIMATION 
library(moveVis)
library(anipaths)
library(tidyverse)
library(lubridate)
library(igraph)

setwd("E:/Data/3_Liddell_Ecology_proc/T002/W1/BORIS")
tracks <- read.csv("DUR_MERGE_RT_BORIS_T002_W1.csv") 




# Create outline of Liddell Paddock
referencecoords <- data.frame("df"= 1:2, "X" = c(0,0,75,75), "Y" = c(0,175,175,0))

plot(NA, xlim=c(0,180),
     ylim=c(0,180),
     xlab='',ylab='',
     axes=FALSE)

outline.xs <- referencecoords$X[c(1,2,3,4,1)]
outline.ys <- referencecoords$Y[c(1,2,3,4,1)]
lines(outline.xs,outline.ys)

#Create Resource Zones
z1x <- c(15, 30, 30, 15)
z1y <- c(20, 20, 35, 35)
polygon(z1x,z1y)
#center coords (20,25)

z2x <- c(45,60, 60, 45)
z2y <- c(20, 20, 35, 35)
polygon(z2x,z2y)
#center coords (50,25)

z3x <- c(15, 30, 30, 15)
z3y <- c(60, 60, 75, 75)
polygon(z3x,z3y)
#center coords (20,65)

z4x <- c(45,60, 60, 45)
z4y <- c(60, 60, 75, 75)
polygon(z4x,z4y)
#center coords (50,65)


z5x <- c(15, 30, 30, 15)
z5y <- c(100, 100, 115, 115)
polygon(z5x,z5y)
#center coords (20,105)


z6x <- c(45,60, 60, 45)
z6y <- c(100, 100, 115, 115)
polygon(z6x,z6y)
#center coords (50,105)


z7x <- c(15, 30, 30, 15)
z7y <- c(140, 140, 155, 155)
polygon(z7x,z7y)
#center coords (20,145)


z8x <- c(45,60, 60, 45)
z8y <- c(140, 140, 155, 155)
polygon(z8x,z8y)
#center coords (50,145)

clean_tracks <- subset(tracks, select = c(Subject, x, y, RWT_START, RWT_STOP))



#Example code
library(moveVis)
library(move)

data("move_data", package = "moveVis") # move class object
# if your tracks are present as data.frames, see df2move() for conversion
head(move_data)
summary(move_data)
# align move_data to a uniform time  scale
m <- align_move(move_data, res = 240, digit = 0, unit = "secs")

# create spatial frames with a OpenStreetMap watercolour map
frames <- frames_spatial(m, path_colours = c("red", "green", "blue"),
                         map_service = "osm", map_type = "watercolor", alpha = 0.5) %>% 
  add_labels(x = "Longitude", y = "Latitude") %>% # add some customizations, such as axis labels
  add_northarrow() %>% 
  add_scalebar() %>% 
  add_timestamps(m, type = "label") %>% 
  add_progress()

frames[[100]] # preview one of the frames, e.g. the 100th frame

# animate frames
animate_frames(frames, out_file = "moveVis.gif")


