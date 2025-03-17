## Liddell2018Script.R
## Caleb C. Vogt, PhD Student Cornell University
## Updated on 6.24.2020
## TO DO LIST


# STEP 1: CREATE 1 or 10 Htz OCR FILE FROM VIDEO -------------------------------------------

library(tesseract)
library(magick)
library(beepr)
eng <- tesseract("eng")

#Set R and system working directories
# CHECK THAT ALL VIDEOS ARE 1080P. 
wd <- setwd("Z:/3_Liddell_2018/3_Liddell_2018_Field/T001/T001_no_RCNN_videos/T001 Daily Videos Scoring/videos")
system("cd /d Z:/3_Liddell_2018/3_Liddell_2018_Field/T001/T001_no_RCNN_videos/T001 Daily Videos Scoring/videos")
getwd()

#Create jpgs from video and list of all the generated jpgs. 
avi_list <- dir(wd, pattern = "*.avi") #Note that this code works with .avi's in mpeg4 codec
avi_list


# Need to look into making the cropped image bigger.
# follow the guidelines to make OCR detection better. https://cran.r-project.org/web/packages/tesseract/vignettes/intro.html

# BEST RUN ON FOLDERS CONTAINING 10-15 VIDEOS
for (aa in 1:length(avi_list)) {
  cmdstring <- paste("ffmpeg -i", avi_list[aa], "-vf fps=10,crop=315:35:50:970 frame%09d.jpg") #### ADJUST TO 1 OR 10 FPS. w:h:x:y
  shell(cmdstring)
  jpg_list <- dir(wd, pattern ="*.jpg")
  for (bb in 1:length(jpg_list)) {
    image <- image_read(jpg_list[bb])
    txt <- tesseract::ocr(image, engine = eng)
    ocr_txt <- as.character(txt)
    if(bb==1) {
      ocr_out <- ocr_txt
    } else {
      ocr_out <- rbind(ocr_out, ocr_txt)
    }
  }
  write.csv(ocr_out, paste(avi_list[aa],"_OCR",".csv", sep = ''))
  unlink("*.jpg")
  beep(1)
  beep(1)
  beep(1)
}




# STEP 2: MERGE OCR AND BORIS DATAFRAMES INTO _MERGED_DF ----------------------------------

library(data.table)
library(tidyverse)
library(readr)
library(reshape)
library(xlsx)
library(lubridate)
library(scales)

# RUN THIS ON NEWLY EXPORTED BORIS OBSERVATION FROM BORIS PROJECT FILE 
# for (i in 1:length(file.list)){
#   file.rename(file.list[i], paste(file.list[i],"_BORIS",".csv", sep=''))
# }

# SET THE PARENT WORKING DIRECTORY
wd <- setwd("Z:/3_Liddell_2018/3_Liddell_2018_Field")

# GET THE FILES YOU WANT
fl <- list.files(wd, pattern = glob2rx("*T002*.csv*"), full.names = TRUE, recursive = TRUE, include.dirs = TRUE)

# DROP THE FILES YOU DONT. YOU NEED EQUAL NUMBERS OF BORIS AND OCR FILES
fl <- fl[-grep("*RCNN*", fl)]

# GET YOUR OCR FILES
OCR_list <- fl[grep("*_OCR*", fl)]

# GET YOUR BORIS FILES
BORIS_list <- fl[grep("*_BORIS*", fl)]

# CHECK OCR FPS AND ADJUST WITHIN IN LOOP. CHECK IF MODIFIERS ARE INCLUDED IN YOUR BORIS FILE. DO NOT WANT TO LOSE THEM!
a=1
# CREATE EMPTY MERGED_DF DATAFRAME
merged_df <- data.frame()
for (a in 1:length(BORIS_list)) {
  ocr.df <- read.csv(OCR_list[a], stringsAsFactors = FALSE)
  ocr.df <- subset(ocr.df, select = -c(X))
  ocr.Frames <- rownames(ocr.df)
  ocr.df <- cbind(ocr.Frames=ocr.Frames, ocr.df)
  ocr.df$ocr.Frames <- ocr.Frames
  names(ocr.df) <- c("OCR.Frame", "Field.Time")
  
  # ADJUST SKIP NUMBER DEPENDING ON BORIS FILE CSVs. Get rid of the metadata
  bor.df <- read.csv(BORIS_list[a], skip=15, stringsAsFactors = FALSE) 
  names(bor.df)[1]<-"OCR.Frame"
  
  
  ## IS OCR AT 10hz? USE THIS. #WILL EVENTUALLY NEED TO AUTOMATE THIS SHIT.                          
  #bor.df[,1] <- round(bor.df[,1]*10)
  
  ## IS OCR AT 1hz? USE THIS. CURRENTLY ONLY APPLIES TO T002 FILES
  bor.df[,1] <- round(bor.df[,1]) 
  
  # CHANGE ANY 0 OCR.FRAME VALUES TO 1 IN BOR.DF
  for (i in 1:nrow(bor.df)) {
    if (bor.df$OCR.Frame[i] == 0) {
      bor.df$OCR.Frame[i] <- 1
    }}
  
  ## ADD MODIFIER.1 AND MODIFIER.2 COLUMNS IF NOT PRESENT. 
  if(!"Modifier.1" %in% colnames(bor.df)) {
    bor.df$Modifier.1 <- NA
  }
  
  if(!"Modifier.2" %in% colnames(bor.df)) {
    bor.df$Modifier.2 <- NA
  }
  
  # CHANGE COLUMNS TO CHARACTERS
  bor.df$OCR.Frame<-as.numeric(as.character(bor.df$OCR.Frame))
  ocr.df$OCR.Frame<-as.numeric(as.character(ocr.df$OCR.Frame))
  bor.df$Subject <- as.character(bor.df$Subject)
  
  # CHANGE F1 TO F01 TO AVOID OVERWRITE ISSUE
  bor.df$Subject[bor.df$Subject == "F1"] <- "F01"
  
  # CHANGE BORIS SUBJECT NAMES IF NOT IN CORRECT FORMAT (APPLIES TO SOME T001 BORIS FILES)
  bor.df$Subject <- gsub("F170.*","F01",bor.df$Subject, perl = TRUE)
  bor.df$Subject <- gsub("F171.*","F4",bor.df$Subject, perl = TRUE)
  bor.df$Subject <- gsub("F172.*","F3",bor.df$Subject, perl = TRUE)
  bor.df$Subject <- gsub("F173.*","F5",bor.df$Subject, perl = TRUE)
  bor.df$Subject <- gsub("F174.*","F2",bor.df$Subject, perl = TRUE)
  bor.df$Subject <- gsub("F175.*","F7",bor.df$Subject, perl = TRUE)
  bor.df$Subject <- gsub("F176.*","F6",bor.df$Subject, perl = TRUE)
  bor.df$Subject <- gsub("F177.*","F9",bor.df$Subject, perl = TRUE)
  bor.df$Subject <- gsub("F178.*","F10",bor.df$Subject, perl = TRUE)
  bor.df$Subject <- gsub("F179.*","F8",bor.df$Subject, perl = TRUE)
  bor.df$Subject <- gsub("LEWES.*","M1",bor.df$Subject, perl = TRUE)
  bor.df$Subject <- gsub("WSB.*","M2",bor.df$Subject, perl = TRUE)
  bor.df$Subject <- gsub("NY1.*","M3",bor.df$Subject, perl = TRUE)
  bor.df$Subject <- gsub("NY2.*","M4",bor.df$Subject, perl = TRUE)
  bor.df$Subject <- gsub("NY3.*","M5",bor.df$Subject, perl = TRUE)
  bor.df$Subject <- gsub("C57.*","M6",bor.df$Subject, perl = TRUE)
  
  # MERGE OCR AND BORIS DATAFRAMES INTO MERGED
  # Orders the OCR.Frame column oddly. dropping last row of bor.df
  merged <- merge(bor.df,ocr.df, by = "OCR.Frame", all.y = TRUE) 
  
  # REORDER COLUMNS AND DROP ANY COLUMNS NOT LISTED HERE. 
  merged <- merged[ , c("OCR.Frame", "Field.Time", "Media.file.path", "Subject", "Behavior", "Modifier.1", "Modifier.2", "Status")] 
  merged[c("M1", "M2", "M3", "M4", "M5","M6","F01","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15")] <- NA
  cols <- as.character(colnames(merged))
  merged$Media.file.path <- OCR_list[a]
  
  #STATE EVENT DURATIONS: Create mini dataframe for figuring out the frames to which state events are applied!
  mini <- merged[order(merged$Status, merged$Subject), ]
  mini <- subset(mini, select=c(OCR.Frame, Field.Time, Subject, Behavior, Status)) 
  
  #PULL ENTER ZONE BEHAVIORS
  mini <- mini[mini$Behavior == 'Enter Zone', ] 
  mini <- na.omit(mini)
  
  # PULL START/STOP ROWS
  START_rows <- mini[mini$Status == 'START', ] 
  STOP_rows <- mini[mini$Status == 'STOP', ] 
  
  # SPECIAL CASE: IF NUMBER OF START AND STOP ROWS UNEVEN, JUST THROW OUT THE UNPAIRED OBSERVATION. COMMENT OUT. 
  # START_rows <- START_rows[-nrow(START_rows),]
  # CONCATENATE START/STOP ROWS INTO SINGLE ROW. 
  mini <- cbind(START_rows, STOP_rows)
  mini <- na.omit(mini)
  
  # EXTRACT START AND STOP ROWS FROM MINI
  for(q in 1:nrow(mini)){
    print(paste0("Processing ",q," out of ",nrow(mini), " events in file ",BORIS_list[a]))
    x <- mini[q, ]
    
    # SET START AND STOP OCR.FRAME COLUMNS (1 AND 6 USUALLY). OCR.FRAME DEFINES START AND STOP OF 1'S. 
    replacement1s <- c(as.numeric(x[1]):as.numeric(x[6])) 
    mouse <- as.character(x$Subject)
    
    # ADD 1'S TO THE APPROPRIATE SUBJECT COLUMNS INDICATING THE MOUSE IS IN THE ZONE
    for (z in 1:length(replacement1s)){
      merged[which(merged$OCR.Frame == replacement1s[z]),grep(mouse,colnames(merged))] <- 1
    }
  }
  # DROP NAS
  merge1 <- merged[rowSums(is.na(merged[,c("M1", "M2", "M3", "M4", "M5","M6","F01","F2","F3","F4","F5",
                                           "F6","F7","F8","F9","F10","F11","F12","F13","F14","F15")])) !=21, ]
  
  # DROP ROWS WITH FULL MATCHING ROWS OF DESIGNATED COLUMNS. 
  merge2 <- merge1[!duplicated(merge1[c("Field.Time", "Media.file.path", "Subject", "Behavior", "Status")]), ] 
  
  # RBIND CURRENT ITERATION OF THE LOOP TO MERGED_DF
  merged_df <- rbind(merged_df, merge2)
  #write.csv(merged, paste0(OCR_list[a],"_MERGED_DF",".csv"), row.names = FALSE)
}




# STEP 3: COMBINE MERGED_DF INTO DATA1 -------------------------------
data <- merged_df

# CREATE ZONE COLUMN. 
data$Zone <- ifelse(grepl("Z1", data$Media.file.path), "1", 
                    ifelse(grepl("Z2", data$Media.file.path), "2",
                           ifelse(grepl("Z3", data$Media.file.path), "3",
                                  ifelse(grepl("Z4", data$Media.file.path), "4",
                                         ifelse(grepl("Z5", data$Media.file.path), "5",
                                                ifelse(grepl("Z6", data$Media.file.path), "6",
                                                  
             ifelse(grepl("Z7", data$Media.file.path), "7",
                    ifelse(grepl("Z8", data$Media.file.path), "8",
                           ifelse(grepl("Z9", data$Media.file.path), "9", 
                                  ifelse(grepl("Z10", data$Media.file.path), "10", 
                                         ifelse(grepl("Z11", data$Media.file.path), "11", 
                                                ifelse(grepl("Z12", data$Media.file.path), "12", 
                                                                                                 
               ifelse(grepl("Z13", data$Media.file.path), "13", 
                      ifelse(grepl("Z14", data$Media.file.path), "14", 
                             ifelse(grepl("Z15", data$Media.file.path), "15",
                                    ifelse(grepl("Z16", data$Media.file.path), "16", "NONE"))))))))))))))))
# MOVE ZONE COLUMN
data <- data %>%
  select(Zone, everything())

# CREATE TRIAL COLUMN. 
data$Trial <- ifelse(grepl("T001", data$Media.file.path), "T001", 
                     ifelse(grepl("T002", data$Media.file.path), "T002",
                            ifelse(grepl("T003", data$Media.file.path), "T003",
                                   ifelse(grepl("T004", data$Media.file.path), "T004", "NONE"))))

# MOVE TRIAL COLUMN
data <- data %>%
  select(Trial, everything())

# CHANGE SUBJECT COLUMNS FOR WHICH THERE ARE DATA TO FACTORS. 
sapply(data, class)
data[sapply(data, is.character)] <- lapply(data[sapply(data, is.character)], as.factor) #Change 
summary(data)

# CREATE DATA1
data1 <- data

# WRITE DATA1 TO FILE
write.csv(data1, "data1.csv", row.names = FALSE)


# STEP 4: CREATE STATIC PLOTS FROM COMBO_DF.CSV ----------------------------

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

## SET WORKING DIRECTORY AND IMPORT DATA1
data <- fread("data1.csv", stringsAsFactors = TRUE) # nrows = , header=, etc... 
data$Zone <- as.numeric(data$Zone)

# CONVERT FIELD TIME TO POSIXT FORMAT. THIS WILL CHANGE OCR MISREADS TO NAS
data$Field.Time <- ymd_hms(data$Field.Time) 

# DROP ANY OCR MISREAD NAS FROM THE DATAFRAME. 
data <- na.omit(data, cols = c("Field.Time"))

# LOOPS THROUGH THE SUBJECT COLUMNS, M1 THROUGH F15
# THE CODE FOR CREATING GRAPHS ARE WITHIN THE LOOP. GO TO LINE 


for(i in 11:31){
  # KEEP METADATA COLS 1:10, THEN ADD CURRENT SUBJECT COLUMN TO COL 11 
  move_df <- data[,c(1:10, ..i)]
  
  # SAVE SUBJECT AS VARIABLE, WHOEVER IS IN COL 11
  current_mouse <- names(move_df[,11])
  
  # CHANGE COLUMN 11 NAME TO MOUSE
  names(move_df)[11] <- "Mouse"
  
  # REMOVE NAS FROM MOUSE COLUMN
  move_df <- na.omit(move_df, cols = 11)
  
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
    
    # CHANGE FIELD TIME TO POSIX FORMAT
    move_df$Field.Time <- as.POSIXct(move_df$Field.Time)
    move_df <- as.data.frame(move_df)
    
    # REMOVE DUPLICATED FIELD TIMES AND ZONES FROM OTHER TRIALS THAT MIGHT BE INCLUDED IN DATA1
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
    
    
    # REMOVE OBSERVATIONS THAT LAST ONLY A SINGLE SECOND. ADD 'DELETE' TO SINGLE SECOND OBSERVATIONS.
    for (bb in 2:nrow(move_df)-1) {
      tryCatch({
        
        if((difftime(move_df$Field.Time[bb+1], move_df$Field.Time[bb], units="secs") > 30) && (difftime(move_df$Field.Time[bb], move_df$Field.Time[bb-1], units="secs") > 30)) {
          move_df$bout_status[bb] <- "DELETE"
        } else {
        }
        # ERROR FUNCTION HERE IS A BULLSHIT WAY TO AVOID THROWING AN ERROR. AH WELL IT WORKS. 
      }, error=function(e){})
      
    }
    
    # I FORGET WHICH ERROR THIS IS ATTEMPTING TO RESOLVE, BUT CODE LOOKS STRAIGHTFORWARD
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
    
    # PULL OUT START AND STOP ROWS FROM BOUT_STATUS 
    START_rows <- subset(move_df, (move_df$bout_status == "START")) 
    STOP_rows <- subset(move_df, (move_df$bout_status == 'STOP'))
    
    # CHECK TO SEE THAT NROWS FOR STARTS = NROWS FOR STOPS
    if(nrow(START_rows) != nrow(STOP_rows)){
      next 
    }
    
    # CREATE MOVE NEW WITH START AND STOP ROWS
    move_new <- cbind(START_rows, STOP_rows)
    names(move_new)[2] <- "RWT_START" 
    names(move_new)[9] <- "RWT_STOP"
    
    # CHANGE TO CORRECT TIME FORMAT
    move_new$RWT_START <- ymd_hms(move_new$RWT_START)  
    move_new$RWT_STOP <- ymd_hms(move_new$RWT_STOP)
    
    # CALCULATE ESTIMATED BOUT DURATIONS
    move_new$duration <- move_new$RWT_STOP - move_new$RWT_START
    
    # CHECK THAT THERE ARE NO NEGATIVE DURATIONS, POSSIBLE RESULT OF POOR OCR READING.  
    move_bouts <- move_new[move_new$duration >= 0, ]
    as.data.table(move_bouts)
    move_bouts$duration <- as.integer(move_bouts$duration)
    
    # CHANGES DURATION TO MINUTES IN MOVE_BOUTS
    move_bouts$duration <- move_bouts$duration/60 
    
    
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
    # IF A ZONE HAD A DURATION OF ZERO, CHANGE NA TO 0
    stats_df[is.na(stats_df)] <- 0 
    stats_df$Duration <- stats_df$Duration/60 # CHANGES DURATION TO MINUTES                       ...AND ## HERE##
    
    
    # YOU HAVE NOW CREATED...
    # STATS_DF = ZONES WITH DURATIONS AND IN MINUTES
    # MOVE_DF
    # MOVE_BOUTS
    
    ## CREATE STATIC PLOTS
    
    ### wANT TO START GRAPHING? PUTTING THE GRAPHING CODE IN THE LOOP ALLOWS YOU TO GET ALL SUBJECT GRAPHS AT ONCE. 
    
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



# STEP 6: CREATE ZONE MOVEMENT VISUALIZATIONS AND PLOTS FROM COMBO --------

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






# STEP7: CREATE LIDDELL MOVEMENT ANIMATION --------------------------------

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



