# merge_ocr_boris.R
# Caleb Clifton Vogt, PhD Cornell University
# Updated 4.20.2020


# Required Packages ###########
# some of these may no longer be required

library(data.table)
library(tidyverse)
library(readr)
library(reshape)
library(xlsx)
library(lubridate)
library(scales)

# OCR + BORIS Integration -------------------------------------------------

#Add _BORIS to the END of all the BORIS observation csv files if not yet done. 
#Make sure folder only contains BORIS csv files before doing this. 
wd <- setwd("Z:/3_Liddell_2018/3_Liddell_2018_Field/T003")
file.list <- list.files(wd, pattern = "*.csv")


# RUN THIS ON NEWLY EXPORTED BORIS OBSERVATIONS. 
for (i in 1:length(file.list)){
  file.rename(file.list[i], paste(file.list[i],"_BORIS",".csv", sep=''))
}

# #for removing any character prior
# gsub(".*_", "", a) 
# file.rename(filename,gsub(replace,with,tolower(filename)))

# MOVE BORIS OBSERVATIONS BACK TO MAIN FOLDER AND CHANGE WORKING DIRECTORY 
wd <- setwd("Z:/3_Liddell_2018/3_Liddell_2018_Field/T003/T003_W2_RCNN_10Htz")

# STEP 2
# CREATE MERGED_DF CSV FOR ALL OBSERVATIONS. 
# OVERWRITE PREVIOUS FILE LIEST IF NECESSARY
file.list <- list.files(wd, pattern = "*.csv")
file.list

OCR_list <- dir(wd, pattern = "*_OCR*") #make sure there are no folders. 
BORIS_list <- dir(wd, pattern = "*_BORIS*")
# Make sure list lenghts are the same


# CHECK OCR FPS AND ADJUST WITHIN IN LOOP
# CHECK IF MODIFIERS ARE INCLUDED IN YOUR BORIS FILE. DO NOT WANT TO LOSE THEM!

for (a in 1:length(BORIS_list)) {
  ocr.df <- read.csv(OCR_list[a], stringsAsFactors = FALSE)
  ocr.df <- subset(ocr.df, select = -c(X))
  ocr.Frames <- rownames(ocr.df)
  ocr.df <- cbind(ocr.Frames=ocr.Frames, ocr.df)
  ocr.df$ocr.Frames <- ocr.Frames
  names(ocr.df) <- c("OCR.Frame", "Field.Time")
  
  bor.df <- read.csv(BORIS_list[a], skip=15, stringsAsFactors = FALSE) # ADJUST SKIP NUMBER DEPENDING ON BORIS FILE CSVs. Get rid of the metadata
  names(bor.df)[1]<-"OCR.Frame"
  
  
  ## IS OCR AT 10 FPS? USE THIS.                         
  bor.df[,1] <- round(bor.df[,1]*10)
  
  # ## IS OCR AT 1 FPS? USE THIS. 
  # bor.df[,1] <- round(bor.df[,1]) 
  
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
  
  #Change character 
  bor.df$OCR.Frame<-as.numeric(as.character(bor.df$OCR.Frame))
  ocr.df$OCR.Frame<-as.numeric(as.character(ocr.df$OCR.Frame))
  bor.df$Subject <- as.character(bor.df$Subject)
  
  # CHANGE F1 TO F01 TO AVOID OVERWRITE ISSUE
  bor.df$Subject[bor.df$Subject == "F1"] <- "F01"
  # CHANGE BORIS SUBJECT NAMES IF NOT IN CORRECT FORMAt
  # This applies to Adam's day-long scored videos. Too cumbersome to change names within BORIS project file. 
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
  
  #merge ocr and boris dataframes into merge
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
  
  merged <- merged[rowSums(is.na(merged[,c("M1", "M2", "M3", "M4", "M5","M6","F01","F2","F3","F4","F5",
                                           "F6","F7","F8","F9","F10","F11","F12","F13","F14","F15")])) !=21, ]
  
  # DROP ROWS WITH FULL MATCHING ROWS OF DESIGNATED COLUMNS. 
  merged <- merged[!duplicated(merged[c("Field.Time", "Media.file.path", "Subject", "Behavior", "Status")]), ] 
  write.csv(merged, paste0(OCR_list[a],"_MERGED_DF",".csv"), row.names = FALSE)
}
