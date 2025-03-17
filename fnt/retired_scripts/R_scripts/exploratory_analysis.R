## exploratory_analysis.R
## Caleb C. Vogt, Cornell University

## TO-DO LIST:
# 1. Break data down by hour, by tub, by individual, by social group. 
# 3. How do we make the claim that inbred and outbred mice form territories, not dominance hierarchies, in the field. 
# 5. Change days to go from noon to noon as opposed to midnight-to-midnight. THIS IS IMPORTANT>>>>>>
# 6. Handling Large Data Sets: https://rstudio-pubs-static.s3.amazonaws.com/72295_692737b667614d369bd87cb0f51c9a4b.html
# 7 . 

# LOAD PACKAGES
library(tidyverse)
library(readxl)
library(plyr)
library(gganimate)
library(data.table)
library(readr)
library(reshape)
library(lubridate)
library(scales)
library(plot.matrix)
library(asnipe)
library(igraph)
library(transformr)

# SET WD AND LOAD DATA
wd <- setwd("C:/Users/Caleb Vogt/Desktop/Liddell_2020_RFID_full_10day_data")
metadata <- read_excel("Liddell.2020.xlsx", sheet = 1, skip = 1)
filenames <- list.files(wd, pattern = "*RFID_full_data*")

#################
# TESTING CODE

memory.size()  # CHECK
gc(verbose = TRUE)
memory.size()  # CHECK
memory.limit()
memory.limit(8000000)

# WORKING WITH THE CURRENT DF FOR GMM ANALYSIS
# ERRORS:
# T002 DAY 8: Error: cannot allocate vector of size 36.2 Gb
# T006 Day 3: Error: cannot allocate vector of size 31.2 Gb
# T005 Day 3- Error: cannot allocate vector of size 20.9 Gb

for(i in 5:7){
  df <- fread(filenames[i])
  
  # CHANGE FIELD.TIME COL TO POSIXct
  df$Field.Time <- as.POSIXct(df$Field.Time, format="%Y-%m-%d %H:%M:%OS")
  
  # # REMOVE OBSERVATIONS FROM TRIALS WHICH ARE NOT POSSIBLE GIVEN PAIRED TRIAL STRUCTURE . DONE ALREADY!
  # ifelse(names(filelist[i]) == "T001", 
  #   filelist[[i]] <- subset(filelist[[i]], trial %in% c('T001')),
  #       ifelse(names(filelist[i]) == "T002", 
  #              filelist[[i]] <- subset(filelist[[i]], trial %in% c('T002','T003')),
  #                 ifelse(names(filelist[i]) == "T003", 
  #                        filelist[[i]] <- subset(filelist[[i]], trial %in% c('T002','T003')),
  #                             ifelse(names(filelist[i]) == "T004", 
  #                                    filelist[[i]] <- subset(filelist[[i]], trial %in% c('T004','T005')),
  #                                         ifelse(names(filelist[i]) == "T005", 
  #                                                filelist[[i]] <- subset(filelist[[i]], trial %in% c('T004','T005')),
  #                                                 ifelse(names(filelist[i]) == "T006", 
  #                                                        filelist[[i]] <- subset(filelist[[i]], trial %in% c('T006','T007')),
  #                                                            ifelse(names(filelist[i]) == "T007", 
  #                                                                   filelist[[i]] <- subset(filelist[[i]], trial %in% c('T006','T007')),NA)))))))
  
  trial <- substr(filenames[i],1,4)
  id <- unique(df$full_ids)
  Tag <- df$full_ids
  Date <-df$scan.date
  Time <-df$scan.time
  Field_Time <- df$Field.Time
  Antenna <- df$antenna.id
  memory.size()  # CHECK
  
  # CREATE RFID DATA FRAME
  rfid <- cbind.data.frame(Tag, Date, Time, Field_Time, Antenna)
  class(rfid$Field_Time)
  rfid$Time_sec <- as.numeric(difftime(Field_Time,min(Field_Time),units="secs")+1)
  
  origin <- as.POSIXct(paste(rfid$Date[1], "12:00:00", sep =" "), format="%m/%d/%Y %H:%M:%OS")
  rfid$noon_to_noon_day <- ceiling(difftime(rfid$Field_Time, origin,  units = "days"))
  rfid$Days_Antenna <- paste(rfid$noon_to_noon_day,rfid$Antenna,sep="_")
  global_ids <- unique(rfid$Tag)
  days <- unique(rfid$noon_to_noon_day)
  memory.size()  # CHECK
  
  #REMOVE UNNECESSARY COLUMNs FROM RFID
  rfid <- rfid[, c("Tag", "Field_Time", "Time_sec", "noon_to_noon_day", "Days_Antenna")]
  object.size(rfid)
  memory.size()  # CHECK
  
  
  #REMOVE UNNECESSARY OBJECTS FROM ENVIRONMENTAL MEMORY IF NECESSARY!
  rm(df, Antenna, Date, Field_Time, Tag, Time)
  memory.size()  # CHECK
  gc(verbose = TRUE)
  memory.size()  # CHECK
 
  
  ## I FUCKING DID IT BIATCH>>
  flag <-1
  for(aa in 3:length(days)){
    # SPLIT INTO DAILY CHUNKS
    daily_df <- subset(rfid, rfid$noon_to_noon_day == aa)
    memory.size()  # CHECK
    
    # GENERATE GMM DATA
    gmm_data <- gmmevents(time=daily_df$Time_sec,
                          identity=daily_df$Tag,
                          location=daily_df$Days_Antenna,
                          global_ids= global_ids,
                          verbose = TRUE,
                          splitGroups = TRUE) # Try this with FALSE
    
    # SAVE GMM DATA
    save(gmm_data, file=paste0(trial, "_GMM_RFID_Day_",flag,".RData"))
    flag <- flag + 1
    rm(daily_df, gmm_data) #TEST. REMOVE THE DAILY DF BEFORE STARTING WRITING OVER IT!!!!
    gc(verbose=TRUE) #TEST. PERHAPS THIS CAN HELP CLEAR MEMORY AFTER EACH ITERATION OF LOOP. DELETE IF FAILURE. 
  }
}

# rm(list=ls())
# gc(verbose=TRUE)
# .rs.restartR()
# See this for implementing restarting R session: https://stackoverflow.com/questions/48305203/restart-r-session-without-interrupting-the-for-loop
