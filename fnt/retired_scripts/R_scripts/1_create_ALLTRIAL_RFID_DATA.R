## 1_create_ALLTRIAL_RFID_DATA
## Caleb C. Vogt, Cornell University

## LOAD PACKAGES
library(tidyverse)
library(readxl)
library(dplyr)
library(plyr)
library(data.table)
library(readr)
library(reshape)
library(plot.matrix)

## IMPORT METADATA 
## DOWNLOAD METADATA AS EXCEL FILE INTO FOLDER AGAIN FROM GOOGLE DRIVE IF THE METADATA NEEDS UPDATING
wd <- setwd("C:/Users/caleb vogt/Box/0_CV_Shared_Analyses/7_LID_2020")
metadata <- read_excel("LID_2020_metadata.xlsx", sheet = 1, skip = 1)
weather <- read_excel("LID_2020_metadata.xlsx", sheet = 2, skip = 0)
weather$Weather_Time <- as.POSIXct(weather$Weather_Time, format = "%m/%d/%Y %H:%M") 
class(weather$Weather_Time)

## CREATE TRIAL_RFID_FULL_DATA.CSV IN EACH TRIAL FOLDER
## CREATE FOLDER LIST FOR T001-T007
folders <- list.dirs(wd, recursive=FALSE, full.names = TRUE)
folders <- folders[4:10] ## change
folders
folders_short <- list.dirs(wd, recursive=FALSE, full.names = FALSE)
folders_short <- folders_short[4:10]  # change
folders_short

## TO RUN FOR SPECIFIC TRIALS, JUST CHANGE THE FOLDER SEARCH IN THE FOR LOOP. 
## SET FLAG TO 1 WHEN RUNNING OVER ALL FOLDERS. SET TO TRIAL FOR SPECIFIC RUN. 
i=folders[1]
flag=1
all_trial_list <- list()
for (i in folders[1:7]) {
  setwd(i)
  
  # Pull trial variable from which data came from. 
  trial_var <- folders_short[flag]
  
  # IMPORT ALL RFID DATA SAVED AS TXT FILES
  txt <- list.files(pattern = "*.txt")
  
  # IMPORT ALL TEXT FILES. BLIND TO TEXT FILE NAME. BIND_ROWS INSTEAD OF RBIND TO AVOID ERROR. 
  dfall <- do.call(bind_rows, lapply(txt, function(x) read_table(file = x, col_names = TRUE, col_types = NULL,
                                                                 locale = default_locale(), na = "NA", skip = 4, n_max = Inf,
                                                                 progress = show_progress(), comment = "")))
  
  # SELECT COLUMNS
  dfall <- dfall[ , c("Scan Date", "Scan Time", "Reader ID", "Antenna ID", "DEC Tag ID")] 
  
  # RENAME COLUMNS
  colnames(dfall) <- c("scan.date", "scan.time", "reader.id", "antenna.id", "dec.tag.id")
  
  # DROP ROWS WITH NO DEC.TAG.ID (BASICALLY ALL TRIALS WHERE ANIMALS HAVENT BEEN INJECTED YET)
  dfall <- dfall[!is.na(dfall$dec.tag.id), ]
  
  # DROP TRAILING 0'S FROM TAG DEC IDS. METADATA DOESNT SAVE THE TRAILING ZERO ON IMPORT INTO R
  dfall$dec.tag.id <- sub("0*$", "", dfall$dec.tag.id)
  
  #REMOVE DUPLICATED ROWS FROM TXT WITH REPEATED OBSERVATIONS ACROSS TIME.  
  dfall <- unique(dfall)
  
  ## DUPLICATE TAG COLUMN TO KEEP RECORD OF ORIGINALLY RECORDED RFID TAG.
  dfall$read_tag <- dfall$dec.tag.id
  
  # MERGE METADATA AND DFNEW BY DEC.TAG.ID. THIS ONLY KEEPS MATCHES FOR TAG_1
  dfnew <- merge(dfall, metadata, by.x ="dec.tag.id", by.y = "dec.tag.id")
  
  # MERGE METADATA AND DFNEW BY DEC.TAG.ID. THIS ONLY KEEPS MATCHES FOR TAG_2
  dfnew1 <- merge(dfall, metadata, by.x = "dec.tag.id", by.y = "dec.tag.id_2")
  
  ## MERGE THE DATAFRAMES TO COMBINE DATA BY INDIVIDUAL REGARDLESS OF WHICH RFID TAG WAS READ BY THE READER
  dfall <- rbind.fill(dfnew, dfnew1)
  
  # CHOOSE COLUMNS TO KEEP
  dfall <- dfall[ , c("trial", "paddock", "strain", "sex", "ID", "name", "code", 
                      "reader.id", "antenna.id", "scan.date", "scan.time", "read_tag")] 
  
  
  
  # CREATE FIELD.TIME COLUMN. NOTE THAT MILISECONDS ARE PRESENT, BUT NOT PRINTED. 
  dfall$field.time <- as.POSIXct(paste(dfall$scan.date, dfall$scan.time), format="%m/%d/%Y %H:%M:%OS")
  
  # SORT DATAFRAME BY DATE AND TIME AND CREATE DATA
  data <- dfall[order(dfall$field.time), ]
  
  # CHANGE ANTENNA IDS/ZONES TO NUMERICS
  data$antenna.id <- as.numeric(data$antenna.id)
  data$ID <- as.numeric(data$ID)
  data$reader.id <- as.numeric(data$reader.id)
  
  # CREATE FULL_IDS 
  data$full_ids <- paste0(data$strain,"-", data$sex,"-", data$ID)
  
  # note that if more than 10 antennas are present, grepl will take the first digit and change it
  # CREATE X AND Y COORDINATES FOR THE ZONES FOR PLOTTING ON A GRID
  data$x <- ifelse(grepl("1", data$antenna.id), "A", 
                   ifelse(grepl("2", data$antenna.id), "B",
                          ifelse(grepl("3", data$antenna.id), "A",
                                 ifelse(grepl("4", data$antenna.id), "B",
                                        ifelse(grepl("5", data$antenna.id), "A",
                                               ifelse(grepl("6", data$antenna.id), "B",
                                                      ifelse(grepl("7", data$antenna.id), "A",
                                                             ifelse(grepl("8", data$antenna.id), "B",
                                                                    "none"))))))))
  
  
  data$y <- ifelse(grepl("1", data$antenna.id), "A", 
                   ifelse(grepl("2", data$antenna.id), "A",
                          ifelse(grepl("3", data$antenna.id), "B",
                                 ifelse(grepl("4", data$antenna.id), "B",
                                        ifelse(grepl("5", data$antenna.id), "C",
                                               ifelse(grepl("6", data$antenna.id), "C",
                                                      ifelse(grepl("7", data$antenna.id), "D",
                                                             ifelse(grepl("8", data$antenna.id), "D",
                                                                    "none"))))))))
  
  
  # REMOVE OBSERVATIONS FROM TRIALS WHICH ARE NOT POSSIBLE GIVEN PAIRED TRIAL STRUCTURE. 
  ifelse(grepl(pattern = "T001", i), 
         data <- subset(data, trial %in% c('T001')),
         ifelse(grepl(pattern = "T002", i), 
                data <- subset(data, trial %in% c('T002','T003')),
                ifelse(grepl(pattern = "T003", i), 
                       data <- subset(data, trial %in% c('T002','T003')),
                       ifelse(grepl(pattern = "T004", i), 
                              data <- subset(data, trial %in% c('T004','T005')),
                              ifelse(grepl(pattern = "T005", i), 
                                     data <- subset(data, trial %in% c('T004','T005')),
                                     ifelse(grepl(pattern = "T006", i), 
                                            data <- subset(data, trial %in% c('T006','T007')),
                                            ifelse(grepl(pattern = "T007", i), 
                                                   data <- subset(data, trial %in% c('T006','T007')),NA)))))))
  
  
  colnames(data)
  colnames(data) <- c("trial", "paddock", "strain", "sex", "ID", "name", "code", "reader_ID", 
                      "antenna", "date", "time", "read_tag", "field_time", "full_ID", "zone_x", "zone_y")
  
  # change trial to the ASSIGNED TRIAL BASED ON THE METADATA!!! Which is what the trial column currently reflects. 
  names(data)[names(data) == 'trial'] <- "assigned_trial"
  
  # create a new column called trial which reflects where (which paddock) and when (temporally) the data were derived from. 
  # thus, if an animal from T004 somehow crossed into T005 (which ran simultaneously), assigned trial for that obs would read T004, while Trial would 
  # equal T005, which is the trial the mouse was detected in. 
  data$trial <- trial_var
  
  rfid <- data
  class(rfid$field_time)
  #CREATE TIME FROM THE FIRST RECORDED RFID READ AT THE START OF THE TRIAL IN SECONDS
  rfid$time_sec <- as.numeric(difftime(rfid$field_time,min(rfid$field_time),units="secs")+1)
  
  ## CREATE ORIGIN FOR THE FIRST DAY OF THE TRIAL AT NOON. EACH 24 HOUR PERIOD IS CALLED A NIGHT
  origin <- as.POSIXct(paste(rfid$date[1], "12:00:00", sep =" "), format="%m/%d/%Y %H:%M:%OS")
  rfid$noon_to_noon_day <- ceiling(difftime(rfid$field_time, origin,  units = "days"))
  rfid$days_antenna <- paste(rfid$noon_to_noon_day,rfid$antenna,sep="_")
  
  ## MERGE WEATHER DATA WITH RFID DATA
  rfid$Weather_Time <- round(as.POSIXct(rfid$field_time), "hour")
  rfid <- left_join(rfid,weather)
  
  ## WRITE THE DATA TO CSV INTO EACH FOLDER
  write.csv(rfid, file = paste0(folders_short[flag],'_RFID_FULL_DATA.csv'))
  
  ## PUT EACH TRIALS ADJUSTED DATA INTO A LIST TO CREATE ALLTRIAL_RFID_DATA.CSV
  all_trial_list[[i]] <- rfid
  
  ## SET YOUR FLAG WHEN RUNNING THE LOOP. OMIT WHEN RUNNING ON SINGLE FOLDER. 
  flag <- flag+1
  
}

## MERGE ALL TRIAL DATA INTO SINGLE FILE in the working directory
all_trial_data = do.call(bind_rows, all_trial_list)

## REMOVE DATA WITH KNOWN INVALIDATING ISSUES
## ISSUE #1, RESOLVED: T005 MITANI (NEW NAME?) LOST 8741 RFID TAG ON 7/24/2020 AT 8PM WHICH WAS RECOVERED IN PERSON. REMOVE ALL SUBSEQUENT READS OF THIS TAG.
all_trial_data <- subset(all_trial_data, !(read_tag == "982.126057708741" & date >= "07/24/2020")) #remove


## CREATE ALLTRIAL_RFID_DATA.CSV
setwd(wd)
write.csv(all_trial_data, "LID_2020_ALLTRIAL_RFID_DATA.csv")
