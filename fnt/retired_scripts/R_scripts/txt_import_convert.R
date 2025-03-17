### txt_import_convert.R
### Caleb C. Vogt, PhD Candidate, Cornell University
## Things to add:
# 1. allow conversion process to loop through trial subfolders instead of manually changing wd each trial. 

library(tidyverse)
library(readxl)
library(dplyr)
library(plyr)
library(data.table)
library(readr)
library(reshape)
library(lubridate)

#DOWNLOAD METADATA AS EXCEL FILE INTO FOLDER FROM GOOGLE DRIVE IF THE METADATA NEEDS UPDATING

wd <- setwd("C:/Users/Caleb Vogt/Desktop/txt")
metadata <- read_excel("Liddell.2020.xlsx", sheet = 1, skip = 1)
 
#CHANGE TRIAL NUMBER
setwd("C:/Users/Caleb Vogt/Desktop/txt")


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

dfall <- rbind.fill(dfnew, dfnew1)

# CHOOSE COLUMNS TO KEEP
dfall <- dfall[ , c("trial", "reader.id", "paddock", "antenna.id", "scan.date", "scan.time", "strain", "sex", "ID", "field.age", "dec.tag.id")] 

# CREATE FIELD.TIME COLUMN. NOTE THAT MILISECONDS ARE PRESENT, BUT NOT PRINTED. 
dfall$Field.Time <- as.POSIXct(paste(dfall$scan.date, dfall$scan.time), format="%m/%d/%Y %H:%M:%OS")

# SORT DATAFRAME BY DATE AND TIME AND CREATE DATA
data <- dfall[order(dfall$Field.Time), ]

# CHANGE ANTENNA IDS/ZONES TO NUMERICS
data$antenna.id <- as.numeric(data$antenna.id)
data$ID <- as.numeric(data$ID)

# CREATE FULL_IDS 
data$full_ids <- paste0(data$strain,"-", data$sex,"-", data$ID)

# CREATE X AND Y COORDINATES FOR THE ZONES. 
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


write.csv(data, "full_data.csv")

