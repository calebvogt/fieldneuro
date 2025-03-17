# combine_merged_df.R
# Caleb Clifton Vogt, PhD Cornell University
# Updated 4.22.2020

# Required Packages ###########
library(tidyverse)

#### Working with MERGED_DF Files #############
wd <- setwd("Z:/3_Liddell_2018/3_Liddell_2018_Field/T003/T003_W2_RCNN_10Htz/merged_df")

# COMBINE ALL MERGED_DF FILES INTO ONE LARGE data DATA FRAME AND CREATE ZONE COLUMN.
## READ IN ALL MERGED_DF FILES. 
file.list <- list.files(wd, pattern = "*MERGED_DF.csv")
file.list
data <- do.call(rbind, 
                lapply(file.list, read.csv))

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


# CHANGE SUBJECT COLUMNS FOR WHICH THERE ARE DATA TO FACTORS. 
sapply(data, class)
data[sapply(data, is.character)] <- lapply(data[sapply(data, is.character)], as.factor) #Change 
summary(data)

# WRITE COMBO_MERGED CSV, CHANGE 
write.csv(data, "T003_W2_RCNN_COMBO_DF.csv", row.names = FALSE)




# CAN LIKELY STOP HERE
# CREATE FULL TRIAL ANALYSIS DF FROM ALL COMBO_DF FILES ###########

wd <- setwd("G:/My Drive/RCNN_Copies for scoring/NBB_Symposium_Analysis")
file.list <- list.files(wd, pattern = "*COMBO_DF.csv")
file.list
data <- do.call(rbind, 
                lapply(file.list, read.csv))

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

# WRITE data CSV 
write.csv(data, "LID.2018_overlord.csv", row.names = FALSE)

