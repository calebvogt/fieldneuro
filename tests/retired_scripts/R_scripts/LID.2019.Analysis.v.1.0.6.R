## LID.2019.Analysis.v.1.0.3.R
## Caleb Clifton Vogt, PhD Cornell University
## Updated 10.9.2019
# # PROJECT ANALYSIS: 4_OXT_Liddell_master ----------------------------------
# library(googlesheets)
# library(ggplot2)
# # gs_auth(new_user = TRUE)
# # gs_ls()
# for_gs <- gs_title("4_OXT_Liddell_master")
# oxt_master <- gs_read(for_gs)
# str(oxt_master) #DISPLAY THE STRUCTURE OF THE R OBJECT




# LIDDELL 2019: OCR + BORIS Integration -------------------------------------------------
library(data.table)
library(plyr)
library(dplyr)
library(tidyverse)
library(readr)
library(reshape)
library(xlsx)

# STEP 1: 
#Add _BORIS to the END of all the BORIS observation csv files if not yet done. 
#Make sure folder only contains BORIS csv files before doing this. 
wd <- setwd("C:/Users/Caleb Vogt/Desktop/10FPS OCR Test")
file.list <- list.files(wd, pattern = "*.csv")


# # RUN THIS ON NEWLY EXPORTED BORIS OBSERVATIONS. 
# for (i in 1:length(file.list)){
#  file.rename(file.list[i], paste(file.list[i],"_BORIS",".csv", sep=''))
# }

# gsub(".*_", "", a) for removing any character prior
#file.rename(filename,gsub(replace,with,tolower(filename)))

#Move boris observations and OCR outputs into the same working directory folder. 


# sTEP 2: 
# cREATE MERGED_DF CSV FOR ALL OBSERVATIONS. 
#Overwrite previous file.list
file.list <- list.files(wd, pattern = "*.csv")
file.list

OCR_list <- dir(wd, pattern = "*_OCR*") #make sure there are no folders. 
BORIS_list <- dir(wd, pattern = "*_BORIS*")
OCR_list
BORIS_list


for (a in 1:length(BORIS_list)) {
  #Create OCR dataframe. 
  ocr.df <- read.csv(OCR_list[a], stringsAsFactors = FALSE)
  ocr.df <- subset(ocr.df, select = -c(X))
  ocr.Frames <- rownames(ocr.df)
  #ocr.Frames <- 0:(nrow(ocr.df)-1)
  ocr.df <- cbind(ocr.Frames=ocr.Frames, ocr.df)
  ocr.df$ocr.Frames <- ocr.Frames
  names(ocr.df) <- c("OCR.Frame", "Field.Time") #Depends on OCR FPS
  
  #Create boris dataframe. 
  bor.df <- read.csv(BORIS_list[a], skip=15, stringsAsFactors = FALSE) # ADJUST SKIP NUMBER DEPENDING ON BORIS FILE CSVS. 
  names(bor.df)[1]<-"OCR.Frame"
  # If OCR file is at 10 FPS...
  bor.df[,1] <- round(bor.df[,1]*10)
  # If OCR file is at 1 FPS...
  #bor.df[,1] <- round(bor.df[,1]) 
  #
  if (bor.df$OCR.Frame[1] == 0) {
    bor.df$OCR.Frame[1] <- 1
  }
  bor.df$OCR.Frame<-as.numeric(as.character(bor.df$OCR.Frame))
  ocr.df$OCR.Frame<-as.numeric(as.character(ocr.df$OCR.Frame))
  bor.df$Subject <- as.character(bor.df$Subject)
  #change F1 to F01
  bor.df$Subject[bor.df$Subject == "F1"] <- "F01" #Change F1 to F01
  #Below for LID.2018.T001
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
  merged <- merge(bor.df,ocr.df, by = "OCR.Frame", all.y = TRUE) # Orders the OCR.Frame column oddly. dropping last row of bor.df
  #should include modifiers
  #merged <- merged[ , c("OCR.Frame", "Field.Time", "Media.file.path", "Subject", "Behavior", "Modifier.1", "Modifier.2", "Status")] # REORDER COLUMNS AND DROP ANY COLUMNS NOT LISTED HERE. 
  merged <- merged[ , c("OCR.Frame", "Field.Time", "Media.file.path", "Subject", "Behavior", "Status")]
  merged[c("M1", "M2", "M3", "M4", "M5","M6","F01","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15")] <- NA
  #sub_cols <- c("M1", "M2", "M3", "M4", "M5","F01","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15")
  cols <- as.character(colnames(merged))
  merged$Media.file.path <- OCR_list[a]
  
  #STATE EVENT DURATIONS: Create mini dataframe for figuring out the frames to which state events are applied!
  mini <- merged[order(merged$Status, merged$Subject), ]
  mini <- subset(mini, select=c(OCR.Frame, Field.Time, Subject, Behavior, Status)) 
  mini <- mini[mini$Behavior == 'Enter Zone', ] #Specifies that we are merely pulling out the enter zone. 
  START_rows <- mini[mini$Status == 'START', ] # PULL Mini rows with sTARTS. Ignores point events. but doesnt discriminate between "enter zone" and other state events. 
  STOP_rows <- mini[mini$Status == 'STOP', ] # Pull mini rows with STOPS. 
  mini <- cbind(START_rows, STOP_rows)
  mini <- na.omit(mini)
  
  #Use mini to extract START and STOP frame rows for adding to merged dataframe. 
  for(q in 1:nrow(mini)){
    print(paste0("Processing ",q," out of ",nrow(mini), " events in file ",BORIS_list[a]))
    x <- mini[q, ]
    replacement1s <- c(as.numeric(x[1]):as.numeric(x[6])) # Set where to look for video start and stop times in columns 1 and 6
    mouse <- as.character(x$Subject)
    for (z in 1:length(replacement1s)){
      merged[which(merged$OCR.Frame == replacement1s[z]),grep(mouse,colnames(merged))] <- mouse
    }
  }
  merged <- merged[rowSums(is.na(merged[,c(7:27)])) !=21, ] #Drop all rows where NAs are present for subject column 7:27 (check)
  merged <- merged[!duplicated(merged[c("Field.Time", "Media.file.path", "Subject", "Behavior", "Status")]), ] # Drops all duplicated Field.Time rows and keeps rows with STARTS and STOPS
  # The above line of code dramatically improves the resolution of the BORIS scoring. Instead of rounding to the nearest seconds, rounds to the nearest frame. 
 write.csv(merged, paste0(OCR_list[a],"_MERGED_DF",".csv"), row.names = FALSE)
}


# STEP 3: 
# cOMBINE ALL MERGED_DF FILES INTO ONE LARGE MASTER DATA FRAME AND CREATE ZONE COLUMN.
# EXPORT THE MASTER DATA FRAME AS A CSV. 

wd <- setwd("G:/My Drive/S-Lab Analysis/T001_test")


## READ IN ALL MERGED_DF FILES. 
file.list <- list.files(wd, pattern = "*MERGED_DF.csv")
master<- do.call(rbind, 
                 lapply(file.list, read.csv))

# CREATE ZONE COLUMN. 
master$Zone <- ifelse(grepl("Z1", master$Media.file.path), "1", 
                      ifelse(grepl("Z2", master$Media.file.path), "2",
                             ifelse(grepl("Z3", master$Media.file.path), "3",
                                    ifelse(grepl("Z4", master$Media.file.path), "4",
                                           ifelse(grepl("Z5", master$Media.file.path), "5",
                                                  ifelse(grepl("Z6", master$Media.file.path), "6",
                                                         ifelse(grepl("Z7", master$Media.file.path), "7",
              ifelse(grepl("Z8", master$Media.file.path), "8",
                     ifelse(grepl("Z9", master$Media.file.path), "9", 
                            ifelse(grepl("Z10", master$Media.file.path), "10", 
                                   ifelse(grepl("Z11", master$Media.file.path), "11", 
                                          ifelse(grepl("Z12", master$Media.file.path), "12", 
                                                 ifelse(grepl("Z13", master$Media.file.path), "13", 
                                                        ifelse(grepl("Z14", master$Media.file.path), "14", 
                                                               ifelse(grepl("Z15", master$Media.file.path), "15",
                                                                      ifelse(grepl("Z16", master$Media.file.path), "16", "NONE"))))))))))))))))


# cHANGE SUBJECT COLUMNS FOR WHICH THERE ARE DATA TO FACTORS. 
sapply(master, class)
master[sapply(master, is.character)] <- lapply(master[sapply(master, is.character)], as.factor) #Change 
summary(master)

# RENAME YOUR FILE. 
write.csv(master, "LID.2018.T003_master.csv", row.names = FALSE)



# GRAPHING LIDDELL MASTER CSV ---------------------------------------------
# To-Do list. 
# 1. Set 30s-1min threshold for entry events
# 2. Plot entries and exits into tubs across zones. 
# 3. Plot social interactions
# 4. Plot total seconds of observation I have for each animal across all zones. 
# 5. Plot # of seconds of observation time within each zone. 

wd <- setwd("G:/My Drive/RCNN_Copies for scoring/2018_Analysis_11.7.2019")

library(ggplot2)
library(scales)
library(plyr)
library(data.table)
library(lubridate)
library(tidyr)


data <- fread("LID.2018.T003_master.csv", stringsAsFactors = TRUE) # nrows = , header=, etc... 

summary(data)
#data$Subject[data$Subject == "F1"] <- "F01" #Change F1 to F01
data$Zone <- as.numeric(data$Zone)
data$Field.Time <- ymd_hms(data$Field.Time) 
#subset by date range using lubridate
data <- with(data,data[day(Field.Time) >= 20 & day(Field.Time) <=21])



### Create new STOP and START column by set time range conditional on zone and subject. \





















## CREATE DUR_M1 DATAFRAME (In progress)

# for (aa in 1:nrow(M1)) {
#   if 
# }



# CALCULATE DURATION OF TIME SPENT IN EACH ZONE FOR EACH MALE. (Incomplete)
df1 <- data.frame(matrix(vector(), 0, 3,
                             dimnames=list(c(), c("Subject", "Zone", "Duration"))),
                                    stringsAsFactors=F)


subs <- paste(unique(data$Subject))
msubs <- c("M1", "M2", "M3", "M4", "M5", "M6")
zoner <- paste(unique(data$Zone))               

datalist = list()
flag=1
for (i in subs) {
  for (j in 1:length(zoner)) {
    df1 <- setNames(data.frame(matrix(ncol = 3, nrow = 1)), 
                    c("Subject", "Zone", "Duration"))
    #df1$Duration <- nrow(paste0('data[data$',i,' == "',i,'" & data$Zone == ',j,',]')) # this line does not work. 
    df1$Duration <- nrow(data[data$i == i & data$Zone == j,',]'))
    
    df1$Subject <- i
    df1$Zone <- j
    datalist[[flag]] <- df1
    flag=flag+1
  }
}
 
df_total = do.call(rbind, datalist)

#nrow(data[data$M1 == "M1" & data$Zone == 2,])

#FUNCTIONAL for durations in zones

for (i in 1:8) {
  x <- nrow(data[data$F7 == "F7" & data$Zone == i])
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$M2 == "M2" & data$Zone == i])
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$M3 == "M3" & data$Zone == i])
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$M4 == "M4" & data$Zone == i])
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$M5 == "M5" & data$Zone == i])
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$M6 == "M6" & data$Zone == i])
  print(x)
}


#FUNCTIONAL for # of behaviors
#mating
msubs <- c("M1", "M2", "M3", "M4", "M5", "M6")

nrow(data[data$Subject == "M1" & data$Modifier.1 == "Mating"])

nrow(data[data$Subject == "M2" & data$Modifier.1 == "Mating"])

nrow(data[data$Subject == "M3" & data$Modifier.1 == "Mating"])

nrow(data[data$Subject == "M4" & data$Modifier.1 == "Mating"])

nrow(data[data$Subject == "M5" & data$Modifier.1 == "Mating"])

nrow(data[data$Subject == "M6" & data$Modifier.1 == "Mating"])




# Mating in zone...
df <- data.frame(matrix(NA, nrow=8, ncol=6))

for (i in 1:8) {
  x <- nrow(data[data$Subject == "M1" & data$Zone == i & data$Modifier.1 == "Mating"])
  print(x)
  #df[i,1] <- x
}

m2_Mating <- rep(0,8)
for (i in 1:8) {
  x <- nrow(data[data$Subject == "M2" & data$Zone == i & data$Modifier.1 == "Mating"])
  #df[i,2] <- x
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$Subject == "M3" & data$Zone == i & data$Modifier.1 == "Mating"])
  #df[i,3] <- x
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$Subject == "M4" & data$Zone == i & data$Modifier.1 == "Mating"])
  #df[i,4] <- x
  print(x)
}


for (i in 1:8) {
  x <- nrow(data[data$Subject == "M5" & data$Zone == i & data$Modifier.1 == "Mating"])
  #df[i,5] <- x
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$Subject == "M6" & data$Zone == i & data$Modifier.1 == "Mating"])
  #df[i,6] <- x
  print(x)
}




#attacking
nrow(data[data$Subject == "M1" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M2"])
nrow(data[data$Subject == "M1" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M3"])
nrow(data[data$Subject == "M1" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M4"])
nrow(data[data$Subject == "M1" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M5"])
nrow(data[data$Subject == "M1" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M6"])



nrow(data[data$Subject == "M2" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M1"])
nrow(data[data$Subject == "M2" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M3"])
nrow(data[data$Subject == "M2" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M4"])
nrow(data[data$Subject == "M2" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M5"])
nrow(data[data$Subject == "M2" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M6"])





nrow(data[data$Subject == "M3" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M1"])
nrow(data[data$Subject == "M3" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M2"])
nrow(data[data$Subject == "M3" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M4"])
nrow(data[data$Subject == "M3" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M5"])
nrow(data[data$Subject == "M3" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M6"])



nrow(data[data$Subject == "M4" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M1"])
nrow(data[data$Subject == "M4" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M2"])
nrow(data[data$Subject == "M4" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M3"])
nrow(data[data$Subject == "M4" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M5"])
nrow(data[data$Subject == "M4" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M6"])


nrow(data[data$Subject == "M5" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M1"])
nrow(data[data$Subject == "M5" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M2"])
nrow(data[data$Subject == "M5" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M3"])
nrow(data[data$Subject == "M5" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M4"])
nrow(data[data$Subject == "M5" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M6"])



nrow(data[data$Subject == "M6" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M1"])
nrow(data[data$Subject == "M6" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M2"])
nrow(data[data$Subject == "M6" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M3"])
nrow(data[data$Subject == "M6" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M4"])
nrow(data[data$Subject == "M6" & data$Modifier.1 == "Attacking" & data$Modifier.2 == "M5"])


# Aggression in zone...
for (i in 1:8) {
  x <- nrow(data[data$Subject == "M1" & data$Zone == i & data$Modifier.1 == "Attacking"])
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$Subject == "M2" & data$Zone == i & data$Modifier.1 == "Attacking"])
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$Subject == "M3" & data$Zone == i & data$Modifier.1 == "Attacking"])
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$Subject == "M4" & data$Zone == i & data$Modifier.1 == "Attacking"])
  print(x)
}


for (i in 1:8) {
  x <- nrow(data[data$Subject == "M5" & data$Zone == i & data$Modifier.1 == "Attacking"])
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$Subject == "M6" & data$Zone == i & data$Modifier.1 == "Attacking"])
  print(x)
}




#Drinking
nrow(data[data$Subject == "M1" & data$Behavior == "Drinking"])

nrow(data[data$Subject == "M2" & data$Behavior == "Drinking"])

nrow(data[data$Subject == "M3" & data$Behavior == "Drinking"])

nrow(data[data$Subject == "M4" & data$Behavior == "Drinking"])

nrow(data[data$Subject == "M5" & data$Behavior == "Drinking"])

nrow(data[data$Subject == "M6" & data$Behavior == "Drinking"])



# Drinking in zone...
for (i in 1:8) {
  x <- nrow(data[data$Subject == "M1" & data$Zone == i & data$Behavior == "Drinking"])
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$Subject == "M2" & data$Zone == i & data$Behavior == "Drinking"])
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$Subject == "M3" & data$Zone == i & data$Behavior == "Drinking"])
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$Subject == "M4" & data$Zone == i & data$Behavior == "Drinking"])
  print(x)
}


for (i in 1:8) {
  x <- nrow(data[data$Subject == "M5" & data$Zone == i & data$Behavior == "Drinking"])
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$Subject == "M6" & data$Zone == i & data$Behavior == "Drinking"])
  print(x)
}



#Feeding
nrow(data[data$Subject == "M1" & data$Behavior == "Feeding"])

nrow(data[data$Subject == "M2" & data$Behavior == "Feeding"])

nrow(data[data$Subject == "M3" & data$Behavior == "Feeding"])

nrow(data[data$Subject == "M4" & data$Behavior == "Feeding"])

nrow(data[data$Subject == "M5" & data$Behavior == "Feeding"])

nrow(data[data$Subject == "M6" & data$Behavior == "Feeding"])




# Feeding in zone...
for (i in 1:8) {
  x <- nrow(data[data$Subject == "M1" & data$Zone == i & data$Behavior == "Feeding"])
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$Subject == "M2" & data$Zone == i & data$Behavior == "Feeding"])
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$Subject == "M3" & data$Zone == i & data$Behavior == "Feeding"])
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$Subject == "M4" & data$Zone == i & data$Behavior == "Feeding"])
  print(x)
}


for (i in 1:8) {
  x <- nrow(data[data$Subject == "M5" & data$Zone == i & data$Behavior == "Feeding"])
  print(x)
}

for (i in 1:8) {
  x <- nrow(data[data$Subject == "M6" & data$Zone == i & data$Behavior == "Feeding"])
  print(x)
}






#Create individual data frames
M1 <- subset(data, (M1 == "M1")) # Get all M1 OCR frames
M1 <- subset(M1, Status != "POINT" | is.na(Status)) # Remove all point events, but keeps na values.
M1 <- subset(M1, select = c(Field.Time, Zone, Subject, Behavior, Status, M1)) #subset columns
M1 <- subset(M1, Subject == "M1" | is.na(Status)) # Only take M1 observations, remove other subjects. 
M1 <- na.omit(M1, cols = c("Field.Time"))
M1$Field.Time <- as.POSIXct(M1$Field.Time)
M1 <- as.data.frame(M1)


ggplot(data=M1,aes(x=Field.Time,y=Zone)) +
  geom_point(size=1) +
  labs(x="Time", y="Zone", title="M1 Movement, Week 2") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))



#Create individual data frames
M2 <- subset(data, (M2 == "M2")) # Get all M2 OCR frames
M2 <- subset(M2, Status != "POINT" | is.na(Status)) # Remove all point events, but keeps na values.
M2 <- subset(M2, select = c(Field.Time, Zone, Subject, Behavior, Status, M2)) #subset columns
M2 <- subset(M2, Subject == "M2" | is.na(Status)) # Only take M2 observations, remove other subjects. 
M2 <- na.omit(M2, cols = c("Field.Time"))
M2$Field.Time <- as.POSIXct(M2$Field.Time)
M2 <- as.data.frame(M2)

ggplot(data=M2,aes(x=Field.Time,y=Zone)) +
  geom_point(size=1) +
  labs(x="Time", y="Zone", title="M2 Movement, Week 2") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))


#Create individual data frames
M3 <- subset(data, (M3 == "M3")) # Get all M3 OCR frames
M3 <- subset(M3, Status != "POINT" | is.na(Status)) # Remove all point events, but keeps na values.
M3 <- subset(M3, select = c(Field.Time, Zone, Subject, Behavior, Status, M3)) #subset columns
M3 <- subset(M3, Subject == "M3" | is.na(Status)) # Only take M3 observations, remove other subjects. 
M3 <- na.omit(M3, cols = c("Field.Time"))
M3$Field.Time <- as.POSIXct(M3$Field.Time)
M3 <- as.data.frame(M3)

ggplot(data=M3,aes(x=Field.Time,y=Zone)) +
  geom_point(size=1) +
  labs(x="Time", y="Zone", title="M3 Movement, Week 2") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))


#Create individual data frames
M4 <- subset(data, (M4 == "M4")) # Get all M4 OCR frames
M4 <- subset(M4, Status != "POINT" | is.na(Status)) # Remove all point events, but keeps na values.
M4 <- subset(M4, select = c(Field.Time, Zone, Subject, Behavior, Status, M4)) #subset columns
M4 <- subset(M4, Subject == "M4" | is.na(Status)) # Only take M4 observations, remove other subjects. 
M4 <- na.omit(M4, cols = c("Field.Time"))
M4$Field.Time <- as.POSIXct(M4$Field.Time)
M4 <- as.data.frame(M4)

ggplot(data=M4,aes(x=Field.Time,y=Zone)) +
  geom_point(size=1) +
  labs(x="Time", y="Zone", title="M4 Movement, Week 2") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))


#Create individual data frames
M5 <- subset(data, (M5 == "M5")) # Get all M5 OCR frames
M5 <- subset(M5, Status != "POINT" | is.na(Status)) # Remove all point events, but keeps na values.
M5 <- subset(M5, select = c(Field.Time, Zone, Subject, Behavior, Status, M5)) #subset columns
M5 <- subset(M5, Subject == "M5" | is.na(Status)) # Only take M5 observations, remove other subjects. 
M5 <- na.omit(M5, cols = c("Field.Time"))
M5$Field.Time <- as.POSIXct(M5$Field.Time)
M5 <- as.data.frame(M5)

ggplot(data=M5,aes(x=Field.Time,y=Zone)) +
  geom_point(size=1) +
  labs(x="Time", y="Zone", title="M5 Movement, Week 2") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))



#Create individual data frames
M6 <- subset(data, (M6 == "M6")) # Get all M6 OCR frames
M6 <- subset(M6, Status != "POINT" | is.na(Status)) # Remove all point events, but keeps na values.
M6 <- subset(M6, select = c(Field.Time, Zone, Subject, Behavior, Status, M6)) #subset columns
M6 <- subset(M6, Subject == "M6" | is.na(Status)) # Only take M6 observations, remove other subjects. 
M6 <- na.omit(M6, cols = c("Field.Time"))
M6$Field.Time <- as.POSIXct(M6$Field.Time)
M6 <- as.data.frame(M6)

ggplot(data=M6,aes(x=Field.Time,y=Zone)) +
  geom_point(size=1) +
  labs(x="Time", y="Zone", title="M6 Movement, Week 2") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))






# Extra Code: -------------------------------------------------------------

 class(data)
 nrow(data)
 str(data)
 dplyr::glimpse(data)
 plot(data)
 identify(data)
 xtabs(data)
 names(data)
 
 subs <- unique(df$Subject)
 subs <- na.omit(subs)
 
 
 
 
 qplot(Field.Time, Zone, data=df)
 
 x <- !is.na(df$F01)
 
 
 
 ggplot(df, aes(fill=Subject, y=duration, x=zone)) + 
   geom_bar(stat="identity") +
   labs(title = "Female Zone Occupancy (Week 1)") +
   facet_wrap(~Subject) #Creates multiple plots by subject. 
 
 
 
 
 ggplot(data=df, aes(x=df$Field.Time, y=df$Zone, group=F01)) +
   geom_line(size = 10) +
   labs(x="Time", y="Zone", title="F1 Week 1 Exploration") +
   scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))
 
 
 
 
 
 
 
 master$Field.Time <- as.POSIXct(master$Field.Time,format="%Y-%m-%d %H:%M:%S")
 ggplot() + geom_point(data=master, aes(x=master$Field.Time,y=master$F01))+
   geom_point(data=master, aes(x=master$Field.Time,y=master$F7))
 
 masterF01 <- subset(master, select = c('F01'))
 ggplot() + geom_point(data=masterF01, aes(x=master$Field.Time,y=master$Zone))
 
 
 # #this is what was used to produce a graph
 # z <- subset(graph_master2,(!is.na(graph_master2$F1)))
 # w <- subset(graph_master2,(!is.na(graph_master2$F2)))
 # z$Field.Time <- as.POSIXct(z$Field.Time,format="%Y-%m-%d %H:%M:%S")
 # w$Field.Time <- as.POSIXct(w$Field.Time,format="%Y-%m-%d %H:%M:%S")
 # ggplot() + geom_point(data=z, aes(x=z$Field.Time,y=z$F1))+
 #   geom_point(data=w, aes(x=w$Field.Time,y=w$F2))
 
 
 
 
 
 
 identify() #function allows you click data points of interest on the graph
 
 
 
 ggplot(data=M1, aes(x=M1[,"Field.Time"], y=M1[,"Zone"])) +
   geom_point(size = 1) +
   labs(x="Time", y="Zone", title="M1 Movement, Week 2") +
   scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))
 
 ggplot(data=M1, aes(x=M1$Field.Time, y=M1$Zone)) +
   geom_point(size = 1) +
   labs(x="Time", y="Zone", title="M1 Movement, Week 2") +
   scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))
 
 
 