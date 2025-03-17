## LID.2019.Analysis.v.1.0.0.R
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
library(readr)
library(reshape)
library(xlsx)

# STEP 1: 
#Add _BORIS to the END of all the BORIS observation csv files if not yet done. 
#Make sure folder only contains BORIS csv files before doing this. 
wd <- setwd("C:/Users/Caleb Vogt/Desktop/T001_test")
file.list <- list.files(wd, pattern = "*.csv")


for (i in 1:length(file.list)){
 file.rename(file.list[i], paste(file.list[i],"_BORIS",".csv", sep=''))
}

# gsub(".*_", "", a) for removing any character prior
#file.rename(filename,gsub(replace,with,tolower(filename)))
#Move boris observations and OCR outputs into the same working directory folder. 

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
  names(ocr.df) <- c("Video.Time", "Field.Time") #Depends on OCR FPS
  
  #Create boris dataframe. 
  bor.df <- read.csv(BORIS_list[a], skip=15, stringsAsFactors = FALSE) # ADJUST SKIP NUMBER DEPENDING ON BORIS FILE CSVS. 
  names(bor.df)[1]<-"Video.Time"
  bor.df[,1] <- round(bor.df[,1]) #Round Video.Time Column of _BORIS.csv
  if (bor.df$Video.Time[1] == 0) {
    bor.df$Video.Time[1] <- 1
  }
  bor.df$Video.Time<-as.numeric(as.character(bor.df$Video.Time))
  ocr.df$Video.Time<-as.numeric(as.character(ocr.df$Video.Time))
  bor.df$Subject <- as.character(bor.df$Subject)
  #change F1 to F01
  #bor.df$Subject[bor.df$Subject == "F1"] <- "F01" #Change F1 to F01
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
  
  
  
  #merge ocr and boris dataframes into master dataframe
  master <- merge(bor.df,ocr.df, by = "Video.Time", all.y = TRUE) # Orders the Video.Time column oddly. dropping last row of bor.df
  master <- master[,c("Video.Time", "Field.Time", "Media.file.path", "Subject", "Behavior", "Status")] # REORDER COLUMNS AND DROP ANY COLUMNS NOT LISTED HERE. 
  master[c("M1", "M2", "M3", "M4", "M5","M6","F01","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15")] <- NA
  #sub_cols <- c("M1", "M2", "M3", "M4", "M5","F01","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15")
  cols <- as.character(colnames(master))
  master$Media.file.path <- OCR_list[a]
  
  #Create mini dataframe
  mini <- master[order(master$Status, master$Subject), ]
  mini <- subset(mini, select=c(Video.Time, Field.Time, Subject, Behavior, Status)) 
  mini <- mini[mini$Behavior == 'Enter Zone', ] #Specifies that we are merely pulling out the enter zone. 
  START_rows <- mini[mini$Status == 'START', ] # PULL Mini rows with sTARTS. Ignores point events. but doesnt discriminate between "enter zone" and other state events. 
  STOP_rows <- mini[mini$Status == 'STOP', ] # Pull mini rows with STOPS. 
  mini <- cbind(START_rows, STOP_rows)
  mini <- na.omit(mini)
  
  #Use mini to extract START and STOP frame rows for adding to Master dataframe. 
  for(q in 1:nrow(mini)){
    print(paste0("Processing ",q," out of ",nrow(mini), " events in file ",BORIS_list[a]))
    x <- mini[q, ]
    replacement1s <- c(as.numeric(x[1]):as.numeric(x[6])) # Set where to look for video start and stop times in columns 1 and 6
    mouse <- as.character(x$Subject)
    for (z in 1:length(replacement1s)){
      master[which(master$Video.Time == replacement1s[z]),grep(mouse,colnames(master))] <- mouse
    }
  }
  master <- master[rowSums(is.na(master[,c(7:27)])) !=21, ] #
  write.csv(master, paste0(OCR_list[a],"_DF",".csv"), row.names = FALSE)
}


## READ IN ALL DF FILES. 
file.list <- list.files(wd, pattern = "*_DF.csv")
master <- ldply(file.list, read_csv) #this might be killing some of my observations. 
master<- do.call(rbind, 
                 lapply(file.list, read.csv))


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




sapply(master, class)
master[sapply(master, is.character)] <- lapply(master[sapply(master, is.character)], as.factor) #Change 
summary(master)
do.call(cbind, lapply(master, summary))

# CHANGE THIS
write.csv(master, "LID.2018.T001_master.csv", row.names = FALSE)


## READ IN MASTER FILE. 
library(ggplot2)
library(data.table)
library(lubridate)
library(tidyr)

data <- fread("LID.2018.T001_master.csv", stringsAsFactors = TRUE)
summary(data)


df <- read.csv("LID.2018.T002_master.csv",  
                           stringsAsFactors=T, header=T, nrows=500)  


View(df)

class(data)
nrow(data)
str(data)
dplyr::glimpse(data)
plot(data)
identify(data)
xtabs(data)
names(data)


data$Zone <- as.numeric(data$Zone)

## Finally!
#Subset the merge file. #Check columns. 
df2 <- subset(data, select = c(Field.Time, Zone, Subject, Behavior, Status, M4))
df2$Field.Time <- ymd_hms(df2$Field.Time)  
df2 <- df2 %>% drop_na(M4)


ggplot(data=df2, aes(x=df2$Field.Time, y=df2$Zone)) +
  geom_point(size = 1) +
  labs(x="Time", y="Zone", title="LID.2018.T001, M4") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))





identify() #function allows you click data points of interest on the graph




# Question: How many total seconds of observation do I have for each animal across all zones. 
# Each row = one second of real world observation. 

# Question: how many seconds of total observations do I have for each zone for all animals. 
# How much time does one animal spend in each individual zone. 


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



