## LID.2019.Analysis.v.1.0.0.R
## Caleb Clifton Vogt, PhD Cornell University
## Updated 10.9.2019



# PROJECT ANALYSIS: 4_OXT_Liddell_master ----------------------------------
library(googlesheets)
library(ggplot2)
# gs_auth(new_user = TRUE)
# gs_ls()
for_gs <- gs_title("4_OXT_Liddell_master")
oxt_master <- gs_read(for_gs)
str(oxt_master) #DISPLAY THE STRUCTURE OF THE R OBJECT


# LIDDELL 2019: OCR + BORIS Integration -------------------------------------------------
library(data.table)
library(plyr)
library(dplyr)
library(readr)
library(reshape)
library(xlsx)

# STEP 1: 
#Add _BORIS to the END of all the BORIS observation csv files. 
#Make sure folder only contains BORIS csv files before doing this. 
wd <- setwd("E:/3_Liddell_2018_PROC/T002 (W3 needs OCR + BORIS Scoring)/W2")
file.list <- list.files(wd, pattern = "*.csv")

#for (i in 1:length(file.list)){
# file.rename(file.list[i], paste(file.list[i],"_BORIS",".csv", sep=''))
#}



# gsub(".*_", "", a) for removing any character prior
#file.rename(filename,gsub(replace,with,tolower(filename)))
#Move boris observations and OCR outputs into the same working directory folder. 

#Overwrite previous file.list
file.list <- list.files(wd, pattern = "*.csv")
file.list

OCR_list <- dir(wd, pattern = "*_OCR*") #make sure there are no folders. 
BORIS_list <- dir(wd, pattern = "*_BORIS*")
OCR_list #Check yo shit. 
BORIS_list #Check yo shit. 

for (a in 1:length(BORIS_list)) {
  #Create OCR dataframe. 
  ocr.df <- read.csv(OCR_list[a], stringsAsFactors = FALSE)
  ocr.df <- subset(ocr.df, select = -c(X))
  ocr.Frames <- rownames(ocr.df)
  ocr.df <- cbind(ocr.Frames=ocr.Frames, ocr.df)
  names(ocr.df) <- c("Video.Time", "Field.Time") #Depends on OCR FPS
  #Create boris dataframe. 
  bor.df <- read.csv(BORIS_list[a], skip=15) #Check skip #
  names(bor.df)[1]<-"Video.Time"
  bor.df[,1] <- round(bor.df[,1]) #Round Video.Time Column of _BORIS.csv
  bor.df$Video.Time<-as.numeric(as.character(bor.df$Video.Time))
  ocr.df$Video.Time<-as.numeric(as.character(ocr.df$Video.Time))
  #merge ocr and boris dataframes into master dataframe
  master<-merge(bor.df,ocr.df, by = "Video.Time", all.y = TRUE) # Orders the Video.Time column oddly. 
  master<- master[,c(1,10,2:9)]
  master<- subset(master, select=-c(Total.length, FPS)) #Drop cols by name. 
  master[c("M1", "M2", "M3", "M4", "M5","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15")] <- NA
  #Create mini dataframe
  mini <- master[order(master$Status, master$Subject), ]
  mini <- subset(mini, select=c(Video.Time, Field.Time, Subject, Status))
  START_rows <- mini[mini$Status == 'START', ]
  STOP_rows <- mini[mini$Status == 'STOP', ]
  mini <- cbind(START_rows, STOP_rows)
  mini <- na.omit(mini)
  #Use mini to extract START and STOP frame rows for adding to Master dataframe. 
  for(q in 1:nrow(mini)){
    #print(paste0("Processing ",q," out of ",nrow(mini)))
    x<-mini[q,]
    replacement1s<-c(as.numeric(x[1]):as.numeric(x[5]))
    mouse<-as.character(x$Subject)
    for (z in 1:length(replacement1s)){
      master[which(master$Video.Time == replacement1s[z]),grep(mouse,colnames(master))] <- 1
    }
    #master[replacement1s,grep(mouse,colnames(master))]<-1
  }
  
  #order() master rows by Field.Time
  write.csv(master, paste0(BORIS_list[a],"_master",".csv"))
  
}



