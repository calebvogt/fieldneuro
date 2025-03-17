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
#Add _BORIS to the END of all the BORIS observation csv files if not yet done. 
#Make sure folder only contains BORIS csv files before doing this. 
wd <- setwd("C:/Users/Caleb Vogt/Desktop/R-test")
file.list <- list.files(wd, pattern = "*.csv")


#for (i in 1:length(file.list)){
 #file.rename(file.list[i], paste(file.list[i],"_BORIS",".csv", sep=''))
#}

# gsub(".*_", "", a) for removing any character prior
#file.rename(filename,gsub(replace,with,tolower(filename)))
#Move boris observations and OCR outputs into the same working directory folder. 

#Overwrite previous file.list
file.list <- list.files(wd, pattern = "*.csv")
file.list

OCR_list <- dir(wd, pattern = "*_OCR*") #make sure there are no folders. 
BORIS_list <- dir(wd, pattern = "*_BORIS*")

for (a in 1:length(BORIS_list)) {
  #Create OCR dataframe. 
  ocr.df <- read.csv(OCR_list[a], stringsAsFactors = FALSE)
  ocr.df <- subset(ocr.df, select = -c(X))
  ocr.Frames <- rownames(ocr.df)
  ocr.df <- cbind(ocr.Frames=ocr.Frames, ocr.df)
  names(ocr.df) <- c("Video.Time", "Field.Time") #Depends on OCR FPS
  
  #Create boris dataframe. 
  bor.df <- read.csv(BORIS_list[a], skip=15, stringsAsFactors = FALSE) # ADJUST SKIP NUMBER DEPENDING ON BORIS FILE CSVS. 
  names(bor.df)[1]<-"Video.Time"
  bor.df[,1] <- round(bor.df[,1]) #Round Video.Time Column of _BORIS.csv
  bor.df$Video.Time<-as.numeric(as.character(bor.df$Video.Time))
  ocr.df$Video.Time<-as.numeric(as.character(ocr.df$Video.Time))
  bor.df$Subject <- as.character(bor.df$Subject)
  bor.df$Subject[bor.df$Subject == "F1"] <- "F01" #Change F1 to F01
  
  #merge ocr and boris dataframes into master dataframe
  master <- merge(bor.df,ocr.df, by = "Video.Time", all.y = TRUE) # Orders the Video.Time column oddly. 
  master <- master[,c("Video.Time", "Field.Time", "Media.file.path", "Subject", "Behavior", "Status")] # REORDER COLUMNS AND DROP ANY COLUMNS NOT LISTED HERE. 
  master[c("M1", "M2", "M3", "M4", "M5","F01","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15")] <- NA
  sub_cols <- c("M1", "M2", "M3", "M4", "M5","F01","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15")
  cols <- as.character(colnames(master))
  
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
    print(paste0("Processing ",q," out of ",nrow(mini)))
    x <- mini[q, ]
    replacement1s <- c(as.numeric(x[1]):as.numeric(x[6])) # Set where to look for video start and stop times in columns 1 and 6
    mouse <- as.character(x$Subject)
    #master[replacement1s,grep(mouse,colnames(master))] <- 1 #This does NOT work over long time periods. 
    #master[which(master$Video.Time == replacement1s), grep(mouse,colnames(master))] <- 1 Throws an error fuck. 
    
    for (z in 1:length(replacement1s)){
      master[which(master$Video.Time == replacement1s[z]),grep(mouse,colnames(master))] <- mouse #Code here is quite slow. Adds 1's to columns. Problem! F1 adds 1s for F10,F11, and so on. 
      #master[which(master$Video.Time == replacement1s[z]),which(colnames(master) == mouse)] <- 1 #Doesnt work
      #master[which(master$Video.Time == replacement1s[z]),master$which(colnames(master) == mouse)] <- 1 #DOESNT WORK!!
      #master[which(master$Video.Time == replacement1s[z]),grep(paste0("^",mouse,"$"),colnames(master),fixed=T, value=T)] <- 1 #doesnt perfectly pattern match. # DOESNT FUCKING WORK!!!
      #master[which(master$Video.Time == replacement1s[z]),grep(paste0("^",mouse,"$"),cols,fixed=F)] <- 1 #still doesnt work. 
      #master[which(master$Video.Time == replacement1s[z]),which(cols == mouse)] # THIS DOESNT WORK EITHER!!! WTF. 
      
    }
    
  }
  # Add code here to remove rows with no data in subject columns. 
  #master <- na.omit(master, cols = sub_cols) # removes row if ANY of the cols has NAs. 
  #test <- master[rowSums(is.na(master[,3:26])) !=23, ]
  #order() master rows by Field.Time
  
  write.csv(master, paste0(OCR_list[a],"_DF",".csv"), row.names=F)
  
}







#TEST CODE THAT CAN BE USED FOR GRAPHING
mice <- c("M1", "M2", "M3", "M4", "M5","F01","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15")
graph_master <- master
graph_master2 <- master

for (m in 1:nrow(master)){
  for (r in 1:length(mice)){
    if (!is.na(master[m,r+8])){
      graph_master[m,r+8] <- graph_master[m,2]
    }
  }
}

for (m in 1:nrow(master)){
  for (r in 1:length(mice)){
    if (!is.na(master[m,r+8])){
      graph_master2[m,r+8] <- colnames(graph_master2)[r+8]
    }
  }
}


graph_master2$Field.Time <- as.POSIXct(graph_master2$Field.Time,format="%Y-%m-%d %H:%M:%S")


#this is what was used to produce a graph
z <- subset(graph_master2,(!is.na(graph_master2$F1)))
w <- subset(graph_master2,(!is.na(graph_master2$F2)))
z$Field.Time <- as.POSIXct(z$Field.Time,format="%Y-%m-%d %H:%M:%S")
w$Field.Time <- as.POSIXct(w$Field.Time,format="%Y-%m-%d %H:%M:%S")
ggplot() + geom_point(data=z, aes(x=z$Field.Time,y=z$F1))+
  geom_point(data=w, aes(x=w$Field.Time,y=w$F2))




