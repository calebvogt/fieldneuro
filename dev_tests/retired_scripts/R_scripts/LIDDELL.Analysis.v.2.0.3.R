## LID.2019.Analysis.v.2.0.3_testing
## TESTING. 
## Caleb Clifton Vogt, PhD Cornell University
## Updated 12.2.2019
## CODING LESSON #1: SOMETIMES, YOU JUST DONT HAVE TIME TO MAKE THAT SHIT LOOK PRETTY. 


##### INSTALL ALL PACKAGES AND DEPENDENCIES FIRST ###########
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


######### METADATA PROJECT NALYSIS: 4_OXT_Liddell_data ----------------------------------
gs_auth(new_user = TRUE)
gs_ls()
for_gs <- gs_title("4_OXT_Liddell_data")
oxt_data <- gs_read(for_gs)
str(oxt_data) #DISPLAY THE STRUCTURE OF THE R OBJECT


# LIDDELL 2019: OCR + BORIS Integration -------------------------------------------------
############## STEP 1: ###########################

#Add _BORIS to the END of all the BORIS observation csv files if not yet done. 
#Make sure folder only contains BORIS csv files before doing this. 
wd <- setwd("G:/My Drive/RCNN_Copies for scoring/NBB_Symposium_Analysis/T003_W2_OCR_10")
file.list <- list.files(wd, pattern = "*.csv")


# RUN THIS ON NEWLY EXPORTED BORIS OBSERVATIONS. 
 for (i in 1:length(file.list)){
  file.rename(file.list[i], paste(file.list[i],"_BORIS",".csv", sep=''))
 }

#for removing any character prior
gsub(".*_", "", a) 
file.rename(filename,gsub(replace,with,tolower(filename)))

# MOVE BORIS OBSERVATIONS BACK TO MAIN FOLDER AND CHANGE WORKING DIRECTORY 


############## STEP 2: #########################
# CREATE MERGED_DF CSV FOR ALL OBSERVATIONS. 
# OVERWRITE PREVIOUS FILE LIEST IF NECESSARY
file.list <- list.files(wd, pattern = "*.csv")
file.list

OCR_list <- dir(wd, pattern = "*_OCR*") #make sure there are no folders. 
BORIS_list <- dir(wd, pattern = "*_BORIS*")
OCR_list
BORIS_list

# CHECK OCR FPS AND ADJUST WITHIN IN LOOP
# CHECK IF MODIFIERS ARE INCLUDED IN YOUR BORIS FILE. DO NOT WANT TO LOSE THEM!

for (a in 1:length(BORIS_list)) {
  ocr.df <- read.csv(OCR_list[a], stringsAsFactors = FALSE)
  ocr.df <- subset(ocr.df, select = -c(X))
  ocr.Frames <- rownames(ocr.df)
  ocr.df <- cbind(ocr.Frames=ocr.Frames, ocr.df)
  ocr.df$ocr.Frames <- ocr.Frames
  names(ocr.df) <- c("OCR.Frame", "Field.Time")
  
  bor.df <- read.csv(BORIS_list[a], skip=15, stringsAsFactors = FALSE) # ADJUST SKIP NUMBER DEPENDING ON BORIS FILE CSVS. 
  names(bor.df)[1]<-"OCR.Frame"
  
  ## IS OCR AT 10 FPS? USE THIS.                         
  #bor.df[,1] <- round(bor.df[,1]*10)
  
  ## IS OCR AT 1 FPS? USE THIS. 
  bor.df[,1] <- round(bor.df[,1]) 
  
  if (bor.df$OCR.Frame[1] == 0) {
    bor.df$OCR.Frame[1] <- 1
  }
  
  ## ADD MODIFIER.1 AND MODIFIER.2 COLUMNS IF NOT PRESENT. 
  if(!"Modifier.1" %in% colnames(bor.df)) {
    bor.df$Modifier.1 <- NA
  }
  
  if(!"Modifier.2" %in% colnames(bor.df)) {
    bor.df$Modifier.2 <- NA
  }
  
  bor.df$OCR.Frame<-as.numeric(as.character(bor.df$OCR.Frame))
  ocr.df$OCR.Frame<-as.numeric(as.character(ocr.df$OCR.Frame))
  bor.df$Subject <- as.character(bor.df$Subject)
  #CHANGE F1 TO F01 TO AVOID OVERWRITE ISSUE. 
  bor.df$Subject[bor.df$Subject == "F1"] <- "F01"
  #CHANGE BORIS SUBJECT NAMES IF NOT IN CORRECT FORMAT. 
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
  merged <- merged[ , c("OCR.Frame", "Field.Time", "Media.file.path", "Subject", "Behavior", "Modifier.1", "Modifier.2", "Status")] # REORDER COLUMNS AND DROP ANY COLUMNS NOT LISTED HERE. 
  merged[c("M1", "M2", "M3", "M4", "M5","M6","F01","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15")] <- NA
  cols <- as.character(colnames(merged))
  merged$Media.file.path <- OCR_list[a]
  
  #STATE EVENT DURATIONS: Create mini dataframe for figuring out the frames to which state events are applied!
  mini <- merged[order(merged$Status, merged$Subject), ]
  mini <- subset(mini, select=c(OCR.Frame, Field.Time, Subject, Behavior, Status)) 
  #PULL ENTER ZONE BEHAVIORS
  mini <- mini[mini$Behavior == 'Enter Zone', ] 
  # PULL START/STOP ROWS
  START_rows <- mini[mini$Status == 'START', ] 
  STOP_rows <- mini[mini$Status == 'STOP', ] 
  # CONCATENATE START/STOP ROWS INTO SINGLE ROW. 
  mini <- cbind(START_rows, STOP_rows)
  mini <- na.omit(mini)
  
  # EXTRACT START AND STOP ROWS FROM MINI
  for(q in 1:nrow(mini)){
    print(paste0("Processing ",q," out of ",nrow(mini), " events in file ",BORIS_list[a]))
    x <- mini[q, ]
    # SET START AND STOP OCR.FRAME COLUMNS (1 AND 6 USUALLY)
    replacement1s <- c(as.numeric(x[1]):as.numeric(x[6])) 
    mouse <- as.character(x$Subject)
    for (z in 1:length(replacement1s)){
      merged[which(merged$OCR.Frame == replacement1s[z]),grep(mouse,colnames(merged))] <- 1
    }
  }
  merged <- merged[rowSums(is.na(merged[,c("M1", "M2", "M3", "M4", "M5","M6","F01","F2","F3","F4","F5",
                                           "F6","F7","F8","F9","F10","F11","F12","F13","F14","F15")])) !=21, ]
  
  # DROP DUPLICATED FIELD.TIME ROWS, BUT KEEP ROWS WITH STARTS AND STOPS.
  merged <- merged[!duplicated(merged[c("Field.Time", "Media.file.path", "Subject", "Behavior", "Status")]), ] 
 
  write.csv(merged, paste0(OCR_list[a],"_MERGED_DF",".csv"), row.names = FALSE)
  
}

############## STEP 3: #############
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

# WRITE data CSV 
write.csv(data, "LID.2018.T003_W2_master.csv", row.names = FALSE)



############## STEP 4: ###########
## CREATE LIDDELL 2018 OVERLORD ANALYSIS CSV
file.list <- list.files(wd, pattern = "*master.csv")
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


############# STEP 5: ###########
# To-Do list. 
# 1. Set 30s-1min threshold for entry events
# 2. Plot entries and exits into tubs across zones. 
# 3. Plot social interactions
# 4. Plot total seconds of observation I have for each animal across all zones. 
# 5. Plot # of seconds of observation time within each zone. 

## SET WORKING DIRECTORY
wd <- setwd("G:/My Drive/RCNN_Copies for scoring/NBB_Symposium_Analysis")
data <- fread("LID.2018_overlord.csv", stringsAsFactors = TRUE) # nrows = , header=, etc... 
summary(data)
data$Zone <- as.numeric(data$Zone)
data$Field.Time <- ymd_hms(data$Field.Time) 


## CREATE MOVE_DF DATAFRAME FOR PLOTTING AN INDIVIDUAL'S STATE EVENTS. FIND AND REPLACE SUBJECT  BETWEEN...     ## HERE ## 
move_df <- subset(data, (Trial == "T003")) # SUBSET BY TRIAL

#Week 1 or Week 2
#move_df <- with(move_df,move_df[day(Field.Time) >= 13 & day(Field.Time) <= 19]) # SUBSET BY DATE/DAY IF DESIRED
#move_df <- with(move_df,move_df[day(Field.Time) >= 20 & day(Field.Time) <= 26]) # SUBSET BY DATE/DAY IF DESIRED

move_df <- subset(move_df, (M6 == 1)) # SUBSET BY SUBJECT
move_df <- subset(move_df, Status != "POINT" | is.na(Status)) # REMOVE ALL POINT EVENTS EXCEPT WHEN STATUS IS NOT NA
move_df <- subset(move_df, select = c(Trial, Zone, Field.Time, Subject, Behavior, Status, M6)) #subset columns
move_df <- subset(move_df, Subject == "M6" | is.na(Status)) # Only take M6 observations, remove other subjects. 
move_df <- na.omit(move_df, cols = c("Field.Time"))
move_df$Field.Time <- as.POSIXct(move_df$Field.Time)
move_df <- as.data.frame(move_df)

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

## CREATE POINT_DF FOR PLOTTING POINT BEHAVIORAL EVENTS
point_df <- subset(data, (Status == "POINT")) # SUBSET BY TRIAL
point_df <- subset(point_df, (Trial == "T003")) # SUBSET BY TRIAL
x<-unique(point_df$Subject)
x


## BUBBLE PLOT OF TIME SPENT IN PARTICULAR ZONE 
plot1 <- ggplot(stats_df, aes(x, y, size = Duration, color = Resource)) +
  ggtitle("M6 Resource Zone Activity Duration (minutes)") +
  geom_point(alpha=0.5) +
  scale_y_discrete(limits=c(0,1,2,3,4,5)) +
  scale_x_discrete(limits=c(0,1,2,3)) +
  scale_size(range = c(1, 30), name="Duration (minutes)") + # SCALE BY AREA, NOT BY RADIUS! RADIUS = QUADRATIC SCALING
  geom_text(aes(x,y, label = round(Duration, 1)), 
            position = position_dodge(width = 0.5), size = 2.5, inherit.aes = FALSE) + 
  guides(size = FALSE) + # CHOOSE WHICH LEGENDS TO OMIT
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size=3))

plot(plot1)

# CHANGE WIDTH FROM 5 -> 3 WHEN OMMITTING LEGENDS
ggsave(filename="M6_Zone_Duration_Bubble.png", plot=plot1, width = 5, height = 4, dpi = 300, units = "in", device='png')


## PLOT INDIVIDUAL MOVEMENT
plot2 <- ggplot(data=move_df, aes(Field.Time, Zone)) +
            geom_point(na.rm=TRUE, color="red", size=1) +
            ggtitle("M6 Zone Movement") +
            xlab("Date") + ylab("Zone") +
            scale_x_datetime(breaks = "1 day", labels=date_format("%m-%d")) +
            #scale_y_discrete(limits=c(1","2","3","4","5","6","7","8"), drop = FALSE) +
            scale_y_continuous(breaks = seq(1,8,1), limits=c(1,8)) +
            theme(axis.text.x = element_text(angle = 45, hjust = 1),
                  axis.line = element_line(color = "black", size = 1, linetype = "solid"))
            
plot(plot2)

ggsave(filename="M6_Zone_Movement_Scatter.png", plot=plot2, width = 5, height = 4, dpi = 300, units = "in", device='png')



## PLOT TIME SPENT IN EACH ZONE 
plot3 <- ggplot(stats_df, aes(Zone, Duration)) +
              ggtitle("M6 Zone Duration") +
              scale_x_discrete(limits=c(1,2,3,4,5,6,7,8)) +
              geom_bar(stat="identity") +
              xlab("Zone") + ylab("Time (minutes)")
              
plot(plot3)
ggsave(filename="M6_Zone_Duration_Bar.png", plot=plot3, width = 5, height = 4, dpi = 300, units = "in", device='png')




## ANIMATE BUBBLE PLOT OF TIME SPENT IN ZONES OVER TIME (IN PROGRESS)

g <- ggplot(move_df, aes(Field.Time, Zone)) +
      geom_point(alpha = 0.7, show.legend = FALSE) +
      scale_colour_manual(values = country_colors) +
      scale_size(range = c(2, 12)) +
      scale_x_log10() +
      facet_wrap(~continent) +
      # Here comes the gganimate specific bits
      labs(title = 'Time: {Field.Time}', x = 'GDP per capita', y = 'life expectancy') +
      transition_time(Field.Time) +
      ease_aes('linear')

print(g)

# Save at gif:
anim_save("271-ggplot2-animated-gif-chart-with-gganimate2.gif")
anim_save(filename = "movie.gif", animation = g)



# Extra Code: -------------------------------------------------------------

########### TESTING CODE FOR DETERMINING LENGTH OF ENTRY BOUTS############ 
## ADD THIS TO STEP 5
# merged$Field.Time <- ymd_hms(merged$Field.Time) 
# dur <- merged
# subs <- c("M1", "M2", "M3", "M4", "M5","M6","F01","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15")
# dur <-  subset(dur, M2 == 'M2')
# 
# dur$bout_status <- NA
# dur$bout_status[1] <- "START"
# 
# #for
# jumpflag<-1
# for(i in 2:(nrow(dur)-1)){
# 
#   if (difftime(dur$Field.Time[i+1], dur$Field.Time[i], units="secs") >= 30) {
#      #print(paste("jumpcounter=",jumpflag))
#       jumpflag<-jumpflag+1
#     if(difftime(dur$Field.Time[i], dur$Field.Time[i-1], units="secs") >= 30){
#      
#      
#       tempframe<-dur[i,,drop=FALSE] 
#       tempframe$bout_status <- paste("START")
#       duptemp<-tempframe
#       duptemp$bout_status<-paste("STOP")
#       duptemp$Field.Time<-duptemp$Field.Time+1
#       tempframe<-rbind(tempframe,duptemp)
#       
#     } else {
#       tempframe<-dur[i,,drop=FALSE] 
#       tempframe$bout_status<- paste("STOP")
#     }
#   
#   } else {
#       if(difftime(dur$Field.Time[i], dur$Field.Time[i-1], units="secs") >= 30){
#         tempframe<-dur[i,,drop=FALSE] 
#         tempframe$bout_status<- paste("START")
#       } else {
#         tempframe<-dur[i,,drop=FALSE]
#       }
#   }
#  if(i==2){
#      bigboy<-tempframe
#    } else {
#      bigboy<-rbind(bigboy,tempframe)
#    }
#   ## End testing code
# }
# 
# 
# 
























subs <- paste(unique(data$Subject))
# REMOVE IRRELVANT SUBS
subs <- subs[-c(2,12)]
list <- c(19:28)


df <- as.data.frame(data)

for (i in 1:length(subs)) {
  temp <- subset(df, (subs[[1]] == 1))
  
}


for (i in 1:length(list)) {
  #temp <- data[, == 1]
  #temp <- subset(data, data[,19] == 1)
  #temp <- filter(data, F6 == 1)
  #temp <- filter(data,  == 1)
  temp <- data[F6 == 1]
  
  
  temp <- data[which(data[,1 == 1), ]
                     
                     
                     
}

temp <- subset(temp, Status != "POINT" | is.na(Status))
#temp <- subset(temp, select = c(Field.Time, Zone, Subject, Behavior, Status, a))
temp <- subset(temp, Subject == a | is.na(Status))
temp <- na.omit(temp, cols = c("Field.Time"))
temp$Field.Time <- as.POSIXct(temp$Field.Time)
temp <- as.data.frame(temp)









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








subs <- c("M1", "M2", "M3", "M4", "M5","M6","F01","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15")
data <- data[data$M1 == subs[1],]


data <- with(data, data[])

for (i in 1:nrow(merged)) {
  merged$zbout[i+1] <- difftime(merged$Field.Time[i+1], merged$Field.Time[i], units="secs")
}


data <- with(data,data[day(Field.Time) >= 20 & day(Field.Time) <=21])


x <- nrow(data[data$M5 == "M5" & data$Zone == i])


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
 
 
 
 data$Field.Time <- as.POSIXct(data$Field.Time,format="%Y-%m-%d %H:%M:%S")
 ggplot() + geom_point(data=data, aes(x=data$Field.Time,y=data$F01))+
   geom_point(data=data, aes(x=data$Field.Time,y=data$F7))
 
 dataF01 <- subset(data, select = c('F01'))
 ggplot() + geom_point(data=dataF01, aes(x=data$Field.Time,y=data$Zone))
 
 
 # #this is what was used to produce a graph
 # z <- subset(graph_data2,(!is.na(graph_data2$F1)))
 # w <- subset(graph_data2,(!is.na(graph_data2$F2)))
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
 
 
 