
# Scripts for Boris and Video data combination ---------------------------
#extract timestamp from batch of smi files

library(data.table)
library(dplyr)
library(readr)
library(reshape)

wd <- setwd("G:/My Drive/Alex Liddell Data/Week 2 Analysis/T001_Bravo_Zone5_CAM10")  # DEFINE
list.files(wd)

filelist = list.files(pattern = "*.smi")
myfiles = lapply(filelist, read.delim)

timestamp <- vector()
for (i in 1:length(myfiles)) {
  y <- subset(myfiles[i][[1]], startsWith(as.character(X.SAMI.), "2018"))
  timestamp <- append(timestamp, y)
}

full_timestamp <- unlist(timestamp)

write.csv(full_timestamp, "timestamp.csv")


# Merge dataframes to create master analysis file -------------------------

DF1 <- read_csv("timestamp.csv")
DF2 <- read_csv("T001_Bravo_Zone5_CAM10.csv", skip = 18)    ## DEFINE
mydata <- DF2[,c("Subject", "Time", "X9")]

md <- melt(mydata, id=(c("Subject", "X9")))
new <- cast(md, Subject+value~X9)
new_clean <- new[,c("Subject","START","STOP")]
temp_stop <- filter(new_clean, STOP != "NA")
temp_start <- filter(new_clean, START != "NA")
temp <- cbind(temp_start,temp_stop)
temp_clean <- temp[,c(1:2,6)]
zone <- rep(5, nrow(temp_clean))            #adds a zone column: DEFINE EACH ITERATION
temp_zone <- cbind(zone, temp_clean)

denominator <- as.numeric(unlist(DF2[1,3]))
temp_zone$START_convert <- (temp_zone$START)/denominator*(length(full_timestamp))
temp_zone$STOP_convert <- (temp_zone$STOP)/denominator*(length(full_timestamp))
temp_zone$START_round <- round(temp_zone$START_convert)
temp_zone$STOP_round <- round(temp_zone$STOP_convert)

sapply(full_timestamp, class) #data stored as a factor
new_timestamp <- as.character(full_timestamp) #convert data to a character

START_real <- rep(1, nrow(temp_zone))
STOP_real <- rep(1, nrow(temp_zone))
as.factor(START_real)
as.factor(STOP_real)

temp_real <- cbind(temp_zone, START_real, STOP_real)

for (i in 1:nrow(temp_real)) {
  temp_real$START_real[i] <- new_timestamp[temp_zone$START_round[i]]
}


for (i in 1:nrow(temp_real)) {
  temp_real$STOP_real[i] <- new_timestamp[temp_zone$STOP_round[i]]
}


temp_real$START_real <- strptime(temp_real$START_real, "%Y-%m-%d %H:%M:%S")
temp_real$STOP_real <- strptime(temp_real$STOP_real, "%Y-%m-%d %H:%M:%S")

duration <- temp_real$STOP_real - temp_real$START_real
temp_duration <- cbind(temp_real, duration)


duration <- temp_duration$STOP_real - temp_duration$START_real

write.csv(temp_duration, "Zone5_temp_duration.csv") # DEFINE EACH ITERATION


#this code doesnt quite match the appropriate timing of the videos... 




# Once all temp_duration csvs have been created ---------------------------
# 

wd <- setwd("G:/My Drive/Alex Liddell Data/Week 2 Analysis/temp_duration_Zones")
list.files(wd)
dir()
 
# setwd("G:/My Drive/Alex Liddell Data/temp_duration_Zones")
data <- lapply(dir(),read.csv)
 
full_zone_DF <- rbind(data[[2]], data[[3]], data[[4]], data[[5]], data[[6]], data[[7]], data[[8]], data[[9]])
View(full_zone_DF)
write.csv(full_zone_DF, "full_zone_df.csv")
# 


# # Graphing full_zone_DF

# library(ggplot2)

#simple as shit graphs

# Fig. 1.  ----------------------------------------------------------------

# 
# y <- table(full_zone_DF$zone)
# plot(y)
# 
# 
# # Fig. 2.  ----------------------------------------------------------------
# 
# x <- table(full_zone_DF$Subject)
# plot(x)
# 
# 
# # Fig. 3.  ----------------------------------------------------------------
# 
# 
# 
# 
# 
# time <- as.character(full_zone_DF$START_real)
# time1 <- strsplit(time, " ")
# 
# View(unlist(time1))
# 
# 
# 
# ggplot(data=full_zone_DF, aes(x = Zone, y = )) +
#   geom_bar(stat = )
# 



# match start_round stop_round to full_timestamp --------------------------

# new_timestamp <- as.character(full_timestamp)
# 
# 
# start_round <- temp_zone[,7]
# stop_round <- temp_zone[,8]
# 
# 
# 
# start_stamp <- character()
# for (i in 1:length(start_round)) {
#   y <- new_timestamp[i]
#   start_stamp <- append(start_stamp, y)
}










# 
# write.csv(DF2, "boris_output.csv")
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# masterDF <- data.frame(zone = numeric(), 
#                        id = character(), 
#                        start_video = numeric(), 
#                        stop_video = numeric(), 
#                        start_real = numeric(), 
#                        stop_real = numeric(), 
#                        duration = numeric(),
#                        stringsAsFactors = FALSE)
# 
# 
# # build masterDF ----------------------------------------------------------
# 
# newrow <- data.frame(zone=1,
#                      id=2)
# 
# masterDF1 <- rbind(masterDF, newrow)


# subject <- DF2$Subject
# masterDF$id <- subject
# 
# masterDF[nrow(masterDF$id)+1, ] <- c(1,2,3,4,5,6,7) 
#   
# 
# masterDF$id <- filter(DF2[,5], Subject == "F1", X9 == "START")
# 
# 






# df4 <- merge(df1, df3, all.x = TRUE) # also doesnt work




# merged_data <- merge(data1, data2)# results in nothing dawg.








# 
# 
# 
# 
# subset(My.Data, startsWith(as.character(x), "G45"))
# subset(My.Data, grepl("^G45", My.Data$x))
# 
# my_value <-character()
# 
# subset(filelist[1], grep('2018'))
# 
# 
# for (i in 1:length(myfiles)) {
#   timestamp <- subset([i], grep("2018"))
# }
# 
# 
# 
# #
# 
# timestamp <- subset(myfiles[1][[1]], startsWith(as.character(X.SAMI.), "2018"))
# 
# 
# 
# timestamp <- subset(myfiles[1][[1]], startsWith(as.character(X.SAMI.), "2018"))

# for (i in 1:length(myfiles)) {
#   y <- subset(myfiles[i][[1]], startsWith(as.character(X.SAMI.), "2018"))
#   full_timestamp <- append(timestamp, y)
#   
# }
