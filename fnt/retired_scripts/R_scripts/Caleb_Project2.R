setwd("D:/Alex Data/T001_Bravo_Zone1_CAM2")
library(readr)
library(dplyr)
library(reshape)

filelist = list.files(pattern = "*.smi")
myfiles = lapply(filelist, read.delim)

timestamp <- vector()
for (i in 1:length(myfiles)) {
y <- subset(myfiles[i][[1]], startsWith(as.character(X.SAMI.),"2018"))
timestamp <- append(timestamp, y)
}

full_timestamp<-unlist(timestamp)

write.csv(full_timestamp, "timestamp.csv")

#Merge dataframes to create master file

DF1 <- read_csv("timestamp.csv")
DF2 <- read_csv("T001_Bravo_Zone1_CAM2.csv", skip = 18)
mydata <- DF2[,c("Subject", "Time", "X9")]

md <- melt(mydata, id=(c("Subject", "X9")))
new <- cast(md, Subject+value~X9)
new_clean <- new[,c("Subject","START","STOP")]
temp_stop <- filter(new_clean, STOP != "NA")
temp_start <- filter(new_clean, START != "NA")
temp <- cbind(temp_start,temp_stop)
temp_clean <- temp[,c(1:2,6)]
zone <- rep(1, nrow(temp_clean))            #adds a zone column
temp_zone <- cbind(zone, temp_clean)

denominator <- as.numeric(unlist(DF2[1,3]))
temp_zone$START_convert <- (temp_zone$START)/denominator*(length(full_timestamp))
temp_zone$STOP_convert <- (temp_zone$STOP)/denominator*(length(full_timestamp))
temp_zone$START_round <- round(temp_zone$START_convert)
temp_zone$STOP_round <- round(temp_zone$STOP_convert)

sapply(full_timestamp, class) #data stored as a factor
new_timestamp <- as.character(full_timestamp)  #convert data to a character

START_real <- rep(1, nrow(temp_zone))
STOP_real <- rep(1, nrow(temp_zone))
as.factor(START_real)
as.factor(STOP_real)

temp_real <- cbind(temp_zone, START_real, STOP_real)

for (i in 1:nrow(temp_real)){
temp_real$START_real[i] <- new_timestamp[temp_zone$START_round[i]]
}

for (i in 1:nrow(temp_real)){
  temp_real$STOP_real[i] <- new_timestamp[temp_zone$STOP_round[i]]
}

temp_real$START_real <- strptime(temp_real$START_real, "%Y-%m-%d %H:%M:%S")
temp_real$STOP_real <- strptime(temp_real$STOP_real, "%Y-%m-%d %H:%M:%S")

duration <- temp_real$STOP_real-temp_real$START_real
temp_duration <- cbind(temp_real, duration)

duration <- temp_duration$STOP_real-temp_duration$START_real

y <-  format(temp_duration$START_real, "%b%d\n%H%p")
x <- table(y)
plot(x)







#scratch work
masterDF <- data.frame(zone = numeric(), id = character(), start_video = numeric(), stop_video = numeric(), start_real = numeric(), stop_real = numeric(), duration = numeric())

DF1 <- read_csv("timestamp.csv")
DF2 <- read_csv("T001_Bravo_Zone1_CAM2.csv", skip = 18)

temp <- filter(DF2, Subject == "F1", X9 == "START")
temp2 <- t(temp$Subject)
masterDF$id <- cbind(masterDF$id, t(temp2))


masterDF <- data.frame(zone = numeric(), id = character(), start_video = numeric(), stop_video = numeric(), start_real = numeric(), stop_real = numeric(), duration = numeric())

for (file in dir()) {
  temp <- read.csv2(file, sep = " ", header = FALSE)
  names(temp) <- c("zone", "id", "start_video", "stop_video", "start_real", "stop_real", "duration")
  masterDF <- rbind(masterDF, temp)
}


#example
data <- data.frame(ticker=character(),
  value=numeric(),
  date = as.Date(character()),
  stringsAsFactors=FALSE)

for (file in dir())
{
  stocks <- read.csv2(file, sep = " ", header = FALSE)
  stocks$date <- sub(".csv", "", file)
  names(stocks)<- c("ticker", "value", "date")
  data <- rbind(data, stocks)
}


