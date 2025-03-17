## LID.2018.Analysis.v.1.0.0.R
## Caleb Clifton Vogt, PhD Cornell University
## Updated 10.9.2019

# PROJECT ANALYSIS: 3_Liddell_2018_master ----------------------------------------
library(googlesheets)
library(ggplot2)
gs_auth(new_user = TRUE)
# gs_ls()
for_gs <- gs_title("3_Liddell_2018_master")
liddell_master <- gs_read(for_gs)
str(liddell_master) #DISPLAY THE STRUCTURE OF THE R OBJECT

library(ggplot2)

ggplot(data = liddell_master) +
  aes(x = strain, y = `mass.%.change`) +
  geom_boxplot(fill = "#b4de2c") +
  labs(title = "Liddell 2018: Male Percent Body Mass Change",
       x = "Strain",
       y = "% Mass Change",
       subtitle = "(Post-Field mass / Pre- Field Mass)") +
  theme_stata()
as.numeric(liddell_master$testes.library(ggplot2)
           
           
           
           
           
           
           ggplot(data = liddell_master) +
             aes(x = strain, y = testes.g) +
             geom_boxplot(fill = "#b4de2c") +
             labs(title = "Liddell 2018: Male Testes Mass",
                  x = "Strain",
                  y = "Testes Mass (g)") +
             theme_stata()
           g)

library(ggplot2)

ggplot(data = liddell_master) +
  aes(x = strain, y = oft.dist.m) +
  geom_boxplot(fill = "#b4de2c") +
  labs(title = "OFT: Male Distance Travelled",
       x = "Strain",
       y = "Distance (m)") +
  theme_stata()

ggplot(data = liddell_master) +
  aes(x = strain, y = oft.mass) +
  geom_boxplot(fill = "#b4de2c") +
  labs(title = "OFT: Male Mass",
       x = "Strain",
       y = "Mass (g)") +
  theme_stata()




ggplot(data = liddell_master) +
  aes(x = strain, y = oft.total.dist.mm) +
  geom_boxplot(fill = "#b4de2c") +
  labs(title = "Open Field Trial",
       x = "Strain",
       y = "Distance (mm)",
       subtitle = "Liddell 2018") +
  theme_minimal()

library(ggplot2)

ggplot(data = liddell_master) +
  aes(x = strain, y = oft.mass) +
  geom_boxplot(fill = "#b4de2c") +
  labs(title = "Liddell 2018: Body Mass",
       x = "Mouse Strain",
       y = "Mass (g)") +
  theme_stata()







# LIDDELL 2018 BORIS + OCR INTEGRATION, DURATION ANALYSIS -----------------

# Old Duration Analysis Framework 2018:
# Check the Boris observation files for the unnecessary metadata rows. 
# Make sure to check that the number of "STARTs" and "STOPS" are the same in the csv
# Otherwise this will likely throw an error. 
aa=1
for (aa in 1:length(BORIS_list)) {
  DF1 <- read.csv(OCR_list[aa])
  DF2 <- read.csv(BORIS_list[aa], skip = 15) #worth checking if BORIS files have 15 worthless rows
  DF2["Real_World_Time"] <- NA # Add column 
  rw_time <- DF1[ ,2] #pulls out the real world times
  # NOTE: FPS ADJUSTMENT/MULTIPLICATION FACTOR NEEDS TO BE ADDED HERE.
  BOR_time <- DF2[ ,1] #pulls out the BORIS video time in seconds
  # Rounds the BORIS video times to nearest second.
  # ADD MULTIPLICATION FACTOR HERE TO BOR_TIME?
  round_BOR_time <- round(BOR_time)   
  DF2$Real_World_Time <-rw_time[round_BOR_time] #Populates RWT col with RWT according to rounded BOR_Time frame # = row
  write.csv(DF2, paste(BORIS_list[1],"_RT",".csv", sep = ''))
}


### STOP

#Functional, but silly. # of columns must match. Change csv names. 
D1 <- read.csv("RT_BORIS_T002_Z1_W1.csv")
D2 <- read.csv("RT_BORIS_T002_Z2_W1.csv")
D3 <- read.csv("RT_BORIS_T002_Z3_W1.csv")
D4 <- read.csv("RT_BORIS_T002_Z4_W1.csv")
D5 <- read.csv("RT_BORIS_T002_Z5_W1.csv")
D6 <- read.csv("RT_BORIS_T002_Z6_W1.csv")
D7 <- read.csv("RT_BORIS_T002_Z7_W1.csv")
D8 <- read.csv("RT_BORIS_T002_Z8_W1.csv")

#Delete extra column from RT file by name if necessary. Actually, better method would be to add an empty column 
D5 <- subset(D5, select = -c(Modifier.1))

#merge RT data. Rename as needed. 
merged <- do.call("rbind", list(D1, D2, D3, D4, D5, D6, D7, D8)) #merge1 <- rbind(D1,D2). 

#Note that zone information is included in the file name column. 
merged$Zone <- ifelse(grepl("Z1", merged$Media.file.path), "1", 
                      ifelse(grepl("Z2", merged$Media.file.path), "2",
                             ifelse(grepl("Z3", merged$Media.file.path), "3",
                                    ifelse(grepl("Z4", merged$Media.file.path), "4",
                                           ifelse(grepl("Z5", merged$Media.file.path), "5",
                                                  ifelse(grepl("Z6", merged$Media.file.path), "6",
                                                         ifelse(grepl("Z7", merged$Media.file.path), "7",
                                                                ifelse(grepl("Z8", merged$Media.file.path), "8", "NONE"))))))))
merged$zone <- as.numeric(merged$zone)

#Subset the merge file. #Check columns. 
short_merge <- subset(merged, select = c(X,
                                         Time,
                                         Media.file.path,
                                         zone,
                                         Subject,
                                         Behavior,
                                         #Modifier.1, #Ethogram specific, check
                                         #Modifier.2, #ethogram specific, check
                                         Comment,     #Ethogram specific
                                         Status,
                                         Real_World_Time))

#Write a csv here if you want START/STOP in single column. 
write.csv(short_merge, "MERGE_RT_BORIS_T002_W1.csv")

# Extract START and STOP rows and combine into new dataframe. 
DF1 <- short_merge[order(short_merge$Status, short_merge$Subject), ]
START_rows <- DF1[DF1$Status == 'START', ]
STOP_rows <- DF1[DF1$Status == 'STOP', ]
DF2 <- cbind(START_rows, STOP_rows)

#check which columns you are renaming the correct columns in DF2
names(DF2)[9] <- "RWT_START" 
names(DF2)[18] <- "RWT_STOP"

#change time format into appropriate class
library(lubridate)
#Converts factor to "POSIXct" and POSIXt"
DF2$RWT_START <- ymd_hms(DF2$RWT_START)  
DF2$RWT_STOP <- ymd_hms(DF2$RWT_STOP)

# Converts factor to POSIXlt" and "POSIXt"
#DF2$RWT_START <- strptime(DF2$RWT_START, "%Y-%m-%d %H:%M:%S") 
#DF2$RWT_STOP <- strptime(DF2$RWT_STOP, "%Y-%m-%d %H:%M:%S")

# Calculate duration of state events
duration <- DF2$RWT_STOP - DF2$RWT_START

#Merge dataframes and durations
caleb <- cbind(DF2, duration)

#Decide what final columns you want. 
dur_merge <- subset(caleb, select = c(X,
                                      zone,
                                      Subject,
                                      Behavior,
                                      Modifier.1,
                                      Modifier.2,
                                      #Comment,
                                      RWT_START,
                                      RWT_STOP,
                                      duration))


#Write DUR_MERGE_RT.CSV
write.csv(dur_merge, "DUR_MERGE_RT_BORIS_T002_W2.csv")




# LIDDELL GRAPHING --------------------------------
library(tidyverse)
library(lubridate)
library(ggplot2)
#Figure 1. Total durations of females in RZ's for the entire week. 
#Figure 2. Development of female space use over time. 
#Figure 3. Total durations of male space use in RZ's for the entire week. 
#Figure 4. Development of male space use over time. Also say which resources animals were released in. 
#add red square for release point. 
#Figure 5. 

#T002_W1 Graphing with MERGE file. 
setwd("C:/Users/caleb/Desktop/data")
merge <- read.csv("MERGE_RT_BORIS_T002_W1.csv")
merge <- subset(merge, select = -c(Comment))
merge <- na.omit(merge)
merge$Real_World_Time <- ymd_hms(merge$Real_World_Time)


caleb1 <- subset(merge, Status == 'START')
caleb2 <- subset(merge, Status == 'STOP')

caleb1$ID <- seq.int(nrow(caleb1))
caleb2$ID <- seq.int(nrow(caleb2))

caleb3 <- rbind(caleb1,caleb2) 

F1_1 <- subset(caleb3, (Subject == "F1"))
F2_1 <- subset(caleb3, (Subject == "F2"))
F3_1 <- subset(caleb3, (Subject == "F3"))
F4_1 <- subset(caleb3, (Subject == "F4"))
F5_1 <- subset(caleb3, (Subject == "F5"))
F6_1 <- subset(caleb3, (Subject == "F6"))
F7_1 <- subset(caleb3, (Subject == "F7"))
F8_1 <- subset(caleb3, (Subject == "F8"))
F9_1 <- subset(caleb3, (Subject == "F9"))
F10_1 <- subset(caleb3, (Subject == "F10"))

#T002_W1 ALL FEMALE ZONE OCCUPANCY (WEEK 1) 
#Prepare dataframes
dur1 <- read.csv("DUR_MERGE_RT_BORIS_T002_W1.csv")
dur1 <- na.omit(dur1)
dur1 <- dur1[!grepl("Vole", dur1$Subject), ] #remove voles
#Plot "FEMALE ZONE OCCUPANCY (WEEK 1)"

ggplot(dur1, aes(fill=Subject, y=duration, x=zone)) + 
  geom_bar(stat="identity") +
  labs(title = "Female Zone Occupancy (Week 1)") +
  facet_wrap(~Subject) #Creates multiple plots by subject. 



# T002_W1 Female Individual Movement Graphs 
#F1_1 Exploration
ggplot(data=F1_1, aes(x=F1_1$Real_World_Time, y=F1_1$zone, group=F1_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F1 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F2_1 Exploration
ggplot(data=F2_1, aes(x=F2_1$Real_World_Time, y=F2_1$zone, group=F2_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F2 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F3_1 Exploration
ggplot(data=F3_1, aes(x=F3_1$Real_World_Time, y=F3_1$zone, group=F3_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F3 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F4_1 Exploration
ggplot(data=F4_1, aes(x=F4_1$Real_World_Time, y=F4_1$zone, group=F4_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F4 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))


#F5_1 Exploration
ggplot(data=F5_1, aes(x=F5_1$Real_World_Time, y=F5_1$zone, group=F5_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F5 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F6_1 Exploration
ggplot(data=F6_1, aes(x=F6_1$Real_World_Time, y=F6_1$zone, group=F6_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F6 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F7_1 Exploration
ggplot(data=F7_1, aes(x=F7_1$Real_World_Time, y=F7_1$zone, group=F7_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F7 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F8_1 Exploration
ggplot(data=F8_1, aes(x=F8_1$Real_World_Time, y=F8_1$zone, group=F8_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F8 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F9_1 Exploration
ggplot(data=F9_1, aes(x=F9_1$Real_World_Time, y=F9_1$zone, group=F9_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F9 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F10_1 Exploration
ggplot(data=F10_1, aes(x=F10_1$Real_World_Time, y=F10_1$zone, group=F10_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F10 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))




############T002_W2 ALL FEMALE + MALE ZONE OCCUPANCY (WEEK 2)
#Prepare dataframes
dur2 <- read.csv("DUR_MERGE_RT_BORIS_T002_W2.csv")
dur2 <- subset(dur2, select = -c(Modifier.1, Modifier.2))
dur2 <- na.omit(dur2)
dur2f <- dur2[!grepl("M", dur2$Subject), ] #remove male rows
dur2m <- dur2[!grepl("F", dur2$Subject), ] #remove female rows

#Plot "FEMALE ZONE OCCUPANCY (WEEK 2)"
ggplot(dur2f, aes(fill=Subject, y=duration, x=zone)) + 
  geom_bar(stat="identity") +
  labs(title = "Female Zone Occupancy (Week 2)") +
  facet_wrap(~Subject) #Creates multiple plots by subject. 


#Plot "MALE ZONE OCCUPANCY (WEEK 2)"
ggplot(dur2m, aes(fill=Subject, y=duration, x=zone)) + 
  geom_bar(stat="identity") +
  labs(title = "Male Zone Occupancy (Week 2)") +
  facet_wrap(~Subject) #Creates multiple plots by subject. 


# male and female full week zone use
ggplot(dur2, aes(fill=Subject, y=duration, x=zone)) + 
  geom_bar(stat="identity") + 
  labs(title = "Male Zone Occupancy (Week 2") +
  facet_wrap(~Subject)




########## T002_W2 INDIVIDUAL MOVEMENT GRAPHS 
merge2 <- read.csv("MERGE_RT_BORIS_T002_W2.csv")
merge2 <- subset(merge2, select = -c(Modifier.1, Modifier.2,Comment))
merge2 <- na.omit(merge2)
merge2$Real_World_Time <- ymd_hms(merge2$Real_World_Time)


#### Prepare dataframes
alex1 <- subset(merge2, Status == 'START')
alex2 <- subset(merge2, Status == 'STOP')

alex1$ID <- seq.int(nrow(alex1))
alex2$ID <- seq.int(nrow(alex2))
alex3 <- rbind(alex1,alex2) 

#Prepare female individual data. 
F1_2 <- subset(alex3, (Subject == "F1"))
F2_2 <- subset(alex3, (Subject == "F2"))
F3_2 <- subset(alex3, (Subject == "F3"))
F4_2 <- subset(alex3, (Subject == "F4"))
F5_2 <- subset(alex3, (Subject == "F5"))
F6_2 <- subset(alex3, (Subject == "F6"))
F7_2 <- subset(alex3, (Subject == "F7"))
F8_2 <- subset(alex3, (Subject == "F8"))
F9_2 <- subset(alex3, (Subject == "F9"))
F10_2 <- subset(alex3, (Subject == "F10"))


#Prepare male individual data.
M1_2 <- subset(alex3, (Subject == "M1"))
M2_2 <- subset(alex3, (Subject == "M2"))
M3_2 <- subset(alex3, (Subject == "M3"))
M4_2 <- subset(alex3, (Subject == "M4"))
M5_2 <- subset(alex3, (Subject == "M5"))
M6_2 <- subset(alex3, (Subject == "M6"))



#F1_2 Exploration
ggplot(data=F1_2, aes(x=F1_2$Real_World_Time, y=F1_2$zone, group=F1_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F1 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F2_2 Exploration
ggplot(data=F2_2, aes(x=F2_2$Real_World_Time, y=F2_2$zone, group=F2_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F2 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F3_2 Exploration
ggplot(data=F3_2, aes(x=F3_2$Real_World_Time, y=F3_2$zone, group=F3_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F3 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F4_2 Exploration
ggplot(data=F4_2, aes(x=F4_2$Real_World_Time, y=F4_2$zone, group=F4_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F4 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))


#F5_2 Exploration
ggplot(data=F5_2, aes(x=F5_2$Real_World_Time, y=F5_2$zone, group=F5_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F5 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F6_2 Exploration
ggplot(data=F6_2, aes(x=F6_2$Real_World_Time, y=F6_2$zone, group=F6_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F6 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F7_2 Exploration
ggplot(data=F7_2, aes(x=F7_2$Real_World_Time, y=F7_2$zone, group=F7_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F7 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F8_2 Exploration
ggplot(data=F8_2, aes(x=F8_2$Real_World_Time, y=F8_2$zone, group=F8_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F8 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F9_2 Exploration
ggplot(data=F9_2, aes(x=F9_2$Real_World_Time, y=F9_2$zone, group=F9_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F9 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F10_2 Exploration
ggplot(data=F10_2, aes(x=F10_2$Real_World_Time, y=F10_2$zone, group=F10_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F10 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))



# T002_W2 Male Individual Movement Graphs 
#M1_W2 Exploration
ggplot(data=M1_2, aes(x=M1_2$Real_World_Time, y=M1_2$zone, group=M1_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="M1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#M2_W2 Exploration
ggplot(data=M2_2, aes(x=M2_2$Real_World_Time, y=M2_2$zone, group=M2_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="M2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#M3_W2 Exploration
ggplot(data=M3_2, aes(x=M3_2$Real_World_Time, y=M3_2$zone, group=M3_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="M3 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#M4_W2 Exploration
ggplot(data=M4_2, aes(x=M4_2$Real_World_Time, y=M4_2$zone, group=M4_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="M4 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#M5_W2 Exploration
ggplot(data=M5_2, aes(x=M5_2$Real_World_Time, y=M5_2$zone, group=M5_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="M5 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#M6_W2 Exploration
ggplot(data=M6_2, aes(x=M6_2$Real_World_Time, y=M6_2$zone, group=M6_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="M6 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))





##########Unclear what this does. 
attach(dur2)
F1wut <- aggregate(duration ~ zone, data = subset(dur2,Subject == 'F1'),sum)
F2 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'F2'),sum)
M1 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M1'),sum)
M2 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M2'),sum)
M3 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M3'),sum)
M4 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M4'),sum)
M5 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M5'),sum)
M6 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M6'),sum)

ggplot(data=F1wut, aes(x=F1$zone,y=F1$duration)) + geom_bar(stat='identity')
ggplot(data=F2, aes(x=F2$zone,y=F2$duration)) + geom_bar(stat='identity')
ggplot(data=M2, aes(x=M2$zone,y=M2$duration)) + geom_bar(stat='identity')
ggplot(data=M3, aes(x=M3$zone,y=M3$duration)) + geom_bar(stat='identity')
ggplot(data=M4, aes(x=M4$zone,y=M4$duration)) + geom_bar(stat='identity')
ggplot(data=M5, aes(x=M5$zone,y=M5$duration)) + geom_bar(stat='identity')
ggplot(data=M6, aes(x=M6$zone,y=M6$duration)) + geom_bar(stat='identity')

ggplot(data=subset(dat,Subject == "M5"), aes(x=zone,y=duration)) + geom_point()




# LIDDELL MOVEMENT ANIMATION -----------------------
library(moveVis)
library(anipaths)
library(tidyverse)
library(lubridate)
library(igraph)

setwd("E:/Data/3_Liddell_Ecology_proc/T002/W1/BORIS")
tracks <- read.csv("DUR_MERGE_RT_BORIS_T002_W1.csv") 




# Create outline of Liddell Paddock
referencecoords <- data.frame("df"= 1:2, "X" = c(0,0,75,75), "Y" = c(0,175,175,0))

plot(NA, xlim=c(0,180),
     ylim=c(0,180),
     xlab='',ylab='',
     axes=FALSE)

outline.xs <- referencecoords$X[c(1,2,3,4,1)]
outline.ys <- referencecoords$Y[c(1,2,3,4,1)]
lines(outline.xs,outline.ys)

#Create Resource Zones
z1x <- c(15, 30, 30, 15)
z1y <- c(20, 20, 35, 35)
polygon(z1x,z1y)
#center coords (20,25)

z2x <- c(45,60, 60, 45)
z2y <- c(20, 20, 35, 35)
polygon(z2x,z2y)
#center coords (50,25)

z3x <- c(15, 30, 30, 15)
z3y <- c(60, 60, 75, 75)
polygon(z3x,z3y)
#center coords (20,65)

z4x <- c(45,60, 60, 45)
z4y <- c(60, 60, 75, 75)
polygon(z4x,z4y)
#center coords (50,65)


z5x <- c(15, 30, 30, 15)
z5y <- c(100, 100, 115, 115)
polygon(z5x,z5y)
#center coords (20,105)


z6x <- c(45,60, 60, 45)
z6y <- c(100, 100, 115, 115)
polygon(z6x,z6y)
#center coords (50,105)


z7x <- c(15, 30, 30, 15)
z7y <- c(140, 140, 155, 155)
polygon(z7x,z7y)
#center coords (20,145)


z8x <- c(45,60, 60, 45)
z8y <- c(140, 140, 155, 155)
polygon(z8x,z8y)
#center coords (50,145)

#add xy coordinate columns to tracks dataframe. would be good to add some jitter to that shit. 

tracks$x <- ifelse(grepl("1", tracks$zone), "20", 
                   ifelse(grepl("2", tracks$zone), "50",
                          ifelse(grepl("3", tracks$zone), "20",
                                 ifelse(grepl("4", tracks$zone), "50",
                                        ifelse(grepl("5", tracks$zone), "20",
                                               ifelse(grepl("6", tracks$zone), "50",
                                                      ifelse(grepl("7", tracks$zone), "20",
                                                             ifelse(grepl("8", tracks$zone), "50",
                                                                    "none"))))))))

tracks$y <- ifelse(grepl("1", tracks$zone), "25", 
                   ifelse(grepl("2", tracks$zone), "25",
                          ifelse(grepl("3", tracks$zone), "65",
                                 ifelse(grepl("4", tracks$zone), "65",
                                        ifelse(grepl("5", tracks$zone), "105",
                                               ifelse(grepl("6", tracks$zone), "105",
                                                      ifelse(grepl("7", tracks$zone), "145",
                                                             ifelse(grepl("8", tracks$zone), "145",
                                                                    "none"))))))))

clean_tracks <- subset(tracks, select = c(Subject, x, y, RWT_START, RWT_STOP))



#Example code
library(moveVis)
library(move)

data("move_data", package = "moveVis") # move class object
# if your tracks are present as data.frames, see df2move() for conversion
head(move_data)
summary(move_data)
# align move_data to a uniform time  scale
m <- align_move(move_data, res = 240, digit = 0, unit = "secs")

# create spatial frames with a OpenStreetMap watercolour map
frames <- frames_spatial(m, path_colours = c("red", "green", "blue"),
                         map_service = "osm", map_type = "watercolor", alpha = 0.5) %>% 
  add_labels(x = "Longitude", y = "Latitude") %>% # add some customizations, such as axis labels
  add_northarrow() %>% 
  add_scalebar() %>% 
  add_timestamps(m, type = "label") %>% 
  add_progress()

frames[[100]] # preview one of the frames, e.g. the 100th frame

# animate frames
animate_frames(frames, out_file = "moveVis.gif")


# LIDDELL 2018: SOCIAL NETWORK: Create Adjacency matrix -------------------
#SOCIAL NETWORK ASSOCIATION MATRIX FOR ALL OF WEEK 1. 
setwd("E:/Data/3_Liddell_Ecology_proc/T002/W2/BORIS")
library(tidyverse)
library(lubridate)
library(igraph)

vog <- read.csv("DUR_MERGE_RT_BORIS_T002_W2.csv") 

#Depending on basis for the social network, need to remove unnecessary columns
# A network based on co-occupancy of the resource zones should not have columns containing nas,
# for instance from the modifier columns. 

vog <- subset(vog, select = c(X, 
                              zone,
                              Subject,
                              Behavior, 
                              RWT_START,
                              RWT_STOP,
                              duration))


#make sure to check that your RWT_START and STOP dates havent lost the seconds. 
#for some reason was running into this issue quite a bit. 
vog$RWT_START <- ymd_hms(vog$RWT_START)  
vog$RWT_STOP <- ymd_hms(vog$RWT_STOP)

# Remove any rows with NA's. 
vog <- na.omit(vog)


vog <- vog[order(vog$RWT_START, na.last=FALSE), ]

#Quickly check that your durations are... normal?
#vog$duration2<-vog$RWT_STOP-vog$RWT_START
#plot(duration~duration2,data=vog)


zones<-sort(unique(vog$zone))
flag<-0
for (aa in 1:length(zones)) {
  zone_num <- zones[aa] #Current zone being analyzed
  zonewise<-vog[which(vog$zone==zone_num),] #add rows for observations in current zone
  print(paste("Processing zone ",zone_num," out of ",length(zones),sep=''))
  for(bb in 1:(nrow(zonewise)-1)){ #unclear to me the purpose of taking off the last row. 
    c1 <- zonewise[bb,,drop=FALSE] #create dataframe with all observations within current zone
    for(cc in (bb+1):(nrow(zonewise))){ #for i in 
      c2 <- zonewise[cc,,drop=FALSE]
      if(c1$Subject!=c2$Subject){
        if(c2$RWT_START<c1$RWT_STOP & c2$RWT_START>c1$RWT_START){
          xtemp <- matrix(c(as.character(c1$Subject),as.character(c2$Subject),c1$zone),nrow=1)
          colnames(xtemp)<-c("ID1","ID2","zone")
          if(flag<1){
            socialinteractions <- xtemp
          } else {
            socialinteractions <- rbind(socialinteractions,xtemp)
          }
          flag <- flag+1
          
        }
      }
      
    }
  }
  
}

# This will create a directed data frame as combinations are repeated. no time association
socialinteractions <- as.data.frame(socialinteractions)
socialinteractions$ID2 <- factor(socialinteractions$ID2)
socialinteractions$ID1 <- factor(socialinteractions$ID1)
socialinteractions$zone <- factor(socialinteractions$zone)
summary(socialinteractions)

# Create directed adjacency matrix 
g.unit <- (table(socialinteractions))
caca <- as.data.frame(g.unit)
#caca[which(max(caca$Freq)==caca$Freq),]
ids <- list()
ids[[1]]<-sort(unique(caca$ID1))
ids[[2]]<-sort(unique(caca$ID2))

#Create the totsmagoats. 
for(z in 1:length(zones)){
  diszone <- caca[which(caca$zone==z),]
  ##change number to reflect all subjects in ids. 
  present <- matrix(diszone$Freq,nrow=16,ncol=16,dimnames = ids) ### Change this. 
  if(z<2){
    totsmagoats<-present
  } else {
    totsmagoats<-totsmagoats+present
  }
}

sum(totsmagoats)
# This is a directed adjaceny matrix, but unclear how. 
# Weighted towards which individual comes into the tub/approaches?
write.csv(totsmagoats, "directed_adjacency_matrix.csv")



# SOCIAL NETWORK: iGraph Plotting ------------------------------

#Working with the directed adjacency matrix. 
totsmagoats <- read.csv("directed_adjacency_matrix.csv")

# Directed social network 
network_df <- totsmagoats
net_graph <- graph.adjacency(network_df, mode="directed", weighted = NULL)
g <- simplify(net_graph)
V(g)$label <- V(g)$name
V(g)$degree <- degree(g)
set.seed(3952) # what does this doe. 
layout1 <- layout.fruchterman.reingold(g)
E(g)$weight <- edge.betweenness(g)

plot(g,
     vertex.color = "grey", # change color of nodes
     vertex.size = 30,
     vertex.label.color = "black", # change color of labels
     vertex.label.cex = 2.0, # change size of labels to 75% of original size
     edge.curved=.25, # add a 25% curve to the edges
     edge.color="grey20",# change edge color to grey
     edge.width = E(g)$weight*0.6
     # edge.width=edge.betweenness(g)
) 

# plot(g, layout=layout1)
# plot.igraph(g)


# Undirected weighted social network 
summary(socialinteractions)
df <- subset(socialinteractions, select = -c(3))
# coerces the data into a two-column matrix format that igraph likes
el=as.matrix(df)
el[,1]=as.character(el[,1])
el[,2]=as.character(el[,2])
# turns the edgelist into a 'graph object'
g=graph.edgelist(el,directed=FALSE) 

#create adjacency matrix from edgelist
g <- get.adjacency(g,sparse=FALSE) 

# create igraph object from undirected adjacency matrix
g <- graph.adjacency(g, mode="undirected", weighted =TRUE)

#simplify igraph object. removes mulitiple edges and loop edges
g <- simplify(g)

V(g)$label <- V(g)$name
V(g)$degree <- degree(g)

set.seed(3952)#what does this do? I have literally no clue---hahaha
layout1 <- layout.fruchterman.reingold(g)
# standard plot
plot(g,
     layout=layout1,
     vertex.color = "green",
     vertex.size = 25,
     edge.color = 'black'
)

# Pretty plot!
E(g)$weight <- edge.betweenness(g)
plot(g,
     vertex.color = "grey", # change color of nodes
     vertex.size = 30,
     vertex.label.color = "black", # change color of labels
     vertex.label.cex = 2.0, # change size of labels to 75% of original size
     edge.curved=.25, # add a 25% curve to the edges
     edge.color="grey20",# change edge color to grey
     edge.width = E(g)$weight*0.7
     # edge.width=edge.betweenness(g)
) 

# Network Measures 

degree.cent <- centr_degree(g, mode = "all")
degree.cent$res
degree(g_undir, mode='all')
degree(g_undir, mode='in')

# Create directed adjacency matrix
g.unit<-(table(socialinteractions))
caca<-as.data.frame(g.unit)
#caca[which(max(caca$Freq)==caca$Freq),]
ids<-list()
ids[[1]]<-sort(unique(caca$ID1))
ids[[2]]<-sort(unique(caca$ID2))
for(z in 1:length(zonetypes)){
  diszone<-caca[which(caca$zone==z),]
  present<-matrix(diszone$Freq,nrow=10,ncol=10,dimnames = ids)
  if(z<2){
    totsmagoats<-present
  } else {
    totsmagoats<-totsmagoats+present
  }
}

totsmagoats
View(totsmagoats)
sum(totsmagoats)
write.csv(totsmagoats, "directed_adjacency_matrix.csv")

# CCV code directed adjacency matrrix 

library(igraph)
network_df <- totsmagoats
net_graph <- graph.adjacency(network_df, mode="directed", weighted = TRUE)
g_dir <- simplify(net_graph)
V(g)$label <- V(g)$name
V(g)$degree <- degree(g)
set.seed(3952)
layout1 <- layout.fruchterman.reingold(g)
plot(g, layout=layout1)
plot.igraph(g)

# Obtain unweighted adjacency matrix from socialinteractions
# Undirected weighted social network 
library(igraph)
summary(socialinteractions)
df <- subset(socialinteractions, select = -c(3))
# coerces the data into a two-column matrix format that igraph likes
el=as.matrix(df)
el[,1]=as.character(el[,1])
el[,2]=as.character(el[,2])
# turns the edgelist into a 'graph object'
g=graph.edgelist(el,directed=FALSE) 


#create adjacency matrix from edgelist
g <- get.adjacency(g,sparse=FALSE) 

# create igraph object from undirected adjacency matrix
g <- graph.adjacency(g, mode="undirected", weighted =TRUE)

#simplify igraph object. removes mulitiple edges and loop edges
g <- simplify(g)

V(g)$label <- V(g)$name
V(g)$degree <- degree(g)

set.seed(3952)
layout1 <- layout.fruchterman.reingold(g)
# standard plot
plot(g, 
     layout=layout1,
     vertex.color = "green",
     vertex.size = 25,
     edge.color = 'black'
)

# Pretty plot!
E(g)$weight <- edge.betweenness(g)
plot(g,
     vertex.color = "grey", # change color of nodes
     vertex.label.color = "black", # change color of labels
     vertex.label.cex = 3.0, # change size of labels to 75% of original size
     edge.curved=.25, # add a 25% curve to the edges
     edge.color="grey20",# change edge color to grey
     edge.width = E(g)$weight
     # edge.width=edge.betweenness(g)
) 




# Network Measures
degree.cent <- centr_degree(g, mode = "all")
degree.cent$res
degree(g_undir, mode='all')
degree(g_undir, mode='in')


