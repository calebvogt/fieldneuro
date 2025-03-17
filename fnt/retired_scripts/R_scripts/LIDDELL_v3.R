# LIDDELL.v.3
# Caleb Clifton Vogt, PhD Cornell University
# Updated 3.10.2020


# Required Packages ###########


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
library(plot.matrix)

# Google Sheet Metadata Download and Analysis ----------------------------------


gs_auth(new_user = TRUE)
gs_ls()
for_gs <- gs_title("4_OXT_Liddell_data")
oxt_data <- gs_read(for_gs)
str(oxt_data) #DISPLAY THE STRUCTURE OF THE R OBJECT



# Analyzing Overlord.csv ###########


# To-Do list. 
# 3. Plot social interactions
# 5. Plot # of seconds of observation time within each zone. 

## SET WORKING DIRECTORY
wd <- setwd("G:/My Drive/RCNN_Copies for scoring/NBB_Symposium_Analysis")
data <- fread("LID.2018_overlord.csv", stringsAsFactors = TRUE) # nrows = , header=, etc... 
summary(data)
data$Zone <- as.numeric(data$Zone)
data$Field.Time <- ymd_hms(data$Field.Time) 
data <- na.omit(data, cols = c("Field.Time"))

# temp[1116916, 6] <- "M5" 
# write.csv(temp, "LID.2018_overlord.csv", row.names = FALSE)
# 


# CREATE DATA FRAMES

# Unclear what this data frame is used for --------------------------------


zoner_df <- subset(data, (Trial=="T002"))
zoner_df <- data
zoner_df$Field.Time <- ymd_hms(zoner_df$Field.Time) 

date1 <- as.POSIXct("2018-07-13 12:00:00")
date2 <- as.POSIXct("2018-07-14 12:00:00")
int <- interval(date1, date2)

zoner <- zoner_df[zoner_df$Field.Time %within% int,]
zoner <- subset(zoner, (Zone== "1"))

x <- unique(zoner$Subject)


 
zoner_dat <- rbind(c(8.7,9.2,5.3,8.3,9.8,9.3,9.8,9.7),c(6.0,6.2,7.2,10.0,10.0,9.5,9.3,9.7))
rownames(zoner_dat) <- c("W1","W2")
colnames(zoner_dat) <- c("Z1","Z2","Z3","Z4","Z5","Z6","Z7","Z8")

zoner_dat <- t(zoner_dat)

as.data.frame(zoner_dat)

plot(zoner_dat)

g <- ggplot(zoner_dat, ) +
  geom_point()

zoner_dat[,1]



w1 <- zoner_dat[,1]
w2 <- zoner_dat[,2]

library(ggpubr)

ggpaired(zoner_dat, cond1= "w1", cond2 = "w2", fill = "condition", pallete = "jco")

write.csv(zoner_dat, "zoner_dat.csv")

move_df <- with(move_df,move_df[day(Field.Time) >= 13 & day(Field.Time) <= 19]) # SUBSET BY DATE/DAY IF DESIRED


## CREATE POINT_DF FOR PLOTTING POINT BEHAVIORAL EVENTS (IN PROGRESS)
point_df <- subset(data, (Status == "POINT")) # SUBSET BY POINT EVENTS
point_df[point_df$Subject == ""] <- NA # CHANGE BLANK SUBJECTS TO NAS
point_df <- point_df[!is.na(point_df$Subject),] # REMOVE NAS FROM SUBJECT COLUMN
point_df <- subset(point_df, (Trial == "T002")) # SUBSET BY TRIAL 

split <- split(point_df, point_df$Subject)
## SUBSET MALES FROM LIST
split <- split[c(12:17)]
datalist = list()

for (i in names(split)) {
  temp <- data.frame(Subject = NA,
                     total.mating.bouts = NA,
                     #total.females.mated = NA,
                     total.attacking.bouts = NA,
                     M1.attacks = NA,
                     M2.attacks= NA,
                     M3.attacks = NA,
                     M4.attacks = NA,
                     M5.attacks = NA,
                     M6.attacks = NA,
                     F01.mating = NA,
                     F3.mating = NA,
                     F4.mating = NA,
                     F5.mating = NA,
                     F6.mating = NA,
                     F7.mating = NA,
                     F8.mating = NA,
                     F9.mating = NA,
                     F10.mating = NA,
                     stringsAsFactors = FALSE)
  temp$Subject <- i
  temp$total.mating.bouts <- nrow(split[[i]][split[[i]]$Modifier.1 == "Mating" | split[[i]]$Modifier.1 == "Attempted Mount"])
  #temp$total.females.mated <- 
  temp$total.attacking.bouts <- nrow(split[[i]][split[[i]]$Modifier.1 == "Attacking" | split[[i]]$Modifier.1 == "Chasing"])
  
  
  #Attack Counts
  temp$M1.attacks <- nrow(split[[i]][split[[i]]$Modifier.1 == "Attacking" & split[[i]]$Modifier.2 == "M1"])
  temp$M2.attacks <- nrow(split[[i]][split[[i]]$Modifier.1 == "Attacking" & split[[i]]$Modifier.2 == "M2"])
  temp$M3.attacks <- nrow(split[[i]][split[[i]]$Modifier.1 == "Attacking" & split[[i]]$Modifier.2 == "M3"])
  temp$M4.attacks <- nrow(split[[i]][split[[i]]$Modifier.1 == "Attacking" & split[[i]]$Modifier.2 == "M4"])
  temp$M5.attacks <- nrow(split[[i]][split[[i]]$Modifier.1 == "Attacking" & split[[i]]$Modifier.2 == "M5"])
  temp$M6.attacks <- nrow(split[[i]][split[[i]]$Modifier.1 == "Attacking" & split[[i]]$Modifier.2 == "M6"])
  
  temp$M1.attacks <- temp$M1.attacks + nrow(split[[i]][split[[i]]$Modifier.1 == "Chasing" & split[[i]]$Modifier.2 == "M1"])
  temp$M2.attacks <- temp$M2.attacks + nrow(split[[i]][split[[i]]$Modifier.1 == "Chasing" & split[[i]]$Modifier.2 == "M2"])
  temp$M3.attacks <- temp$M3.attacks + nrow(split[[i]][split[[i]]$Modifier.1 == "Chasing" & split[[i]]$Modifier.2 == "M3"])
  temp$M4.attacks <- temp$M4.attacks + nrow(split[[i]][split[[i]]$Modifier.1 == "Chasing" & split[[i]]$Modifier.2 == "M4"])
  temp$M5.attacks <- temp$M5.attacks + nrow(split[[i]][split[[i]]$Modifier.1 == "Chasing" & split[[i]]$Modifier.2 == "M5"])
  temp$M6.attacks <- temp$M6.attacks + nrow(split[[i]][split[[i]]$Modifier.1 == "Chasing" & split[[i]]$Modifier.2 == "M6"])
  
  # Mating Counts
  temp$F01.mating <- nrow(split[[i]][split[[i]]$Modifier.1 == "Mating" & split[[i]]$Modifier.2 == "F01"])
  temp$F2.mating <- nrow(split[[i]][split[[i]]$Modifier.1 == "Mating" & split[[i]]$Modifier.2 == "F2"])
  temp$F3.mating <- nrow(split[[i]][split[[i]]$Modifier.1 == "Mating" & split[[i]]$Modifier.2 == "F3"])
  temp$F4.mating <- nrow(split[[i]][split[[i]]$Modifier.1 == "Mating" & split[[i]]$Modifier.2 == "F4"])
  temp$F5.mating <- nrow(split[[i]][split[[i]]$Modifier.1 == "Mating" & split[[i]]$Modifier.2 == "F5"])
  temp$F6.mating <- nrow(split[[i]][split[[i]]$Modifier.1 == "Mating" & split[[i]]$Modifier.2 == "F6"])
  temp$F7.mating <- nrow(split[[i]][split[[i]]$Modifier.1 == "Mating" & split[[i]]$Modifier.2 == "F7"])
  temp$F8.mating <- nrow(split[[i]][split[[i]]$Modifier.1 == "Mating" & split[[i]]$Modifier.2 == "F8"])
  temp$F9.mating <- nrow(split[[i]][split[[i]]$Modifier.1 == "Mating" & split[[i]]$Modifier.2 == "F9"])
  temp$F10.mating <- nrow(split[[i]][split[[i]]$Modifier.1 == "Mating" & split[[i]]$Modifier.2 == "F10"])
  
  
  
  datalist[[i]] <- temp
  
}

point_df <- do.call(rbind, datalist)
as.data.table(point_df)


## CREATE POINT_DF2 FOR PLOTTING INDIVIDUAL ATTACK/CHASING LOCATIONS AND TIMES

point_df <- subset(data, (Status == "POINT")) # SUBSET BY POINT EVENTS
point_df[point_df$Subject == ""] <- NA # CHANGE BLANK SUBJECTS TO NAS
point_df <- point_df[!is.na(point_df$Subject),] # REMOVE NAS FROM SUBJECT COLUMN
point_df <- subset(point_df, (Trial == "T002")) # SUBSET BY TRIAL 
point_df2 <- point_df
point_df2 <- subset(point_df2, (Subject %in% c("M1", "M2", "M3", "M4", "M5", "M6")))
point_df2 <- subset(point_df2, (Modifier.1 == "Attacking" | Modifier.1 == "Chasing"))


## CREATE MOVE_DF DATAFRAME FOR PLOTTING AN INDIVIDUAL'S STATE EVENTS. FIND AND REPLACE SUBJECT  BETWEEN...     ## HERE ## 
move_df <- subset(data, (Trial == "T003")) # SUBSET BY TRIAL

#Week 1 or Week 2
#move_df <- with(move_df,move_df[day(Field.Time) >= 13 & day(Field.Time) <= 19]) # SUBSET BY DATE/DAY IF DESIRED
#move_df <- with(move_df,move_df[day(Field.Time) >= 20 & day(Field.Time) <= 26]) # SUBSET BY DATE/DAY IF DESIRED

move_df <- subset(move_df, (M6 == 1)) # SUBSET BY SUBJECT
move_df <- subset(move_df, Status != "POINT" | is.na(Status)) # REMOVE ALL POINT EVENTS EXCEPT WHEN STATUS IS NOT NA
move_df <- subset(move_df, select = c(Trial, Zone, Field.Time, Subject, Behavior, Status, M6)) #subset columns
move_df <- subset(move_df, Subject == "M6" | is.na(Status)) # Only take M6 observations, remove other subjects. 
move_df <- na.omit(move_df, cols = c("Field.Time")) # THIS LINE MAY BE SUPERFLUOUS. 

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


## CREATE DUR_BOUTS FOR ESTIMATING DURATION OF ZONE VISITATION BOUTS FOR AN INDIVIDUAL (IN PROGRESS)
# 1. MAY NEED TO SUBSET BY ZONE AS WELL AS TRIAL + INDIVIDUAL. 
# 2. DEFINITELY NEED TO APPLY CODE TO EACH INDIVIDUAL BY ZONE FIRST!!!! THE CODE CANT HANDLE THE SKIP BETWEEN ZONES. 
dur_df <- move_df
dur_df <- dur_df[!duplicated(dur_df[c("Field.Time", "Zone","Trial")]),] # 
# Order by field time
dur_df <- dur_df[order(dur_df$Zone, dur_df$Field.Time), ]
split <- split(dur_df, dur_df$Zone)
datalist = list()
for (i in (1:length(split))) {
  temp <- split[[i]]
  temp$bout_status <- NA
  temp$bout_status[1] <- "START"
  n <- nrow(temp)
  temp$bout_status[n] <- "STOP"
  datalist[[i]] <- temp
}
dur_df <- do.call(rbind, datalist)
as.data.table(dur_df)



# REMOVE OBSERVATIONS OF ONLY A SINGLE SECOND. 
for (i in 2:nrow(dur_df)-1) {
  if((difftime(dur_df$Field.Time[i+1], dur_df$Field.Time[i], units="secs") > 30) && (difftime(dur_df$Field.Time[i], dur_df$Field.Time[i-1], units="secs") > 30)) {
    dur_df$bout_status[i] <- "DELETE"
  } else {
  }
}
x <- grep('DELETE',dur_df$bout_status)

if (length(x) != 0) {
  dur_df <- dur_df[-c(x[1:length(x)]),] #eliminates issue of NA
} else {}

# ADD STARTS AND STOPS TO BOUT.STATUS. 
for (i in 2:nrow(dur_df)-1) {
  if(difftime(dur_df$Field.Time[i+1], dur_df$Field.Time[i], units="secs") > 30) { # NEGATIVE NUMBERS HERE ARE IGNORED!!! HUZZAH!
    dur_df$bout_status[i] <- "STOP"
    dur_df$bout_status[i+1] <- "START"
  } else {
  }
}



## SPECIFIC FOR DELETING PROBLEM ROW WHERE ANIMAL START AND STOP HAPPENS IN SINGLE FRAME. 
## APPLIES TO T003 Male 2 Z8 W2 DATA
# dur_df <- dur_df[-20813,]

START_rows <- subset(dur_df, (dur_df$bout_status == "START")) 
STOP_rows <- subset(dur_df, (dur_df$bout_status == 'STOP'))
dur_new <- cbind(START_rows, STOP_rows)
names(dur_new)[3] <- "RWT_START" 
names(dur_new)[11] <- "RWT_STOP"
dur_new$RWT_START <- ymd_hms(dur_new$RWT_START)  
dur_new$RWT_STOP <- ymd_hms(dur_new$RWT_STOP)
dur_new$duration <- dur_new$RWT_STOP - dur_new$RWT_START

# NEARLY FUCKING NAILED IT! ONE LAST MOTHER FUCKING BOUT THAT COMES OUT NEGATIVE. POSSIBLY FROM FUCKUP IN THE OCR 
dur_bouts <- dur_new[dur_new$duration >= 0, ]
as.data.table(dur_bouts)
dur_bouts$duration <- as.integer(dur_bouts$duration)
class(dur_bouts$duration)
dur_bouts$duration <- dur_bouts$duration/60 # CHANGES DURATION TO MINUTES 

#USE DUR BOUTS
write.csv(dur_bouts, "T003_M6_dur_bouts.csv", row.names = FALSE)



# Graphing and Creating Plots ####
## PLOT #1: BUBBLE PLOT OF TIME SPENT IN PARTICULAR ZONE 
plot1 <- ggplot(stats_df, aes(x, y, size = Duration, color = Resource)) +
  ggtitle("T003: M6 Resource Zone Activity Duration (minutes)") +
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
ggsave(filename="T003_M6_Zone_Duration_Bubble.png", plot=plot1, width = 5, height = 4, dpi = 300, units = "in", device='png')


## PLOT #2: INDIVIDUAL MOVEMENT
plot2 <- ggplot(data=move_df, aes(Field.Time, Zone)) +
  geom_point(na.rm=TRUE, color="red", size=1) +
  ggtitle("T003: M6 Zone Movement") +
  xlab("Date") + ylab("Zone") +
  scale_x_datetime(breaks = "1 day", labels=date_format("%m-%d")) +
  #scale_y_discrete(limits=c(1","2","3","4","5","6","7","8"), drop = FALSE) +m  
  scale_y_continuous(breaks = seq(1,8,1), limits=c(1,8)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.line = element_line(color = "black", size = 1, linetype = "solid"))

plot(plot2)

ggsave(filename="T003_M6_Zone_Movement_Scatter.png", plot=plot2, width = 5, height = 4, dpi = 300, units = "in", device='png')



## PLOT 3: TIME SPENT IN EACH ZONE 
plot3 <- ggplot(stats_df, aes(Zone, Duration)) +
  ggtitle("T003: M6 Zone Duration") +
  scale_x_discrete(limits=c(1,2,3,4,5,6,7,8)) +
  ylim(0,600) +
  geom_bar(stat="identity") +
  xlab("Zone") + ylab("Time (minutes)")

plot(plot3)
ggsave(filename="T003_M6_Zone_Duration_Bar.png", plot=plot3, width = 5, height = 4, dpi = 300, units = "in", device='png')



## PLOT4: VISIT DURATIONS ACROSS ALL ZONES HISTOGRAM

plot4 <- qplot(dur_bouts$duration,
               geom="histogram",
               binwidth = 1,  # 1
               main = "T003: M6 Visit Durations (All Zones)", 
               xlab = "Visit Duration (minutes)", 
               xlim = c(-1.25,31), #-1.25, 31
               ylab = "Frequency",
               ylim = c(0,250),  #250
               fill=I("blue"), 
               col=I("red"), 
               alpha=I(.2))

plot(plot4)              
ggsave(filename="T003_M6_Visit_Duration_Histogram_Bar.png", plot=plot4, width = 5, height = 4, dpi = 300, units = "in", device='png')



## PLOT5: VISIT DURATIONS ACROSS ALL ZONES HISTOGRAM (6 SECOND BIN)

plot5 <- qplot(dur_bouts$duration,
               geom="histogram",
               binwidth = 0.1,  # 1
               main = "T003: M6 Visit Durations (All Zones)", 
               xlab = "Visit Duration (minutes)", 
               xlim = c(0,10), #-1.25, 31
               ylab = "Frequency",
               ylim = c(0,50),  #250
               fill=I("blue"), 
               col=I("red"), 
               alpha=I(.2))

plot(plot5)              
ggsave(filename="T003_M6_Visit_Duration_Histogram_Bar_6s.png", plot=plot5, width = 5, height = 4, dpi = 300, units = "in", device='png')


## PLOT 7: TIME AND PLACE OF ATTACKS (INPROGRESS)

plot6 <- ggplot(data=point_df2, aes(Field.Time, Zone, color = Subject)) +
  geom_point(na.rm=TRUE,size=3) +
  ggtitle("T003: Directed Attacks by Subject") +
  xlab("Date") + ylab("Zone") +
  scale_x_datetime(breaks = "1 day", labels=date_format("%m-%d")) +
  #scale_y_discrete(limits=c(1","2","3","4","5","6","7","8"), drop = FALSE) +m  
  scale_y_continuous(breaks = seq(1,8,1), limits=c(1,8)) +
  scale_color_manual(values = c("red", "green", "blue", "yellow", "black")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.line = element_line(color = "black", size = 1, linetype = "solid"))

plot(plot6)

ggsave(filename="T003_Male_Attack_Map_Scatter.png", plot=plot6, width = 5, height = 4, dpi = 300, units = "in", device='png')


## PLOT A: PLOT STRAIN DIFFERENCES IN AGGRESSION

plotA<- ggplot(point_df) +
  aes(x = reorder(Subject, total.attacking.bouts), weight = total.attacking.bouts) +
  geom_bar(fill = "#ef562d") +
  labs(x = "Strain", y = "Attack Frequency", title = "T003: Male Aggression By Strain") +
  scale_x_discrete(labels=c("M1" = "LEWES - M1", "M2" = "WSB - M2", "M3" = "NY1 - M3", 
                            "M4" = "NY2 - M4", "M5" = "NY3 - M5", "M6" = "C57 - M6"))

plot(plotA)

ggsave(filename="T003_Male_Aggression_Freq_Bar.png", plot=plotA, width = 5, height = 4, dpi = 300, units = "in", device='png')


## PLOT B: PLOT STRAIN DIFFERENCES IN Mating
plotB<- ggplot(point_df) +
  aes(x = Subject, weight = total.mating.bouts) +
  geom_bar(fill = "#ef562d") +
  labs(x = "Strain", y = "Mating Bout Frequency", title = "T003: Male Mating Bouts By Strain") +
  scale_x_discrete(labels=c("M1" = "LEWES - M1", "M2" = "WSB - M2", "M3" = "NY1 - M3", 
                            "M4" = "NY2 - M4", "M5" = "NY3 - M5", "M6" = "C57 - M6"))

plot(plotB)

ggsave(filename="T003_Male_Mating_Freq_Bar.png", plot=plotB, width = 5, height = 4, dpi = 300, units = "in", device='png')


## PLOT C: CREATE POINT_HM FOR PLOTTING TRIAL ATTACK HEATMAP.
point_hm <- point_df
point_hm <- point_hm[,c(4:9)]
colnames(point_hm) <- c("M1", "M2", "M3", "M4", "M5", "M6")
point_hm <- as.matrix(point_hm)
#point_hm <- t(point_hm)


png('T003_Male_Aggression_Heatmap.png')
plot(point_hm, col = c("beige", "yellow", "orange", "red"), 
     breaks=c(0,1,2,4,8), 
     main = "T003: Heatmap of Male Aggression",
     xlab= "Male Receiving Aggression", ylab = "Male Aggressor")


dev.off()



# Extra Code to draw from -------------


# Alexandra Extra Code

wd <- setwd("G:/My Drive/RCNN_Copies for scoring/NBB_Symposium_Analysis")
data <- fread("LID.2018_overlord.csv", stringsAsFactors = TRUE) # nrows = , header=, etc... 
summary(data)
data$Zone <- as.numeric(data$Zone)
data$Field.Time <- ymd_hms(data$Field.Time) 
data <- na.omit(data, cols = c("Field.Time"))

dur_df <- subset(data, (Trial == "T003")) # SUBSET BY TRIAL
#Week 1 or Week 2
#move_df <- with(move_df,move_df[day(Field.Time) >= 13 & day(Field.Time) <= 19]) # SUBSET BY DATE/DAY IF DESIRED
#dur_df <- with(dur_df,dur_df[day(Field.Time) >= 20 & day(Field.Time) <= 26]) # SUBSET BY DATE/DAY IF DESIRED
dur_df <- subset(dur_df, (M6 == 1)) # SUBSET BY SUBJECT
dur_df <- subset(dur_df, Status != "POINT" | is.na(Status)) # REMOVE ALL POINT EVENTS EXCEPT WHEN STATUS IS NOT NA
dur_df <- subset(dur_df, select = c(Trial, Zone, Field.Time, Subject, Behavior, Status, M6)) #subset columns
dur_df <- subset(dur_df, Subject == "M6" | is.na(Status)) # Only take M6 observations, remove other subjects. 
dur_df <- na.omit(dur_df, cols = c("Field.Time"))
dur_df <- as.data.frame(dur_df)

dur_df$bout_status <- NA
dur_df$bout_status[1] <- "START"
n <- nrow(dur_df)
dur_df$bout_status[n] <- "STOP"
jumpcounter <- 1

#Add START and STOP times to data frame

for (i in 2:nrow(dur_df)-1) {
  if(difftime(dur_df$Field.Time[i+1], dur_df$Field.Time[i], units="secs") > 30) {
    if (is.na((dur_df$bout_status[i] != "START"))) { #r does not like this due to NA's and logical(0)
      jumpcounter <- jumpcounter+1
      dur_df$bout_status[i] <- "STOP"
      dur_df$bout_status[i+1] <- "START"
    } else {
      dur_df$bout_status[i] <- "DELETE"
    }
  } else {
    
  }
}




for (i in 2:nrow(dur_df)-1) {
  if(difftime(dur_df$Field.Time[i+1], dur_df$Field.Time[i], units="secs") >= 30) {
    if (is.na(dur_df$Status[i])) { #r does not like this due to NA's and logical(0)
      jumpcounter <- jumpcounter+1
      dur_df$bout_status[i] <- "STOP"
      dur_df$bout_status[i+1] <- "START"
    } else {
      dur_df$bout_status[i] <- "DELETE"
    }
  } else {
    
  }
}


for (i in 2:nrow(dur_df)-1) {
  if(difftime(dur_df$Field.Time[i+1], dur_df$Field.Time[i], units="secs") >= 30) {
    if (!is.na(dur_df$Status[i])) { #r does not like this due to NA's and logical(0)
      dur_df$bout_status[i] <- "DELETE"
    } else {
      jumpcounter <- jumpcounter+1
      dur_df$bout_status[i] <- "STOP"
      dur_df$bout_status[i+1] <- "START"
    }
  } else {
    
  }
}

for (i in 2:nrow(dur_df)-1) {
  if(difftime(dur_df$Field.Time[i+1], dur_df$Field.Time[i], units="secs") >= 30) {
    jumpcounter <- jumpcounter+1
    dur_df$bout_status[i] <- "STOP"
    dur_df$bout_status[i+1] <- "START"
  } else {
    
  }
}


for (i in 3:nrow(dur_df)-1) {
  if((difftime(dur_df$Field.Time[i+1], dur_df$Field.Time[i], units="secs") >= 30) && (difftime(dur_df$Field.Time[i], dur_df$Field.Time[i-1], units="secs") >= 30)) {
    dur_df$bout_status[i] <- "DELETE"
  } else {
    jumpcounter <- jumpcounter+1
    dur_df$bout_status[i] <- "STOP"
    dur_df$bout_status[i+1] <- "START"
  }
}




x <- grep('DELETE',dur_df$bout_status)


#Drop single frame events
dur_df <- subset(dur_df, dur_df$bout_status != "DELETE")

START_rows <- dur_df[dur_df$bout_status == 'START', ] 
STOP_rows <- dur_df[dur_df$bout_status == 'STOP', ]
dur_dur <- cbind(START_rows, STOP_rows)
names(dur_dur)[3] <- "RWT_START" 
names(dur_dur)[11] <- "RWT_STOP"
dur_dur$RWT_START <- ymd_hms(dur_dur$RWT_START)  
dur_dur$RWT_STOP <- ymd_hms(dur_dur$RWT_STOP)
dur_dur$duration <- dur_dur$RWT_STOP - dur_dur$RWT_START

unique(data[which(is.na(data$Field.Time)),2])









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




#8. VIDEO DATA PIPELINE: LIDDELL: SOCIAL NETWORK: Create Adjacency matrix 
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



# 9. VIDEO DATA PIPELINE: SOCIAL NETWORK: iGraph Plotting 

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


# 10. VIDEO DATA PIPELINE: LIDDELL GRAPHING 
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
nrow(point_df[point_df$Subject == "M1" & point_df$Modifier.1 == "Mating"])
nrow(point_df[point_df$Subject == "M2" & point_df$Modifier.1 == "Mating"])




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
 
 
 