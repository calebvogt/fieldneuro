## Liddell2020Script.R
## Caleb C. Vogt, PhD Student Cornell University

## Updated on 7.21.2020
## TO DO LIST
# 1. Generate social networks from rfid data. some code in code_graveyard, but maybe should start from scratch 
# 2. Add Dark/Light transitions to Plot 3a 
# 3. Add weather animations to plot 3a. 
# 4. https://goodekat.github.io/presentations/2019-isugg-gganimate-spooky/slides.html#1
# 5. https://jevansbio.wordpress.com/cv/

#lil edit here. 

library(tidyverse)
library(googlesheets)
library(readxl)
library(dplyr)
library(plyr)
library(ggplot2)
library(gganimate)
library(data.table)
library(readr)
library(reshape)
library(lubridate)
library(scales)
library(plot.matrix)
library(asnipe)
library(igraph)
library(transformr)


# IMPORT METADATA ---------------------------------------------------------
# 
#DOWNLOAD METADATA AS EXCEL FILE INTO FOLDER AGAIN FROM GOOGLE DRIVE IF THE METADATA NEEDS UPDATING
# COMMENT THIS OUT IF YOU WANT TO SOURCE THE FILE AND RUN THE CODE OVER MULTIPLE SUBFOLDERS. 

setwd("C:/Users/caleb/Box/Liddell.2020.RFID")
metadata <- read_excel("Liddell.2020.xlsx", sheet = 1, skip = 1)


# # CHOOSE TRIAL, IMPORT RFID DATA, AND CLEAN -----------------------------------------------------

folders <- list.files("C:/Users/caleb/Box/Liddell.2020.RFID", recursive=FALSE, full.names = TRUE)
folders 
# REMOVE ALL EXTRA FOLDERS, ONLY HAVE FOLDERS FOR T001-T007
folders <- folders[-c(8:9)]
folders


# TO RUN FOR SPECIFIC TRIALS, JUST CHANGE THE FOLDER SEARCH IN THE FOR LOOP.
# Limit to one trial at a time. Then can hit Control + A and run all code. 
for (i in folders[7]) {
  setwd(i)
  
  # IMPORT ALL RFID DATA SAVED AS TXT FILES
  txt <- list.files(pattern = "*.txt")
  
  # IMPORT ALL TEXT FILES. BLIND TO TEXT FILE NAME. 
  dfall <-   do.call(rbind, lapply(txt, function(x) read_table(file = x, col_names = TRUE, col_types = NULL,
                                                               locale = default_locale(), na = "NA", skip = 4, n_max = Inf,
                                                               progress = show_progress(), comment = "")))
  
  # SELECT COLUMNS
  dfall <- dfall[ , c("Scan Date", "Scan Time", "Reader ID", "Antenna ID", "DEC Tag ID")] 
  
  # RENAME COLUMNS
  colnames(dfall) <- c("scan.date", "scan.time", "reader.id", "antenna.id", "dec.tag.id")
  
  # DROP ROWS WITH NO DEC.TAG.ID (BASICALLY ALL TRIALS WHERE ANIMALS HAVENT BEEN INJECTED YET)
  dfall <- dfall[!is.na(dfall$dec.tag.id), ]
  
  # DROP TRAILING 0'S FROM TAG DEC IDS. METADATA DOESNT SAVE THE TRAILING ZERO ON IMPORT
  dfall$dec.tag.id <- sub("0*$", "", dfall$dec.tag.id)
  
  #REMOVE DUPLICATED ROWS FROM TXT WITH REPEATED OBSERVATIONS ACROSS TIME.  
  dfall <- unique(dfall)
  
  # MERGE METADATA AND DFNEW BY DEC.TAG.ID. THIS ONLY KEEPS MATCHES. 
  dfnew <- merge(dfall, metadata, by.x ="dec.tag.id", by.y = "dec.tag.id")
  
  dfnew1 <- merge(dfall, metadata, by.x = "dec.tag.id", by.y = "dec.tag.id_2")
  
  dfall <- rbind.fill(dfnew, dfnew1)
  
  # CHOOSE COLUMNS TO KEEP
  dfall <- dfall[ , c("trial", "reader.id", "paddock", "antenna.id", "scan.date", "scan.time", "strain", "sex", "ID", "field.age", "dec.tag.id")] 
  
  # CREATE FIELD.TIME COLUMN. NOTE THAT MILISECONDS ARE PRESENT, BUT NOT PRINTED. 
  dfall$Field.Time <- as.POSIXct(paste(dfall$scan.date, dfall$scan.time), format="%m/%d/%Y %H:%M:%OS")
  
  # SORT DATAFRAME BY DATE AND TIME AND CREATE DATA
  data <- dfall[order(dfall$Field.Time), ]
  
  # CHANGE ANTENNA IDS/ZONES TO NUMERICS
  data$antenna.id <- as.numeric(data$antenna.id)
  data$ID <- as.numeric(data$ID)
  
  # CREATE FULL_IDS 
  data$full_ids <- paste0(data$strain,"-", data$sex,"-", data$ID)
  
  # CREATE X AND Y COORDINATES FOR THE ZONES. 
  data$x <- ifelse(grepl("1", data$antenna.id), "A", 
                   ifelse(grepl("2", data$antenna.id), "B",
                          ifelse(grepl("3", data$antenna.id), "A",
                                 ifelse(grepl("4", data$antenna.id), "B",
                                        ifelse(grepl("5", data$antenna.id), "A",
                                               ifelse(grepl("6", data$antenna.id), "B",
                                                      ifelse(grepl("7", data$antenna.id), "A",
                                                             ifelse(grepl("8", data$antenna.id), "B",
                                                                    "none"))))))))
  
  
  data$y <- ifelse(grepl("1", data$antenna.id), "A", 
                   ifelse(grepl("2", data$antenna.id), "A",
                          ifelse(grepl("3", data$antenna.id), "B",
                                 ifelse(grepl("4", data$antenna.id), "B",
                                        ifelse(grepl("5", data$antenna.id), "C",
                                               ifelse(grepl("6", data$antenna.id), "C",
                                                      ifelse(grepl("7", data$antenna.id), "D",
                                                             ifelse(grepl("8", data$antenna.id), "D",
                                                                    "none"))))))))
  
  # REMOVE OBSERVATIONS FROM TRIALS WHICH ARE NOT POSSIBLE GIVEN PAIRED TRIAL STRUCTURE  
  ifelse(grepl(pattern = "T001", i), 
         data <- subset(data, trial %in% c('T001')),
         ifelse(grepl(pattern = "T002", i), 
                data <- subset(data, trial %in% c('T002','T003')),
                ifelse(grepl(pattern = "T003", i), 
                       data <- subset(data, trial %in% c('T002','T003')),
                       ifelse(grepl(pattern = "T004", i), 
                              data <- subset(data, trial %in% c('T004','T005')),
                              ifelse(grepl(pattern = "T005", i), 
                                     data <- subset(data, trial %in% c('T004','T005')),
                                     ifelse(grepl(pattern = "T006", i), 
                                            data <- subset(data, trial %in% c('T006','T007')),
                                            ifelse(grepl(pattern = "T007", i), 
                                                   data <- subset(data, trial %in% c('T006','T007')),NA)))))))
  
  write.csv(data, file = "RFID_full_data.csv")

}


# INDIVIDUAL MOVEMENT SCATTERPLOTS --------------------------------------------------------

# GET UNIQUE IDS OF INDIVIDUALS PRESENT IN THIS TRIAL. 
ids <- unique(data$full_ids)

# THE CODE FOR CREATING GRAPHS ARE WITHIN THE LOOP. CYCLES THROUGH IDS 

for(i in ids[1:length(ids)]) {
  # CREATE MOVE_DF
  move_df <- data
  
  # SAVE CURRENT MOUSE AS VARIABLE
  current_mouse <- print(i)
  
  # ONLY KEEP OBSERVATIONS FROM CURRENT_MOUSE, REMOVE OTHER MICE
  move_df <- subset(move_df, full_ids == current_mouse) 
  
  # SAVE CURRENT SUBJECT AS VARIABLE
  current_mouse_info <- paste(unique(move_df$trial), unique(move_df$strain), unique(move_df$sex), unique(move_df$ID), sep = "_")
  
  # SAVE CURRENT TRIAL AS VARIABLE
  current_trial <- paste(move_df$trial)
  
  ### wANT TO START GRAPHING? PUTTING THE GRAPHING CODE IN THE LOOP ALLOWS YOU TO GET ALL SUBJECT GRAPHS AT ONCE. 
  
  ##############
  ## PLOT #1: STATIC INDIVIDUAL MOVEMENT ACROSS TUBS FOR ENTIRE TIME PERIOD.
  p1 <- ggplot(data=move_df, aes(Field.Time, antenna.id)) +
    geom_point(na.rm=TRUE, color="red", size=1) +
    ggtitle(paste0(current_mouse_info, " Zone Movement")) +
    xlab("Date") + ylab("Zone") +
    scale_x_datetime(breaks = "1 day", labels=date_format("%m-%d")) +
    #scale_y_discrete(limits=c(1","2","3","4","5","6","7","8"), drop = FALSE) +m
    scale_y_continuous(breaks = seq(1,8,1), limits=c(1,8)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          axis.line = element_line(color = "black", size = 1, linetype = "solid"))

  plot(p1)
  ggsave(filename=paste0(current_mouse_info, "_PLOT_1.png"), plot=p1, width = 5, height = 4, dpi = 300, units = "in", device='png')

  #############
  ## PLOT #2 + 2A: LINE GRAPH INDIVIDUAL MOVEMENT ACROSS TUBS FOR FULL TRIAL
  p2 <- ggplot(move_df, aes(Field.Time, antenna.id, color = factor(full_ids))) +
    geom_line(na.rm=TRUE, color="red", size=1) +
    ggtitle(paste0(current_mouse_info, " Zone Movement")) +
    scale_color_viridis_d() +
    labs(x = "Date", y = "Zone") +
    scale_y_continuous(breaks = seq(1,8,1), limits=c(1,8)) +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1),
          axis.line = element_line(color = "black", size = 1, linetype = "solid"))

  # RENDER THE ANIMATION
  p2a <- p2 +
            geom_point(aes(group=seq_along(Field.Time))) +
            transition_reveal(Field.Time)

  # SAVE THE ANIMATION
  anim_save(filename=paste0(current_mouse_info, "_PLOT_2A.gif"),animation = p2a)

#   #############
# ## PLOT 3 + 3A: ANIMATION OF RFID POINTS OVER FIELD SCHEMATIC  FOR FULL TRIAL
  p3 <- ggplot(move_df, aes(x, y)) +
    ggtitle(paste0(current_mouse_info, " Zone Movement")) +
    geom_point(show.legend = FALSE, alpha = 0.7, size = 2) +
    xlim("A","B") +
    ylim("A","B","C","D") +
    geom_jitter(width = 0.1, height = 0.1) +
    scale_color_viridis_d() +
    labs(x = "none", y = "none") +
    theme(plot.background = element_blank(),
          panel.grid.major = element_line(colour="black", size = 0.5),
          panel.grid.minor = element_line(colour = "black"),
          panel.border = element_rect(colour = "black", fill = NA, size = 3),
          panel.background = element_blank(),
          axis.line = element_blank(),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.ticks = element_blank(),
          axis.text = element_blank(),
          legend.title = element_blank(),
          legend.text = element_text(size = 15),
          plot.title = element_text(size = 20))

  # RENDER THE ANIMATION
  p3a <- p3 +
    # geom_line(aes(group=seq_along(Field.Time))) + #remove?
    transition_time(Field.Time) +
    labs(subtitle = "Time: {frame_time}")

  # SAVE THE ANIMATION
  anim_save(filename=paste0(current_mouse_info, "_PLOT_3A.gif"),animation = p3a,
            renderer = ffmpeg_renderer(format = ".gif"),
            fps = 24, #24 hours
            duration = 30) # 1 second per day

####
  
  
} ## END OF FOR LOOP



# FACET WRAP PLOTS --------------------------------------------------------
#########
# PLOT 4: ALL MALE ANIMATION
male_df <- subset(data, data$sex == "M")

p4 <- ggplot(male_df, aes(x, y, colour = full_ids)) +
  ggtitle("Male Movement") +
  geom_point(show.legend = TRUE, alpha = 0.7, size = 2) +
  xlim("A","B") +
  ylim("A","B","C","D") +
  geom_jitter(width = 0.1, height = 0.1) +
  scale_color_viridis_d() +
  labs(x = "none", y = "none") +
  facet_wrap(~full_ids) +
  theme(plot.background = element_blank(),
        panel.grid.major = element_line(colour="black", size = 0.5),
        panel.grid.minor = element_line(colour = "black"),
        panel.border = element_rect(colour = "black", fill = NA, size = 3),
        panel.background = element_blank(),
        axis.line = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.ticks = element_blank(),
        axis.text = element_blank(),
        legend.title = element_blank(),
        legend.text = element_text(size = 15),
        plot.title = element_text(size = 20))

# RENDER THE ANIMATION
p4a <- p4 +
  transition_time(Field.Time) +
  labs(subtitle = "Time: {frame_time}")

# SAVE THE ANIMATION AS GIF
anim_save(filename="P4A_MALE.gif", animation = p4a,
          renderer = ffmpeg_renderer(format = ".gif"),
          fps = 24, #24 hours
          duration = 30) # 1 second per day 

# SAVE THE ANIMATION AS MP4
anim_save(filename="P4A_MALE.mp4", animation = p4a,
          renderer = ffmpeg_renderer(format = ".mp4"),
          fps = 24, #24 hours
          duration = 30) # 1 second per day 


###############
# PLOT 5: ALL FEMALE ANIMATION
female_df <- subset(data, data$sex == "F")

p5 <- ggplot(female_df, aes(x, y, colour = full_ids)) +
  ggtitle("Female Movement") +
  geom_point(show.legend = TRUE, alpha = 0.7, size = 2) +
  xlim("A","B") +
  ylim("A","B","C","D") +
  geom_jitter(width = 0.1, height = 0.1) +
  scale_color_viridis_d() +
  labs(x = "none", y = "none") +
  facet_wrap(~full_ids) +
  theme(plot.background = element_blank(),
        panel.grid.major = element_line(colour="black", size = 0.5),
        panel.grid.minor = element_line(colour = "black"),
        panel.border = element_rect(colour = "black", fill = NA, size = 3),
        panel.background = element_blank(),
        axis.line = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.ticks = element_blank(),
        axis.text = element_blank(),
        legend.title = element_blank(),
        legend.text = element_text(size = 15),
        plot.title = element_text(size = 20))

# RENDER THE ANIMATION
p5a <- p5 +
  transition_time(Field.Time) +
  labs(subtitle = "Time: {frame_time}")

# SAVE THE ANIMATION AS GIF
anim_save(filename="P5A_FEMALE.gif", animation = p5a,
          renderer = ffmpeg_renderer(format = ".gif"),
          fps = 24, #24 hours
          duration = 30) # 1 second per day 

# SAVE THE ANIMATION AS MP4
anim_save(filename="P5A_FEMALE.mp4", animation = p5a,
          renderer = ffmpeg_renderer(format = ".mp4"),
          fps = 24, #24 hours
          duration = 30) # 1 second per day 


