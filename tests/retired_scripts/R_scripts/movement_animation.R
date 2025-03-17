## movement_animation.R
## Caleb C. Vogt, Cornell University

## TO DO LIST

# LOAD PACKAGES
library(tidyverse)
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


# SET WD AND LOAD DATA
wd <- setwd("C:/Users/Caleb Vogt/Desktop/Liddell_2020_RFID_full_10day_data")
metadata <- read_excel("Liddell.2020.xlsx", sheet = 1, skip = 1)
filenames <- list.files(wd, pattern = "*RFID_full_data*")
filelist <- lapply(filenames, read.csv)
names(filelist) <- c("T001", "T002", "T003", "T004", "T005", "T006", "T007")

# CHANGE FIELD.TIME COL TO POSIXct
for(i in 1:7){ #1:7 for all 7 trials represented
  filelist[[i]]$Field.Time <- as.POSIXct(filelist[[i]]$Field.Time, format="%Y-%m-%d %H:%M:%OS")
}



# RUN THROUGH ALL DATA FRAMES TO CREATE ANIMATIONS FOR EACH INDIVIDUAL MOUSE. 
for(a in length(filelist)){
  move_df <- filelist[[a]]
  
  # GET UNIQUE IDS OF INDIVIDUALS PRESENT IN THIS TRIAL. 
  ids <- unique(move_df$full_ids)
  
  # THE CODE FOR CREATING GRAPHS ARE WITHIN THE LOOP. CYCLES THROUGH IDS 
  for(i in ids[1:length(ids)]) {
    # SAVE CURRENT MOUSE AS VARIABLE
    current_mouse <- print(i)
    
    # ONLY KEEP OBSERVATIONS FROM CURRENT_MOUSE, REMOVE OTHER MICE
    move_df <- subset(move_df, full_ids == current_mouse) 
    
    # SAVE CURRENT SUBJECT AS VARIABLE
    current_mouse_info <- paste(unique(move_df$trial), unique(move_df$strain), unique(move_df$sex), unique(move_df$ID), sep = "_")
    
    # SAVE CURRENT TRIAL AS VARIABLE
    current_trial <- paste(move_df$trial)
    
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
    
    
    ## PLOT 3 + 3A: ANIMATION OF RFID POINTS OVER FIELD SCHEMATIC  FOR FULL TRIAL
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
    
    
  } 
}## END OF FOR LOOP


for(b in length(filelist)){
  data <- filelist[[b]]
  
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
  
}
