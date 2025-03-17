# 5_create_ALLTRIAL_MOVEBOUT_GBI+SUMMARY.R
## Created by Caleb C. Vogt, Cornell University

# load packages
library(tidyverse)
library(readxl)
library(plyr)
library(data.table)
library(lubridate)

# SET WD, LOAD DATA, FORMAT TIME SERIES
# wd <- setwd("C:/Users/caleb/Box/0_CV_Shared_Analyses/7_LID_2020")
wd <- setwd("C:/Users/caleb vogt/Box/0_CV_Shared_Analyses/7_LID_2020")
# output_fp <- paste("C:/Users/caleb/Desktop")
output_fp <- paste("C:/Users/caleb vogt/Desktop")
metadata <- read_excel("LID_2020_metadata.xlsx", sheet = 1, skip = 1)

data <- as.data.frame(fread("LID_2020_ALLTRIAL_MOVEBOUT.csv", stringsAsFactors = FALSE))

# Triage specific mice prior to making MOVEBOUT_GBI files. GBI_Summary can be triaged after (which mike did. )
data <- data %>% 
  #T004: George only mouse to cross between trials on Day 3. triage. 
  filter(!(name == "George")) %>% 
  #T003: Anubis visually confirmed dead by seizure on day 5.  
  filter(!(name == "Anubis" & noon_to_noon_day >= 5)) %>% 
  #T003: Rae appears once on the first day, but she is captured at the end of the trial. Only female to do this, so excluded. 
  filter(!(name == "Rae" & noon_to_noon_day >= 2)) %>%  
  #T004: Hare only appears day 1. Not recovered, presumed dead. 
  filter(!(name == "Hare" & noon_to_noon_day >= 2)) %>% 
  #T004: Isis lost after day 2. Not recovered, presumed dead. #T004: Gilmour lost on day 10 only but recovered/trapped. Keep. 
  filter(!(name == "Isis" & noon_to_noon_day >= 3)) %>% 
  #T003: Rose lost on Day 10, but trapped WITHOUT RFID tag. triage day 10 data. 
  filter(!(name == "Rose" & noon_to_noon_day >= 10))  

data$field_time <- as.POSIXct(data$field_time, format="%Y-%m-%d %H:%M:%OS")
data$field_time_STOP <- as.POSIXct(data$field_time_STOP, format="%Y-%m-%d %H:%M:%OS")
data$field_time <- with_tz(data$field_time, "UTC")
data$field_time_STOP <- with_tz(data$field_time_STOP, "UTC")
data$zone <- paste(data$paddock, data$antenna, sep = "_")
trials <- unique(data$trial)


# Create MOVEBOUT_GBI for each trial --------------------------------------

 
trial_list <- list()
aa = trials[4]
for(aa in trials[1:length(trials)]){
  df <- data %>%
    filter(trial == aa) %>% 
    mutate(mouse = paste(strain,sex,name, sep = "-")) %>% 
    select(trial, antenna, zone, strain, sex, full_ID, name, mouse, code, noon_to_noon_day, field_time, field_time_STOP, duration_s)
  
  mice <- unique(df$mouse)  
  zone_list <- list()
  zones <- unique(df$zone)
  bb = zones[1]
  for(bb in zones[1:length(zones)]){
    df1 <- df %>% 
      filter(zone == bb)
    df1 <- df1[order(df1$field_time, na.last=FALSE), ]
    df1$field_time <- as.POSIXct(df1$field_time, format="%Y-%m-%d %H:%M:%OS")
    df1$field_time_STOP <- as.POSIXct(df1$field_time_STOP, format="%Y-%m-%d %H:%M:%OS")
    START <- as.data.frame(df1$field_time)
    colnames(START) <- "x"
    STOP <- as.data.frame(df1$field_time_STOP)
    colnames(STOP) <- "x"
    start_stop <- as.data.frame(rbind(START, STOP)) # create long row of starts and stops and order them
    colnames(start_stop) <- "x"
    start_stop <- as.data.frame(start_stop[order(start_stop$x, na.last=FALSE),])
    colnames(start_stop) <- "x"
    
    list <- list()
    cc=1
    for(cc in 1:(nrow(start_stop)-1)) {
      field_time_start <- as.character(start_stop$x[cc], format="%Y-%m-%d %H:%M:%OS")
      field_time_stop <- as.character(start_stop$x[cc+1], format="%Y-%m-%d %H:%M:%OS")
      list[[cc]] <- cbind(field_time_start, field_time_stop)
    }
    gbi <- as.data.frame(do.call("rbind", list)) # Create group by individual matrix
    gbi$field_time_start <- as.POSIXct(gbi$field_time_start, format="%Y-%m-%d %H:%M:%OS") #convert to posixct
    gbi$field_time_stop <- as.POSIXct(gbi$field_time_stop, format="%Y-%m-%d %H:%M:%OS")
    gbi$field_time_start <- force_tz(gbi$field_time_start, tzone = "UTC") #force convert to UTC without changing time
    gbi$field_time_stop <- force_tz(gbi$field_time_stop, tzone = "UTC")
    
    gbi$duration_s <- gbi$field_time_stop - gbi$field_time_start #create duration
    gbi_cols <- c("trial", "day", "zone")
    gbi[gbi_cols] <- NA
    gbi$center_time <- gbi$field_time_start + (gbi$duration_s/2) # create center_time which will be compared later to determine participation in grouping event
    gbi[mice] <- NA     ## add columns for all trial mice
    
    dd = 1
    for(dd in 1:nrow(gbi)) {
      center <- as.POSIXct(gbi$center_time[dd]) # find the center point of the visitation event. 
      print(paste("Processing row ",dd," out of ",nrow(gbi), " for zone ",bb," in trial ",aa, sep=''))
      ff =1
      for(ff in 1:nrow(df1)){
        int <- interval(df1$field_time[ff], df1$field_time_STOP[ff])
        int
        if(center %within% int) {
          cool_mouse <- df1$mouse[ff]
          # add a 1 to the mouse column present in the interaction
          gbi[[cool_mouse]][[dd]] <- 1 ## critical step. 
          gbi$day[dd] <- df1$noon_to_noon_day[ff]
        }
      }
    }
    
    gbi2 <- gbi[!is.na(gbi$day),] # now, remove grouping bouts with no detected animals. 
    gbi2$trial <- paste(aa) # add other details
    gbi2$zone <- paste(bb)
    gbi2[is.na(gbi2)] <- 0 #replace na's with 0s
    zone_list[[bb]] <- subset(gbi2, select = -c(center_time)) #drop center time
  }
  trial_list[[aa]] <- do.call("rbind", zone_list) 
}

## Wait for the end of the first loop! 

meta_short <- metadata %>% 
  mutate(mouse = paste(strain, sex, name, sep = "-")) %>% 
  select(trial, paddock, strain, sex, name, code, mouse,family_group)

## CREATE LISTS OF NAMES FOR MATCHING COLUMNS
males <- meta_short %>% 
  filter(sex == "M") %>% 
  select(mouse) %>% 
  filter(!is.na(mouse))
male_list <- dplyr::pull(males, mouse)

## female names list
females <- meta_short %>% 
  filter(sex == "F", na.rm = TRUE) %>% 
  select(mouse) %>% 
  filter(!is.na(mouse))
female_list <- dplyr::pull(females, mouse)

trial_stats <- list()
aa = 1
for(aa in 1:length(trial_list)){
  ## PULL OUT EACH trial'S DATAFRAME
  df <- trial_list[[aa]]
  current_trial <- unique(df$trial)
  df2 <- df %>% 
    mutate(m_sum = rowSums(select(., contains(male_list)))) %>% 
    mutate(f_sum = rowSums(select(., contains(female_list)))) %>% 
    mutate(mf_sum = rowSums(select(., contains(c(male_list,female_list))))) %>% 
    relocate(m_sum,f_sum,mf_sum)
  
  df3 <- df2 %>% 
    relocate(trial, day, zone, field_time_start, field_time_stop, duration_s, m_sum, f_sum, mf_sum)
  
  write.csv(df3, paste0(output_fp, "/", current_trial, "_MOVEBOUT_GBI.csv"))
}


# CREATE ALLtrial_MOVEBOUT_GBI_SUMMARY  -----------------------------------

filenames <- list.files(wd, pattern = "*MOVEBOUT_GBI.csv")
myfiles = lapply(filenames, fread) ## READ IN ALL FILES
trial_stats <- list()
aa = 1
for(aa in 1:length(trial_list)){
  df <- trial_list[[aa]] ## PULL OUT EACH trial'S DATAFRAME
  df2 <- df %>% 
    mutate(m_sum = rowSums(select(., contains(male_list)))) %>% 
    mutate(f_sum = rowSums(select(., contains(female_list)))) %>% 
    mutate(mf_sum = rowSums(select(., contains(c(male_list,female_list))))) %>% 
    relocate(m_sum,f_sum,mf_sum)
  
  ## get mouse column names starting at col 11 (Check this and confirm)
  col_ids <- colnames(df2[,10:ncol(df2)]) ## CHANGE
  col_ids
  stats <- list()
  bb = col_ids[1]
  for(bb in col_ids[1:length(col_ids)]) {
    df3 <- df2 %>% 
      # select(Day, zone, Start, End, Duration, field_time_START,field_time_STOP, (bb)) %>% 
      filter((!!as.symbol(bb)) == 1) %>% 
      mutate(Name = bb) %>% 
      select(day, field_time_start, field_time_stop, zone, duration_s, name,  m_sum, f_sum, mf_sum)
    
    df4 <- merge(df3, meta_short, by.x = "name", by.y = "mouse") ## ADD RELEVANT METADATA INFORMATION. 
    df5 <- df4 %>% 
      relocate(trial, paddock, Day, zone, 
               strain, sex, name, code, family_group, Name, 
               field_time_Start, field_time_Stop, duration_s, m_sum, f_sum, mf_sum)
    stats[[bb]] <- df5
  }
  trial_stats[[aa]] <- do.call("rbind", stats)
}
master_stats <- do.call("rbind", trial_stats)
write.csv(master_stats, paste0(output_fp, "/", "LID_2020_ALLtrial_MOVEBOUT_GBI_SUMMARY.csv"))
