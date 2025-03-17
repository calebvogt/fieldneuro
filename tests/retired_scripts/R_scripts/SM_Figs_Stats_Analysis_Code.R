# Figs_Stats_Analysis_Code.R
## Created by Caleb C. Vogt, PhD Candidate @ Cornell University
# Load packages -----------------------------------------------------------
library(tidyverse)
library(readxl)
library(data.table)
library(lme4)
library(lmerTest)
library(emmeans)
library(asnipe)
library(igraph)
library(lubridate)
# library(gganimate)
# library(Hmisc)
# library(hrbrthemes)
# library(gifski)
# library(av)
# library(reshape)
# library(plot.matrix)
# library(transformr)
# library(rstatix)
# library(psych)
# library(sna)
# library(sjstats)
# library(MuMIn)
# library(broom)
# library(knitr)
# library(ggfortify)
# library(svglite)
# library(jtools)
# library(sjPlot)
# library(effects)
# library(ggeffects)
# library(ggrepel)
# library(rgl) # for 3D PCA plot. 
# library(ggpubr)
# library(rstatix)
# library(plotrix)
# library(devtools)
# library(export)


# Set directories and load metadata ---------------------------------------------------------
wd <- setwd("C:/Users/caleb/Box/7_LID_2020")
# wd <- setwd("C:/Users/Caleb Vogt/Box/7_LID_2020")
output_fp <- paste("C:/Users/caleb/Desktop")
# output_fp <- paste("C:/Users/Caleb Vogt/Desktop")
meta <- read_excel("LID_2020_metadata.xlsx", sheet = 1, skip = 1)
meta_short <- meta %>% 
  dplyr::select(trial, paddock, strain, sex, name, code, family_group)

# Load rfid_data and clean ----------------------------------------------------------
rfid_data <- as.data.frame(fread("LID_2020_ALLTRIAL_RFID_DATA.csv", stringsAsFactors = FALSE, fill = TRUE))

# Clean data to 10 days and triage relevant subjects as identified by descriptive analyses. 
rfid_data <- rfid_data %>%
  filter(noon_to_noon_day >=1 & noon_to_noon_day <= 10) %>% 
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

rfid_data$strain_sex <- paste0(rfid_data$strain, "-", rfid_data$sex)
rfid_data$field_time <- as.POSIXct(rfid_data$field_time, format="%Y-%m-%d %H:%M:%OS")

# Load move_data, clean, summarize ----------------------------------------------------------
move_data <- as.data.frame(fread("LID_2020_ALLTRIAL_MOVEBOUT.csv", stringsAsFactors = FALSE))
move_data <- subset(move_data, select = -c(V1))
move_data$field_time <- as.POSIXct(move_data$field_time, format="%Y-%m-%d %H:%M:%OS") # CONVERT TIME FORMAT
move_data$field_time_STOP <- as.POSIXct(move_data$field_time_STOP, format="%Y-%m-%d %H:%M:%OS")
move_data$strain_sex <- paste0(move_data$strain, "-", move_data$sex)

#Data cleaning
move_data <- move_data %>% 
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


## Mean estimated hours in the tubs per trial
df <- move_data %>% 
  mutate(duration_hours = duration_s / 60 / 60) %>%
  group_by(trial) %>% 
  tally(sum(duration_hours)) 
sum(df$n)
summary(df$n)
mean(df$n)
std.error(df$n)

## Mean estimated hours in the tub per mouse per night
df <- move_data %>% 
  mutate(duration_hours = duration_s / 60 / 60) %>%
  group_by(name, noon_to_noon_day) %>% 
  tally(sum(duration_hours)) 
sum(df$n)
summary(df$n)
mean(df$n)
std.error(df$n)

## Mean resource zones visited per mouse per night 
df <- move_data %>% # 
  group_by(name, noon_to_noon_day, antenna) %>% 
  tally() %>%  # get number of visits to each zone
  group_by(name, noon_to_noon_day) %>%
  tally() # get number of distinct zone visits
mean(df$n)
std.error(df$n)



# Load social_data and clean -----------------------------------------------------------
filenames <- list.files(wd, pattern = "*MOVEBOUT_GBI.csv")
social_data = lapply(filenames, fread) ## READ IN ALL FILES

# clean social data for triaged mice from social interaction bouts. 
# note that this cleaning step merely deletes columns and adds 0s where appropriate. Does not adjust GBI summary information (m_sum, f_sum, mf_sum)
aa = 4
for(aa in 1:length(social_data)){
  df <- social_data[[aa]] ## PULL OUT EACH TRIAL'S DATAFRAME
  
  df2 <- df %>% 
    #T004: George only mouse to cross between trials on Day 3. triage completely (drop column)
    dplyr::select(-one_of(c( "V1", "NYOB-M-George"))) %>% 
    #T003: Anubis visually confirmed dead by seizure on day 5.  
    mutate_at(vars(one_of(c("C57-M-Anubis"))), ~ ifelse(day >= 5, 0, .)) %>%
    #T003: Rae appears once on the first day, but she is captured at the end of the trial. Only female to do this, so excluded. 
    mutate_at(vars(one_of(c("C57-F-Rae"))), ~ ifelse(day >= 2, 0, .)) %>%
    #T004: Hare only appears day 1. Not recovered, presumed dead. 
    mutate_at(vars(one_of(c("NYOB-M-Hare"))), ~ ifelse(day >= 2, 0, .)) %>%
    #T004: Isis lost after day 2. Not recovered, presumed dead. #T004: Gilmour lost on day 10 only but recovered/trapped. Keep. 
    mutate_at(vars(one_of(c("NYOB-F-Isis"))), ~ ifelse(day >= 3, 0, .)) %>%
    #T003: Rose lost on Day 10, but trapped WITHOUT RFID tag. triage day 10 data. 
    mutate_at(vars(one_of(c("C57-F-Rose"))), ~ ifelse(day >= 10, 0, .))
    
  social_data[[aa]] <- df2
}

# Figure 1b: Zone visit heatmap with GraphPad  --------
# GP heat map scale should be set to the max visits for any given category. Suri = 184 visits. 
df <- move_data %>%
  filter(trial == "T001", sex == "M") %>%
  # filter(trial == "T001", sex == "F") %>%
  # filter(trial == "T002", sex == "M") %>%
  # filter(trial == "T002", sex == "F") %>%
  # filter(trial == "T003", sex == "M") %>%
  # filter(trial == "T003", sex == "F") %>%
  # filter(trial == "T004", sex == "M") %>%
  # filter(trial == "T004", sex == "F") %>%
  # filter(trial == "T005", sex == "M") %>%
  # filter(trial == "T005", sex == "F") %>%
  # filter(trial == "T006", sex == "M") %>%
  # filter(trial == "T006", sex == "F") %>%
  # filter(trial == "T007", sex == "M") %>%
  # filter(trial == "T007", sex == "F") %>%
  group_by(name, noon_to_noon_day, antenna) %>%
  tally() #number of visits

#check maximum visit number! adjust GP heatmap settings accordingly. 
# Suri = 123 visits

ids <- unique(df$name)#CREATE LOOP FOR ADDING NUMBER OF VISITS TO STATS FOR AN ENTIRE TRIAL. 
idlist <- list() # NOTE THAT DATA FRAMED WILL BE IN ORDER OF THIS LIST
daylist<- list()
flag=1
# aa = ids[1]
for(aa in ids[1:length(ids)]){ # LOOP THROUGH EACH INDIVIDUAL AND PULL OUT NUMBER OF VISITS PER UNIQUE zone AND PUT INTO 2X4 GRID THAT LOCALIZES TO THE 
  # CREATE STATS DATAFRAME TO MIMIC LAYOUT OF FIELD SITE
  stats <- data.frame(matrix(0, nrow = 4, ncol = 2)) # ENCLOSURE SETUP. PUT EACH NIGHT OF ACTIVITY TO THE RIGHT FOR 10 NIGHTS. COPY AND PASTE DIRECLY INTO PRISM. 
  df1 <- subset(df, name == aa)
  for(bb in 1:10){
    df2 <- subset(df1, noon_to_noon_day == bb)
    # CHECK IF THERE WERE ANY DETECTED VISITS THAT NIGHT. IF YES, JUST PUT A ZERO. IF NO, ADD VISIT NUMBERS BY zone
    if(nrow(df2) == 0){
      stats <- data.frame(matrix(0, nrow = 4, ncol = 2))
    } else {
      
      for(cc in 1:nrow(df2)){
        if(df2$antenna[cc] == 1){
          stats[4,1] <- print(df2$n[cc])
        } else if(df2$antenna[cc] == 2){
          stats[4,2] <-print(df2$n[cc])
        } else if(df2$antenna[cc] == 3){
          stats[3,1] <- print(df2$n[cc])
        } else if(df2$antenna[cc] == 4){
          stats[3,2] <- print(df2$n[cc])
        } else if(df2$antenna[cc] == 5){
          stats[2,1] <- print(df2$n[cc])
        } else if(df2$antenna[cc] == 6){
          stats[2,2] <- print(df2$n[cc])
        } else if(df2$antenna[cc] == 7){
          stats[1,1] <- print(df2$n[cc])
        } else if (df2$antenna[cc] == 8){
          stats[1,2] <- print(df2$n[cc])
        } else {print("barnacles")}
        
      }
    }
    daylist[[bb]] <- stats
    stats <- data.frame(matrix(0, nrow = 4, ncol = 2)) #CHANGE FROM NA 'S TO 0 'S
  }
  master_class <- do.call("cbind",daylist) #THIS THROWS THE ERROR
  idlist[[flag]] <- master_class
  flag=flag+1
}
master_SASS <- do.call("rbind",idlist) # RBIND DATA FOR EACH INDIVIDUAL
# View(master_SASS)
head(master_SASS)
write.table(master_SASS, "clipboard", sep="\t", row.names=FALSE) # COPY THE OUTPUT TO THE CLIPBOARD  
ids # PRINT ORDER OF THE DATA SET, COPY INTO GRAPHPAD



# Figure 2a & S2a: Minimum distance traveled -------------------------------------------------------------------
df <- move_data %>% 
  dplyr::select(trial, strain_sex,sex,strain,name, noon_to_noon_day, zone, zone_x, zone_y, antenna)

df$zone_x[df$zone_x =="A"] <- 3.75
df$zone_x[df$zone_x =="B"] <- 11.25
df$zone_y[df$zone_y =="A"] <- 7.6
df$zone_y[df$zone_y =="B"] <- 15.2
df$zone_y[df$zone_y =="C"] <- 22.8
df$zone_y[df$zone_y =="D"] <- 30.4
df$zone_x <- as.numeric(df$zone_x)
df$zone_y <- as.numeric(df$zone_y)
ids <- unique(df$name)
data_list <- list()
aa = ids[60]
for(aa in ids[1:length(ids)]){
  ## create df of percent time
  df1 <- df %>% 
    filter(name == aa) 
  # delete consecutive repeat antenna hits, only keep rows where antennas change. 
  df2 <- as.data.table(df1)[, .SD[1], by = rleid(df1$zone)]
  df2$dist <- NA
  n <- nrow(df2)
  if(n==1){
    df2$dist[1] <- 0
    data_list[[aa]] <- df2
  } else{
    df2$dist[2:n] <- sqrt((df2$zone_x[2:n] - df2$zone_x[1:n-1]) ^ 2 + (df2$zone_y[2:n] - df2$zone_y[1:n-1]) ^ 2)
    df2$dist[1] <- 0
    data_list[[aa]] <- df2
  }
}
df3 <- do.call("rbind", data_list)
summary(df3)
df3$noon_to_noon_day <- as.numeric(df3$noon_to_noon_day)
df4 <- df3 %>% 
  group_by(trial, strain_sex,strain, sex, name, noon_to_noon_day) %>% 
  tally(sum(dist)) %>% 
  complete(noon_to_noon_day = 1:10, fill = list(n = 0)) %>%  # fill in days where mouse doesnt appear with 0s.
  dplyr::rename(dist = n)

# data cleaning. 
df5 <- df4 %>% 
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

df6 <- df5 %>% 
  group_by(strain_sex, noon_to_noon_day) %>%
  summarise(mean_n = mean(dist), sd_n = sd(dist),count = n(), se_n = (sd_n/(sqrt(count))))

## Figure 2a
(p <- ggplot(df6, aes(x = noon_to_noon_day, y = mean_n, color = strain_sex)) +
    geom_line(size = 0.75) + 
    geom_point(size = 1.5) +
    geom_errorbar(aes(ymin = mean_n - se_n, ymax = mean_n + se_n), width = 0.2) +
    scale_x_continuous(limits = c(0.8,10.3), breaks = seq(1, 10, by = 1)) +
    scale_y_continuous(breaks = seq(0, 150, by = 20)) +
    scale_color_manual(breaks = c("C57-F", "C57-M", "NYOB-F", "NYOB-M"),
                       values=c("sienna1", "sienna", "skyblue", "skyblue4")) +
                       # values=c("red1", "red4", "steelblue", "steelblue4")) +
                       # values=c("goldenrod1", "goldenrod4", "slateblue", "slateblue4")) +
    theme_classic() +
    xlab("Day") +
    ylab("Min. dist. travelled (m)") +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=8),
          legend.position = "right" )
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.5, height=2.15, bg = "transparent")

# STATS  
df7 <- df5 %>% 
  group_by(trial, strain, sex, name, noon_to_noon_day) %>% 
  tally(sum(dist)) %>% 
  group_by(trial, strain, sex, name, noon_to_noon_day) %>% 
  dplyr::rename(sum_dist = n)

write.table(df7, "clipboard-16384", sep="\t", row.names=FALSE, col.names = TRUE)
df7 <- read_excel("Figure_Data.xlsx", sheet = "Fig2a")

df7$trial <- as.factor(df7$trial)
df7$strain <- as.factor(df7$strain)
df7$sex <- as.factor(df7$sex)
df7$name <- as.factor(df7$name)
df7$c57_F<-ifelse(df7$strain!="C57"|df7$sex!="F","other","pC57_F")

mod1 = lmer(sum_dist ~ strain*sex*log(noon_to_noon_day) + (1|trial) + (log(noon_to_noon_day)|name), data = df7) 
mod2 = lmer(sum_dist ~ strain*sex*log(noon_to_noon_day) + (log(noon_to_noon_day)|name), data = df7) 
mod3 = lmer(sum_dist ~ strain*sex*log(noon_to_noon_day) + (1|trial) + (1|name), data = df7) 
mod4 = lmer(sum_dist ~ strain+sex+log(noon_to_noon_day) + (1|trial) + (log(noon_to_noon_day)|name), data = df7) 
AIC(mod1,mod2, mod3)
qqnorm(resid(mod1))
anova(mod1)
summary(mod1)
emmeans(mod1, pairwise ~ strain*sex*log(noon_to_noon_day))
write.table(summary(mod1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)

# daily models: change for days 1 through 10 and report
d = lmer(sum_dist ~ strain*sex + (1|trial), data = subset(df7, noon_to_noon_day == 10)) 
emmeans(d, pairwise ~ strain*sex) #tukey adjusted
write.table(em.d, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)


# Figure S2a
df6a <- df5 %>% 
  group_by(trial, strain_sex, name) %>% 
  tally(sum(dist)) %>% 
  dplyr::rename(total_dist = n)

# write.table(df6a, "clipboard-16384", sep="\t", row.names=FALSE, col.names = TRUE)
df6a <- read_excel("Figure_Data.xlsx", sheet = "FigS2a")

(p <- ggplot(df6a, aes(x=strain_sex, y=total_dist, fill = strain_sex)) + 
    geom_violin(width=1, alpha = 1) +
    geom_boxplot(width=0.1, color = "black", alpha=0.5, size = 0.5) +
    scale_y_continuous(limits = c(0,2000)) +
    scale_fill_manual(breaks = c("C57-F", "C57-M", "NYOB-F", "NYOB-M"),
                      values=c("sienna1", "sienna", "skyblue", "skyblue4")) +
    xlab("") +
    ylab("Total estimated dist. travelled (m)") +
    theme_classic() +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"),
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"),
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.position = "none")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.75, height=2.15)

m1 <- lmer(total_dist ~ strain_sex + (1|trial), data = df6a)
summary(m1)
anova(m1)
emmeans(m1, pairwise ~ strain_sex, adjust = 'none') 
write.table(summary(m1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)

# Figure 2b: Unique zones visited -------------------------------------------------------------------
df <- move_data %>% # 
  group_by(trial, strain_sex, sex, strain, name, noon_to_noon_day, antenna) %>% 
  # filter(trial == "T007") %>% #Used for per trial graphs
  tally() %>%  # get number of visits to each zone
  group_by(trial, strain_sex, sex, strain, name, noon_to_noon_day, .drop = FALSE) %>%
  tally() %>% # get number of unique zone visits
  complete(noon_to_noon_day = 1:10, fill = list(n = 0)) %>% # fill in days where mouse doesnt appear with 0s
  dplyr::rename(unique_zones_visited = n)

# data cleaning as mice are filled in 
df <- df %>% 
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

# Output Fig2b data
write.table(df, "clipboard-16384", sep="\t", row.names=FALSE, col.names = TRUE)
df <- read_excel("Figure_Data.xlsx", sheet = "Fig2b")

mean(df$unique_zones_visited)
std.error(df$unique_zones_visited)

df1 <- df %>% 
  group_by(strain_sex, noon_to_noon_day) %>%
  summarise(mean_zone = mean(unique_zones_visited), sd_zone = sd(unique_zones_visited),count = n(), se_zone = (sd_zone/(sqrt(count))))

(p <- ggplot(df1, aes(x = noon_to_noon_day, y = mean_zone, color = strain_sex)) +
    geom_line(size = 0.75) + 
    geom_point(size = 1.5) +
    geom_errorbar(aes(ymin = mean_zone - se_zone, ymax = mean_zone + se_zone), width = 0.2) +
    scale_x_continuous(limits = c(0.8,10.3), breaks = seq(1, 10, by = 1)) + 
    scale_y_continuous(limits = c(1,8), breaks = seq(1, 8, by = 1)) +
    scale_color_manual(breaks = c("C57-F", "C57-M", "NYOB-F", "NYOB-M"),
                       values=c("sienna1", "sienna", "skyblue", "skyblue4")) +
    theme_classic() +
    xlab("Day") +
    ylab("Num. unique zones visited") +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=7),
          legend.background = element_rect(fill='transparent'),
          # legend.position = c(0.21,0.8))
          legend.position = "")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.5, height=2.15, bg = "transparent") 

# STATS
df$trial <- as.factor(df$trial)
df$sex <- as.factor(df$sex)
df$strain <- as.factor(df$strain)
df$noon_to_noon_day <- as.numeric(df$noon_to_noon_day)
df$c57F <- ifelse(df$strain=="C57"&df$sex=="F",1,0)

m1 = lmer(unique_zones_visited ~ strain*sex*log(noon_to_noon_day) + (1|trial) + (1+log(noon_to_noon_day)|name), data = df) 
# m2 = lmer(unique_zones_visited ~ strain*sex+log(noon_to_noon_day) + (1|trial) + (1+log(noon_to_noon_day)|name), data = df) 
# Removing C57s demonstrates they are solely responsible for effect. 
# m3 = lmer(unique_zones_visited ~ strain_sex*noon_to_noon_day + (1|name), data = subset(df, df$strain_sex != "C57-F"))
m4 = lmer(unique_zones_visited ~ c57F*log(noon_to_noon_day) + (1|trial) + (1|name), data = df) # keeps C57 females in model, but evaluates them separately. 
AIC(m1,m2,m3,m4)
qqnorm(resid(m1))
qqline(resid(m1))
summary(m1)
anova(m1)
emmeans(m1, pairwise ~ strain*sex*log(noon_to_noon_day)) #tukey
write.table(summary(m1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)
write.table(summary(m4)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)

ggpredict(mod1, terms = c("strain", "sex", "noon_to_noon_day"), type = "fe") %>% 
  plot()

# daily models: change for days 1 through 10. 
d = lmer(unique_zones_visited ~ strain*sex + (1|trial), data = subset(df, noon_to_noon_day == 10)) 
emmeans(d, pairwise ~ strain*sex) # tukey adjusted


# Figure 2c: Cumulative unique zones visited  --------
df <- move_data %>% 
  dplyr::select(trial, strain_sex, strain, sex, name, noon_to_noon_day, zone) %>%
  group_by(trial, strain_sex, strain, sex, name, zone) %>% 
  distinct(zone, .keep_all = TRUE) %>% # get distinct zones ever visited, keep associated day it was visited
  group_by(trial, strain_sex,strain, sex, name, noon_to_noon_day) %>% #regroup by day
  tally() %>% #tally unique zones visited per day
  mutate(csum_novel_zones = cumsum(n)) %>%  
  complete(name, noon_to_noon_day = full_seq(1:10, period = 1)) %>% #fill in missing day rows for each mouse
  dplyr::arrange(name, noon_to_noon_day) %>% 
  fill(csum_novel_zones) %>% ## fill cumulative sum data from last observed day
  dplyr::select(trial, strain_sex, strain, sex, name, noon_to_noon_day, csum_novel_zones)

#Data cleaning
df <- df %>% 
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


# output Fig2c data to clipboard
write.table(df, "clipboard-16384", sep="\t", row.names=FALSE, col.names = TRUE)
df <- read_excel("Figure_Data.xlsx", sheet = "Fig2c")

df1 <- df %>% 
  group_by(strain_sex, noon_to_noon_day) %>% 
  summarise(mean_n = mean(csum_novel_zones), sd_n = sd(csum_novel_zones),count = n(), se_n = (sd_n/(sqrt(count))))

# PLOT
(p <- ggplot(df1, aes(x=noon_to_noon_day, y=mean_n, color = strain_sex)) + 
    geom_line(size = 0.75) + 
    geom_point(size = 1.5) +
    geom_errorbar(aes(ymin = mean_n - se_n, ymax = mean_n + se_n), width = 0.2) +
    scale_x_continuous(limits = c(0.8,10.3), breaks = seq(1,11,by=1)) +
    scale_y_continuous(limits =c(1,8), breaks = seq(1,8,by=1)) +
    scale_color_manual(breaks = c("C57-F", "C57-M", "NYOB-F", "NYOB-M"),
                       values=c("sienna1", "sienna", "skyblue", "skyblue4")) +
    xlab("Day") +
    ylab("Csum. novel zones visited") +
    theme_classic() +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.position = "none") +
    guides(color=guide_legend(override.aes=list(fill=NA)))
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.5, height=2.15, bg = "transparent")

# STATS
summary(df)
df[sapply(df, is.character)] <- lapply(df[sapply(df, is.character)],as.factor)
df$csum_novel_zones <- as.numeric(df$csum_novel_zones)

#complex way of doing t-tests but allows you to control for the random effect of trial. 
mod2 = lmer(csum_novel_zones ~ sex*strain + (1|trial), data = subset(df, noon_to_noon_day == 10))
anova(mod2)
summary(mod2)
write.table(summary(mod2)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)
em.mod2 = emmeans(mod2, pairwise ~ sex*strain)
write.table(em.mod2$emmeans, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)
write.table(em.mod2$contrasts, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)


#logistic regression model of differences in all zones visited by strain*sex interaction
df2 <- df
df2$every_zone <- ifelse(df2$csum_novel_zones == 8, 1, 0)
df3 <- subset(df2, noon_to_noon_day == 10)
mod3 = glmer(every_zone ~ strain_sex + (1|trial), data = subset(df2, noon_to_noon_day == 10), family ="binomial") #family binomial because response variable is 0 or 1. 
mod3.1 = glmer(every_zone ~ (1|trial), data = subset(df2, noon_to_noon_day == 10), family ="binomial") #family binomial because response variable is 0 or 1. 
aov(mod3, mod3.1)
anova(mod3,mod3.1)
anova(mod3)
summary(mod3)
write.table(summary(mod3)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)


# Figure 2d: Proportion time in most occupied zone -----------------------
df <- move_data %>% 
  mutate(duration_min = duration_s / 60) %>% 
  group_by(name, antenna) %>% 
  tally(sum(duration_min)) %>% 
  mutate(percent_time = n / sum(n)) %>% 
  arrange(desc(percent_time)) %>% 
  group_by(name) %>% 
  slice_max(percent_time, n = 2) %>% 
  mutate(rank_order = rank(desc(percent_time))) %>% 
  complete(rank_order = 1:2, fill = list(n = 0, percent_time = 0)) %>% 
  mutate(rank_order = as.factor(rank_order))
df <- merge(df, meta, by  = "name") 

df1 <- df %>% 
  dplyr::select(trial, name, strain, sex, rank_order, antenna, n, percent_time) %>% 
  mutate(strain_sex = paste0(strain, "-", sex)) %>% 
  filter(rank_order == 1) %>% 
  dplyr::rename(duration_min = n)

# output stats df1
# write.table(df1, "clipboard-16384", sep="\t", row.names=FALSE, col.names = TRUE)
df <- read_excel("Figure_Data.xlsx", sheet = "Fig2d")

table(round(df1$percent_time, 6), df1$strain_sex)
mode(df1$percent_time)

(p <- ggplot(df1, aes(x=strain_sex, y=percent_time, fill = strain_sex)) + 
    geom_violin(width=1) +
    geom_boxplot(width=0.1, color = "black", alpha=0.5, size = 0.5) +
    scale_y_continuous(limits = c(0,1)) +
    scale_fill_manual(breaks = c("C57-F", "C57-M", "NYOB-F", "NYOB-M"),
                      values=c("sienna1", "sienna", "skyblue", "skyblue4")) +
    
    xlab("") +
    ylab("Prop. time in top occupied zone") +
    theme_classic() +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"),
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"),
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.position = "none")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.5, height=2.15)

# STATS
df1[sapply(df1, is.character)] <- lapply(df1[sapply(df1, is.character)],as.factor)
m1 = lmer(asin(sqrt(percent_time)) ~ strain*sex + (1|trial), data = df1) # random effects
AIC(m1)
anova(m1)
summary(m1)
write.table(summary(m1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)
emmeans(m1, pairwise ~ strain*sex, adjust = 'none')

summary(m1)
#C57-F untransformed estimate
sin(0.94079)^2
#C57-M untransformed estimate
sin(0.26718+0.94079)^2
#OB-F untransformed estimate
sin(0.25090 +0.94079)^2
#OB-M untransformed estimate
sin(0.34030+0.94079)^2

# Figure 2e-f & Figure S2c-d: Priority Access Scores -----------------------
df <- move_data %>% 
  # filter(sex  == "M") %>%
  filter(sex  == "F") %>%
  # filter(trial == "T001") %>%
  # filter(strain == "C57") %>%
  # filter(strain == "NYOB") %>%
  mutate(duration_min = duration_s / 60) %>% 
  group_by(trial, noon_to_noon_day, antenna) %>% 
  tally(sum(duration_min)) %>% 
  mutate(trial_day_antenna = paste0(trial, "_",noon_to_noon_day, "_", antenna)) %>% 
  dplyr::rename(total_duration_min = n) %>% #specify dplyr due to conflict
  ungroup() %>% 
  dplyr::select(trial_day_antenna, total_duration_min)

df1 <- move_data %>% 
  # filter(sex  == "M") %>%
  filter(sex  == "F") %>%
  # filter(trial == "T001") %>%
  # filter(strain == "C57") %>%
  # filter(strain == "NYOB") %>%
  mutate(duration_min = duration_s / 60) %>% 
  mutate(trial_day_antenna = paste0(trial, "_", noon_to_noon_day, "_", antenna)) %>% 
  group_by(name, trial_day_antenna, noon_to_noon_day, antenna) %>% 
  tally(sum(duration_min)) %>%
  dplyr::rename(mus_duration_min = n)#specify dplyr due to conflict

df2 <- merge(df1, df, by = "trial_day_antenna", all = TRUE) # bring in rest of males that did not win any days. 

df3 <- merge(df2, meta, by = "name", all = FALSE) # bring in metadata

df4 <- df3 %>%
  dplyr::select(trial, name, noon_to_noon_day, antenna, mus_duration_min, total_duration_min)

#summing daily adjusted scores and taking single number to avoid guessing of slice_max when scores vacillate in the negative range. 
df5 <- df4 %>% 
  mutate(mus_percent_capture = (mus_duration_min / total_duration_min)) %>% 
  group_by(trial, name, noon_to_noon_day) %>% 
  mutate(penalty = if(any(mus_percent_capture > 0.5)) 0 else -1) %>% # PENALTY #1: If on any day you dont capture greater than 50% for any zone, take off 1 point 
  group_by(trial, name) %>%
  complete(noon_to_noon_day = 1:10, fill = list(penalty = -1, mus_percent_capture = 0)) %>% # PENALTY #2: Not observed at all penalty. Doesnt effect anyone after filtering.  
  arrange(name, noon_to_noon_day) %>% 
  group_by(name, noon_to_noon_day) %>% 
  mutate(sum_daily_capture = sum(mus_percent_capture)) %>% # for each day sum percent capture pre-penalty application. 
  group_by(name,noon_to_noon_day) %>% 
  mutate(disc_col = paste0(name, "_", noon_to_noon_day, "_", sum_daily_capture)) %>% #Need to do this to drop repeated rows. 
  distinct(disc_col, .keep_all = TRUE) %>% # drop repeated daily sums on days with two zone rows. 
  mutate(sum_daily_capture_penalty = sum(sum_daily_capture+penalty)) %>% 
  group_by(name) %>% 
  mutate(csum_daily_capture_penalty = cumsum(sum_daily_capture_penalty))

## Cleaning to remove filled in entries for triaged mice. 
df6 <- df5 %>% 
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

df7 <- merge(df6, meta, by = "name", all = FALSE) # bring in metadata

df8 <- df7 %>%
  mutate(strain_sex = paste0(strain, "-", sex)) %>% 
  dplyr::select(trial.x, strain_sex, strain,sex, name, noon_to_noon_day, sum_daily_capture, penalty, sum_daily_capture_penalty, csum_daily_capture_penalty) %>% 
  dplyr::rename(trial = trial.x) %>% 
  mutate(label = if_else(noon_to_noon_day == max(noon_to_noon_day), as.character(name), NA_character_))

# output csv
write.table(df8, "clipboard-16384", sep="\t", row.names=F, col.names=F) #col.names = false for second time. 
# df8 <- read_excel("Figure_Data.xlsx", sheet = "Fig2e-f") #note that this is combined male and female data. you will need to filter by sex to produce the graphs below. 

# Figure S2 c and d. line plot
(p <- ggplot(df8, aes(x=noon_to_noon_day, y=csum_daily_capture_penalty, group = name, color = name)) + #y=csum_adj_mus_percent_capture_score
    geom_line(size =1, alpha = 0.5) +
    scale_x_continuous(breaks = seq(1,10,by=1), limits = c(1,10)) +
    scale_y_continuous(breaks = seq(-10,20, by = 5), limits = c(-10,20)) +
    xlab("Day") +
    ylab("Cumulative PA Score") +
    geom_hline(yintercept=0, linetype="dashed", color = "black", size = 1) +
    theme_classic() +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"),
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=8), 
          legend.position = "none") 
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.75, height=2.15, bg = "transparent") #main figs

# density plot of day 10 priority access scores. 
(p <- ggplot(subset(df8, noon_to_noon_day == 10), aes(x=csum_daily_capture_penalty, group = strain_sex, fill = strain_sex)) + 
    geom_density(adjust = 0.25,alpha = 0.8) +
    scale_x_continuous(breaks = seq(-10,20,by=5), limits = c(-10,21)) +
    scale_y_continuous(breaks = seq(0,0.2,by=0.05), limits = c(0, 0.18)) +
    scale_fill_manual(breaks = c("C57-F", "C57-M", "NYOB-F", "NYOB-M"),
                      values=c("sienna1", "sienna", "skyblue", "skyblue4")) +
    xlab("Day 10 Priority Access Score Distribution") +
    ylab("Density") +
    geom_vline(xintercept=0, linetype="dashed", color = "black", size = 0.75) +
    theme_classic() +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=7), 
          legend.key.size = unit(0.25, 'cm'),
          legend.position = "top")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.5, height=2.15, bg = "transparent") #main figs

library(diptest)
df9 <- df8 %>% 
  filter(noon_to_noon_day == 10, strain == "NYOB") 
dip.test(df9$csum_daily_capture_penalty)
dS <- (dip(df9$csum_daily_capture_penalty, full.result = TRUE))
plot(dS)

#Combined male and female plot. 
mf <- read_excel("Data_Stats_v5.xlsx", sheet = "Fig2E-F")

# density plot of day 10 priority access scores. 
(p <- ggplot(subset(mf, noon_to_noon_day == 10), aes(x=csum_daily_capture_penalty, group = sex, fill = sex)) + 
    geom_density(adjust = 0.25,alpha = 0.4) +
    scale_x_continuous(breaks = seq(-10,20,by=5), limits = c(-10,21)) +
    scale_y_continuous(breaks = seq(0,0.2,by=0.05), limits = c(0, 0.18)) +
    scale_fill_manual(breaks = c("F", "M"),
                      values=c("red", "blue")) +
    xlab("Day 10 Priority Access Score Distribution") +
    ylab("Density") +
    geom_vline(xintercept=0, linetype="dashed", color = "black", size = 0.75) +
    theme_classic() +
    theme(axis.text.x = element_text(color = "black", size = 8, face = "bold"),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8, face = "bold"),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=6), 
          legend.key.size = unit(0.25, 'cm'),
          legend.position = "top")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.75, height=2.15, bg = "transparent") #main figs

# Figure 3a: Time per interaction type ------------------------------------
# Data from combinedSurface.xlsx, transferred to Figure_Data.xlsx; plots made in GraphPad using proportion data. 

# Figure 3b-c & Fig. S3a-b: Prop. Social Grouping Times  -------------------------------------------------------------
# setwd("C:/Users/caleb/Box/7_LID_2020/1_MS_files/Fig3.jan18.2022")
# setwd("C:/Users/caleb vogt/Box/7_LID_2020/1_MS_files/Fig3.jan18.2022")
# time.Social.total <- read_csv("time.Social.total.csv") 
# time.Social.filtered <- time.Social.total[ which(time.Social.total$sumHrs>5), ] 
# time.Social.filtered$strain_sex <- paste0(time.Social.filtered$strain, "-", time.Social.filtered$sex)
# write.table(time.Social.filtered, "clipboard-16384", sep="\t", row.names=F, col.names=T) 
# wd <- setwd("C:/Users/caleb/Box/7_LID_2020")
# output_fp <- paste("C:/Users/caleb/Desktop")
time.Social.filtered <- read_excel("Figure_Data.xlsx", sheet = "Fig3b-c_S3a-b")

### Figure 3B. Proportion time spent alone
(p <- ggplot(data=time.Social.filtered, aes(x=strain_sex, y=asqrtAlone, fill=strain_sex)) +
   geom_violin(alpha = 1) +
    geom_boxplot(width=0.1, color = "black", alpha=0.5, size = 0.5) +
   geom_hline(yintercept=asin(sqrt(0.5))) +
   scale_fill_manual(breaks = c("C57-F", "C57-M", "NYOB-F", "NYOB-M"),
                     values=c("sienna1", "sienna", "skyblue", "skyblue4")) +
   ylab("Prop. time alone (arcsine)") +
   xlab("") +
   theme_classic() +
   theme(axis.text.x = element_text(color = "black", size = 8),
         axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
         axis.text.y = element_text(color = "black", size = 8),
         axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
         plot.background = element_rect(fill = "transparent", color = NA),
         panel.background = element_rect(fill = "transparent"), 
         legend.title = element_blank(),
         legend.text = element_text(size=7),
         legend.background = element_rect(fill='transparent'),
         legend.position = "")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.5, height=2.15, bg = "transparent") 

## STATS
m1 = lmer(asqrtAlone ~ strain*sex + (1|trial), data = time.Social.filtered)
AIC(m1)
anova(m1)
summary(m1)
write.table(summary(m1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)
emmeans(m1, pairwise ~ strain*sex)

## Total time spent alone across 10 days. 
## Get stats for the multitude of statements made by mike here. 


### Figure 3C. Prop. social time with opp. sex (arcsin)
(p <- ggplot(data=time.Social.filtered, aes(x=strain_sex, y=asqrtOppSocial, fill=strain_sex)) +
    geom_violin(alpha = 1) +
    geom_boxplot(width=0.1, color = "black", alpha=0.5, size = 0.5) +
    geom_hline(yintercept=asin(sqrt(0.5))) +
    scale_fill_manual(breaks = c("C57-F", "C57-M", "NYOB-F", "NYOB-M"),
                      values=c("sienna1", "sienna", "skyblue", "skyblue4")) +
    ylab("Prop. social time with opp. sex (arcsine)") +
    xlab("") +
    theme_classic() +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=7),
          legend.background = element_rect(fill='transparent'),
          legend.position = "")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.5, height=2.15, bg = "transparent") 
 
## Stats
# timeTotalmodel7=lm(asqrtOppSocial~strain*sex, data=time.Social.filtered)
# anova(timeTotalmodel7)
m1 = lmer(asqrtOppSocial ~ strain*sex + (1|trial), data = time.Social.filtered)
AIC(m1)
anova(m1)
summary(m1)
write.table(summary(m1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)
emmeans(m1, pairwise ~ strain*sex)


### Fig. S3a: Proportion of time spent with mixed sex
(p <- ggplot(data=time.Social.filtered, aes(x=strain_sex, y=asqrtFM, fill=strain_sex)) +
    geom_violin(alpha = 1) +
    geom_boxplot(width=0.1, color = "black", alpha=0.5, size = 0.5) +
    # geom_hline(yintercept=asin(sqrt(0.5))) +
    scale_fill_manual(breaks = c("C57-F", "C57-M", "NYOB-F", "NYOB-M"),
                      values=c("sienna1", "sienna", "skyblue", "skyblue4")) +
    ylab("Prop. time in mixed sex groups (arcsine)") +
    xlab("") +
    theme_classic() +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=6),
          legend.background = element_rect(fill='transparent'),
          legend.position = "")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.5, height=2.15, bg = "transparent") 

## STATS
m1 = lmer(asqrtFM ~ strain*sex + (1|trial), data = time.Social.filtered)
AIC(m1)
anova(m1)
summary(m1)
write.table(summary(m1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)
emmeans(m1, pairwise ~ strain*sex)


### Figure S3b. Proportion of time spent with asqrt Same Sex
(p <- ggplot(data=time.Social.filtered, aes(x=strain_sex, y=asqrtSS, fill=strain_sex)) +
    geom_violin(alpha = 1) +
    geom_boxplot(width=0.1, color = "black", alpha=0.5, size = 0.5) +
    # geom_hline(yintercept=asin(sqrt(0.5))) +
    scale_fill_manual(breaks = c("C57-F", "C57-M", "NYOB-F", "NYOB-M"),
                      values=c("sienna1", "sienna", "skyblue", "skyblue4")) +
    ylab("Prop. time in same sex groups (arcsine)") +
    xlab("") +
    theme_classic() +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=6),
          legend.background = element_rect(fill='transparent'),
          legend.position = "")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.5, height=2.15, bg = "transparent") 


## STATS
m1 = lmer(asqrtSS ~ strain*sex + (1|trial), data = time.Social.filtered)
AIC(m1)
anova(m1)
summary(m1)
write.table(summary(m1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)
emmeans(m1, pairwise ~ strain*sex)



# Figure 3d: Male Priority Access Score Ranking and time spent in mixed sex groups --------
# setwd("C:/Users/caleb/Box/7_LID_2020/1_MS_files/Fig3.jan18.2022")
# setwd("C:/Users/caleb vogt/Box/7_LID_2020/1_MS_files/Fig3.jan18.2022")
# terrSocial <- read_csv("terr.TimeSocial.csv")
# terrSocial <- terrSocial %>% 
  # mutate(strain_sex = paste0(strain, "-", sex))

# output to clipboard
# write.table(terrSocial, "clipboard-16384", sep="\t", row.names=FALSE, col.names = TRUE)
terrSocial <- read_excel("Figure_Data.xlsx", sheet = "Fig3d")

##Plot
(p <- ggplot(terrSocial, aes(x=rankTerr, y=asqrtMixedSex, color=strain_sex)) +
    # geom_point(aes(color=strain_sex)) +
    geom_point() +
    geom_smooth(method="lm") +
    scale_color_manual(breaks = c("C57-F", "C57-M", "NYOB-F", "NYOB-M"),
                      values=c("sienna1", "sienna", "skyblue", "skyblue4")) +
    scale_x_continuous(limits = c(0.8,10.3), breaks = seq(1, 10, by = 1)) +
    xlab("Male Priority Access Score, Ranked") +
    ylab("Prop. time with females (arcsine)") +
    # stat_cor(aes(color = strain_sex), label.x = 4, method = "pearson", p.accuracy = 0.001) +
    # stat_cor(label.x = 4, method = "pearson", p.accuracy = 0.001) +
    theme_classic() +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=7),
          legend.background = element_rect(fill='transparent'),
          legend.position = c(0.9,0.9))
          # legend.position = "left")
          # legend.position = "")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.5, height=2.15, bg = "transparent") 

terrSocial.mod1 = lmer(asqrtMixedSex ~ rankTerr*strain + (1|trial), data=terrSocial)
AIC(terrSocial.mod1)
anova(terrSocial.mod1)
summary(terrSocial.mod1)
write.table(summary(terrSocial.mod1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)


# Figure 3e and S3c-d: Proportion and duration of time spent in MM, FF, MF interactions ------------------------------------------
# setwd("C:/Users/caleb/Box/7_LID_2020/1_MS_files/Fig3.jan18.2022")
# ff=read_csv("liddell2020.ff.csv")
# fm=read_csv("liddell2020.fm.csv")
# mm=read_csv("liddell2020.mm.csv")
# mm$TrialTime <- mm$TrialTime+1
# ff$TrialTime <- ff$TrialTime+1
# fm$TrialTime <- fm$TrialTime+1

# Fig. 3e: MM scatter plots
# write.table(mm, "clipboard-16384", sep="\t", row.names=FALSE, col.names = TRUE)
mm <- read_excel("Figure_Data.xlsx", sheet = "Fig3e")

## graph
(p <- ggplot(data=mm, aes(y=(duration_s/60), x=TrialTime, color=Strain))+
  geom_point(size = 0.9) +
  ylab("Male-male social bout durations (min)") + 
  xlab("Day") +
  facet_wrap(~Strain) +
  scale_color_manual(breaks = c("C57", "NY"), values=c("sienna", "skyblue4")) +
  scale_x_continuous(limits = c(0.8,11), breaks = seq(0, 10, by = 1)) +
  ylim(0,40) +
  theme_classic() +
  theme(axis.text.x = element_text(color = "black", size = 8),
        axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
        axis.text.y = element_text(color = "black", size = 8),
        axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
        plot.background = element_rect(fill = "transparent", color = NA),
        panel.background = element_rect(fill = "transparent"), 
        legend.title = element_blank(),
        legend.text = element_text(size=7),
        legend.background = element_rect(fill='transparent'),
        legend.position = c(0.9,0.9))
        # legend.position = "")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=7, height=2.5, bg = "transparent") 

mm.mod1=lmer((duration_s/60)~Strain*TrialTime+(1|Trial), data=mm)
anova(mm.mod1)
summary(mm.mod1)
write.table(summary(mm.mod1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)

# Fig. S3c: FM, mixed sex interactions
fm <- read_excel("Figure_Data.xlsx", sheet = "FigS3c")

(p <- ggplot(data=fm, aes(y=(duration_s/60), x=TrialTime, color=Strain))+
    geom_point(size = 0.9) +
    ylab("Female-male social bout durations (min)") + 
    xlab("Day") +
    facet_wrap(~Strain) +
    scale_color_manual(breaks = c("C57", "NY"), values=c("goldenrod1", "goldenrod4")) +
    scale_x_continuous(limits = c(0.8,11), breaks = seq(0, 10, by = 1)) +
    ylim(0,40) +
    theme_classic() +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=7),
          legend.background = element_rect(fill='transparent'),
          legend.position = c(0.9,0.9))
  # legend.position = "")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=7, height=2.5, bg = "transparent") 

##stats
write.table(fm, "clipboard-16384", sep="\t", row.names=FALSE, col.names = TRUE)
fm.mod1=lmer((duration_s/60)~Strain*TrialTime+(1|Trial), data= fm)
anova(fm.mod1)
summary(fm.mod1)
write.table(summary(fm.mod1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)

# Fig. S3d: FF
# write.table(ff, "clipboard-16384", sep="\t", row.names=FALSE, col.names = TRUE)
ff <- read_excel("Figure_Data.xlsx", sheet = "FigS3d")

(p <- ggplot(data=ff, aes(y=(duration_s/60), x=TrialTime, color=Strain))+
    geom_point(size = 0.9) +
    ylab("Female-female social bout durations (min)") + 
    xlab("Day") +
    facet_wrap(~Strain) +
    scale_color_manual(breaks = c("C57", "NY"), values=c("sienna1", "skyblue")) +
    scale_x_continuous(limits = c(0.8,11), breaks = seq(0, 10, by = 1)) +
    ylim(0,40) +
    theme_classic() +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=7),
          legend.background = element_rect(fill='transparent'),
          legend.position = c(0.9,0.9))
  # legend.position = "")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=7, height=2.5, bg = "transparent") 

## Stats
ff.mod1=lmer((duration_s/60)~Strain*TrialTime+(1|Trial), data=ff)
anova(ff.mod1)
summary(ff.mod1)
write.table(summary(ff.mod1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)




# Figure 4a-b: C57 and Outbred social networks over time (visit overlap counts) -----------------------------------------
# Get priority access scores for node attribute. 
pas <- read_excel("Figure_Data.xlsx", sheet = "Fig2e-f")

# get trial social GBI data. 
df <- social_data[[7]] # goes in order of trials. 
print(unique(df$trial))
net_stats_list <- list()
vertex_stats_list <- list()
node_stats_list <- list()

# The important loop net graphs, stats, the whole shebang. 
i=1
for(i in 1:10) { # loop through days
  # create gbi
  gbi <- df %>%  # select mouse columns
    filter(day == i) %>% #comment out if you want the full 10 day network and run from here below
    dplyr::select(matches(c("*C57*", "*NYOB*")))   # choose your strain.
    # dplyr::select(matches("*-F-*")) # choose your sex, add %>%  above
  
  # remove strain-sex info from colnames of strain you are working with. 
  colnames(gbi) <- gsub("C57-M-","",colnames(gbi))
  colnames(gbi) <- gsub("C57-F-","",colnames(gbi))
  colnames(gbi)<-gsub("NYOB-M-","",colnames(gbi))
  colnames(gbi)<-gsub("NYOB-F-","",colnames(gbi))
  # colnames(gbi) <- gsub("C57-","",colnames(gbi))
  # colnames(gbi) <- gsub("C57-","",colnames(gbi))
  # colnames(gbi)<-gsub("NYOB-","",colnames(gbi))
  # colnames(gbi)<-gsub("NYOB-","",colnames(gbi))
  ids <- colnames(gbi)
  
  ## create network
  undir_matrix <- get_network(association_data = gbi,    # ASNIPE FUNCTION, # TURN GBI DATA INTO UNDIRECTED WEIGHTED ADJACENCY MATRIX FOR USE IN IGRAPH. 
                              data_format = "GBI",
                              association_index = "SRI") #CHOOSE SIMPLE RATIO INDEX OR SAMPLING PERIODS ARRAY
  g <- graph.adjacency(undir_matrix, mode="undirected", weighted = TRUE, diag = FALSE) # CREATE IGRAPH OBJECT FROM ADJACENCY MATRIX. GRAPHS SHOULD BE UNDIRECTED. 
  g <- simplify(g) # SIMPLIFY IGRAPH OBJECT. REMOVES MULTIPLE EDGES AND LOOP EDGES. 
  
  # set up vertex attributes
  V(g)$day <- i 
  V(g)$sex = as.character(meta_short$sex[match(V(g)$name,meta_short$name)])
  V(g)$label <- V(g)$name # Add names to graph graphs (full 10 day plot)
  V(g)$family_group = as.character(meta_short$family_group[match(V(g)$name,meta_short$name)])
  
  # get days priority access score
  pas1 <- pas %>% 
    filter(noon_to_noon_day == i) %>% 
    dplyr::select(name, csum_daily_capture_penalty) %>% 
    dplyr::rename(priority_access_score = csum_daily_capture_penalty)
  
  V(g)$node_priority_access_score <- as.character(pas1$priority_access_score[match(V(g)$name,pas1$name)])
  
  # set up vertex colors "sienna1", "sienna", "skyblue", "skyblue4")) +
  V(g)$color = V(g)$sex #assign the "Sex" attribute as the vertex color
  
  #C57 trials
  # V(g)$color = gsub("F","sienna1",V(g)$color) #Females will be red
  # V(g)$color = gsub("M","sienna",V(g)$color) #Males will be blue
  # # 
  #Outbred trials
  V(g)$color = gsub("F","skyblue",V(g)$color) #Females will be red
  V(g)$color = gsub("M","skyblue4",V(g)$color) #Males will be blue
  
  
  ## Vertex Measures
  V(g)$node_degree_centrality <- degree(g, mode="all") 
  V(g)$node_eigen_centrality <- eigen_centrality(g)$vector 
  V(g)$node_closeness_centrality <- closeness(g, mode = "all", normalized = TRUE, weights = NA) 
  V(g)$node_betweeness_centrality <- betweenness(g, directed = FALSE, weights = NA) 
  V(g)$node_edge_strength <- graph.strength(g) 
  V(g)$node_page_rank <- page_rank(g)$vector
  V(g)$node_authority_score <- authority_score(g)$vector
  
  # create vertex stats
  vertex_stats <- igraph::as_data_frame(g, "vertices")
  vertex_stats_list[[i]] <- vertex_stats
  
  ## Edge measures
  # sort(edge_betweenness(g, directed = FALSE, weights = NA)) 
  
  # Set up Network stats
  net_stats <- data.frame(matrix(ncol = 1,nrow = 1)) # create empty dataframe
  
  ## Network  Measures
  graph_centrality = centr_degree(g, mode = "all", loops = T, normalized = T) 
  net_stats$net_centrality <- graph_centrality$centralization 
  net_stats$net_eigen_centrality <- centr_eigen(g, directed =F, scale = T, normalized = T)$value 
  net_stats$net_mean_dist <- mean_distance(g, directed = FALSE) 
  net_stats$net_edge_density <- edge_density(g, loops = FALSE) 
  net_stats$net_transitivity <- transitivity(g) 
  net_stats$net_components_num <- components(g)$no 
  net_stats$net_modularity_infomap <- modularity(cluster_infomap(g)) 
  net_stats$net_modularity_infomap_group_n <- length(cluster_infomap(g))
  net_stats$net_modularity_fast_greedy <- modularity(cluster_fast_greedy(g))
  net_stats$net_modularity_fast_greedy_group_n <- length(cluster_fast_greedy(g))
  net_stats$net_assortativity_priority_access_score <- assortativity(g, V(g)$node_priority_access_score)
  net_stats$net_assortativity_family_group <- assortativity(g, as.factor(V(g)$family_group))

  #clean nets_stats
  net_stats[1] <- NULL
  net_stats_list[[i]] <- net_stats
  

  ##################  Fig 4a - b: Daily networks ############
  svg(file=paste0(output_fp,"/","P1_Day_", i, ".svg"), bg = "transparent") ## M+F plot
  par(mar = c(0.4, 0.1, 2, 0.1))
  plot(g,
       layout = layout.fruchterman.reingold,
       # layout = layout_nicely,
       # vertex.size = V(g)$node_degree_centrality*2,
       vertex.size = scales::rescale(V(g)$node_edge_strength, to = c(5,25)), #rescale vertex strength to reasonable min/max size
       # vertex.label= V(g)$name, # include labels
       vertex.label= NA, # Remove vertex labels
       # vertex.label.font = 2,
       # vertex.label.color = "black",
       # vertex.label.cex = 1,
       # vertex.label.degree = 2,
       edge.width = scales::rescale(E(g)$weight, to = c(0.5,30)), # rescale edge weight to reasonable min/max size
       edge.color = "darkgray",
       edge.curved = 0.2,

       ### COMMUNITY CLUSTERING.
       # Decide if you actually want this. Maybe for full 10 day plot only.
       # mark.groups = cluster_fast_greedy(g),
       # mark.border = "darkgray"
  )
  title(paste0("Day ", i), cex.main=3) #
  dev.off()
  
  
  ############## PLOT 2: Circular Network ##################
  # x <- get.adjacency(g)
  # plot(g)
  # graph.strength(g) #GET NODE STRENGTHS FOR THIS NETWORK. STRENGTH = SUM OF ALL EDGE WEIGHTS FOR A SINGLE NODE (BETWEEN 0-1)
  # V(g)$label <- V(g)$name
  # V(g)$degree <- degree(g)
  # 
  # svg(file=paste0(output_fp,"/","P2_Day_", i, ".svg", bg = "transparent"))
  # par(mar = c(0.4, 0.1, 2, 0.1))
  # plot.igraph(g,
  #             vertex.color = "lightblue", #change
  #             # vertex.color = "red",
  #             vertex.size = 50, #20
  #             # vertex.size = igraph::degree(g)*5, #SET NODE SIZE AS A FUNCTION OF DEGREE CENTRALITY MULTIPLIED BY A SCALAR
  #             vertex.label.color = "black",
  #             vertex.label.font = 4, #changes font type
  #             vertex.label.cex = 1.5, #0.75
  #             edge.width = E(g)$weight*100, #maintain original weights
  #             edge.color = 'black',
  #             edge.curved = 0.5,
  #             layout = layout_in_circle(g, order = ids) # SORT ALPHABETICALLY FOR REPEATED GRAPHS ACROSS DAYS
  # )
  # title(paste0("Day ", i), cex.main=3)
  # dev.off() 
  
}
# Get net Stats and copy the output directly 
net_stats <- do.call("rbind", net_stats_list)
# options(scipen = 999)
write.table(net_stats, "clipboard-16384", sep="\t", row.names=F, col.names = F) 

## Get vertex stats # COPY THE OUTPUT TO THE CLIPBOARD, paste directly into excel. 
vertex_stats <- do.call("rbind", vertex_stats_list)
vertex_stats <- merge(vertex_stats, meta_short, by.x = "name", by.y = "name")
vertex_stats <- vertex_stats %>% 
  dplyr::rename(sex = sex.x, family_group = family_group.x) %>% 
  dplyr::select(trial, strain, sex, name, code, family_group, day, 
                node_priority_access_score, node_degree_centrality, node_eigen_centrality, 
                node_closeness_centrality, node_betweeness_centrality, node_edge_strength, 
                node_page_rank, node_authority_score) %>% 
  filter(!(name == "George")) %>% 
  #T003: Anubis visually confirmed dead by seizure on day 5.  
  filter(!(name == "Anubis" & day >= 5)) %>% 
  #T003: Rae appears once on the first day, but she is captured at the end of the trial. Only female to do this, so excluded. 
  filter(!(name == "Rae" & day >= 2)) %>%  
  #T004: Hare only appears day 1. Not recovered, presumed dead. 
  filter(!(name == "Hare" & day >= 2)) %>% 
  #T004: Isis lost after day 2. Not recovered, presumed dead. #T004: Gilmour lost on day 10 only but recovered/trapped. Keep. 
  filter(!(name == "Isis" & day >= 3)) %>% 
  #T003: Rose lost on Day 10, but trapped WITHOUT RFID tag. triage day 10 data. 
  filter(!(name == "Rose" & day >= 10)) %>% 
  arrange(name, day)
write.table(vertex_stats, "clipboard-16384", sep="\t", row.names=F, col.names = F) 

# Figure 4c: Node degree centrality ---------------------------------------
df <- read_excel("Figure_Data.xlsx", sheet = "Fig4_node_data")
df$trial <- as.factor(df$trial)
df$sex <- as.factor(df$sex)
df$strain <- as.factor(df$strain)
df$day <- as.numeric(df$day)

df1 <- df %>% 
  mutate(strain_sex = paste0(strain, "-", sex)) %>% 
  group_by(strain_sex, day) %>%
  summarise(mean = mean(node_degree_centrality), 
            sd = sd(node_degree_centrality), 
            count = n(), 
            sem = (sd/(sqrt(count))))

(p <- ggplot(df1, aes(x=day, y=mean, color = strain_sex)) + 
    geom_line(size = 0.75) + 
    geom_point(size = 1.5) +
    geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem), width = 0.2) +
    scale_x_continuous(limits = c(0.8,10.3), breaks = seq(1, 10, by = 1)) +
    scale_y_continuous(breaks = seq(1, 20, by = 1)) +
    scale_color_manual(breaks = c("C57-F", "C57-M", "NYOB-F", "NYOB-M"),
                       values=c("sienna1", "sienna", "skyblue", "skyblue4")) +
    theme_classic() +
    xlab("Day") +
    ylab("Node degree centrality") +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=6),
          legend.background = element_rect(fill='transparent'),
          # legend.position = c(0.10,0.9))
          # legend.position = "left")
          legend.position = "")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.5, height=2.15, bg = "transparent") 

# STATS
mod1 = lmer(node_degree_centrality ~ strain*sex*log(day) + (1|trial) + (1+log(day)|name), data = df) 
AIC(mod1)
anova(mod1)
summary(mod1)
write.table(summary(mod1)$coef, "clipboard-16384", sep="\t", row.names=TRUE, col.names = TRUE)
ggpredict(mod1, terms = c("strain", "sex", "day"), type = "fe") %>% 
  plot()

# # daily models: change for days 1 through 10. 
# # i = 4
# for(i in 1:10){
#   d = lmer(node_degree_centrality ~ strain*sex + (1|trial), data = subset(df, day == i)) 
#   #get daily contrast estimates
#   contrasts <- emmeans(d, pairwise ~ strain*sex)
#   print(i)
#   print(contrasts$contrasts)
#   # anova(d)
#   # summary(d)
# }




# Figure 4d: Cumulative novel mice mice ------------
## CREATE LISTS OF NAMES FOR MATCHING COLUMNS
males <- meta_short %>% 
  filter(sex == "M") %>% 
  dplyr::select(name) %>% 
  filter(!is.na(name))
male_list <- dplyr::pull(males, name)

females <- meta_short %>% 
  filter(sex == "F", na.rm = TRUE) %>% 
  dplyr::select(name) %>% 
  filter(!is.na(name))
female_list <- dplyr::pull(females, name)

trial_stats <- list()
aa = 1
for(aa in 1:length(social_data)){
  df <- social_data[[aa]] ## PULL OUT EACH TRIAL'S DATAFRAME
  colnames(df)<-gsub("C57-M-","",colnames(df))
  colnames(df)<-gsub("C57-F-","",colnames(df))
  colnames(df)<-gsub("NYOB-M-","",colnames(df))
  colnames(df)<-gsub("NYOB-F-","",colnames(df))
  
  col_ids <- colnames(df[,10:ncol(df)]) ## get mouse column names starting at col 10
  bb = col_ids[1]
  all_mouse_list <- list()
  first_flag = 1
  for(bb in col_ids[1:length(col_ids)]) {
    df2 <- df %>% 
      filter((!!as.symbol(bb)) == 1) %>% # pull all rows where bb mouse is present. 
      mutate(name = bb) %>% 
      relocate(name)
    
    non_self_ids <- col_ids[!col_ids %in% bb] # remove current mouse from the next loop to compare to other animals
    non_self_ids[1]
    novel_mouse_rows <- list()
    second_flag = 1
    for(i in non_self_ids[1:length(non_self_ids)]) {
      df3 <- df2 %>% 
        filter((!!as.symbol(i)) == 1) %>% 
        mutate(novel_mouse_met = i)
      
      novel_mouse_rows[[second_flag]] <- df3[1,] #save first observed meeting of the focal and novel mouse to list
      second_flag = second_flag + 1
    }
    all_mouse_list[[first_flag]] <- do.call("rbind", novel_mouse_rows)
    first_flag <- first_flag + 1
  }
  df4 <- do.call("rbind", all_mouse_list)
  
  #remove na rows which are introduced when a mouse does not ever meet a particular other mouse. 
  df4[rowSums(is.na(df4)) > 0,]
  df5 <- df4[complete.cases(df4), ] 
  
  ## ADD RELEVANT METADATA INFORMATION. 
  df6 <- merge(df5, meta_short, by.x = "name", by.y = "name")
  df7 <- df6 %>%
    dplyr::rename(trial = trial.x) %>% 
    dplyr::select(trial, paddock, strain, sex, name, code, novel_mouse_met, day, zone, field_time_start, field_time_stop, m_sum, f_sum, mf_sum, duration_s)
  
  trial_stats[[aa]] <- df7
}
df8 <- do.call("rbind", trial_stats)
df9 <- df8[with(df8, order(name, day)),]

# df9 has a dataframe ordered by first meeting time with each mouse in the paddock .
df10 <- df9 %>% 
  mutate(strain_sex = paste0(strain, "-", sex)) %>% 
  group_by(trial, strain_sex,strain, sex,name, day) %>% 
  tally() %>% # 
  complete(name, day = full_seq(1:10, period = 1)) %>% #fill in missing day rows for each mouse, adds NAs. 
  replace(is.na(.), 0) %>% 
  mutate(csum = cumsum(n)) %>% #get cumulative # of novel mice met
  arrange(name, day) %>% 
  fill(csum) %>% ## fill cumulative sum data from last observed day
  dplyr::rename(novel_indivs_met = n, csum_novel_indivs_met = csum)

#data cleaning
df11 <- df10 %>% 
  filter(!(name == "George")) %>% 
  #T003: Anubis visually confirmed dead by seizure on day 5.  
  filter(!(name == "Anubis" & day >= 5)) %>% 
  #T003: Rae appears once on the first day, but she is captured at the end of the trial. Only female to do this, so excluded. 
  filter(!(name == "Rae" & day >= 2)) %>%  
  #T004: Hare only appears day 1. Not recovered, presumed dead. 
  filter(!(name == "Hare" & day >= 2)) %>% 
  #T004: Isis lost after day 2. Not recovered, presumed dead. #T004: Gilmour lost on day 10 only but recovered/trapped. Keep. 
  filter(!(name == "Isis" & day >= 3)) %>% 
  #T003: Rose lost on Day 10, but trapped WITHOUT RFID tag. triage day 10 data. 
  filter(!(name == "Rose" & day >= 10))  

# output csv
# write.table(df11, "clipboard-16384", sep="\t", row.names=F, col.names = T)
df11 <- read_excel("Figure_Data.xlsx", sheet = "Fig4d")


#plot
df12 <- df11 %>% 
  group_by(strain_sex, day) %>%
  summarise(mean = mean(csum_novel_indivs_met), 
            sd = sd(csum_novel_indivs_met), 
            count = n(), 
            sem = (sd/(sqrt(count))))


(p <- ggplot(df12, aes(x=day, y=mean, color = strain_sex)) + 
    geom_line(size = 0.75) + 
    geom_point(size = 1.5) +
    geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem), width = 0.2) +
    scale_x_continuous(limits = c(0.8,10.3), breaks = seq(1, 10, by = 1)) + 
    scale_y_continuous(limits = c(1,20), breaks = seq(2, 20, by = 2)) +
    scale_color_manual(breaks = c("C57-F", "C57-M", "NYOB-F", "NYOB-M"),
                       values=c("sienna1", "sienna", "skyblue", "skyblue4")) +
    theme_classic() +
    xlab("Day") +
    ylab("Csum. novel mice met") +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=6),
          legend.background = element_rect(fill='transparent'),
          # legend.position = c(0.21,0.8))
          legend.position = "none") 
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.5, height=2.15, bg = "transparent") 

# STATS
df <- df11
df$trial <- as.factor(df$trial)
df$strain_sex <- as.factor(df$strain_sex)
df$strain <- as.factor(df$strain)
df$sex <- as.factor(df$sex)
df$day <- as.numeric(df$day)
mod1 = lmer(csum_novel_indivs_met ~ strain*sex + (1|trial), data = subset(df, day == 10)) 
anova(mod1)
summary(mod1)
write.table(summary(mod1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)

# Figure 4e: Node page rank score  ---------------------------------------
df <- read_excel("Figure_Data.xlsx", sheet = "Fig4_node_data")

## Choose your node measure and make your graphs.
df1 <- df %>% 
  mutate(strain_sex = paste0(strain, "-", sex)) %>% 
  group_by(strain_sex, day) %>%
  summarise(mean = mean(node_page_rank), 
            sd = sd(node_page_rank), 
            count = n(), 
            sem = (sd/(sqrt(count))))

(p <- ggplot(df1, aes(x=day, y=mean, color = strain_sex)) + 
    geom_line(size = 0.75) + 
    geom_point(size = 1.5) +
    geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem), width = 0.2) +
    scale_x_continuous(limits = c(0.8,10.3), breaks = seq(1, 10, by = 1)) +
    # scale_y_continuous(limits = cbreaks = seq(1, 20, by = 1)) +
    scale_color_manual(breaks = c("C57-F", "C57-M", "NYOB-F", "NYOB-M"),
                       values=c("sienna1", "sienna", "skyblue", "skyblue4")) +
    theme_classic() +
    xlab("Day") +
    ylab("Node page rank score") +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=6),
          legend.background = element_rect(fill='transparent'),
          # legend.position = c(0.10,0.9))
          # legend.position = "left")
          legend.position = "")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.5, height=2.15, bg = "transparent") 

# STATS
df$trial <- as.factor(df$trial)
df$sex <- as.factor(df$sex)
df$strain <- as.factor(df$strain)
df$day <- as.numeric(df$day)
mod1 = lmer(node_page_rank ~ strain*sex*log(day) + (1|trial) + (1+log(day)|name), data = df) 
mod2 = lmer(node_page_rank ~ strain*sex*log(day) + (1|trial) + (1|name), data = df)
AIC(mod1, mod2)
anova(mod1)
summary(mod1)
write.table(summary(mod1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)

# daily models: change for days 1 through 10. 
# i = 4
for(i in 1:10){
  d = lmer(degree ~ strain*sex + (1|trial), data = subset(df, day == i)) 
  #get daily contrast estimates
  contrasts <- emmeans(d, pairwise ~ strain*sex)
  print(i)
  print(contrasts$contrasts)
  # anova(d)
  # summary(d)
}




# Figure 4f: MRQAP Daily Network Prediction of Day 10 Network -------------------
# get trial social GBI data. 
library(asnipe)
df <- social_data[[1]] # goes in order of trials. 
print(unique(df$trial))
net_stats_list <- list()
vertex_stats_list <- list()
node_stats_list <- list()

# BONUS: Do Day 1 networks predict Day 10 networks?
adj_list <- list()
aa = 1
for(aa in 1:10){
  gbi2 <- df %>%  # select mouse columns
    filter(day == aa) %>%
    dplyr::select(matches(c("*C57*","*NYOB*")))   # choose your strain.
  # convert gbi to ibg to adjacency matrix.
  ibg <- t(gbi2)
  adj_list[[aa]] <- ibg %*% t(ibg)
}

for(bb in 1:10) {
  print(bb)
  print(mrqap.dsp(adj_list[[10]]~adj_list[[bb]], directed = "undirected", test.statistic = "t-value", randomisations = 10))
}
# manually copy and paste outputs into excel. 

df <- read_excel("Figure_Data.xlsx", sheet = "Fig4f")


## Figure 4f
(p <- ggplot(df, aes(x=day, y=adjusted_r_squared, group = trial, color = strain, fill = strain, shape = significant)) + 
    geom_line(size = 0.7) + 
    geom_point(size = 3) +
    # geom_point(aes(color = strain), size = 4, shape = 21) +
    scale_color_manual(breaks = c("C57", "Outbred"), values=c("goldenrod1", "goldenrod4")) +
    scale_x_continuous(limits = c(0.8,10.3), breaks = seq(1, 10, by = 1)) +
    theme_classic() +
    xlab("Day") +
    # ylab("net_mean_dist") +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=8),
          legend.background = element_rect(fill='transparent'),
          # legend.position = c(0.2,0.85))
          legend.position = "right")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=5, height=2.5, bg = "transparent") 


# Table S1: Numbers and Descriptive Stats  -------------------------------------
# Mean RFID reads per trial
library(plotrix)
df <- rfid_data %>% 
  group_by(trial) %>%  
  tally()
mean(df$n)
std.error(df$n)

# Mean mouse RFID reads per day
df <- rfid_data %>% 
  # filter(trial == "T007") %>% #per trial measures. omit for all trials. 
  group_by(name, noon_to_noon_day) %>%  
  tally()
mean(df$n)
std.error(df$n)


# Figure S1d: Daily inter-zone travel times ---------------------------------------
df <- rfid_data
df$zone <- df$antenna
ids <- unique(df$name)
day_list <- list()
data_list <- list()
aa = ids[4]
for(aa in ids[1:length(ids)]){
  df1 <- df %>% 
    filter(name == aa) 
  #get daily inter-travel times. 
  for(bb in 1:10) {
    df2 <- df1 %>% 
      filter(noon_to_noon_day == bb)
    
    # delete consecutive repeat antenna hits, only keep rows where antennas change. 
    df3 <- as.data.table(df2)[, .SD[1], by = rleid(df2$antenna)]
    day_list[[bb]] <- df3 %>% 
      select(trial, strain, sex, name, zone, field_time) %>% 
      mutate(diff = field_time - lag(field_time), 
             diff_secs = as.numeric(diff, units = 'secs'), 
             diff_mins = as.numeric(diff, units = 'mins'), 
             diff_hours = as.numeric(diff, units = 'hours'))
  }
  data_list[[aa]] <- do.call("rbind", day_list)
}
df4 <- do.call("rbind", data_list)
df5 <- na.omit(df4)
min(df5$diff_secs)
mean(df5$diff_mins)
max(df5$diff_hours)

# Graph
# sdat <- summary(df5$diff_mins)
# summStr <- paste(names(sdat), format(sdat, digits = 0), collapse = "; ")
# op <- par(mar = c(7,4,4,2) + 0.1)
hist(df5$diff_mins, 
     xlim = c(0, 1000),
     breaks = 10000,
     # main = stuff,
     main = "",
     xlab = "Inter-zone travel time (min)"
)
# title(sub = summStr, line = 5.5)
# par(op)
# export as svg. 


# Figure S1e: Within-zone inter-RFID interval and time window capture thresholds --------

df <- rfid_data
df$zone <- df$antenna
ids <- unique(df$name)
big_data_list <- list()
data_list <- list()
flag <- 1
aa = ids[1]
for(aa in ids[1:length(ids)]){
  print(paste("Processing mouse ",flag, " out of ", length(ids), sep=''))
  ## create df of percent time
  df1 <- df %>% 
    filter(name == aa) 
  
  days <- unique(df1$noon_to_noon_day)
  day_list <- list()
  cc=days[1]
  for(cc in days[1:length(days)]){
    df2 <- df1 %>% 
      filter(noon_to_noon_day == cc)
    
    zones <- unique(df2$zone)
    zone_list <- list()
    bb = zones[1]
    for(bb in zones[1:length(zones)]){
      zone_list[[bb]] <- df2 %>% 
        filter(zone == bb) %>% 
        select(trial, strain, sex, name, noon_to_noon_day, zone, field_time) %>% 
        mutate(diff = field_time - lag(field_time), 
               diff_secs = as.numeric(diff, units = 'secs')) %>% 
        #remove 0s
        filter(diff_secs > 0)
    }
    #list of a mouses data per day per zone
    data_list [[aa]] <- do.call("rbind", zone_list)
  }
  #list of all mouse data per day per zone
  big_data_list[[cc]] <- do.call("rbind", data_list)
  flag <- flag + 1
  
}

# after loop finishes
df3 <- do.call("rbind", big_data_list)

#remove NAs
df3 <- df3[!is.na(df3$diff_secs),]
sdat <- summary(df3$diff_secs)
sdat
## get time interval where 95% of intervals below that value. 
sort(df3$diff_secs)[0.95*length(df3$diff_secs)]  # 10 days, all time, 95% = 11 seconds
sort(df3$diff_secs)[0.99*length(df3$diff_secs)]  # 10 days, all time, 99% = 153 seconds <<<< We select the most conservative estimate for all mice at all times

#base plot
options(scipen=999)

## Graph
# svg(file = paste0(output_fp, "/", "output.svg"))
summStr <- paste(names(sdat), format(sdat, digits = 2), collapse = "; ")
summStr
op <- par(mar = c(7,4,4,2) + 0.1)
hist(df3$diff_secs, 
     xlim = c(0, 200),
     # log = "y",
     breaks = 30000,
     main = "",
     xlab = "Within Tub inter-read interval (s)"
)
abline(v=c(11,153), col=c("red","blue"), lty=c(1,2), lwd=c(3,3))
title(sub = summStr, line = 5.5)
par(op)
dev.copy(svg, file = paste0(output_fp, "/", "output.svg"))
dev.off()

# Figure S1f: Number of zone visits and time in zone correlation -----------------------
# number of zone visits
df <- move_data %>% 
  mutate(duration_min = duration_s / 60)  %>% 
  group_by(trial, strain, sex, name, antenna) %>% 
  tally()

# duration of time spent in zone
df2 <- move_data %>% 
  mutate(duration_min = duration_s / 60)  %>% 
  group_by(trial, strain, sex, name, antenna) %>% 
  tally(sum(duration_min))

df3 <- as.data.frame(cbind(df, df2$n))
df3$strain_sex <- paste0(df3$strain,"-",df3$sex)
colnames(df3) <- c("trial", "strain", "sex", "name", "antenna", "num_visits", "total_time_min", "strain_sex")

library(ggpubr)
(p <- ggplot(data = df3, aes(x = num_visits, y = total_time_min, color = strain_sex)) +
    geom_point() +
    scale_color_brewer(palette = "PuOr") + 
    geom_smooth(method="lm") + 
    xlab("Number of zone visits") +
    ylab("Time spent in zone (min)") +
    scale_color_manual(breaks = c("C57-F", "C57-M", "NYOB-F", "NYOB-M"),
                       values=c("sienna1", "sienna", "skyblue", "skyblue4")) +
    stat_cor(aes(color = strain_sex), label.x = 4, method = "pearson", p.accuracy = 0.001) +
    theme(axis.line = element_line(colour = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(),
          panel.background = element_blank()) 
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=4.5, height=3)




# Figure S2b: Cumulative estimated time spent in resource zones --------
df <- move_data %>% 
  dplyr::select(trial, strain, sex, name, noon_to_noon_day,duration_s) %>%
  mutate(duration_min = duration_s / 60) %>% 
  group_by(trial,strain, sex, name, noon_to_noon_day) %>% 
  tally(sum(duration_min)) %>% 
  mutate(csum_zone_min = cumsum(n)) %>%  #get cumulative # of novel mice met
  complete(name, noon_to_noon_day = full_seq(1:10, period = 1)) %>% #fill in missing day rows for each mouse
  arrange(name, noon_to_noon_day) %>% 
  fill(csum_zone_min) %>% ## fill cumulative sum data from last observed day
  mutate(group = paste0(strain, "-", sex)) %>% 
  dplyr::select(trial,group, strain, sex, name, noon_to_noon_day, csum_zone_min)

# data cleaning. 
df <- df %>% 
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

# output stats df1
write.table(df, "clipboard-16384", sep="\t", row.names=FALSE, col.names = TRUE)

## Graph
df1 <- df %>% 
  group_by(group, noon_to_noon_day) %>% 
  summarise(mean_n = mean(csum_zone_min), sd_n = sd(csum_zone_min),count = n(), se_n = (sd_n/(sqrt(count))))

# PLOT
(p <- ggplot(df1, aes(x=noon_to_noon_day, y=mean_n, color = group)) + 
    geom_line(size = 0.75) + 
    geom_point(size = 1.5) +
    geom_errorbar(aes(ymin = mean_n - se_n, ymax = mean_n + se_n), width = 0.2) +
    scale_x_continuous(breaks = seq(1,11,by=1), limits = c(1,10.2)) +
    scale_color_manual(breaks = c("C57-F", "C57-M", "NYOB-F", "NYOB-M"),
                       values=c("sienna1", "sienna", "skyblue", "skyblue4")) +
    xlab("Day") +
    ylab("Cumulative time (min)") +
    theme_classic() +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          legend.title = element_blank(),
          legend.text = element_text(size=6),
          # legend.position = c(0.15,0.8))
          legend.position = "") +
    guides(color=guide_legend(override.aes=list(fill=NA)))
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.75, height=2.15) #supp figs. 

# STATS
summary(df)
df[sapply(df, is.character)] <- lapply(df[sapply(df, is.character)],as.factor)

mod1 = lmer(csum_zone_min ~ sex*strain + (1|trial), data = subset(df, noon_to_noon_day == 10))
anova(mod1)
summary(mod1)
write.table(summary(mod1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)
em.mod1 = emmeans(mod1, pairwise ~ sex*strain)
write.table(em.mod1$emmeans, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)
write.table(em.mod1$contrasts, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)




# Figure S4a-c: Network  measures and stats -------------------------------------------
df <- read_excel("Figure_Data.xlsx", sheet = "Fig4_net_data")
df$trial <- as.factor(df$trial)
df$strain <- as.factor(df$strain)
df$day <- as.numeric(df$day)

## Figure S4a: num. network components
df1 <- df %>% 
  group_by(strain, day) %>%
  summarise(mean = mean(net_components_num), 
            sd = sd(net_components_num), 
            count = n(), 
            sem = (sd/(sqrt(count))))

(p <- ggplot(df1, aes(x=day, y=mean, color = strain)) + 
    geom_line(size = 0.75) + 
    geom_point(size = 1.5) +
    geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem), width = 0.2) +
    scale_x_continuous(limits = c(0.8,10.3), breaks = seq(1, 10, by = 1)) +
    scale_color_manual(breaks = c("C57", "Outbred"), values=c("goldenrod1", "goldenrod4")) +
    theme_classic() +
    xlab("Day") +
    ylab("Num. network components") +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=6),
          legend.background = element_rect(fill='transparent'),
          # legend.position = c(0.2,0.85))
          legend.position = "")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.5, height=2.15, bg = "transparent") 

## STATS 
mod1 = lmer(net_components_num ~ strain*log(day) + (1|trial), data = df) 
anova(mod1)
summary(mod1)
write.table(summary(mod1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)


## Figure S4b: network edge density
df1 <- df %>% 
  group_by(strain, day) %>%
  summarise(mean = mean(net_edge_density), 
            sd = sd(net_edge_density), 
            count = n(), 
            sem = (sd/(sqrt(count))))

(p <- ggplot(df1, aes(x=day, y=mean, color = strain)) + 
    geom_line(size = 0.75) + 
    geom_point(size = 1.5) +
    geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem), width = 0.2) +
    scale_x_continuous(limits = c(0.8,10.3), breaks = seq(1, 10, by = 1)) +
    scale_color_manual(breaks = c("C57", "Outbred"), values=c("goldenrod1", "goldenrod4")) +
    theme_classic() +
    xlab("Day") +
    ylab("Network edge density") +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=6),
          legend.background = element_rect(fill='transparent'),
          # legend.position = c(0.2,0.85))
          legend.position = "")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.5, height=2.15, bg = "transparent") 

## STATS 
mod1 = lmer(net_edge_density ~ strain*log(day) + (1|trial), data = df) 
anova(mod1)
summary(mod1)
write.table(summary(mod1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)



## Figure S4c: network edge density
df1 <- df %>% 
  group_by(strain, day) %>%
  summarise(mean = mean(net_eigen_centrality), 
            sd = sd(net_eigen_centrality), 
            count = n(), 
            sem = (sd/(sqrt(count))))

(p <- ggplot(df1, aes(x=day, y=mean, color = strain)) + 
    geom_line(size = 0.75) + 
    geom_point(size = 1.5) +
    geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem), width = 0.2) +
    scale_x_continuous(limits = c(0.8,10.3), breaks = seq(1, 10, by = 1)) +
    scale_color_manual(breaks = c("C57", "Outbred"), values=c("goldenrod1", "goldenrod4")) +
    theme_classic() +
    xlab("Day") +
    ylab("Network eigenvector centrality") +
    theme(axis.text.x = element_text(color = "black", size = 8),
          axis.title.x = element_text(color = "black", size = 8, face = "bold"), 
          axis.text.y = element_text(color = "black", size = 8),
          axis.title.y = element_text(color = "black", size = 8, face = "bold"), 
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent"), 
          legend.title = element_blank(),
          legend.text = element_text(size=6),
          legend.background = element_rect(fill='transparent'),
          # legend.position = c(0.2,0.85))
          legend.position = "")
)
ggsave(p, file = "Rplot.svg", device = "svg", output_fp, width=2.5, height=2.15, bg = "transparent") 

## STATS 
mod1 = lmer(net_eigen_centrality ~ strain*log(day) + (1|trial), data = df) 
anova(mod1)
summary(mod1)
write.table(summary(mod1)$coef, "clipboard", sep="\t", row.names=TRUE, col.names = TRUE)

 
# # daily models: change for days 1 through 10. 
# i = 4
# for(i in 1:10){
#   d = lm(net_components_num ~ strain, data = subset(df, day == i)) 
#   #get daily contrast estimates
#   contrasts <- emmeans(d, pairwise ~ strain)
#   print(i)
#   print(contrasts$contrasts)
#   # anova(d)
#   # summary(d)
# }

