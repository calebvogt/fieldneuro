
# master_summary analysis -------------------------------------------------


master.summary <- read.csv("master_summary.csv")


#Create new columns in master.summary

master.summary$newcolumn <- NA


## summary of boop information

## m_contact_zone

library(dplyr)
data(master.summary)

master <- tbl_df(master.summary)
master
data.frame(head(master))
master[master$sex == 1 & master$trial==1, ]

filter(master, sex==1, trial==1)
