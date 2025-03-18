
# Caleb Vogt, 8x8 Trial Master_Summary.csv code --------------------------------------------------------------


### Perform after running through Tracking.ROI.R code

getwd()
master.summary <- read.csv("master_summary.csv")
master.summary
View(master.summary)


colnames(master.summary)[10] <- paste(colnames(master.summary)[10], "dist_meters", sep='')
master.summary

colnames(Location)[2:3]<-paste(colnames(Location)[2:3],".A",Caleb,sep='')


