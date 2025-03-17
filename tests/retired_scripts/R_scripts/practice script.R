
# Rusty Data Manipulation Practice ----------------------------------------

install.packages("XLConnect")

library(XLConnect)

setwd("C:/Users/Caleb/Desktop/Rusty Data Practice")

master <- read.csv("8x8_master.csv", na.strings=c("NA", "", " "))  #take items into parenthes and turn them into true NAs

master <- master[c(1:140), ] #subsets to get rid of unnecessary NA rows at bottom

install.packages("sm")
library(sm)


strains <- unique(master$strain)
strains



