
# In Progress. Liddell 2018: Machine Learning & Image Classification -------------------
#Yolo image classification framework

#install.packages("devtools")
#devtools::install_github("bnosac/image", subdir = "image.darknet", build_vignettes = TRUE)
library(image.darknet)
setwd("C:/Users/Caleb Vogt/Desktop/YOLO")

yolo_tiny_voc <- image_darknet_model(type = 'detect', #detect or classify. Classify appears to abort the R session
                                     model = "tiny-yolo-voc.cfg",
                                     weights = system.file(package="image.darknet", "models", "tiny-yolo-voc.weights"),
                                     labels = system.file(package = "image.darknet", "include", "darknet", "data", "voc.names"))

x <- image_darknet_detect("C:/Users/Caleb Vogt/Desktop/YOLO/mouse.png", #full file path needed
                          object = yolo_tiny_voc)
#threshold = 0.19) #Hilarious






#1. Extract jpgs from all videos in folder and place them in new subdirectories
#2. Manually extract empty frames and 




# EXTRA CODE  --------------------------------------------------------------
# EXTRA CODE #1: .smi Timestamp Extraction + BORIS Integration -----------

library(data.table)
library(dplyr)
library(readr)
library(reshape)

wd <- setwd("C:/Users/Caleb Vogt/Desktop/test")
list.files(wd)

filelist = list.files(pattern = "*.smi")
myfiles = lapply(filelist, read.delim)

# Extract timestamp information from all .smi files. 
timestamp <- vector()
for (i in 1:length(myfiles)) {
  y <- subset(myfiles[i][[1]], startsWith(as.character(X.SAMI.), "2018"))
  timestamp <- append(timestamp, y)
}

full_timestamp <- unlist(timestamp)
write.csv(full_timestamp, "timestamp.csv")

# Merge timestamp data frame and BORIS observation dataframe to create master analysis file 
DF1 <- read_csv("timestamp.csv")
DF2 <- read_csv("T001_Bravo_Zone1_CAM2.csv", skip = 18)  #Check how many rows need to be skipped. 
mydata <- DF2[,c("Subject", "Time", "X9")]
View(mydata)

md <- melt(mydata, id=(c("Subject", "X9")))
View(md)

new <- cast(md, Subject+value~X9)
View(new)

new_clean <- new[,c("Subject","START","STOP")]
View(new_clean)

temp_stop <- filter(new_clean, STOP != "NA")
View(temp_stop)

temp_start <- filter(new_clean, START != "NA")
View(temp_start)

temp <- cbind(temp_start,temp_stop)
View(temp)


temp_clean <- temp[,c(1:2,6)]
View(temp_clean)

zone <- rep(1, nrow(temp_clean))            #adds a zone column: DEFINE EACH ITERATION
View(zone)

temp_zone <- cbind(zone, temp_clean)
View(temp_zone)

View(DF2)
denominator <- as.numeric(unlist(DF2[1,3])) #grabs total length of the video
View(denominator)

temp_zone$START_convert <- (temp_zone$START)/denominator*(length(full_timestamp)) #create new column. 
View(temp_zone)

temp_zone$STOP_convert <- (temp_zone$STOP)/denominator*(length(full_timestamp))
View(temp_zone)

temp_zone$START_round <- round(temp_zone$START_convert)
View(temp_zone)

temp_zone$STOP_round <- round(temp_zone$STOP_convert)
View(temp_zone)

sapply(full_timestamp, class) #This tells you that all the data is stored as a factor 

new_timestamp <- as.character(full_timestamp) #convert data to a character
View(new_timestamp)

START_real <- rep(1, nrow(temp_zone))
View(START_real)


STOP_real <- rep(1, nrow(temp_zone))

as.factor(START_real)
as.factor(STOP_real)

temp_real <- cbind(temp_zone, START_real, STOP_real)
View(temp_real)


for (i in 1:nrow(temp_real)) {
  temp_real$START_real[i] <- new_timestamp[temp_zone$START_round[i]]
}


for (i in 1:nrow(temp_real)) {
  temp_real$STOP_real[i] <- new_timestamp[temp_zone$STOP_round[i]]
}


temp_real$START_real <- strptime(temp_real$START_real, "%Y-%m-%d %H:%M:%S") #changes time format into appropriate class. 
temp_real$STOP_real <- strptime(temp_real$STOP_real, "%Y-%m-%d %H:%M:%S")

duration <- temp_real$STOP_real - temp_real$START_real
View(duration)

temp_duration <- cbind(temp_real, duration)
View(temp_duration)


duration <- temp_duration$STOP_real - temp_duration$START_real


write.csv(temp_duration, "Zone5_temp_duration.csv") # DEFINE EACH ITERATION


#this code doesnt quite match the appropriate timing of the videos... 




# Once all temp_duration csvs have been created 
# 

wd <- setwd("G:/My Drive/Alex Liddell Data/Week 2 Analysis/temp_duration_Zones")
list.files(wd)
dir()

# setwd("G:/My Drive/Alex Liddell Data/temp_duration_Zones")
data <- lapply(dir(),read.csv)

full_zone_DF <- rbind(data[[2]], data[[3]], data[[4]], data[[5]], data[[6]], data[[7]], data[[8]], data[[9]])
View(full_zone_DF)
write.csv(full_zone_DF, "full_zone_df.csv")
# 


# # Graphing full_zone_DF

# library(ggplot2)

#simple as shit graphs

# Fig. 1. 

y <- table(full_zone_DF$zone)
plot(y)


# # Fig. 2.  
# 
# x <- table(full_zone_DF$Subject)
# plot(x)
# 
# 
# # Fig. 3.  
# 
# 
# 
# 
# 
# time <- as.character(full_zone_DF$START_real)
# time1 <- strsplit(time, " ")
# 
# View(unlist(time1))
# 
# 
# 
# ggplot(data=full_zone_DF, aes(x = Zone, y = )) +
#   geom_bar(stat = )
# 



# match start_round stop_round to full_timestamp 

# new_timestamp <- as.character(full_timestamp)
# 
# 
# start_round <- temp_zone[,7]
# stop_round <- temp_zone[,8]
# 
# 
# 
# start_stamp <- character()
# for (i in 1:length(start_round)) {
#   y <- new_timestamp[i]
#   start_stamp <- append(start_stamp, y)
#}

# 
# write.csv(DF2, "boris_output.csv")
# 
# 
#  
# 
# 
# masterDF <- data.frame(zone = numeric(), 
#                        id = character(), 
#                        start_video = numeric(), 
#                        stop_video = numeric(), 
#                        start_real = numeric(), 
#                        stop_real = numeric(), 
#                        duration = numeric(),
#                        stringsAsFactors = FALSE)
# 
# 
# # build masterDF
# 
# newrow <- data.frame(zone=1,
#                      id=2)
# 
# masterDF1 <- rbind(masterDF, newrow)


# subject <- DF2$Subject
# masterDF$id <- subject
# 
# masterDF[nrow(masterDF$id)+1, ] <- c(1,2,3,4,5,6,7) 
#   
# 
# masterDF$id <- filter(DF2[,5], Subject == "F1", X9 == "START")
# 
# 
# df4 <- merge(df1, df3, all.x = TRUE) # also doesnt work

# merged_data <- merge(data1, data2)# results in nothing dawg.


# subset(My.Data, startsWith(as.character(x), "G45"))
# subset(My.Data, grepl("^G45", My.Data$x))
# 
# my_value <-character()
# 
# subset(filelist[1], grep('2018'))
# 
# 
# for (i in 1:length(myfiles)) {
#   timestamp <- subset([i], grep("2018"))
# }
# 
# 
# 
# #
# 
# timestamp <- subset(myfiles[1][[1]], startsWith(as.character(X.SAMI.), "2018"))
# 
# 
# 
# timestamp <- subset(myfiles[1][[1]], startsWith(as.character(X.SAMI.), "2018"))

# for (i in 1:length(myfiles)) {
#   y <- subset(myfiles[i][[1]], startsWith(as.character(X.SAMI.), "2018"))
#   full_timestamp <- append(timestamp, y)
#   
# }







# EXTRA CODE #2: R for Data Science, Wickham------------------------------------------

#FOR DATA SCIENCE3 
# Started on 3/6/2018 by Caleb Clifton Vogt
# http://r4ds.had.co.nz/index.html

#Load the tidyverse
install.packages("tidyverse")

library(tidyverse) #loads key elements of the tidyverse


# 3.2 First Steps
library(ggplot2)

mpg
head(mpg)
?mpg
View(mpg)

ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy))

ggplot(data = mpg) #creates coordinate frame which you then add layers to.

#each geom function takes a mapping argument. defines how variables in the data set are mapped to visual properties. mapping arguments are always paired with aes() and the x and y arguments of aes() specify which variales to map to the xy axes. 


# 3.2.3. A Graphing Template
# Download Turdus metadata from Xeno-Canto
download.file("https://marceloarayasalas.weebly.com/uploads/2/5/5/2/25524573/metadata_turdus_recordings_feb-2018.csv", destfile = "metadata_turdus_recordings_feb-2018.csv")

# Read in the data
turdus <- read.csv("metadata_turdus_recordings_feb-2018.csv", stringsAsFactors = FALSE)

ggplot(data = turdus) +
  geom_point(mapping = aes(x = Longitude, y = Latitude))

# 3.2.4. Exercieses

#4. Make a scatterplot of hwy vs. cyl
ggplot(data = mpg) +
  geom_point(mapping = aes(x = hwy, y = cyl))

#. Scatterplot of class vs. drv. why is the plot not useful?
ggplot(data = mpg) + geom_point(mapping = aes(x = class, y = drv))






# DATA WRANGLING IN R 
#http://jules32.github.io/2016-07-12-Oxford/dplyr_tidyr/


# 2.1. Install our first package, dplyr 

install.packages("gapminder")
library(gapminder)    
str(gapminder) #calls structure of gapminder data set. 
View(gapminder)


# 2.2. Use dplyr::filter to subset data rows (observations) 

filter(gapminder, lifeExp < 29)
filter(gapminder, country == "Mexico")
filter(gapminder, country %in% c("Mexico", "Afghanistan"))

# The same thing can be accomplished with base r code, but it is repetitive, not as clean.

# 2.3 Meet the New Pipe Editor 

gapminder[gapminder$lifeExp < 29, ]
gapminder[gapminder$country == "Mexico", ]
subset(gapminder, country == "Mexico")

gapminder %>% head() #pipe command pipes the data set into whatever function is on the right. 
head(gapminder)
head(gapminder, 3)  

turdus %>% head()  


# 2.4. Use dplyr::select to subset data on variables(columns) 

select(gapminder, year, continent)

gapminder %>% select(year, continent, gdpPercap)  

gapminder %>% 
  select(year, continent, gdpPercap) %>% 
  head(5)

#Take gapminder, then select the variables year and lifeExp, then show the first 4 rows."


# 2.5. Revel in the Convenience 
gapminder %>% 
  filter(country == "Cambodia") %>% 
  #select(country, year, pop, gdpPercap) ## entering 4 of the 6 cols is shit.
  select(-continent, -lifeExp)
#Typical R base call would look like the following

gapminder[gapminder]




# LYNDA.COM: R FOR DATA SCIENCE, LUNCHBREAK LESSONS
# Download exercise files. 


# Data Frames: Order and Merge 


data("ChickWeight")
ChickWeight$weight
sort(ChickWeight$weight)
order(ChickWeight$weight) #returns row number of where values appear

ChickWeight[196 , ] #pulls weight at specific row, looking across all columns

ChickWeight[order(ChickWeight$weight),] #sorted list of all rows in chickweight sorted by weight

chick.one <- ChickWeight[ChickWeight$Chick ==1, ]
chick.two <- ChickWeight[ChickWeight$Chick == 2, ]
View(chick.one)

match.time.obs <- merge(chick.one, chick.two, by = "Time")
View(match.time.obs)

source("chicknames.R") 
View(ChickWeight)



# R Built in Data Sets 

library(help = "datasets")
?data
data()
data("mtcars") # listed as a promise... this means that it is available for use, when we want to do something with it. 

head(mtcars)
View(mtcars)



# Vector Math 

many.numbers <- c(1:9)

for (anumber in many.numbers) { print(anumber * 2)}
many.numbers * 2
two.times <- many.numbers * 2

many.numbers / 2
more.numbers <- c(2,3,5,1,3,6,6,5,2)
many.numbers + more.numbers
short.vector <- c(2,4,5)
many.numbers + short.vector
short.odd.vector <- c(2,3,4,5)
many.numbers + short.odd.vector


# Subsetting 
# 
# LETTERS
# > LETTERS
# # [1] "A" "B" "C" "D" "E" "F" "G" "H" "I" "J" "K" "L" "M" "N" "O" "P" "Q" "R" "S"
# # [20] "T" "U" "V" "W" "X" "Y" "Z"
# > LETTERS[3]
# # [1] "C"
# > LETTERS[3:5]
# # [1] "C" "D" "E"
# > LETTERSc[c(3,20:25)]
# # Error: object 'LETTERSc' not found
# > LETTERS[c(3,20:25)]
# # [1] "C" "T" "U" "V" "W" "X" "Y"
# > LETTERS[-c(3:5)]
# # [1] "A" "B" "F" "G" "H" "I" "J" "K" "L" "M" "N" "O" "P" "Q" "R" "S" "T" "U" "V"
# [20] "W" "X" "Y" "Z"
# > LETTERS[c(-3:-5)]
# [1] "A" "B" "F" "G" "H" "I" "J" "K" "L" "M" "N" "O" "P" "Q" "R" "S" "T" "U" "V"
# [20] "W" "X" "Y" "Z"
# > rep(c(TRUE,FALSE), 13)
# [1]  TRUE FALSE  TRUE FALSE  TRUE FALSE  TRUE FALSE  TRUE FALSE  TRUE FALSE  TRUE
# [14] FALSE  TRUE FALSE  TRUE FALSE  TRUE FALSE  TRUE FALSE  TRUE FALSE  TRUE FALSE
# > LETTERS[rep(c(TRUE,FALSE), 13)]
# [1] "A" "C" "E" "G" "I" "K" "M" "O" "Q" "S" "U" "W" "Y"

lots.of.letters <- data.frame(LETTERS, letters, positions = 1 : length(letters))

lots.of.letters[3,]
lots.of.letters["LETTERS"]
lots.of.letters[3:8, 2]
lots.of.letters[LETTERS =="R", "letters"]
lots.of.letters[LETTERS == "R" | LETTERS == "T", "letters"] 


# R Data Types: Basic Types 



# LYNDA.COM: DATA WRANGLING IN R 


# 1. Tidy Data 
install.packages("tidyverse")
library(tidyverse)



# 2. Working with Tibbles 
CO2

CO2_tibble <- as_tibble(CO2)
CO2_tibble
print(CO2_tibble)
print(CO2_tibble, n = Inf)

name <- c("mike", "Renee", "Matt", "Chris" , "Ricky") 
birthyear <- c(2000, 2001, 2002, 2003, 2004)
eyecolor <- c("blue", "brown", "poop", "vomit", "nuts")
people <- tibble(name, birthyear, eyecolor)
print(people)

people$eyecolor
unique(people$eyecolor)
people[['eyecolor']]
unique(people[['eyecolor']])

## Filtering
people
filter(people, eyecolor == "blue")
filter(people, birthyear > 2003)
filter(people, eyecolor =='blue' | birthyear >2002)
filter(people, eyecolor == 'blue' & birthyear>2002)


# 3. Importing Data into R 




# EXTRA CODE #3: Nifty/helpful code snippets ------------------------------

#Deleting rows containing particular character element across all columns
df <- df[!grepl("KILL MEEEEEE", df$Name),]

#Select rows containing particular character element. 

dat_time<- dt[grepl("2018-")]

#Mean 
data(iris)
mean(iris[["Petal.Length"]])
# [1] 3.758
mean(iris[,"Petal.Length"])
# [1] 3.758
mean(iris[["Sepal.Width"]])
# [1] 3.057333
mean(iris[,"Sepal.Width"])
# [1] 3.057333

#Calculate average of specific rows. 
mn84_90 <- with(datfrm, mean(score[year >= 1984 & year <= 1990]) )

#Or...
with(dur1, mean(duration[Subject == "F1"]))



# EXTRA CODE #4: RETICULATE: R and PYTHON ------------------------------------------------
library(reticulate)

py_run_string("import numpy as np")
py_run_string("my_python_array = np.array([2,4,6,8])")

py_run_string("print(my_python_array)")


#Write python code directly and send it to the console. 
#repl_python() STARTS the chunk of code you will use in python
repl_python()
my_python_array = np.array([2,4,6,8])
print(my_python_array)

import pandas
# Exit the python chunk by literally typing "exit". That is pretty dope. 
exit


repl_python()
pip install http://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
pip install fastai



#ALTERNATIVELY. YOU CAN RUN PYTHON FROM THE TERMINAL. WITHIN A SCRIPT, USE CTRL + ALT + ETNER TO PASS SINGLE 
# LINE OF CODE TO THE TERMINAL INSTEAD OF THE CONSOLE. 
# You need to add both the C:/users/Python34 file to windows path (environmental variables) as well
# as C:/users/Python34/Scripts so you can use the pip installerrr. 


#Also can run python via a jupyter notebook in your webbrowser. search juptyr notebook and wait. cmd prompt will open







# EXTRA CODE #5: R, Stats, ESQUISSE Graphing ---------------------------------------
#RAPID GRAPHING USING ESQUISSE
library(dplyr)
library(ggplot2)
library(esquisse)




install.packages("ggstatsplot")



# EXTRA CODE #6: Update R -------------------------------------------------

install.packages("installr")
library(installr)

updateR()






# R: Keras + Tensorflow Practice ------------------------------------
# INSTALL KERAS
install.packages("devtools")
library(devtools)
devtools::install_github("rstudio/keras", force = TRUE)
library(keras)
use_python("C:/Anaconda/python.exe")


install_keras()


devtools::install_github("rstudio/tensorflow")
devtools::install_github("rstudio/keras")

tensorflow::install_tensorflow()
tensorflow::tf_config()


devtools::install_github("rstudio/tfestimators")
library(tfestimators)
