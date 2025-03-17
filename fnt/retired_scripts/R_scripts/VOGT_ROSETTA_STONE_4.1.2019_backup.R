
# 1. LIDDELL 2018: Data Processing Instructions ---------------------------------

#- Offload all camera data for a single trial into one folder.
#- Manually separate avi's and matched smi files into invidiual folders by day (D1, D2, D3, etc...)
#- After step 1, you can begin to process the videos in BORIS using the Liddell Ethogram. Remember, one BORIS project file per Trial, with each video as its own observation. 
#- offload dvr_ at 700MB split file size

# 2. LIDDELL 2018: Create OCR output csv file -------------------------------------------
library(tesseract)
library(magick)
eng <- tesseract("eng")

#Set R and system working directories
wd <- setwd("E:/Data/3_Liddell_Ecology_proc/T001")
system("cd /d E:/Data/3_Liddell_Ecology_proc/T001")
getwd()
 
#Create a list of all the generated jpgs. 
avi_list <- dir(wd, pattern = "*.avi") #Note that this code works with .avi's in mpeg4 codec
avi_list

for (aa in 1:length(avi_list)) {
  cmdstring <- paste("ffmpeg -i", avi_list[aa], "-vf fps=1 sec%06d.jpg")
  shell(cmdstring)
  jpg_list <- dir(wd, pattern ="*.jpg")
    for (bb in 1:length(jpg_list)) {
      image <- image_read(jpg_list[bb])
      crop <- image_crop(image, "315x35+50+970")
      txt <- tesseract::ocr(crop, engine = eng)
      ocr_txt <- as.character(txt)
      if(bb==1) {
        ocr_out <- ocr_txt
      } else {
        ocr_out <- rbind(ocr_out, ocr_txt)
      }
    }
  write.csv(ocr_out, paste("OCR_",avi_list[aa],".csv"))
  unlink("*.jpg")
}


# 3. LIDDELL 2018: OCR + BORIS Integration [IN PROGRESS] -------------------------------------------------



# 4. LIDDELL 2018: Network Analysis -----------------------------------------------------

#Create social network adjacency matrix from full_zone_df.csv
setwd("G:/My Drive/Alex Liddell Data/Analysis files")


vog<-read.csv("full_zone_df_v3.csv")
vog<-vog[,c(2:6,13)]
vog$duration2<-vog$STOP-vog$START
# plot(duration~duration2,data=vog)

vog<-vog[order(vog$START, na.last=FALSE) , ]
zonetypes<-sort(unique(vog$zone))

flag<-0
for (zonation in 1:length(zonetypes)) {
  thiszone<-zonetypes[zonation]
  zonewise<-vog[which(vog$zone==thiszone),]
  print(paste("Processing zone ",thiszone," out of ",
              length(zonetypes),sep=''))
  for(rowwise1 in 1:(nrow(zonewise)-1)){
    c1<-zonewise[rowwise1,,drop=FALSE]
    for(rowwise2 in (rowwise1+1):(nrow(zonewise))){
      c2<-zonewise[rowwise2,,drop=FALSE]
      if(c1$Subject!=c2$Subject){
        if(c2$START<c1$STOP & c2$START>c1$START){
          xtemp<-matrix(c(c1$Subject,c2$Subject,c1$zone),nrow=1)
          colnames(xtemp)<-c("ID1","ID2","zone")
          
          if(flag<1){
            socialinteractions<-xtemp
          } else {
            socialinteractions<-rbind(socialinteractions,xtemp)
          }
          flag<-flag+1
          
        }
      }
      
    }
  }
  
}



# This will create a directed data frame as combinations are repeated
socialinteractions<-as.data.frame(socialinteractions)
socialinteractions$ID2<-factor(socialinteractions$ID2)
socialinteractions$ID1<-factor(socialinteractions$ID1)
socialinteractions$zone<-factor(socialinteractions$zone)
summary(socialinteractions)


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



# Scripts for Boris and Video data combination
#extract timestamp from batch of smi files

install.packages(c("data.table", "dplyr", "readr", "reshape"))

library(data.table)
library(dplyr)
library(readr)
library(reshape)

wd <- setwd("G:/My Drive/Alex Liddell Data/Week 2 Analysis/T001_Bravo_Zone5_CAM10")  # DEFINE
list.files(wd)

filelist = list.files(pattern = "*.smi")
myfiles = lapply(filelist, read.delim)

timestamp <- vector()
for (i in 1:length(myfiles)) {
  y <- subset(myfiles[i][[1]], startsWith(as.character(X.SAMI.), "2018"))
  timestamp <- append(timestamp, y)
}

full_timestamp <- unlist(timestamp)

write.csv(full_timestamp, "timestamp.csv")


# Merge dataframes to create master analysis file 

DF1 <- read_csv("timestamp.csv")
DF2 <- read_csv("T001_Bravo_Zone5_CAM10.csv", skip = 18)    ## DEFINE
mydata <- DF2[,c("Subject", "Time", "X9")]

md <- melt(mydata, id=(c("Subject", "X9")))
new <- cast(md, Subject+value~X9)
new_clean <- new[,c("Subject","START","STOP")]
temp_stop <- filter(new_clean, STOP != "NA")
temp_start <- filter(new_clean, START != "NA")
temp <- cbind(temp_start,temp_stop)
temp_clean <- temp[,c(1:2,6)]
zone <- rep(5, nrow(temp_clean))            #adds a zone column: DEFINE EACH ITERATION
temp_zone <- cbind(zone, temp_clean)

denominator <- as.numeric(unlist(DF2[1,3]))
temp_zone$START_convert <- (temp_zone$START)/denominator*(length(full_timestamp))
temp_zone$STOP_convert <- (temp_zone$STOP)/denominator*(length(full_timestamp))
temp_zone$START_round <- round(temp_zone$START_convert)
temp_zone$STOP_round <- round(temp_zone$STOP_convert)

sapply(full_timestamp, class) #data stored as a factor
new_timestamp <- as.character(full_timestamp) #convert data to a character

START_real <- rep(1, nrow(temp_zone))
STOP_real <- rep(1, nrow(temp_zone))
as.factor(START_real)
as.factor(STOP_real)

temp_real <- cbind(temp_zone, START_real, STOP_real)

for (i in 1:nrow(temp_real)) {
  temp_real$START_real[i] <- new_timestamp[temp_zone$START_round[i]]
}


for (i in 1:nrow(temp_real)) {
  temp_real$STOP_real[i] <- new_timestamp[temp_zone$STOP_round[i]]
}


temp_real$START_real <- strptime(temp_real$START_real, "%Y-%m-%d %H:%M:%S")
temp_real$STOP_real <- strptime(temp_real$STOP_real, "%Y-%m-%d %H:%M:%S")

duration <- temp_real$STOP_real - temp_real$START_real
temp_duration <- cbind(temp_real, duration)


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








# 5. LIDDELL 2018: Network Analysis version 2 [CLEAN UP] ------------------
# Create social network adjacency matrix from full_zone_df.csv 

# Social networks for Week 2, all males and females

setwd("G:/My Drive/Alex Liddell Data/Week 2 Analysis/W2 Analysis")


vog<-read.csv("full_zone_df.csv")
vog<-vog[,c(2:6,13)]
vog$duration2<-vog$STOP-vog$START
# plot(duration~duration2,data=vog)

vog<-vog[order(vog$START, na.last=FALSE) , ]
zonetypes<-sort(unique(vog$zone))

flag<-0
for (zonation in 1:length(zonetypes)) {
  thiszone<-zonetypes[zonation]
  zonewise<-vog[which(vog$zone==thiszone),]
  print(paste("Processing zone ",thiszone," out of ",
              length(zonetypes),sep=''))
  for(rowwise1 in 1:(nrow(zonewise)-1)){
    c1<-zonewise[rowwise1,,drop=FALSE]
    for(rowwise2 in (rowwise1+1):(nrow(zonewise))){
      c2<-zonewise[rowwise2,,drop=FALSE]
      if(c1$Subject!=c2$Subject){
        if(c2$START<c1$STOP & c2$START>c1$START){
          xtemp<-matrix(c(as.character(c1$Subject),as.character(c2$Subject),c1$zone),nrow=1)
          #I think this should work now, try to run this (as a starting point)
          #and generate a new socialinteractions df
          colnames(xtemp)<-c("ID1","ID2","zone")
          
          if(flag<1){
            socialinteractions<-xtemp
          } else {
            socialinteractions<-rbind(socialinteractions,xtemp)
          }
          flag<-flag+1
          
        }
      }
      
    }
  }
  
}


# This will create a directed data frame as combinations are repeated
socialinteractions<-as.data.frame(socialinteractions)
socialinteractions$ID2<-factor(socialinteractions$ID2)
socialinteractions$ID1<-factor(socialinteractions$ID1)
socialinteractions$zone<-factor(socialinteractions$zone)
summary(socialinteractions)

View(socialinteractions)


# Create directed adjacency matrix 


g.unit<-(table(socialinteractions))
caca<-as.data.frame(g.unit)
#caca[which(max(caca$Freq)==caca$Freq),]
ids<-list()
ids[[1]]<-sort(unique(caca$ID1))
ids[[2]]<-sort(unique(caca$ID2))
for(z in 1:length(zonetypes)){
  diszone<-caca[which(caca$zone==z),]
  present<-matrix(diszone$Freq,nrow=16,ncol=16,dimnames = ids)
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

# Directed weighted social network 

library(igraph)

network_df <- totsmagoats

net_graph <- graph.adjacency(network_df, mode="directed", weighted = TRUE)

g <- simplify(net_graph)

V(g)$label <- V(g)$name
V(g)$degree <- degree(g)

set.seed(3952)
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



# 
# plot(g, layout=layout1)
# plot.igraph(g)


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

set.seed(3952)#what does this do? I have literally no clue---hahaha
layout1 <- layout.fruchterman.reingold(g)
# standard plot
# plot(g3, 
#      layout=layout1,
#      vertex.color = "green",
#      vertex.size = 25,
#      edge.color = 'black'
#)

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


# 6. Liddell 2018: Machine Learning & Image Classification -------------------

#1. Extract jpgs from all videos in folder and place them in new subdirectories
#2. Manually extract empty frames and 

# 7. LIDDELL 2018: OFT Analysis (pathtrackr) -------------------------------------------

#SECTION 1: install.packages(c("devtools", "beepr", "stringr"))
library(devtools)
library(beepr)
library(stringr)
#install_github("aharmer/pathtrackr")
library(pathtrackr)


#SECTION 1: Split videos into jpgs -> into new subdirectories. 
wd <- setwd("E:/Data/3_Liddell_Ecology_proc/OFT/T003/split")
wd
vids <- list.files(wd, pattern = "*.avi|*.mp4") #lists avis and mp4s
vid_paths <- str_remove(vids, ".avi|.mp4") #vector without extensions
vids
vid_paths

for (aa in 1:length(vids)) {   # only works if videos are all the same size
     splitVideo(vids[aa], 10, 480, -1) #splitVideo(file, fps, xdim, -1 = keep aspect ratio)
}

dir.create("compressed")
comp_vid <- dir(wd, pattern = "*COMPRESSED")
file.copy(comp_vid, paste(wd, "/compressed", sep=""))
file.remove(comp_vid)
beep(4)

#Check to make sure all videos were split into jpgs. if not, add those to file called not track, then re create vids and vids paths. Converting videos to avi appears to help things. also should resize the videos to 1920x1080. Custom code to split videos below. 
# 
# 
# cmdstring <- paste("ffmpeg -i", vids[1], "-vf scale=320:-1,fps=15 sec%06d.jpg")
# shell(cmdstring)
# 
# path.list <- trackPath("E:/Data/3_Liddell_Ecology_proc/OFT/T003/split/jpgs", 444.5, 444.5, fps = 1, box = 1.5, jitter.damp = 0.9)
# plotPath(path.list)
# plotSummary(path.list)
# makeVideo("E:/Data/3_Liddell_Ecology_proc/OFT/T003/split/jpgs", 444.5, 444.5, fps = 1, box = 1.5, jitter.damp = 0.9)


#SECTION 2: Track path paths and add summaries to new csv. 
path_sum_list = list()

for (bb in 1:length(vid_paths)) {
    path.list <- trackPath(paste(wd, '/', vid_paths[bb], sep = ""), 
                 444.5, 444.5, #set height/width of arena. 444.5mm = 17.5". All OFT's measure 17.5"x17.5" excecpt for arena 2, which is 17.5 x 18.5" for some fucking reason
                 fps = 15, 
                 box = 1.5, 
                 jitter.damp = 0.9) 
            sum <- pathSummary(path.list)
            path_sum_list[[bb]] <- sum
            beep(2)
}

DF1 <- as.data.frame(vid_paths)

DF2 <- as.data.frame(path_sum_list[1]) # creates long row, reshape into columns
sum_DF <- merge(DF1, DF2)
bind_rows(list_of_dataframes, .id = "column_label")

write.csv(sum_DF)

#SECTION 3: Make Videos of all tracks. 
for (i in 1:length(vid_paths)) {    
      makeVideo(paste(wd, '/', vid_paths[i], sep = ""),
            444.5, 444.5, 
            fps = 15, 
            box = 1.5, #try to draw box as a square. 
            jitter.damp = 0.9) #0.9 seems to yield best results
  beep(2)
}



plotPath(path.list) #throws an error...
pdf_1 <- plotPath(path.list)
pdf(pdf_1)


#plot(path.list[["position"]])
plotSummary(path.list)
sum <- pathSummary(path.list)
sum


path.list = diagnosticPDF(dirpath, xarena, yarena, fps = 30, box = 1, jitter.damp = 0.9)

#paste(wd, '/', vid_paths[1], sep = "")

path.list <- trackPath("T001_C57_F170", 444.5, 444.5, fps = 5, box = 1, jitter.damp = 0.8) #Track path , xarena species height in mm, yarena specifies arena height in mm


# 8. KW_EightXEight 2019: Honors Thesis ------------------------------------------------

# Load libraries and install all necessary packages. 

#install.packages(c("data.table", "dplyr", "readr", "reshape"))

library(data.table)
library(dplyr)
library(readr)
library(reshape)
library(ggplot2)


# Set the working directory
wd <- setwd("G:/My Drive/Data + Papers/2_KW_8x8")
getwd()

#Create a list of all csv data files

filelist <- list.files(pattern = "*.csv")
myfiles <- lapply(filelist, read.delim, sep = ",", skip = 15, header = TRUE) #import all csvs separated into multiple columns, skipping first 15 rows. 
View(myfiles[[1]]) #substitue number 1-10 to see different data files. 



mydata <- DF2[,c("Subject", "Time", "X9")]

md <- melt(mydata, id=(c("Subject", "X9")))
new <- cast(md, Subject+value~X9)
new_clean <- new[,c("Subject","START","STOP")]
temp_stop <- filter(new_clean, STOP != "NA")
temp_start <- filter(new_clean, START != "NA")
temp <- cbind(temp_start,temp_stop)
temp_clean <- temp[,c(1:2,6)]
zone <- rep(5, nrow(temp_clean))            #adds a zone column: DEFINE EACH ITERATION
temp_zone <- cbind(zone, temp_clean)

denominator <- as.numeric(unlist(DF2[1,3]))
temp_zone$START_convert <- (temp_zone$START)/denominator*(length(full_timestamp))
temp_zone$STOP_convert <- (temp_zone$STOP)/denominator*(length(full_timestamp))
temp_zone$START_round <- round(temp_zone$START_convert)
temp_zone$STOP_round <- round(temp_zone$STOP_convert)

sapply(full_timestamp, class) #data stored as a factor
new_timestamp <- as.character(full_timestamp) #convert data to a character

START_real <- rep(1, nrow(temp_zone))
STOP_real <- rep(1, nrow(temp_zone))
as.factor(START_real)
as.factor(STOP_real)

temp_real <- cbind(temp_zone, START_real, STOP_real)

for (i in 1:nrow(temp_real)) {
  temp_real$START_real[i] <- new_timestamp[temp_zone$START_round[i]]
}


for (i in 1:nrow(temp_real)) {
  temp_real$STOP_real[i] <- new_timestamp[temp_zone$STOP_round[i]]
}


temp_real$START_real <- strptime(temp_real$START_real, "%Y-%m-%d %H:%M:%S")
temp_real$STOP_real <- strptime(temp_real$STOP_real, "%Y-%m-%d %H:%M:%S")

duration <- temp_real$STOP_real - temp_real$START_real
temp_duration <- cbind(temp_real, duration)


duration <- temp_duration$STOP_real - temp_duration$START_real

write.csv(temp_duration, "Zone5_temp_duration.csv") # DEFINE EACH ITERATION






# 9. VIDEO DATA PIPELINE: Downsampling videos and adding hardcoded timestamps -------------------------
###USE THIS CODE TO GENERATE AVI WITH HARDCODED TIMESTAMP WITH BLACK BACKGROUND (AVI/MPEG4)
# ffmpeg -i 2.avi -vcodec copy -an 2_no_sound.avi

setwd(fp<-"C:/Users/Caleb Vogt/Desktop/direct_650") 
system("cd /d C:")
system("cd /d C:/Users/Caleb Vogt/Desktop/direct_650")
getwd()
dir.create("R_proc")

fl <- list.files(fp, full.names=TRUE, pattern=".avi")
fl.short<-list.files(fp,full.names=FALSE,pattern=".avi")
out_fp="C:/Users/Caleb Vogt/Desktop/direct_650/R_proc"

setwd(fp)
for(i in 1:length(fl)){ #THIS CHANGES H264 -> MPEG4 FORMAT, which is INCOMPATIBLE WITH PATHTRACKR
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -an -vf "subtitles=', 
               substr(as.character(fl.short[i]),1,(nchar(fl.short[i])-4)),
               '.smi',
               ":force_style='",
               'FontSize=10,Alignment=1,BorderStyle=3,Outline=1,Shadow=0,MarginV=20" ',
               '-q 3 -r 15 -max_muxing_queue_size 100000 ',  '"', #-q should be set to 3
               paste(out_fp,fl.short[i],sep='/'),'"',sep='')
  # print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}

cd(old_fp)

# 10. VIDEO DATA PIPELINE: Batch appending hardcoded avi videos ------------------------
#I created a python script that does EXACTLY this automatically called batch_appender_functional.py

setwd(fp<-"C:/Users/Caleb Vogt/Desktop/direct_650/R_proc")
setwd(fp)
system("cd /d C:/Users/Caleb Vogt/Desktop/direct_650/R_proc")

system("(for %i in (*.avi) do @echo file '%i') > mylist.txt")
system("ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.avi")

# 11. VIDEO DATA PIPELINE: Batch splitting of videos -------
setwd(fp<-"C:/Users/Caleb Vogt/Desktop/direct_650/R_proc")
setwd(fp)
system("cd /d C:/Users/Caleb Vogt/Desktop/direct_650/R_proc")
getwd()

#Using this code on raw videos offloaded from idvr pro preserves (AVI/H264) formating
system("ffmpeg -i 1.avi -c copy -q 3 -r 15 -ss 00:10:52 -to 00:20:52 OFT.avi")

# system("ffmpeg -i 1.avi -c copy -q 5 -r 15 -ss 00:10:20 -to 00:20:20 T003_C57_F190_OFT.avi")
# system("ffmpeg -i 1.avi -c copy -q 5 -r 15 -ss 00:21:18 -to 00:31:18 T003_C57_F192_OFT.avi")
# system("ffmpeg -i 1.avi -c copy -q 5 -r 15 -ss 00:32:44 -to 00:42:44 T003_C57_F194_OFT.avi")
# system("ffmpeg -i 1.avi -c copy -q 5 -r 15 -ss 00:47:06 -to 00:57:06 T004_C57_F196_OFT.avi")
# system("ffmpeg -i 1.avi -c copy -q 5 -r 15 -ss 00:58:07 -to 01:08:07 T004_C57_F198_OFT.avi")
# system("ffmpeg -i 2.avi -c copy -q 5 -r 15 -ss 00:10:12 -to 00:20:12 T003_C57_F191_OFT.avi")
# system("ffmpeg -i 2.avi -c copy -q 5 -r 15 -ss 00:21:12 -to 00:31:12 T003_C57_F193_OFT.avi")
# system("ffmpeg -i 2.avi -c copy -q 5 -r 15 -ss 00:33:39 -to 00:43:39 T004_C57_F195_OFT.avi") 
# system("ffmpeg -i 2.avi -c copy -q 5 -r 15 -ss 00:47:17 -to 00:57:17 T004_C57_F197_OFT.avi") 
# system("ffmpeg -i 2.avi -c copy -q 5 -r 15 -ss 00:58:16 -to 01:08:16 T004_C57_F199_OFT.avi")
# system("ffmpeg -i 3.avi -c copy -q 5 -r 15 -ss 00:01:10 -to 00:11:10 T004_C57_F200_OFT.avi")
# system("ffmpeg -i 3.avi -c copy -q 5 -r 15 -ss 00:12:10 -to 00:22:10 T004_C57_F202_OFT.avi") 
# system("ffmpeg -i 3.avi -c copy -q 5 -r 15 -ss 00:28:01 -to 00:38:01 T004_C57_F203_OFT.avi") 
# system("ffmpeg -i 3.avi -c copy -q 5 -r 15 -ss 00:40:38 -to 00:50:38 T003_C57_F206_OFT.avi") 
# system("ffmpeg -i 3.avi -c copy -q 5 -r 15 -ss 00:57:35 -to 01:07:35 T003_C57_F208_OFT.avi")
# system("ffmpeg -i 4.avi -c copy -q 5 -r 15 -ss 00:01:42 -to 00:11:42 T004_C57_F201_OFT.avi")
# system("ffmpeg -i 4.avi -c copy -q 5 -r 15 -ss 00:14:38 -to 00:24:38 T004_C57_F204_OFT.avi")
# system("ffmpeg -i 4.avi -c copy -q 5 -r 15 -ss 00:31:06 -to 00:41:06 T003_C57_F205_OFT.avi")
# system("ffmpeg -i 4.avi -c copy -q 5 -r 15 -ss 00:44:30 -to 00:54:30 T003_C57_F207_OFT.avi")
# system("ffmpeg -i 4.avi -c copy -q 5 -r 15 -ss 01:01:21 -to 01:11:21 T003_C57_F209_OFT.avi")


# 12. VIDEO DATA PIPELINE: Convert Video Format & Rescale ---------------------------
#Single conversions...
setwd(fp<-"E:/Data/3_Liddell_Ecology_proc/OFT/not tracked")
system("cd /d E:/Data/3_Liddell_Ecology_proc/OFT/not tracked")
getwd()

system("ffmpeg -i T001_C57_F174.mp4 -c copy -q 3 -r 15 -vf scale=1920:1080 T001_C57_F174.avi")
system("ffmpeg -i T001_C57_F177.mp4 -c copy -q 3 -r 15 T001_C57_F177.avi")

#Batch conversions...
setwd(fp<-"E:/Data/3_Liddell_Ecology_proc/OFT/test") 
system("cd /d E:")
system("cd /d E:/Data/3_Liddell_Ecology_proc/OFT/test")
getwd()
dir.create("R_proc")
#code from arhammer on how to convert my videos so they dont have sound. this should fix the issue I was having and would allow me to use his splitVideo function. 
# ffmpeg -i 2.avi -vcodec copy -an 2_no_sound.avi

fl <- list.files(fp, full.names=TRUE, pattern=".avi")
fl.short<-list.files(fp,full.names=FALSE,pattern=".avi")
out_fp="E:/Data/3_Liddell_Ecology_proc/OFT/test/R_proc"

setwd(fp)
for(i in 1:length(fl)){
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -vcodec copy -an ',  '"', #-an removes audio
               paste(out_fp, fl.short[i], sep='/'),'"',sep='')
  # print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}
 
cd(old_fp)


# EXTRA CODE  --------------------------------------------------------------

jpg_list <- dir(wd, pattern ="*.jpg")

Z1a_jpg_list <- dir(wd, pattern = "Z1")
cmdstring <- paste("ffmpeg -i", avi_list[1], "-vf fps=1 sec%06d.jpg")
cmdstring





#Delete all jpgs
unlink("*.jpg")  

#Writes OCR output to csv for later use
write.csv(ocr_out, "T001_Z1_D1_7.13_ocr.csv") 

head(jpg_list)

mydata <- textfunk(jpg_list)

image <- image_read("frame000001.jpg")
image
crop <- image_crop(image, "315x35+50+970")
crop
text <- tesseract::ocr(crop, engine = eng)
cat(text)
text




#Run loop to load, crop, and OCR jpgs, and load ocr output into dataframe.
for (i in 1:length(jpg_list)) {
  image <- image_read(jpg_list[i])
  crop <- image_crop(image, "315x35+50+970")
  txt <- tesseract::ocr(crop, engine = eng)
  ocr_txt <- as.character(txt)
  if(i==1) {
    ocr_out <- ocr_txt
  } else {
    ocr_out <- rbind(ocr_out, ocr_txt)
  }
}





# cmdstring <- 'for %A in (*.avi) DO ffmpeg -i %A -vf fps=1 sec%06d_%A.jpg'
cmdstring <- 'for %A in (*.avi) DO ffmpeg -i %A -vf fps=1 %A_%06d.jpg'
shell(cmdstring) #


#Create a list of all the video files in the working directory this actually fucking works!

# avi_list[1]
cmdstring <- paste(
  "ffmpeg -i", avi_list[1],
  "-vf fps=1 sec%06d.jpg"
)
cmdstring
