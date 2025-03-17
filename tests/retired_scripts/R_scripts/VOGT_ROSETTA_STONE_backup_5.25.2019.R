# 1. VIDEO DATA PIPELINE: Batch downsampling avi's and adding hardcoded timestamps -------------------------
#Remember you can batch convert all of your video files at once on an external HDD, but not append using the python script.
#Then place the converted videos you want to append in the same folder and copy to Desktop SDD. Go to step 2. 

setwd(fp<-"I:/3_Liddell_2018_RAW/Field/T004/proc") #change this
system("cd /d I:")
system("cd /d I:/3_Liddell_2018_RAW/Field/T004/proc")  #change this
getwd()
dir.create("R_proc")
 
fl <- list.files(fp, full.names=TRUE, pattern=".avi")
fl.short<-list.files(fp,full.names=FALSE,pattern=".avi")
out_fp="I:/3_Liddell_2018_RAW/Field/T004/proc/R_proc"  #change this

setwd(fp)
for(i in 1:length(fl)){ #Changes codec from H264 -> mpeg4
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -an -vf "subtitles=', #-an removes audio from video
               substr(as.character(fl.short[i]),1,(nchar(fl.short[i])-4)),
               '.smi',
               ":force_style='",
               'FontSize=10,Alignment=1,BorderStyle=3,Outline=1,Shadow=0,MarginV=20" ',
               '-q 3 -r 10 -max_muxing_queue_size 100000 ',  '"', #-q should be set to 3. -r should be set to 10 fps
               paste(out_fp,fl.short[i],sep='/'),'"',sep='')
  # print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}

cd(old_fp)




### My Looping attempt (as yet dysfunctional): 

library(beepr)

wd <- setwd("I:/3_Liddell_2018_RAW/Field/T002") #set R directory
system("cd /d I:/3_Liddell_2018_RAW/Field/T002") #set ffmpeg directory
folder.list <- list.files(wd, full.names = TRUE)
folder.list

for (aa in 3:length(folder.list)) { #make sure to skip any irrelevant folders
  fp <- paste(folder.list[aa]) 
  # fp <- setwd(paste(folder.list[aa])) 
  system(paste("cd /d", paste(folder.list[aa], sep =''))) #changes ffmpeg directory *noice
  
  newDir<-paste(folder.list[aa], "/R_proc", sep = '')
  if (dir.exists(newDir)){
    setwd(newDir)
  } else {
    dir.create(newDir)
    setwd(newDir)
  }


  fl <- list.files(fp, full.names=TRUE, pattern=".avi")
  fl.short <- list.files(fp,full.names=FALSE,pattern=".avi")
  out_fp = paste(fp, "/R_proc", sep = '')
  
  # setwd(fp)
  for(i in 1:length(fl)){ #Changes codec from H264 -> mpeg4
    cmdstr=paste('ffmpeg -i "', fl[i], '" -an -vf subtitles="', #-an removes audio from video
                 substr(as.character(fl[i]),1,(nchar(fl[i])-4)),
                 '.smi"',
                 ":force_style=","'",
                 'FontSize=10,Alignment=1,BorderStyle=3,Outline=1,Shadow=0,MarginV=20',"'",
                 ' -q 3 -r 10 -max_muxing_queue_size 100000 ',  '"', #-q should be set to 3
                 paste(out_fp,fl.short[i],sep='/'),'"',sep='')
    
    
    
    # print(cmdstr)#display in command window
    system(cmdstr) #send to windows
  }
  
  # wd <- setwd("I:/3_Liddell_2018_RAW/Field/T002") 
  # system("cd /d I:/3_Liddell_2018_RAW/Field/T002")
  beep(1)
  beep(1)
  beep(1)
}

# 2. VIDEO DATA PIPELINE: Batch appending hardcoded avi videos ------------------------
# Python script "batch_appender_functional.py" does this. 
# However, it will append videos in wrong order if you do this from an external hard drive
# Copy converted videos to desktop and then run .py script. 
# Take converted videos and put them in the appropriate subfolders. 

setwd(fp<-"C:/Users/Caleb Vogt/Desktop/direct_650/R_proc")
setwd(fp)
system("cd /d C:/Users/Caleb Vogt/Desktop/direct_650/R_proc")

system("(for %i in (*.avi) do @echo file '%i') > mylist.txt")
system("ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.avi")

# 3. VIDEO DATA PIPELINE: Batch splitting/trimming of videos with ffmpeg -------
setwd(fp<-"E:/Data/3_Liddell_Ecology_proc/OFT/extra/T003+T004")
setwd(fp)
system("cd /d E:/Data/3_Liddell_Ecology_proc/OFT/extra/T003+T004")
getwd()

#Using this code on raw videos offloaded from idvr pro preserves (AVI/H264) formating

system("ffmpeg -i 4.avi -c copy -q 3 -r 15 -ss 01:53:26 -to 02:03:26 T004_WSB_M127_OFT_trim.avi")


# 4. VIDEO DATA PIPELINE: Convert Video Format & Rescale ---------------------------
# system("ffmpeg -i T001_C57_F174.mp4 -c copy -q 3 -r 15 -vf scale=1920:1080 T001_C57_F174.avi")

setwd(fp<-"E:/Data/3_Liddell_Ecology_proc/OFT/T002") 
system("cd /d E:/Data/3_Liddell_Ecology_proc/OFT/T002")
dir.create("R_proc")

fl <- list.files(fp, full.names=TRUE, pattern=".avi|.mp4")
fl.short <- list.files(fp,full.names=FALSE,pattern=".avi|.mp4")
fl.shorty <-str_remove(fl.short, ".avi|.mp4")
out_fp = "E:/Data/3_Liddell_Ecology_proc/OFT/T002/R_proc"

setwd(fp)
for(i in 1:length(fl)){   #-vf scale=-1:1080 # throws error, degrades quality
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -vcodec copy -an ',  '"', #-an removes audio
               paste(out_fp, paste(fl.shorty[i], ".avi", sep=""), sep='/'),'"',sep='')
  # print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}
 
cd(old_fp)


# 5. VIDEO DATA PIPELINE: Create OCR output csv file -------------------------------------------
library(tesseract)
library(magick)
library(beepr)
eng <- tesseract("eng")

#Set R and system working directories
wd <- setwd("E:/Data/3_Liddell_Ecology_proc/T004/W2")
system("cd /d E:/Data/3_Liddell_Ecology_proc/T004/W2")
getwd()

#Create jpgs from video and list of all the generated jpgs. 
avi_list <- dir(wd, pattern = "*.avi") #Note that this code works with .avi's in mpeg4 codec
avi_list

#note that this for loop is best run on folders containing 10-15 videos. Sometimes aborts the R session, likely due to memory overload when there are more videos. Could also be due to having exceptionally long videos. 
for (aa in 1:length(avi_list)) {
  cmdstring <- paste("ffmpeg -i", avi_list[aa], "-vf fps=1 sec%06d.jpg")
  shell(cmdstring)
  jpg_list <- dir(wd, pattern ="*.jpg")
  for (bb in 1:length(jpg_list)) {
    image <- image_read(jpg_list[bb])
    crop <- image_crop(image, "315x35+50+970") #Note that this defines the crop area. Widthxheigh + distance from side+distance from top. 140x15+25+430 for T004 Z4. 
    txt <- tesseract::ocr(crop, engine = eng)
    ocr_txt <- as.character(txt)
    if(bb==1) {
      ocr_out <- ocr_txt
    } else {
      ocr_out <- rbind(ocr_out, ocr_txt)
    }
  }
  write.csv(ocr_out, paste("OCR_",avi_list[aa],".csv", sep = ''))
  unlink("*.jpg")
  
  beep(1)
  beep(1)
  beep(1)
}


# 6. VIDEO DATA PIPELINE: OCR + BORIS Integration -------------------------------------------------
library(data.table)
library(plyr)
library(dplyr)
library(readr)
library(reshape)
library(xlsx)

#Add BORIS_ to the beginning of all the BORIS observation csv files. Make sure folder only contains BORIS csv files before doing this. 
wd <- setwd("E:/Data/3_Liddell_Ecology_proc/T002/W2/BORIS")
file.list <- list.files(wd, pattern = "*.csv")

for (i in 1:length(file.list)){
  file.rename(file.list[i], paste("BORIS_", file.list[i], sep=''))
}
#Move boris observations and OCR outputs into the same working directory folder. 

#Overwrite previous file.list
file.list <- list.files(wd, pattern = "*.csv")
file.list

OCR_list <- dir(wd, pattern = "*OCR_*") #make sure there are no folders. 
BORIS_list <- dir(wd, pattern = "*BORIS_*")
OCR_list #Check yo shit. 
BORIS_list #Check yo shit. 

#Check the Boris observation files for the unnecessary metadata rows. 
#Make sure to check that the number of "STARTs" and "STOPS" are the same in the csv
# Otherwise this will likely throw an error. 

for (aa in 1:length(BORIS_list)) {
   DF1 <- read.csv(OCR_list[aa])
   DF2 <- read.csv(BORIS_list[aa], skip = 15) #worth checking if BORIS files have 15 worthless rows
   DF2["Real_World_Time"] <- NA
   rw_time <- DF1[ ,2] #pulls out the real world times
   BOR_time <- DF2[ ,1] #pulls out the BORIS video time in seconds
   round_BOR_time <- round(BOR_time) #Rounds the BORIS video times to nearest second
   DF2$Real_World_Time <-rw_time[round_BOR_time] #Populates Real_World_Time column... fucking dope. 
   write.csv(DF2, paste("RT_",BORIS_list[aa], sep = ''))
}

### STOP

#Functional, but silly. # of columns must match. Change csv names. 
D1 <- read.csv("RT_BORIS_T002_Z1_W2.csv")
D2 <- read.csv("RT_BORIS_T002_Z2_W2.csv")
D3 <- read.csv("RT_BORIS_T002_Z3_W2.csv")
D4 <- read.csv("RT_BORIS_T002_Z4_W2.csv")
D5 <- read.csv("RT_BORIS_T002_Z5_W2.csv")
D6 <- read.csv("RT_BORIS_T002_Z6_W2.csv")
D7 <- read.csv("RT_BORIS_T002_Z7_W2.csv")
D8 <- read.csv("RT_BORIS_T002_Z8_W2.csv")

#Delete extra column from RT file by name if necessary. Actually, better method would be to add an empty column 
D5 <- subset(D5, select = -c(Modifier.1))

#merge RT data. Rename as needed. 
merged <- do.call("rbind", list(D1, D2, D3, D4, D5, D6, D7, D8)) #merge1 <- rbind(D1,D2). 

#Note that zone information is included in the file name column. 
merged$zone <- ifelse(grepl("Z1", merged$Media.file.path), "1", 
                  ifelse(grepl("Z2", merged$Media.file.path), "2",
                  ifelse(grepl("Z3", merged$Media.file.path), "3",
                  ifelse(grepl("Z4", merged$Media.file.path), "4",
                  ifelse(grepl("Z5", merged$Media.file.path), "5",
                  ifelse(grepl("Z6", merged$Media.file.path), "6",
                  ifelse(grepl("Z7", merged$Media.file.path), "7",
                  ifelse(grepl("Z8", merged$Media.file.path), "8", "NONE"))))))))
merged$zone <- as.numeric(merged$zone)

#Subset the merge file. #Check columns. 
short_merge <- subset(merged, select = c(X,
                                        #Time,
                                        #Media.file.path,
                                        zone,
                                        Subject,
                                        Behavior,
                                        Modifier.1, #Ethogram specific, check
                                        Modifier.2, #ethogram specific, check
                                        Comment,     #Ethogram specific
                                        Status,
                                        Real_World_Time))

#Write a csv here if you want START/STOP in single column. 
write.csv(short_merge, "MERGE_RT_BORIS_T002_W2.csv")

# Extract START and STOP rows and combine into new dataframe. 
DF1 <- short_merge[order(short_merge$Status, short_merge$Subject), ]
START_rows <- DF1[DF1$Status == 'START', ]
STOP_rows <- DF1[DF1$Status == 'STOP', ]
DF2 <- cbind(START_rows, STOP_rows)

#check which columns you are renaming the correct columns in DF2
names(DF2)[9] <- "RWT_START" 
names(DF2)[18] <- "RWT_STOP"

#change time format into appropriate class
library(lubridate)
#Converts factor to "POSIXct" and POSIXt"
DF2$RWT_START <- ymd_hms(DF2$RWT_START)  
DF2$RWT_STOP <- ymd_hms(DF2$RWT_STOP)

# Converts factor to POSIXlt" and "POSIXt"
#DF2$RWT_START <- strptime(DF2$RWT_START, "%Y-%m-%d %H:%M:%S") 
#DF2$RWT_STOP <- strptime(DF2$RWT_STOP, "%Y-%m-%d %H:%M:%S")

# Calculate duration of state events
duration <- DF2$RWT_STOP - DF2$RWT_START

#Merge dataframes and durations
caleb <- cbind(DF2, duration)

#Decide what final columns you want. 
dur_merge <- subset(caleb, select = c(X,
                                      zone,
                                      Subject,
                                      Behavior,
                                      Modifier.1,
                                      Modifier.2,
                                      #Comment,
                                      RWT_START,
                                      RWT_STOP,
                                      duration))


#Write DUR_MERGE_RT.CSV
write.csv(dur_merge, "DUR_MERGE_RT_BORIS_T002_W2.csv")


# 7. SOCIAL NETWORK: Create Adjacency matrix -------------------
#SOCIAL NETWORK ASSOCIATION MATRIX FOR ALL OF WEEK 1. 
setwd("E:/Data/3_Liddell_Ecology_proc/T002/W2/BORIS")
library(tidyverse)
library(lubridate)
library(igraph)

vog <- read.csv("DUR_MERGE_RT_BORIS_T002_W2.csv") 

#Depending on basis for the social network, need to remove unnecessary columns
#A network based on co-occupancy of the resource zones should not have columns containing nas,
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



# 8. SOCIAL NETWORK: iGraph Plotting ------------------------------


#Working with the directed adjacency matrix. 
totsmagoats <- read.csv("directed_adjacency_matrix.csv")

# Directed social network 
network_df <- totsmagoats
net_graph <- graph.adjacency(network_df, mode="directed", weighted = NULL)
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


# 9. LIDDELL GRAPHING --------------------------------
library(tidyverse)
library(lubridate)

#T002_W1 Graphing
setwd("E:/Data/3_Liddell_Ecology_proc/T002/W1/BORIS")
dat1 <- read.csv("DUR_MERGE_RT_BORIS_T002_W1.csv") 


#Figure 1. Total durations of females in RZ's for the entire week. 
#Figure 2. Development of female space use over time. 
#Figure 3. Total durations of male space use in RZ's for the entire week. 
#Figure 4. Development of male space use over time. Also say which resources animals were released in. 
#add red square for release point. 
#Figure 5. 

ggplot(subset(dat, Subject == c('M1','M2','M3','M4','M5','M6')), aes(fill=Subject, y=duration, x=zone)) + geom_bar(stat="identity") + 
  facet_wrap(~Subject)


ggplot(subset(dat, Subject == c('F1','F2','F3','F4','F5','F6', 'F7', 'F8', 'F9', 'F10' )), aes(fill=Subject, y=duration, x=zone)) + geom_bar(stat="identity") + 
  facet_wrap(~Subject)

ggplot(dat, aes(fill=Subject, y=duration, x=zone)) + geom_bar(stat="identity") + 
  facet_wrap(~Subject)

#PPT slide with color coded ownership RZs. 

# Plot of duration of time spent in zone by animal
ggplot(data = dat) +
  geom_bar(mapping = aes(x = Subject, y = duration))

ggplot(dat, aes(x=Subject, y=duration)) + geom_point()


attach(dat)
F1 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'F1'),sum)
F2 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'F2'),sum)
M1 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M1'),sum)
M2 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M2'),sum)
M3 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M3'),sum)
M4 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M4'),sum)
M5 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M5'),sum)
M6 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M6'),sum)

ggplot(data=F1, aes(x=F1$zone,y=F1$duration)) + geom_bar(stat='identity')
ggplot(data=F2, aes(x=F2$zone,y=F2$duration)) + geom_bar(stat='identity')
ggplot(data=M2, aes(x=M2$zone,y=M2$duration)) + geom_bar(stat='identity')
ggplot(data=M3, aes(x=M3$zone,y=M3$duration)) + geom_bar(stat='identity')
ggplot(data=M4, aes(x=M4$zone,y=M4$duration)) + geom_bar(stat='identity')
ggplot(data=M5, aes(x=M5$zone,y=M5$duration)) + geom_bar(stat='identity')
ggplot(data=M6, aes(x=M6$zone,y=M6$duration)) + geom_bar(stat='identity')

ggplot(data=subset(dat,Subject == "M5"), aes(x=zone,y=duration)) + geom_point()



caleb <- read.csv('MERGE_RT_BORIS_T002_W2.csv')
caleb <- na.omit(caleb)
caleb$Real_World_Time <- ymd_hms(caleb$Real_World_Time)

caleb1 <- subset(caleb, Status == 'START')
caleb2 <- subset(caleb, Status == 'STOP')

caleb1$ID <- seq.int(nrow(caleb1))
caleb2$ID <- seq.int(nrow(caleb1))

caleb3 <- rbind(caleb1,caleb2) 

M3D <- subset(caleb3, (Subject == "M3"))
M4D <- subset(caleb3, (Subject == "M4"))

ggplot(data=M3D, aes(x=M3D$Real_World_Time, y=M3D$zone, group=M3D$ID)) +
  geom_line(size = 5) +
  labs(x="Time", y="Zone", title="M3 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

s <- subset(M3D,Status == "START")
ggplot(data=s, aes(x=s$Real_World_Time, y=s$zone)) + geom_step()caleb <- read.csv('MERGE_RT_BORIS_T002_W2.csv')
caleb <- na.omit(caleb)
caleb$Real_World_Time <- ymd_hms(caleb$Real_World_Time)

caleb1 <- subset(caleb, Status == 'START')
caleb2 <- subset(caleb, Status == 'STOP')

caleb1$ID <- seq.int(nrow(caleb1))
caleb2$ID <- seq.int(nrow(caleb1))

caleb3 <- rbind(caleb1,caleb2) 

M3D <- subset(caleb3, (Subject == "M3"))
M4D <- subset(caleb3, (Subject == "M4"))

ggplot(data=M3D, aes(x=M3D$Real_World_Time, y=M3D$zone, group=M3D$ID)) +
  geom_line(size = 5) +
  labs(x="Time", y="Zone", title="M3 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

s <- subset(M3D,Status == "START")
ggplot(data=s, aes(x=s$Real_World_Time, y=s$zone)) + geom_step()



caleb <- read.csv('MERGE_RT_BORIS_T002_W2.csv')
caleb <- na.omit(caleb)
caleb$Real_World_Time <- ymd_hms(caleb$Real_World_Time)

caleb1 <- subset(caleb, Status == 'START')
caleb2 <- subset(caleb, Status == 'STOP')

caleb1$ID <- seq.int(nrow(caleb1))
caleb2$ID <- seq.int(nrow(caleb1))

caleb3 <- rbind(caleb1,caleb2) 

M3D <- subset(caleb3, (Subject == "M3"))
M4D <- subset(caleb3, (Subject == "M4"))

ggplot(data=M3D, aes(x=M3D$Real_World_Time, y=M3D$zone, group=M3D$ID)) +
  geom_line(size = 5) +
  labs(x="Time", y="Zone", title="M3 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

s <- subset(M3D,Status == "START")
ggplot(data=s, aes(x=s$Real_World_Time, y=s$zone)) + geom_step()caleb <- read.csv('MERGE_RT_BORIS_T002_W2.csv')
caleb <- na.omit(caleb)
caleb$Real_World_Time <- ymd_hms(caleb$Real_World_Time)

caleb1 <- subset(caleb, Status == 'START')
caleb2 <- subset(caleb, Status == 'STOP')

caleb1$ID <- seq.int(nrow(caleb1))
caleb2$ID <- seq.int(nrow(caleb1))

caleb3 <- rbind(caleb1,caleb2) 

M3D <- subset(caleb3, (Subject == "M3"))
M4D <- subset(caleb3, (Subject == "M4"))

ggplot(data=M3D, aes(x=M3D$Real_World_Time, y=M3D$zone, group=M3D$ID)) +
  geom_line(size = 5) +
  labs(x="Time", y="Zone", title="M3 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

s <- subset(M3D,Status == "START")
ggplot(data=s, aes(x=s$Real_World_Time, y=s$zone)) + geom_step()


# 10. VIDEO DATA PIPELINE: LIDDELL MOVEMENT ANIMATION -----------------------
library(moveVis)
library(anipaths)
library(tidyverse)
library(lubridate)
library(igraph)

setwd("E:/Data/3_Liddell_Ecology_proc/T002/W1/BORIS")
tracks <- read.csv("DUR_MERGE_RT_BORIS_T002_W1.csv") 




# Create outline of Liddell Paddock
referencecoords <- data.frame("df"= 1:2, "X" = c(0,0,75,75), "Y" = c(0,175,175,0))

plot(NA, xlim=c(0,180),
     ylim=c(0,180),
     xlab='',ylab='',
     axes=FALSE)

outline.xs <- referencecoords$X[c(1,2,3,4,1)]
outline.ys <- referencecoords$Y[c(1,2,3,4,1)]
lines(outline.xs,outline.ys)

#Create Resource Zones
z1x <- c(15, 30, 30, 15)
z1y <- c(20, 20, 35, 35)
polygon(z1x,z1y)
#center coords (20,25)

z2x <- c(45,60, 60, 45)
z2y <- c(20, 20, 35, 35)
polygon(z2x,z2y)
#center coords (50,25)

z3x <- c(15, 30, 30, 15)
z3y <- c(60, 60, 75, 75)
polygon(z3x,z3y)
#center coords (20,65)

z4x <- c(45,60, 60, 45)
z4y <- c(60, 60, 75, 75)
polygon(z4x,z4y)
#center coords (50,65)


z5x <- c(15, 30, 30, 15)
z5y <- c(100, 100, 115, 115)
polygon(z5x,z5y)
#center coords (20,105)


z6x <- c(45,60, 60, 45)
z6y <- c(100, 100, 115, 115)
polygon(z6x,z6y)
#center coords (50,105)


z7x <- c(15, 30, 30, 15)
z7y <- c(140, 140, 155, 155)
polygon(z7x,z7y)
#center coords (20,145)


z8x <- c(45,60, 60, 45)
z8y <- c(140, 140, 155, 155)
polygon(z8x,z8y)
#center coords (50,145)

#add xy coordinate columns to tracks dataframe. would be good to add some jitter to that shit. 

tracks$x <- ifelse(grepl("1", tracks$zone), "20", 
            ifelse(grepl("2", tracks$zone), "50",
            ifelse(grepl("3", tracks$zone), "20",
            ifelse(grepl("4", tracks$zone), "50",
            ifelse(grepl("5", tracks$zone), "20",
            ifelse(grepl("6", tracks$zone), "50",
            ifelse(grepl("7", tracks$zone), "20",
            ifelse(grepl("8", tracks$zone), "50",
                        "none"))))))))

tracks$y <- ifelse(grepl("1", tracks$zone), "25", 
            ifelse(grepl("2", tracks$zone), "25",
            ifelse(grepl("3", tracks$zone), "65",
            ifelse(grepl("4", tracks$zone), "65",
            ifelse(grepl("5", tracks$zone), "105",
            ifelse(grepl("6", tracks$zone), "105",
            ifelse(grepl("7", tracks$zone), "145",
            ifelse(grepl("8", tracks$zone), "145",
                    "none"))))))))

clean_tracks <- subset(tracks, select = c(Subject, x, y, RWT_START, RWT_STOP))
                       


#Example code
library(moveVis)
library(move)

data("move_data", package = "moveVis") # move class object
# if your tracks are present as data.frames, see df2move() for conversion
head(move_data)
summary(move_data)
# align move_data to a uniform time  scale
m <- align_move(move_data, res = 240, digit = 0, unit = "secs")

# create spatial frames with a OpenStreetMap watercolour map
frames <- frames_spatial(m, path_colours = c("red", "green", "blue"),
                         map_service = "osm", map_type = "watercolor", alpha = 0.5) %>% 
  add_labels(x = "Longitude", y = "Latitude") %>% # add some customizations, such as axis labels
  add_northarrow() %>% 
  add_scalebar() %>% 
  add_timestamps(m, type = "label") %>% 
  add_progress()

frames[[100]] # preview one of the frames, e.g. the 100th frame

# animate frames
animate_frames(frames, out_file = "moveVis.gif")

# In Progress. VIDEO DATA PIPELINE: Overlay transparent .png files over .avi -----------
# In Progress. Liddell 2018: Machine Learning & Image Classification -------------------

#1. Extract jpgs from all videos in folder and place them in new subdirectories
#2. Manually extract empty frames and 




# In Progress. VIDEO DATA PIPELINE: PATHTRACKER for Single Animal Tracking (OFT)  -------------------------------------------

#STEP 1: INSTALL RELEVANT PACKAGES AND SET WORKING DIRECTORY
library(devtools)
library(beepr)
library(stringr)
#install_github("aharmer/pathtrackr")
library(pathtrackr)
library(ggplot2)

wd <- setwd("H:/2_8x8_KW_proc/Pre_SPP") ## RUN THIS LINE TWICE
vids <- list.files(wd, pattern = "*.avi|*.mp4") #lists avis and mp4s
vid_paths <- str_remove(vids, ".avi|.mp4") #vector without extensions

#STEP 2: CREATE NEW FOLDERS AND SPLIT VIDEOS INTO FRAMES
dir.create("COMPRESSED")
for (aa in 1:length(vids)) {   #If this fails, check extra code for manual video split. 
  splitVideo(vids[aa], 15, 1280, -1) #splitVideo(file, fps, xdim, -1 = keep aspect ratio). Use 1280 xwidth for SPP vidoes that have more environmental noise. 
  
  comp_vid <- dir(wd, pattern = "*COMPRESSED")
  file.copy(comp_vid, paste(wd, "/COMPRESSED", sep=""))
  file.remove(comp_vid)
  beep(1)
  beep(1)
  beep(1)
}
beep(4)

#STEP 3: TRACK PATHS AND OUPUT CSVS AND FIGURES.

## OFT CHAMBER: 17.5 x 17.5" (444.5 x 444.5mm)
## SPP CHAMBER: 23 x 8.5" (584 x 216mm)
for (bb in 1:length(vid_paths)) {
  path.list <- trackPath(paste(wd, '/', vid_paths[bb], sep = ""), #video path
                         584, # arena width (mm)444.5
                         216, # arena height (mm)444.5
                         fps = 15, 
                         box = 1, # can mess with this, b/w 1-2
                         jitter.damp = 1.0) # Less than 0.7 seems to have poor performance
  sum <- pathSummary(path.list)
  write.csv(sum, paste(vid_paths[bb], "_summary.csv", sep = ""))
  write.csv(path.list$movement, paste(vid_paths[bb], "_movement.csv", sep =""))
  write.csv(path.list$position, paste(vid_paths[bb], "_position.csv", sep =""))
  
  # pdf(file=paste(vid_paths[bb], "_OFT_track.pdf", sep=""))  
  # plotPath(path.list) 
  # dev.off()
  ## THIS CODE IS TAKEN FROM PATHTRACKR WITH SLIGHT MODIFICATIONS. 
  dat = as.data.frame(path.list$position)
  dat$xpos = dat$xpos * (path.list$dim.arena[1]/path.list$dim.pix[1])
  dat$ypos = dat$ypos * (path.list$dim.arena[2]/path.list$dim.pix[2])
  x_max = path.list$dim.pix[1] * (path.list$dim.arena[1]/path.list$dim.pix[1])
  y_max = path.list$dim.pix[2] * (path.list$dim.arena[2]/path.list$dim.pix[2])
  
  fig <- ggplot(aes(xpos, ypos), data = dat) +
    stat_density2d(aes(fill = ..density.., alpha = ..density..), geom = "tile", contour = FALSE) +
    scale_fill_gradientn(colours = viridis::viridis(256)) +
    geom_path(na.rm = TRUE) +
    geom_point(aes(x = dat[1,1], y = dat[1,2]), size = 3, color = "blue") +
    geom_point(aes(x = dat[nrow(dat),1], y = dat[nrow(dat),2]), size = 3, color = "red") +
    coord_fixed() +
    scale_x_continuous(limits = c(0, x_max), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, y_max), expand = c(0, 0)) +
    xlab("Distance (mm)") +
    ylab("Distance (mm)") +
    theme_bw() +
    theme(legend.position = "none", axis.title = element_text(size = 14, face = "bold"), 
          panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
    labs(title = paste(vid_paths[bb])) ## SHOULD MODIFY THIS CODE TO CENTER TITLE. 
  pdf(file=paste(vid_paths[bb], "_position.pdf", sep=""))
  plot(fig)
  dev.off()
  ## THIS IS FINE FOR NOW, GRAB RAW CODE FROM PATHTRACKR TO ADD TITLES VIA GGPLOT
  pdf(file=paste(vid_paths[bb], "_movement.pdf", sep=""))  
  plotSummary(path.list) 
  dev.off()
  
  beep(1)
  beep(1)
  beep(1)
}
beep(4)


#STEP 4: MAKE TRACKING VIDEOS. 
dir.create("TRACKED")
for (cc in 1:length(vid_paths)) {    
  makeVideo(paste(wd, '/', vid_paths[cc], sep = ""),
            584, # arena width (mm) 584/444.5
            216, # arena height (mm) 216/444.5
            fps = 15, 
            box = 1.0, #try to draw box as a square. 
            jitter.damp = 0.7) #0.9 seems to yield best results
  tracked_vid <- dir(wd, pattern = "*TRACKED")
  file.copy(tracked_vid, paste(wd, "/TRACKED", sep=""))
  file.remove(tracked_vid)
  beep(1)
  beep(1)
  beep(1)
}

beep(4)
beep(4)
beep(4)

# extraneous code: path.list = diagnosticPDF(dirpath, xarena, yarena, fps = 30, box = 1, jitter.damp = 0.9)

#STEP 5 (Under Development): ROI's and location information. 
#mine this code https://github.com/aharmer/pathtrackr/blob/master/R/trackPath.R




# 8x8 Pipeline: UMATracker Output -----------------------------------------




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




# 3. EXTRA CODE: Nifty/helpful code snippets ------------------------------

#Deleting rows containing particular character element
df <- df[!grepl("KILL MEEEEEE", df$Name),]

#Select rows containing particular character element. 

dat_time<- dt[grepl("2018-")]
  
  
  
