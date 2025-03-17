# 1. VIDEO DATA PIPELINE: BATCH ENCODE TIMESTAMPS AND ADJUST VIDEO FRAMERATE/QUALITY  -------------------------
#Remember you can batch convert all of your video files at once on an external HDD, but not append using the python script.
#Then place the converted videos you want to append in the same folder and copy to Desktop SDD. Go to step 2. 
# FOR OPEN FIELD TRIALS, SET FPS TO AT LEAST 15 FPS. 

setwd(fp<-"J:/T001/BRAVO_Treatment") #change this
system("cd /d J:")
system("cd /d J:/T001/BRAVO_Treatment")  #change this
getwd()
dir.create("R_proc")

fl <- list.files(fp, full.names=TRUE, pattern=".avi")
#View(fl)
fl.short<-list.files(fp,full.names=FALSE,pattern=".avi")
out_fp="D:/T018/R_proc"  #change this

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

#TO ADD: SHOULD BE ABLE TO INCLUDE A LINE AT THE END OF THIS FOR LOOP TO APPEND THE VIDEO AFTER EACH ITERATION. 


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

# 2. VIDEO DATA PIPELINE: Batch appending hardcoded avi videos and creating videos from frames ------------------------

# USE PYTHON SCRIPT BATCH_APPENDER_FUNCTIONAL.PY. 
# COPY SCRIPT TO FOLDER WITH FILES TO BE BATCHED AND DOUBLE CLICK. 
# IF THIS DOESNT WORK, SET FILE TO BE OPEN WITH PYTHON.EXE. 
# CHECK MYLIST.TXT FILE FOR CORRECT FILE ORDER. SOMETIMES WORKING OFF HDD RESULTS IN INCORRECT FILE LIST ORDER

#R CODE FOR BATCH APPENDING. 
setwd(fp<-"H:/1_8x8_SingleQuad_proc/1_8x8_IR/T002_C57/R_proc")
setwd(fp)
system("cd /d H:/1_8x8_SingleQuad_proc/1_8x8_IR/T002_C57/R_proc")

system(paste("(for %i in (*.avi) do @echo file '%i') > mylist.txt")) #LINE WONT RUN. COPY INTO TERMINAL AND RUN BELOW. 
system("ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.avi")


# Create Video from folder of image frames
# http://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/



# 3. VIDEO DATA PIPELINE: FFMPEG: Batch Split & PNG Thermal Overlay -------
setwd(fp<-"C:/Users/Caleb Vogt/Desktop/T004_Z1_W1")
setwd(fp)
system("cd /d C:/Users/Caleb Vogt/Desktop/T004_Z1_W1")
getwd()


#USE FOR 10 MINUTE OFT, OR OTHER SPECIFIC DURATION TEST
system("ffmpeg -ss 00:00:03 -i 1.avi -c copy -q 3 -r 15 -t 00:10:00 test.avi")

#USE FOR SPECIFIC START AND STOP TIMES. 
system("ffmpeg -i 1hour.avi -c copy -q 3 -r 10 -ss 00:06:00 -to 00:07:00 1minlight.avi")

#USE TO CHANGE VIDEO FRAMERATE ONLY MAINTAINING DURATION
system("ffmpeg -i T004_Z1_W1.avi -q 3 -r 5 -y 5fps.avi")

#Using this code on raw videos offloaded from idvr pro preserves (AVI/H264) formating
# Calculate the bitrate you need by dividing 1 GB by the video length in seconds. So, for a video of length 16:40 (1000 seconds), use a bitrate of 1000000 bytes/sec:
#   
#ffmpeg -i input.mp4 -b 1000000 output.mp4
# Additional options that might be worth considering is setting the Constant Rate Factor, which lowers the average bit rate, but retains better quality. Vary the CRF between around 18 and 24 - the lower, the higher the bitrate.
# 
# ffmpeg -i input.mp4 -vcodec libx265 -crf 20 output.mp4



# ### PNG Thermal Overlay ###
# cmd = paste('ffmpeg -i T012_a13_D1.avi -pattern_type glob -framerate 1 -i "*.png" -filter_complex overlay video_proc.avi')
# system(cmd)
# 
# cmd1 = paste('ffmpeg -i T012_a13_D1.avi -framerate 1 -i "image/*.png" \
# -filter_complex "[1:v][0:v]scale2ref=iw:ih[ovr][base]; \
# [ovr]colorchannelmixer=aa=0.7[ovrl]; [base][ovrl]overlay[v]"
# -map [v] result.mp4')
# system(cmd1)



# 4. VIDEO DATA PIPELINE: Convert Liddell Video Formats & Rescale ---------------------------
# NOTE: THIS STEP IS NECESSARY FOR RE-ENCODING THE SHITTY ASS VIDEOS THAT PYTHON OPENCV SPITS THE FUCK OUT USING
# THE SHITTY FUCKING LAVF58.20.100 VIDEO ENCODER. NO IDEA WHY THE FUCK IT DOES THIS, BUT OPENCV SHOULD BE USING THE
# LAVF58.12.100 ENCODER TO EXPORT AVI FILES IN THE FMP4 CODEC. FUCKITY FUCK!
# system("ffmpeg -i T001_C57_F174.mp4 -c copy -q 3 -r 15 -vf scale=1920:1080 T001_C57_F174.avi")

# INSTALL PACKAGES
library(stringr)

setwd(fp<-"D:/4_Liddell_2019_PROC_5FPS/T001/CONVERT") 
system("cd /d D:/4_Liddell_2019_PROC_5FPS/T001/CONVERT")
dir.create("R_proc")

fl <- list.files(fp, full.names=TRUE, pattern=".avi|.mp4")
fl.short <- list.files(fp,full.names=FALSE,pattern=".avi|.mp4")
fl.shorty <-str_remove(fl.short, ".avi|.mp4")
out_fp = "D:/4_Liddell_2019_PROC_5FPS/T001/CONVERT/R_proc"

setwd(fp)
for(i in 1:length(fl)){   #-vf scale=-1:1080 # throws error, degrades quality
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -vcodec copy -an ',  '"', #-an removes audio
               paste(out_fp, paste(fl.shorty[i], ".avi", sep=""), sep='/'),'"',sep='')
  # print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}


# 5. VIDEO DATA PIPELINE: PATHTRACKER and TRACKR for OFT ------------------

#PATHTRACKER. 
#STEP 1: INSTALL RELEVANT PACKAGES AND SET WORKING DIRECTORY

library(devtools)
library(beepr)
library(stringr)
#install_github("aharmer/pathtrackr")
library(pathtrackr)
library(ggplot2)

wd <- setwd("F:/T002_Female_MA_proc") ## RUN THIS LINE TWICE
vids <- list.files(wd, pattern = "*.avi|*.mp4") #lists avis and mp4s
vid_paths <- str_remove(vids, ".avi|.mp4") #vector without extensions

#STEP 2: CREATE NEW FOLDERS AND SPLIT VIDEOS INTO FRAMES
# STANDARD VIDEO SIZES:
# 426 x 240 (240p), 640 x 360 (360p), 854 x 480 (480p), 1280 x 720 (720p), 1920 x 1080 (1080p)
dir.create("COMPRESSED")
for (aa in 1:length(vids)) {{   #If this fails, check extra code for manual video split. 
  splitVideo(vids[aa], 15, 1280, -1) #splitVideo(file, fps, xdim, -1 = keep aspect ratio). Use 1280 xwidth for SPP vidoes that have more environmental noise. 
  comp_vid <- dir(wd, pattern = "*COMPRESSED")
  file.copy(comp_vid, paste(wd, "/COMPRESSED", sep=""))
  file.remove(comp_vid) #THIS LINE THROWS A WARNING AND DOES NOT REMOVE THE VIDEO. 
  beep(1)
  beep(1)
  beep(1) 
}
  beep(4)
}

#STEP 3: TRACK PATHS AND OUPUT CSVS AND FIGURES.
#CHECK YOUR ARENA DIMENSIONS
## OFT CHAMBER: 17.5 x 17.5" (444.5 x 444.5mm) /  18" x 18" (457.2 x 457.2mm)
## SPP CHAMBER: 23 x 8.5" (584 x 216mm)
for (bb in 1:length(vid_paths)) {{   #10:length(vid_paths)
  path.list <- trackPath(paste(wd, '/', vid_paths[bb], sep = ""), #video path
                         444.5, # arena width (mm)444.5
                         444.5, # arena height (mm)444.5
                         fps = 15, 
                         box = 2, # can mess with this, b/w 1-2. If you choose a small box, make sure to draw a larger one. 
                         jitter.damp = 0.7) # Less than 0.7 seems to have poor performance
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
}

#STEP 4: MAKE TRACKING VIDEOS. 
#CHECK YOUR ARENA DIMENSIONS
## OFT CHAMBER: 17.5 x 17.5" (444.5 x 444.5mm) /  18" x 18" (457.2 x 457.2mm)
## SPP CHAMBER: 23 x 8.5" (584 x 216mm)
dir.create("TRACKED")
for (cc in 1:length(vid_paths)) {    
  makeVideo(paste(wd, '/', vid_paths[cc], sep = ""),
            444.5, # SET ARENA DIMENSIONS
            444.5, # SET ARENA DIMENSIONS
            fps = 15, 
            box = 1.5, #try to draw box as a square. 
            jitter.damp = 0.9) #0.9 seems to yield best results
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



#TRACKR: https://swarm-lab.github.io/trackR/articles/z3_background.html
devtools::install_github("swarm-lab/trackR")

# To launch trackR using classic background subtraction, type:
library(trackR)
trackR("classic") 

# To launch trackR using quantized background subtraction, type:
library(trackR)
trackR("quanta") 

# To launch trackR using "tracktor" segmentation, type:
library(trackR)
trackR("tracktor") 


# 6. VIDEO DATA PIPELINE: LIDDELL: Create OCR output csv file -------------------------------------------
library(tesseract)
library(magick)
library(beepr)
eng <- tesseract("eng")

#Set R and system working directories
wd <- setwd("K:/4_Liddell_2019_proc_AG/Field/OCR")
system("cd /d K:/4_Liddell_2019_proc_AG/Field/OCR")
getwd()

#Create jpgs from video and list of all the generated jpgs. 
avi_list <- dir(wd, pattern = "*.avi") #Note that this code works with .avi's in mpeg4 codec
avi_list

# BEST RUN ON FOLDERS CONTAINING 10-15 VIDEOS
for (aa in 1:length(avi_list)) {
  # EXTRACT FRAMES AT SET FRAMERATE. NOTE: FRAMES CREATED = 147,435
  cmdstring <- paste("ffmpeg -i", avi_list[aa], "-vf fps=1 frame%06d.jpg") 
  # EXTRACT ALL COMPONENT FRAMES OF VIDEO
  # cmdstring <- paste("ffmpeg -i", avi_list[aa], "frame%07d.jpg") 
  # CHANGE TO SYSTEM?
  shell(cmdstring)
  jpg_list <- dir(wd, pattern ="*.jpg")
  for (bb in 1:length(jpg_list)) {
    image <- image_read(jpg_list[bb])
    # Note that this defines the crop area. Width x heighT + distance from side+distance from top. 
    # 140x15+25+430 for T004 Z4. 
    crop <- image_crop(image, "315x35+50+970") 
    txt <- tesseract::ocr(crop, engine = eng)
    # MODIFY SO FIRST COLUMN SAYS FRAME
    ocr_txt <- as.character(txt)
    if(bb==1) {
      ocr_out <- ocr_txt
    } else {
      ocr_out <- rbind(ocr_out, ocr_txt)
    }
  }
  write.csv(ocr_out, paste(avi_list[aa],"_OCR",".csv", sep = ''))
  unlink("*.jpg")
  
  beep(1)
  beep(1)
  beep(1)
}


# 7. VIDEO DATA PIPELINE: LIDDELL: OCR + BORIS Integration -------------------------------------------------
library(data.table)
library(plyr)
library(dplyr)
library(readr)
library(reshape)
library(xlsx)

# STEP 1: 
#Add _BORIS to the END of all the BORIS observation csv files. 
#Make sure folder only contains BORIS csv files before doing this. 
wd <- setwd("C:/Users/Caleb Vogt/Desktop/Liddell_ML_R_Analysis_Testing")
file.list <- list.files(wd, pattern = "*.csv")

for (i in 1:length(file.list)){
  file.rename(file.list[i], paste(file.list[i],"_BORIS",".csv", sep=''))
}
#Move boris observations and OCR outputs into the same working directory folder. 

#Overwrite previous file.list
file.list <- list.files(wd, pattern = "*.csv")
file.list

OCR_list <- dir(wd, pattern = "*_OCR*") #make sure there are no folders. 
BORIS_list <- dir(wd, pattern = "*_BORIS*")
OCR_list #Check yo shit. 
BORIS_list #Check yo shit. 



# New Machine Learning Framework Analysis 2019
# Major difference here is that I am only assessing frames where I can see the mouse. 
# need to interpolate boris rows between the 
# Need to order the timestamp column in the final master file because some fucking videos had 
# an error during the python video concatenation step, so that first days are appened at the end of the video. 
# not a problem if I reorder the timestamps from early to late. 

a=1
#for (a in 1:length(BORIS_list)) {
  #Create OCR dataframe. 
  ocr.df <-read.csv(OCR_list[a], stringsAsFactors = FALSE)
  ocr.df <- subset(ocr.df, select = -c(X))
  ocr.Frames <- rownames(ocr.df)
  ocr.df <- cbind(ocr.Frames=ocr.Frames, ocr.df)
  names(ocr.df) <- c("Video.Time", "Field.Time") #Depends on OCR FPS
  #Create boris dataframe. 
  bor.df <- read.csv(BORIS_list[a], skip=15) #Check skip #
  names(bor.df)[1]<-"Video.Time"
  bor.df[,1] <- round(bor.df[,1]) #Round Video.Time Column of _BORIS.csv
  bor.df$Video.Time<-as.numeric(as.character(bor.df$Video.Time))
  ocr.df$Video.Time<-as.numeric(as.character(ocr.df$Video.Time))
  #merge ocr and boris dataframes into master dataframe
  master<-merge(bor.df,ocr.df, by = "Video.Time", all.y = TRUE) # Orders the Video.Time column oddly. 
  master<- master[,c(1,12,2:11)]
  master<- subset(master, select=-c(Total.length, FPS)) #Drop cols by name. 
  master[c("M1", "M2", "M3", "M4", "M5","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15")] <- NA
  #Create mini dataframe
  mini <- master[order(master$Status, master$Subject), ]
  mini <- subset(mini, select=c(Video.Time, Field.Time, Subject, Status))
  START_rows <- mini[mini$Status == 'START', ]
  STOP_rows <- mini[mini$Status == 'STOP', ]
  mini <- cbind(START_rows, STOP_rows)
  mini <- na.omit(mini)
  
  
  
  #subs <- c("M1", "M2", "M3", "M4", "M5","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15")
  
#  cc=3
 # for (cc in 1:length(subs)) {
    
    
 # }
  #
  #if(master$Status == "START") {
    subject
  }
  
  master.test<-master
  
  
  apply(mini,1,function(x){
   
  })

for(q in 1:nrow(mini)){
    print(paste0("Processing ",q," out of ",nrow(mini)))
    x<-mini[q,]
    replacement1s<-c(as.numeric(x[1]):as.numeric(x[5]))
    mouse<-as.character(x$Subject)
    master[replacement1s,grep(mouse,colnames(master))]<-1
  }
  
  
  master[mini[,1]:mini[,5], which(grepl(,colnames(master))] <- 1 #in master from 
  # newtest<-master[c(7127:7133),]
  # newtest<-rbind(newtest,master[c(8837:8840),])
  # newtest$Subject[9]<-"M2"
  
  newtest<-master.test
  
  newtest$composite<-paste0(newtest$Subject,newtest$Status)
  newtest$startmouse<-ifelse(newtest$Status=="START",as.factor(as.character(newtest$Subject)),NA)
  
  
  newtest[which(newtest$Status=="START"),
          which(grepl(gsub("START","",
                           newtest$composite[which(grepl("START",newtest$composite))]),
                colnames(newtest))]<-1
  
  newtest[which(newtest$Status=="STOP"),grep(gsub("STOP","",newtest$composite[grep("STOP",newtest$composite)]),colnames(newtest))]<-0
  
  
  library(zoo)
  
  mousecolumns<-c(11:30)
  
  newtest[,mousecolumns]<-apply(newtest[,mousecolumns],2,function(x){na.locf(x,na.rm=FALSE)})
  
  #ok, how to populate subject cols with 0s and 1s?  
  # if subject + START, then add 1 to Subject col until hit subject + STOP.
  # MALES
  b=1
  for (b in 1:5) {
    ifelse(grepl("START",master$Status) & grepl(paste("M",b,sep=""),Subject))
  }
  
  #FEMALES
  
       
  

  
  #}
  sapply(DF3, class)

  
  # Extract START and STOP rows and combine into new dataframe. 

  
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
  
  
  
  
  
  
  
  
  
  
  
  


# Old Duration Analysis Framework 2018:
# Check the Boris observation files for the unnecessary metadata rows. 
# Make sure to check that the number of "STARTs" and "STOPS" are the same in the csv
# Otherwise this will likely throw an error. 
aa=1
for (aa in 1:length(BORIS_list)) {
  DF1 <- read.csv(OCR_list[aa])
  DF2 <- read.csv(BORIS_list[aa], skip = 15) #worth checking if BORIS files have 15 worthless rows
  DF2["Real_World_Time"] <- NA # Add column 
  rw_time <- DF1[ ,2] #pulls out the real world times
  # NOTE: FPS ADJUSTMENT/MULTIPLICATION FACTOR NEEDS TO BE ADDED HERE.
  BOR_time <- DF2[ ,1] #pulls out the BORIS video time in seconds
  # Rounds the BORIS video times to nearest second.
  # ADD MULTIPLICATION FACTOR HERE TO BOR_TIME?
  round_BOR_time <- round(BOR_time)   
  DF2$Real_World_Time <-rw_time[round_BOR_time] #Populates RWT col with RWT according to rounded BOR_Time frame # = row
  write.csv(DF2, paste(BORIS_list[1],"_RT",".csv", sep = ''))
}

### STOP

#Functional, but silly. # of columns must match. Change csv names. 
D1 <- read.csv("RT_BORIS_T002_Z1_W1.csv")
D2 <- read.csv("RT_BORIS_T002_Z2_W1.csv")
D3 <- read.csv("RT_BORIS_T002_Z3_W1.csv")
D4 <- read.csv("RT_BORIS_T002_Z4_W1.csv")
D5 <- read.csv("RT_BORIS_T002_Z5_W1.csv")
D6 <- read.csv("RT_BORIS_T002_Z6_W1.csv")
D7 <- read.csv("RT_BORIS_T002_Z7_W1.csv")
D8 <- read.csv("RT_BORIS_T002_Z8_W1.csv")

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
                                         Time,
                                         Media.file.path,
                                         zone,
                                         Subject,
                                         Behavior,
                                         #Modifier.1, #Ethogram specific, check
                                         #Modifier.2, #ethogram specific, check
                                         Comment,     #Ethogram specific
                                         Status,
                                         Real_World_Time))

#Write a csv here if you want START/STOP in single column. 
write.csv(short_merge, "MERGE_RT_BORIS_T002_W1.csv")

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


# 8. VIDEO DATA PIPELINE: LIDDELL: SOCIAL NETWORK: Create Adjacency matrix -------------------
#SOCIAL NETWORK ASSOCIATION MATRIX FOR ALL OF WEEK 1. 
setwd("E:/Data/3_Liddell_Ecology_proc/T002/W2/BORIS")
library(tidyverse)
library(lubridate)
library(igraph)

vog <- read.csv("DUR_MERGE_RT_BORIS_T002_W2.csv") 

#Depending on basis for the social network, need to remove unnecessary columns
# A network based on co-occupancy of the resource zones should not have columns containing nas,
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



# 9. VIDEO DATA PIPELINE: SOCIAL NETWORK: iGraph Plotting ------------------------------

#Working with the directed adjacency matrix. 
totsmagoats <- read.csv("directed_adjacency_matrix.csv")

# Directed social network 
network_df <- totsmagoats
net_graph <- graph.adjacency(network_df, mode="directed", weighted = NULL)
g <- simplify(net_graph)
V(g)$label <- V(g)$name
V(g)$degree <- degree(g)
set.seed(3952) # what does this doe. 
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


# 10. VIDEO DATA PIPELINE: LIDDELL GRAPHING --------------------------------
library(tidyverse)
library(lubridate)
library(ggplot2)
#Figure 1. Total durations of females in RZ's for the entire week. 
#Figure 2. Development of female space use over time. 
#Figure 3. Total durations of male space use in RZ's for the entire week. 
#Figure 4. Development of male space use over time. Also say which resources animals were released in. 
#add red square for release point. 
#Figure 5. 

#T002_W1 Graphing with MERGE file. 
setwd("C:/Users/caleb/Desktop/data")
merge <- read.csv("MERGE_RT_BORIS_T002_W1.csv")
merge <- subset(merge, select = -c(Comment))
merge <- na.omit(merge)
merge$Real_World_Time <- ymd_hms(merge$Real_World_Time)


caleb1 <- subset(merge, Status == 'START')
caleb2 <- subset(merge, Status == 'STOP')

caleb1$ID <- seq.int(nrow(caleb1))
caleb2$ID <- seq.int(nrow(caleb2))

caleb3 <- rbind(caleb1,caleb2) 

F1_1 <- subset(caleb3, (Subject == "F1"))
F2_1 <- subset(caleb3, (Subject == "F2"))
F3_1 <- subset(caleb3, (Subject == "F3"))
F4_1 <- subset(caleb3, (Subject == "F4"))
F5_1 <- subset(caleb3, (Subject == "F5"))
F6_1 <- subset(caleb3, (Subject == "F6"))
F7_1 <- subset(caleb3, (Subject == "F7"))
F8_1 <- subset(caleb3, (Subject == "F8"))
F9_1 <- subset(caleb3, (Subject == "F9"))
F10_1 <- subset(caleb3, (Subject == "F10"))

#T002_W1 ALL FEMALE ZONE OCCUPANCY (WEEK 1) 
#Prepare dataframes
dur1 <- read.csv("DUR_MERGE_RT_BORIS_T002_W1.csv")
dur1 <- na.omit(dur1)
dur1 <- dur1[!grepl("Vole", dur1$Subject), ] #remove voles
#Plot "FEMALE ZONE OCCUPANCY (WEEK 1)"

ggplot(dur1, aes(fill=Subject, y=duration, x=zone)) + 
  geom_bar(stat="identity") +
  labs(title = "Female Zone Occupancy (Week 1)") +
  facet_wrap(~Subject) #Creates multiple plots by subject. 



# T002_W1 Female Individual Movement Graphs 
#F1_1 Exploration
ggplot(data=F1_1, aes(x=F1_1$Real_World_Time, y=F1_1$zone, group=F1_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F1 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F2_1 Exploration
ggplot(data=F2_1, aes(x=F2_1$Real_World_Time, y=F2_1$zone, group=F2_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F2 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F3_1 Exploration
ggplot(data=F3_1, aes(x=F3_1$Real_World_Time, y=F3_1$zone, group=F3_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F3 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F4_1 Exploration
ggplot(data=F4_1, aes(x=F4_1$Real_World_Time, y=F4_1$zone, group=F4_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F4 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))


#F5_1 Exploration
ggplot(data=F5_1, aes(x=F5_1$Real_World_Time, y=F5_1$zone, group=F5_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F5 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F6_1 Exploration
ggplot(data=F6_1, aes(x=F6_1$Real_World_Time, y=F6_1$zone, group=F6_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F6 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F7_1 Exploration
ggplot(data=F7_1, aes(x=F7_1$Real_World_Time, y=F7_1$zone, group=F7_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F7 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F8_1 Exploration
ggplot(data=F8_1, aes(x=F8_1$Real_World_Time, y=F8_1$zone, group=F8_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F8 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F9_1 Exploration
ggplot(data=F9_1, aes(x=F9_1$Real_World_Time, y=F9_1$zone, group=F9_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F9 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F10_1 Exploration
ggplot(data=F10_1, aes(x=F10_1$Real_World_Time, y=F10_1$zone, group=F10_1$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F10 Week 1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))




############T002_W2 ALL FEMALE + MALE ZONE OCCUPANCY (WEEK 2)
#Prepare dataframes
dur2 <- read.csv("DUR_MERGE_RT_BORIS_T002_W2.csv")
dur2 <- subset(dur2, select = -c(Modifier.1, Modifier.2))
dur2 <- na.omit(dur2)
dur2f <- dur2[!grepl("M", dur2$Subject), ] #remove male rows
dur2m <- dur2[!grepl("F", dur2$Subject), ] #remove female rows

#Plot "FEMALE ZONE OCCUPANCY (WEEK 2)"
ggplot(dur2f, aes(fill=Subject, y=duration, x=zone)) + 
  geom_bar(stat="identity") +
  labs(title = "Female Zone Occupancy (Week 2)") +
  facet_wrap(~Subject) #Creates multiple plots by subject. 


#Plot "MALE ZONE OCCUPANCY (WEEK 2)"
ggplot(dur2m, aes(fill=Subject, y=duration, x=zone)) + 
  geom_bar(stat="identity") +
  labs(title = "Male Zone Occupancy (Week 2)") +
  facet_wrap(~Subject) #Creates multiple plots by subject. 


# male and female full week zone use
ggplot(dur2, aes(fill=Subject, y=duration, x=zone)) + 
  geom_bar(stat="identity") + 
  labs(title = "Male Zone Occupancy (Week 2") +
  facet_wrap(~Subject)




########## T002_W2 INDIVIDUAL MOVEMENT GRAPHS 
merge2 <- read.csv("MERGE_RT_BORIS_T002_W2.csv")
merge2 <- subset(merge2, select = -c(Modifier.1, Modifier.2,Comment))
merge2 <- na.omit(merge2)
merge2$Real_World_Time <- ymd_hms(merge2$Real_World_Time)


#### Prepare dataframes
alex1 <- subset(merge2, Status == 'START')
alex2 <- subset(merge2, Status == 'STOP')

alex1$ID <- seq.int(nrow(alex1))
alex2$ID <- seq.int(nrow(alex2))
alex3 <- rbind(alex1,alex2) 

#Prepare female individual data. 
F1_2 <- subset(alex3, (Subject == "F1"))
F2_2 <- subset(alex3, (Subject == "F2"))
F3_2 <- subset(alex3, (Subject == "F3"))
F4_2 <- subset(alex3, (Subject == "F4"))
F5_2 <- subset(alex3, (Subject == "F5"))
F6_2 <- subset(alex3, (Subject == "F6"))
F7_2 <- subset(alex3, (Subject == "F7"))
F8_2 <- subset(alex3, (Subject == "F8"))
F9_2 <- subset(alex3, (Subject == "F9"))
F10_2 <- subset(alex3, (Subject == "F10"))


#Prepare male individual data.
M1_2 <- subset(alex3, (Subject == "M1"))
M2_2 <- subset(alex3, (Subject == "M2"))
M3_2 <- subset(alex3, (Subject == "M3"))
M4_2 <- subset(alex3, (Subject == "M4"))
M5_2 <- subset(alex3, (Subject == "M5"))
M6_2 <- subset(alex3, (Subject == "M6"))



#F1_2 Exploration
ggplot(data=F1_2, aes(x=F1_2$Real_World_Time, y=F1_2$zone, group=F1_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F1 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F2_2 Exploration
ggplot(data=F2_2, aes(x=F2_2$Real_World_Time, y=F2_2$zone, group=F2_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F2 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F3_2 Exploration
ggplot(data=F3_2, aes(x=F3_2$Real_World_Time, y=F3_2$zone, group=F3_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F3 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F4_2 Exploration
ggplot(data=F4_2, aes(x=F4_2$Real_World_Time, y=F4_2$zone, group=F4_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F4 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))


#F5_2 Exploration
ggplot(data=F5_2, aes(x=F5_2$Real_World_Time, y=F5_2$zone, group=F5_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F5 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F6_2 Exploration
ggplot(data=F6_2, aes(x=F6_2$Real_World_Time, y=F6_2$zone, group=F6_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F6 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F7_2 Exploration
ggplot(data=F7_2, aes(x=F7_2$Real_World_Time, y=F7_2$zone, group=F7_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F7 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F8_2 Exploration
ggplot(data=F8_2, aes(x=F8_2$Real_World_Time, y=F8_2$zone, group=F8_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F8 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F9_2 Exploration
ggplot(data=F9_2, aes(x=F9_2$Real_World_Time, y=F9_2$zone, group=F9_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F9 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#F10_2 Exploration
ggplot(data=F10_2, aes(x=F10_2$Real_World_Time, y=F10_2$zone, group=F10_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="F10 Week 2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))



# T002_W2 Male Individual Movement Graphs 
#M1_W2 Exploration
ggplot(data=M1_2, aes(x=M1_2$Real_World_Time, y=M1_2$zone, group=M1_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="M1 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#M2_W2 Exploration
ggplot(data=M2_2, aes(x=M2_2$Real_World_Time, y=M2_2$zone, group=M2_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="M2 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#M3_W2 Exploration
ggplot(data=M3_2, aes(x=M3_2$Real_World_Time, y=M3_2$zone, group=M3_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="M3 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#M4_W2 Exploration
ggplot(data=M4_2, aes(x=M4_2$Real_World_Time, y=M4_2$zone, group=M4_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="M4 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#M5_W2 Exploration
ggplot(data=M5_2, aes(x=M5_2$Real_World_Time, y=M5_2$zone, group=M5_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="M5 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))

#M6_W2 Exploration
ggplot(data=M6_2, aes(x=M6_2$Real_World_Time, y=M6_2$zone, group=M6_2$ID)) +
  geom_line(size = 10) +
  labs(x="Time", y="Zone", title="M6 Exploration") +
  scale_y_discrete(limits=c(1,2,3,4,5,6,7,8))





##########Unclear what this does. 
attach(dur2)
F1wut <- aggregate(duration ~ zone, data = subset(dur2,Subject == 'F1'),sum)
F2 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'F2'),sum)
M1 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M1'),sum)
M2 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M2'),sum)
M3 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M3'),sum)
M4 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M4'),sum)
M5 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M5'),sum)
M6 <- aggregate(duration ~ zone, data = subset(dat,Subject == 'M6'),sum)

ggplot(data=F1wut, aes(x=F1$zone,y=F1$duration)) + geom_bar(stat='identity')
ggplot(data=F2, aes(x=F2$zone,y=F2$duration)) + geom_bar(stat='identity')
ggplot(data=M2, aes(x=M2$zone,y=M2$duration)) + geom_bar(stat='identity')
ggplot(data=M3, aes(x=M3$zone,y=M3$duration)) + geom_bar(stat='identity')
ggplot(data=M4, aes(x=M4$zone,y=M4$duration)) + geom_bar(stat='identity')
ggplot(data=M5, aes(x=M5$zone,y=M5$duration)) + geom_bar(stat='identity')
ggplot(data=M6, aes(x=M6$zone,y=M6$duration)) + geom_bar(stat='identity')

ggplot(data=subset(dat,Subject == "M5"), aes(x=zone,y=duration)) + geom_point()




# 11. VIDEO DATA PIPELINE: LIDDELL MOVEMENT ANIMATION -----------------------
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

#Deleting rows containing particular character element
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







# EXTRA CODE #5: R, Stats, Graphing ---------------------------------------
#RAPID GRAPHING USING ESQUISSE
library(dplyr)
library(ggplot2)
library(esquisse)




install.packages("ggstatsplot")



# EXTRA CODE #6: Update R -------------------------------------------------

install.packages("installr")
library(installr)

updateR()



# PROJECT ANALYSIS: 1_8x8_SQ_master --------------------------------------
# install.packages("googledrive")
# library(googledrive)
# drive_find("1_8x8_SQ")


library(googlesheets)
library(ggplot2)
gs_auth(new_user = TRUE)
gs_ls()
for_gs <- gs_title("1_8x8_SQ_master")
SQ_master <- gs_read(for_gs)
str(SQ_master)

ggplot(data = SQ_master) +
  aes(x = strain, y = oft.total.dist.mm) +
  geom_boxplot(fill = "#b4de2c") +
  theme_stata()


ggplot(data = SQ_master) +
  aes(x = strain, y = oft.total.dist.mm) +
  geom_boxplot(fill = "#b4de2c") +
  theme_minimal()



# PROJECT ANALYSIS: 3_Liddell_2018_master ----------------------------------------
library(googlesheets)
library(ggplot2)
gs_auth(new_user = TRUE)
# gs_ls()
for_gs <- gs_title("3_Liddell_2018_master")
liddell_master <- gs_read(for_gs)
str(liddell_master) #DISPLAY THE STRUCTURE OF THE R OBJECT

library(ggplot2)

ggplot(data = liddell_master) +
  aes(x = strain, y = `mass.%.change`) +
  geom_boxplot(fill = "#b4de2c") +
  labs(title = "Liddell 2018: Male Percent Body Mass Change",
    x = "Strain",
    y = "% Mass Change",
    subtitle = "(Post-Field mass / Pre- Field Mass)") +
  theme_stata()
as.numeric(liddell_master$testes.library(ggplot2)

           
           

           
           
ggplot(data = liddell_master) +
  aes(x = strain, y = testes.g) +
  geom_boxplot(fill = "#b4de2c") +
  labs(title = "Liddell 2018: Male Testes Mass",
    x = "Strain",
    y = "Testes Mass (g)") +
  theme_stata()
g)

library(ggplot2)

ggplot(data = liddell_master) +
  aes(x = strain, y = oft.dist.m) +
  geom_boxplot(fill = "#b4de2c") +
  labs(title = "OFT: Male Distance Travelled",
    x = "Strain",
    y = "Distance (m)") +
  theme_stata()

ggplot(data = liddell_master) +
  aes(x = strain, y = oft.mass) +
  geom_boxplot(fill = "#b4de2c") +
  labs(title = "OFT: Male Mass",
    x = "Strain",
    y = "Mass (g)") +
  theme_stata()




ggplot(data = liddell_master) +
  aes(x = strain, y = oft.total.dist.mm) +
  geom_boxplot(fill = "#b4de2c") +
  labs(title = "Open Field Trial",
       x = "Strain",
       y = "Distance (mm)",
       subtitle = "Liddell 2018") +
  theme_minimal()

library(ggplot2)

ggplot(data = liddell_master) +
  aes(x = strain, y = oft.mass) +
  geom_boxplot(fill = "#b4de2c") +
  labs(title = "Liddell 2018: Body Mass",
       x = "Mouse Strain",
       y = "Mass (g)") +
  theme_stata()



# PROJECT ANALYSIS: 4_OXT_Liddell_master ----------------------------------
library(googlesheets)
library(ggplot2)
# gs_auth(new_user = TRUE)
# gs_ls()
for_gs <- gs_title("4_OXT_Liddell_master")
oxt_master <- gs_read(for_gs)
str(oxt_master) #DISPLAY THE STRUCTURE OF THE R OBJECT




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
