## ffmpeg_video_editing.R
## Caleb C. Vogt, Ph.D. Candidate


# BATCH CONVERT VIDEOS/GRAYSCALE/RESIZE/SCALE/SEEKABLE/DOWNSAMPLE/NO CODEC CHANGE ---------------------------
# RUN CODE DIRECTLY ON DATA DRIVE CONNECTED VIA USB TO THE COMPUTER. 
# RUNNING FROM A SERVER IS SENSTIVIE TO ETHERNET DROP OUTS! RUN LOCALLY. 


# https://superuser.com/questions/908280/what-is-the-correct-way-to-fix-keyframes-in-ffmpeg-for-dash

library(stringr)
setwd(fp<-"Z:/7_LID_2020/Video_Data/video_analysis_v2/T001") #### CHANGE THIS
system("cd /d Z:/7_LID_2020/Video_Data/video_analysis_v2/T001") #### CHANGE THIS
dir.create("R_proc2")

fl <- list.files(fp, full.names=TRUE, pattern=".avi|.mp4|.mov")
fl.short <- list.files(fp,full.names=FALSE,pattern=".avi|.mp4|.mov")
fl.shorty <-str_remove(fl.short, ".avi|.mp4|.mov")
out_fp="Z:/7_LID_2020/Video_Data/video_analysis_v2/T001/R_proc2"  #change this

setwd(fp)
## 8x8 videos should be downsampled to 20fps. 

## set frame rate -q -crf lower crf values correspond to higher bitrates. 

for(i in 1:length(fl)){ #Changes codec from H264 -> mpeg4
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -an -r 20 -crf 30 -vf "scale=1920:1080:force_original_aspect_ratio=decrease:eval=frame,pad=1920:1080:-1:-1:color=black,format=gray,subtitles=',
               substr(as.character(fl.short[i]),1,(nchar(fl.short[i])-4)),
               '.smi',
               ":force_style='",
               'FontSize=10,Alignment=1,BorderStyle=3,Outline=1,Shadow=0,MarginV=20" ',
               ' -max_muxing_queue_size 10000000 ',  '"', 
               paste(out_fp, paste(fl.shorty[i], ".mp4", sep=""), sep='/'),'"',sep='')
  # print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}



# BATCH CONVERT SEEKABLE/DOWNSAMPLE/H.265 COMPRESSION ---------------------------
# RUN CODE DIRECTLY ON DATA DRIVE CONNECTED VIA USB TO THE COMPUTER. 
# RUNNING FROM A SERVER IS SENSTIVIE TO ETHERNET DROP OUTS! RUN LOCALLY. 

library(stringr)
setwd(fp<-"Z:/1_8x8_SQ_2018/SQ_8x8_2018_analysis_v3/1_8x8/batch_1") #### CHANGE THIS
system("cd /d Z:/1_8x8_SQ_2018/SQ_8x8_2018_analysis_v3/1_8x8/batch_1") #### CHANGE THIS
dir.create("R_proc1")

fl <- list.files(fp, full.names=TRUE, pattern=".avi|.mp4|.mov")
fl.short <- list.files(fp,full.names=FALSE,pattern=".avi|.mp4|.mov")
fl.shorty <-str_remove(fl.short, ".avi|.mp4|.mov")
out_fp="Z:/1_8x8_SQ_2018/SQ_8x8_2018_analysis_v3/1_8x8/batch_1/R_proc1"  #change this

setwd(fp)
#-q should be set to 3. -r should be set to 10 or 15 fps for 8x8 long videos. Anything with fighting should be boosted to minimum 25fps 

for(i in 1:length(fl)){ #Changes codec from H264 -> mpeg4
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -vcodec libx265 -preset ultrafast -crf 26 -pix_fmt yuv420p -an -r 10 -max_muxing_queue_size 10000000 ',
               paste(out_fp, paste(fl.shorty[i], ".mp4", sep=""), sep='/'),'"',sep='')
  # print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}

##############################

# BATCH CONVERT VIDEOS/GRAYSCALE/RESIZE/SCALE/SEEKABLE/DOWNSAMPLE/H.265 COMPRESSION ---------------------------
# RUN CODE DIRECTLY ON DATA DRIVE CONNECTED VIA USB TO THE COMPUTER. 
# RUNNING FROM A SERVER IS SENSTIVIE TO ETHERNET DROP OUTS! RUN LOCALLY. 

library(stringr)
setwd(fp<-"Z:/7_LID_2020/Video_Data/video_analysis_v2/T001") #### CHANGE THIS
system("cd /d Z:/7_LID_2020/Video_Data/video_analysis_v2/T001") #### CHANGE THIS
dir.create("R_proc1")

fl <- list.files(fp, full.names=TRUE, pattern=".avi|.mp4|.mov")
fl.short <- list.files(fp,full.names=FALSE,pattern=".avi|.mp4|.mov")
fl.shorty <-str_remove(fl.short, ".avi|.mp4|.mov")
out_fp="Z:/7_LID_2020/Video_Data/video_analysis_v2/T001/R_proc1"  #change this

setwd(fp)
#-q should be set to 3. -r should be set to 10 or 15 fps for 8x8 long videos. Anything with fighting should be boosted to minimum 25fps 

for(i in 1:length(fl)){ #Changes codec from H264 -> mpeg4
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -vcodec libx265 -preset ultrafast -crf 26 -pix_fmt yuv420p -an -vf "scale=1920:1080:force_original_aspect_ratio=decrease:eval=frame,pad=1920:1080:-1:-1:color=black,format=gray,subtitles=', #-r 
               substr(as.character(fl.short[i]),1,(nchar(fl.short[i])-4)),
               '.smi',
               ":force_style='",
               'FontSize=10,Alignment=1,BorderStyle=3,Outline=1,Shadow=0,MarginV=20" ',
               ' -max_muxing_queue_size 10000000 ',  '"', 
               paste(out_fp, paste(fl.shorty[i], ".mp4", sep=""), sep='/'),'"',sep='')
  # print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}


############################
# BATCH CONVERT VIDEOS TO HAVE RELIABLE FRAME ACCURATE SEEKING ---------------------------
# NOTE THAT THERE IS NO -q or -b:v ARGUMENT INCLUDED HERE. I HAVE TESTED THESE AND THESE DO NOT DECREASE THE FILE SIZE. 
library(stringr)

setwd(fp<-"Z:/15_Caitlin_Female_Preference_Tracking") #### CHANGE THIS
system("cd /d Z:/15_Caitlin_Female_Preference_Tracking") #### CHANGE THIS
dir.create("R_proc")

fl <- list.files(fp, full.names=TRUE, pattern=".avi|.mp4|.mov")
fl.short <- list.files(fp,full.names=FALSE,pattern=".avi|.mp4|.mov")
fl.shorty <-str_remove(fl.short, ".avi|.mp4|.mov")
out_fp = "Z:/15_Caitlin_Female_Preference_Tracking/R_proc"  #### CHANGE THIS

for(i in 1:length(fl)){   
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -c:v libx264 -preset superfast -crf 21 -pix_fmt yuv420p ',  '"', 
               paste(out_fp, paste(fl.shorty[i], ".mp4", sep=""), sep='/'),'"',sep='')
  # print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}


# BATCH CONVERT VIDEOS/CONVERT ALL TO GRAYSCALE/RESIZE WHILE MAINTINAING ASPECT RATIO/ AND MAKE SEEKABLE ---------------------------
# NOTE THAT THERE IS NO -q or -b:v ARGUMENT INCLUDED HERE. I HAVE TESTED THESE AND THESE DO NOT DECREASE THE FILE SIZE. 
library(stringr)

setwd(fp<-"C:/Users/sheehan-lab/Desktop/AD_moth_analysis_v2") #### CHANGE THIS
system("cd /d C:/Users/sheehan-lab/Desktop/AD_moth_analysis_v2") #### CHANGE THIS
dir.create("R_proc")

fl <- list.files(fp, full.names=TRUE, pattern=".avi|.mp4|.mov")
fl.short <- list.files(fp,full.names=FALSE,pattern=".avi|.mp4|.mov")
fl.shorty <-str_remove(fl.short, ".avi|.mp4|.mov")
out_fp = "C:/Users/sheehan-lab/Desktop/AD_moth_analysis_v2/R_proc"  #### CHANGE THIS
i=1
for(i in 1:length(fl)){   
  # cmdstr=paste('ffmpeg -i "', fl.short[i], '" -vf format=gray -vf scale=1280:960,setdar=4:3 ',  '"', 
  #              paste(out_fp, paste(fl.shorty[i], ".mp4", sep=""), sep='/'),'"',sep='')
  # print(cmdstr)#display in command window
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -vf format=gray -vf scale=1280:960:force_original_aspect_ratio=decrease:eval=frame,pad=1280:960:-1:-1:color=black -c:v libx264 -preset superfast -crf 21 -pix_fmt yuv420p ',  '"', 
               paste(out_fp, paste(fl.shorty[i], ".mp4", sep=""), sep='/'),'"',sep='')
  
  system(cmdstr) #send to windows
}


################# BATCH Trim videos to specific time
library(stringr)

setwd(fp<-"C:/Users/sheehan-lab/Desktop/1_CALEB/UNTRIMMED") 
system("cd /d C:/Users/sheehan-lab/Desktop/1_CALEB/UNTRIMMED")
dir.create("R_proc")

fl <- list.files(fp, full.names=TRUE, pattern=".avi|.mp4|.mov")
fl.short <- list.files(fp,full.names=FALSE,pattern=".avi|.mp4|.mov")
fl.shorty <-str_remove(fl.short, ".avi|.mp4|.mov")
out_fp = "C:/Users/sheehan-lab/Desktop/1_CALEB/UNTRIMMED/R_proc"

for(i in 1:length(fl)){   
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -q 3 -r 30 -t 00:15:00 ',  '"', 
               paste(out_fp, paste(fl.shorty[i], ".mp4", sep=""), sep='/'),'"',sep='')
  print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}

# BATCH SPLIT VIDEOS INTO PRE-SET TIME BATCHES ---------------------------
# NOTE THAT THERE IS NO -q or -b:v ARGUMENT INCLUDED HERE. I HAVE TESTED THESE AND THESE DO NOT DECREASE THE FILE SIZE. 
library(stringr)

setwd(fp<-"Z:/1_8x8_SQ_2018/SQ_8x8_2018_analysis_v2") #### CHANGE THIS
system("cd /d Z:/1_8x8_SQ_2018/SQ_8x8_2018_analysis_v2") #### CHANGE THIS
dir.create("R_proc")

fl <- list.files(fp, full.names=TRUE, pattern=".avi|.mp4|.mov")
fl.short <- list.files(fp,full.names=FALSE,pattern=".avi|.mp4|.mov")
fl.shorty <-str_remove(fl.short, ".avi|.mp4|.mov")
out_fp = "Z:/1_8x8_SQ_2018/SQ_8x8_2018_analysis_v2/R_proc"  #### CHANGE THIS

i=1
for(i in 1:length(fl)){   
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -c copy -map 0 -segment_time 01:00:00 -f segment ',  '"', 
               paste(out_fp, paste(fl.shorty[i], "_%02d.mp4", sep=""), sep='/'),'"',sep='')
  # print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}



# BATCH CONVERT VIDEO FORMAT AND SCALE VIDEOS TO 720P ---------------------------
library(stringr)

setwd(fp<-"Z:/14_James_Wasp_Tracking/4_wasps_15min") 
system("cd /d Z:/14_James_Wasp_Tracking/4_wasps_15min")
dir.create("R_proc")

fl <- list.files(fp, full.names=TRUE, pattern=".avi|.mp4|.mov")
fl.short <- list.files(fp,full.names=FALSE,pattern=".avi|.mp4|.mov")
fl.shorty <-str_remove(fl.short, ".avi|.mp4|.mov")
out_fp = "Z:/14_James_Wasp_Tracking/4_wasps_15min/R_proc"


#Loop #2, downgrade quality. 
for(i in 1:length(fl)){   #-vf scale=-1:1080 # throws error, degrades quality #-an removes audio
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -b:v 2000k -r 30 -q 5 -vf scale=960:720 -an ',  '"', 
               paste(out_fp, paste(fl.shorty[i], ".avi", sep=""), sep='/'),'"',sep='')
  # print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}


# CONVERT VIDEO FILE TYPE AND MAINTAIN FPS PROPERTIES -------------------------------------------------


setwd(fp<-"Z:/14_James_Wasp_Tracking/3_wasps_full") #change this
system("cd /d Z:")
system("cd /d Z:/14_James_Wasp_Tracking/3_wasps_full")  #change this
getwd()
dir.create("R_proc")
library(beepr)

fl <- list.files(fp, full.names=TRUE, pattern=".mov")
#View(fl)

fl.short<-list.files(fp,full.names=FALSE,pattern=".mov")

out_fp="Z:/14_James_Wasp_Tracking/3_wasps_full/R_proc"  #change this

fl.out<- gsub("*.mov", ".avi",fl.short) #if you want to change the output video type. See line 18

setwd(fp)

for(i in 1:length(fl)){ #Changes codec from H264 -> mpeg4
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -an ', #-an removes audio from video. -vcodec rawvideo allows video to be opened as imageJ stack. 
               ' -q 3 -r 30 -c:v libx264 -preset superfast -pix_fmt yuv420p -max_muxing_queue_size 100000 ',  '"', #-q should be set to 3. -r should be set to 10 fps
               paste(out_fp, fl.out[i],sep='/'),'"',sep='') # CHANGE FL.SHORT --> TO FL.OUT IF YOU WANT TO CHANGE THE FILE TYPE
  print(cmdstr)#display in command window
  system(cmdstr) #send to windows
  beep(1)
  beep(1)
  beep(1)
  beep(1)
  
}

cd(old_fp)


# BATCH RENAME FILE NAMES ------------------------------------------------------------
wd <- setwd("E:/1_8x8_OFTs/1_8x8_OFTs/project_folder/csv/input_csv")
file.list <- list.files(wd, pattern = "*.csv")

#Use this to remove *DLC and all characters afterwards!
for (i in 1:length(file.list)){
  file.rename(file.list[i], paste(gsub("DLC.*","", file.list[i]),".csv", sep=''))
}



for (i in 1:length(file.list)){
  file.rename(file.list[i], paste(file.list[i],"_BORIS",".csv", sep=''))
}



gsub("DLC.*","", file.list)

gsub(".*_", "", file.list) #for removing any character prior

file.rename(filename,gsub(replace,with,tolower(filename)))


# BATCH HARDCODE SMI TIMESTAMPS ONTO VIDEOS  -------------------------

#Remember you can batch convert all of your video files at once on an external HDD, but not append using the python script.
#Then place the converted videos you want to append in the same folder and copy to Desktop SDD. Go to step 2. 
# FOR OPEN FIELD TRIALS, SET FPS TO AT LEAST 15 FPS. 
# Need to change bitrate = 2000, -r to 8, and scale= 960x720
# For understanding SMI files. 
# https://docs.microsoft.com/en-us/previous-versions/windows/desktop/dnacc/understanding-sami-1.0


setwd(fp<-"Z:/7_Liddell_2020/Video_Data/T006") #change this
system("cd /d Z:")
system("cd /d Z:/7_Liddell_2020/Video_Data/T006")  #change this
getwd()
dir.create("R_proc")

fl <- list.files(fp, full.names=TRUE, pattern=".avi")
#View(fl)
fl.short<-list.files(fp,full.names=FALSE,pattern=".avi")
out_fp="Z:/7_Liddell_2020/Video_Data/T006/R_proc"  #change this

setwd(fp)
#-q should be set to 3. -r should be set to 10 or 15 fps for 8x8 long videos. Anything with fighting should be boosted to minimum 25fps 
for(i in 1:length(fl)){ #Changes codec from H264 -> mpeg4
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -an -vf "subtitles=', #-an removes audio from video
               substr(as.character(fl.short[i]),1,(nchar(fl.short[i])-4)),
               '.smi',
               ":force_style='",
               'FontSize=10,Alignment=1,BorderStyle=3,Outline=1,Shadow=0,MarginV=20" ',
               '-q 3 -r 30 -max_muxing_queue_size 100000 ',  '"', 
               paste(out_fp,fl.short[i],sep='/'),'"',sep='')
  # print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}

cd(old_fp)

#TO ADD: SHOULD BE ABLE TO INCLUDE A LINE AT THE END OF THIS FOR LOOP TO APPEND THE VIDEO AFTER EACH ITERATION. 
# Code to batch modify videos but without adding subtitles


# BATCH RE-ENCODE VIDEO CONTAINERS AND FILE TYPE ----------------------------------------------------

setwd(fp<-"Z:/2_8x8_KW_2018/8x8_RAW") #change this
system("cd /d Z:")
system("cd /d Z:/2_8x8_KW_2018/8x8_RAW")  #change this
getwd()
dir.create("R_proc")
library(beepr)

fl <- list.files(fp, full.names=TRUE, pattern=".avi")
#View(fl)
fl.short<-list.files(fp,full.names=FALSE,pattern=".avi")
out_fp="Z:/2_8x8_KW_2018/8x8_RAW/R_proc"  #change this
# fl.out<- gsub("*.mp4", ".avi",fl.short) #if you want to change the output video type

setwd(fp)
for(i in 1:length(fl)){ #Changes codec from H264 -> mpeg4
  cmdstr=paste('ffmpeg -i "', 
               fl.short[i],
               '" -an ', #-an removes audio from video. -vcodec rawvideo allows video to be opened as imageJ stack. 
               ' -q 3 -r 15 -filter:v "crop=1200:1080:450:800" -max_muxing_queue_size 100000 ',
               '"', #-q should be set to 3. -r should be set to 10 fps
               paste(out_fp, fl.short[i],sep='/'),'"',sep='')
  print(cmdstr)#display in command window
  system(cmdstr) #send to windows
  beep(1)
  beep(1)
  beep(1)
  beep(1)
  
}

cd(old_fp)





# BATCH CONCATENATE/APPEND VIDEOS ------------------------

# USE PYTHON SCRIPT BATCH_APPENDER_FUNCTIONAL.PY. THIS IS AN ALTERNATIVE. 
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



# BATCH TRIM VIDEOS TO SPECIFIC TIMES-------
setwd(fp<-"C:/Users/Caleb Vogt/Desktop/NEST_POSTURE")
setwd(fp)
system("cd /d C:/Users/Caleb Vogt/Desktop/NEST_POSTURE")
getwd()


system("ffmpeg -ss 00:14:21 -i DSC_8562.mov -c copy -q 3 -r 30 -t 00:05:00 DSC_8562_1.avi")

#USE FOR 10 MINUTE OFT, OR OTHER SPECIFIC DURATION TESTS
system("ffmpeg -ss 00:00:03 -i WASPER_1.avi -c -filter:v "setpts=2.0*PTS" copy -q 3 -r 30 -t 00:01:00 RI.avi")


#USE FOR 10 MINUTE OFT, OR OTHER SPECIFIC DURATION TESTS
system("ffmpeg -ss 00:00:00 -i test.mp4 -c copy -q 3 -r 20  -t 00:00:15 sample2.avi")


#USE FOR SPECIFIC START AND STOP TIMES. 
system("ffmpeg -i 1hour.avi -c copy -q 3 -r 10 -ss 00:06:00 -to 00:07:00 1minlight.avi")

#USE FOR SPECIFIC START TIME ONLY
system("ffmpeg -i WASPER_1.avi -c copy -q 3 -r 15 -ss 00:07:52 WASPER_1_short.avi")


#USE TO CHANGE VIDEO FRAMERATE ONLY MAINTAINING DURATION
system("ffmpeg -i T004_Z1_W1.avi -q 3 -r 5 -y 5fps.avi")

system("ffmpeg -i input.avi -vf fps=1 frame%06d.jpg")

# Set bit rate and scale. 
system("ffmpeg -i input.avi -b:v 1742k -q 3 -vf scale=640:480 output.avi")

system("ffmpeg -i input.avi -b:v 1742k -q 2 -vf scale=960:720 720.avi")

system("ffmpeg -i input.avi -b:v 2000k -vf scale=960:720 720_0.avi")

system("ffmpeg -i input.avi -b:v 2000k -vf fps=8 -vf scale=960:720 720_8.avi")

system('ffmpeg -i input.avi -b:v 2000k -vf "fps=8, scale=960:720" 720_test.avi')

system('ffmpeg -i input.avi -b:v 2000k -r 8 -q 2 -vf scale=960:720 720_r.avi')


#Some of the most common resolutions that can be used for each aspect ratio are the following: 
#4:3 aspect ratio resolutions: 640?480, 800?600, 960?720, 1024?768, 1280?960, 1400?1050, 1440?1080, 
#1600?1200, 1856?1392, 1920?1440, and 2048?1536.


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




# BATCH TRIM VIDEOS (PLOTINE CODE) #############

#This script was written by Plotine Jardat in July 2019
#Aim is to split multiple videos at a time into D1 and D2
#based on split times indicated on a csv file, using ffmpeg
#This script is based on lines written by Caleb Vogt, difference is here it is automated
#Don't forget to adapt working directory and any file names

setwd("H:/1_8x8_SingleQuad_proc/1_8x8_IR")
splitting_times = read.table('Splitting_times.csv', header=T, sep = ",")
rows = t(splitting_times["trial"])
splitting_times = splitting_times[,2:dim(splitting_times)[2]] #this removes the column with the trial names (that are now used as row names)
rownames(splitting_times) = rows

subunits = c('D1','D2') #list here the subunits to cut full video into
#has to correspond to the beginning of column names in the csv

for (trial in rownames(splitting_times)){ #for each line of the csv
  line = splitting_times[trial,] #read the line
  strain =(line[['strain']]) #extract the strain
  full_path = paste(getwd(),'/',trial,'_',strain,'/',trial,'_',strain,'_FULL.avi', sep='') #make up the path to the video to be split
  for (day in subunits){ #for each subunit of the video
    cut_path = paste(getwd(),'/',trial,'_',strain,'/',trial,'_',strain,'_',day,'.avi',sep='') #make up the path of the new video
    start=paste(day,'.start',sep='') #make up the column name where start of the video is listed
    stop=paste(day,'.stop',sep='') # same for end
    system(paste('ffmpeg -i ',full_path,' -c copy -q 3 -r 15 -ss ',line[[start]],' -to ',line[[stop]],' ',cut_path,sep='')) #call ffmpeg
  }
}





