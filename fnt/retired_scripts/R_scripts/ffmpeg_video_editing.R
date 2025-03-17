## VIDEO_EDITING_v.2.R
## Caleb C. Vogt, Ph.D. Student Cornell University
## Updated on 3.19.2020


# RENAME FILES ------------------------------------------------------------


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


# BATCH ENCODE TIMESTAMPS  -------------------------
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





setwd(fp<-"Z:/7_Liddell_2020/Video_Data/T007") #change this
system("cd /d Z:")
system("cd /d Z:/7_Liddell_2020/Video_Data/T007")  #change this
getwd()
dir.create("R_proc")

fl <- list.files(fp, full.names=TRUE, pattern=".avi")
#View(fl)
fl.short<-list.files(fp,full.names=FALSE,pattern=".avi")
out_fp="Z:/7_Liddell_2020/Video_Data/T007/R_proc"  #change this

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

# BATCH RE-ENCODE VIDEOS ----------------------------------------------------

setwd(fp<-"C:/Users/Caleb Vogt/Desktop/WASPER") #change this
system("cd /d C:")
system("cd /d C:/Users/Caleb Vogt/Desktop/WASPER")  #change this
getwd()
dir.create("R_proc")

fl <- list.files(fp, full.names=TRUE, pattern=".mp4")
#View(fl)
fl.short<-list.files(fp,full.names=FALSE,pattern=".mp4")
out_fp="C:/Users/Caleb Vogt/Desktop/WASPER/R_proc"  #change this
fl.out<- gsub("*.mp4", ".avi",fl.short) #if you want to change the output video type

setwd(fp)
for(i in 1:length(fl)){ #Changes codec from H264 -> mpeg4
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -an ', #-an removes audio from video. -vcodec rawvideo allows video to be opened as imageJ stack. 
               ' -q 3 -r 15 -max_muxing_queue_size 100000 ',  '"', #-q should be set to 3. -r should be set to 10 fps
               paste(out_fp,fl.out[i],sep='/'),'"',sep='')
  print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}

cd(old_fp)

#  -q 3 -r 10 -vcodec rawvideo -max_muxing_queue_size 100000 This reduces the speed of the conversion by almost an entire factor. 
# without get FPS ~ 450. With it, FPS ~25


#Change video fps without altering video duration
#ffmpeg -i input.mov -r 24 -y output.mov



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

# BATCH APPEND VIDEOS ------------------------

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



# BATCH TRIM VIDEOS -------
setwd(fp<-"C:/Users/Caleb Vogt/Desktop/NEST_POSTURE")
setwd(fp)
system("cd /d C:/Users/Caleb Vogt/Desktop/NEST_POSTURE")
getwd()


system("ffmpeg -ss 00:14:21 -i DSC_8562.mov -c copy -q 3 -r 30 -t 00:05:00 DSC_8562_1.avi")

system("ffmpeg -ss 00:19:22 -i DSC_8562.mov -c copy -q 3 -r 30 -t 00:05:25 DSC_8562_2.avi")

system("ffmpeg -ss 00:24:48 -i DSC_8562.mov -c copy -q 3 -r 30 -t 00:05:00 DSC_8562_3.avi")



system("ffmpeg -ss 00:07:11 -i DSC_8565.mov -c copy -q 3 -r 30 -t 00:05:00 DSC_8565_1.avi")

system("ffmpeg -ss 00:12:22 -i DSC_8565.mov -c copy -q 3 -r 30 -t 00:05:25 DSC_8565_2.avi")

system("ffmpeg -ss 00:17:48 -i DSC_8565.mov -c copy -q 3 -r 30 -t 00:05:00 DSC_8565_3.avi")



system("ffmpeg -ss 00:06:54 -i DSC_8566.mov -c copy -q 3 -r 30 -t 00:05:00 DSC_8566_1.avi")

system("ffmpeg -ss 00:11:55 -i DSC_8566.mov -c copy -q 3 -r 30 -t 00:05:25 DSC_8566_2.avi")

system("ffmpeg -ss 00:17:21 -i DSC_8566.mov -c copy -q 3 -r 30 -t 00:05:00 DSC_8566_3.avi")



system("ffmpeg -ss 00:00:22 -i DSC_8569.mov -c copy -q 3 -r 30 -t 00:05:00 DSC_8569_1.avi")

system("ffmpeg -ss 00:05:23 -i DSC_8569.mov -c copy -q 3 -r 30 -t 00:05:25 DSC_8569_2.avi")

system("ffmpeg -ss 00:10:49 -i DSC_8569.mov -c copy -q 3 -r 30 -t 00:05:00 DSC_8569_3.avi")



system("ffmpeg -ss 00:05:52 -i DSC_8571.mov -c copy -q 3 -r 30 -t 00:05:00 DSC_8571_1.avi")

system("ffmpeg -ss 00:10:53 -i DSC_8571.mov -c copy -q 3 -r 30 -t 00:05:25 DSC_8571_2.avi")

system("ffmpeg -ss 00:16:19 -i DSC_8571.mov -c copy -q 3 -r 30 -t 00:05:00 DSC_8571_3.avi")



system("ffmpeg -ss 00:09:12 -i DSC_8572.mov -c copy -q 3 -r 30 -t 00:05:00 DSC_8572_1.avi")

system("ffmpeg -ss 00:14:13 -i DSC_8572.mov -c copy -q 3 -r 30 -t 00:05:25 DSC_8572_2.avi")

system("ffmpeg -ss 00:19:39 -i DSC_8572.mov -c copy -q 3 -r 30 -t 00:05:00 DSC_8572_3.avi")



system("ffmpeg -ss 00:05:07 -i DSC_8573.mov -c copy -q 3 -r 30 -t 00:05:00 DSC_8573_1.avi")

system("ffmpeg -ss 00:10:08 -i DSC_8573.mov -c copy -q 3 -r 30 -t 00:05:25 DSC_8573_2.avi")

system("ffmpeg -ss 00:15:34 -i DSC_8573.mov -c copy -q 3 -r 30 -t 00:05:00 DSC_8573_3.avi")



system("ffmpeg -ss 00:04:15 -i DSC_8574.mov -c copy -q 3 -r 30 -t 00:05:00 DSC_8574_1.avi")

system("ffmpeg -ss 00:09:16 -i DSC_8574.mov -c copy -q 3 -r 30 -t 00:05:25 DSC_8574_2.avi")

system("ffmpeg -ss 00:14:42 -i DSC_8574.mov -c copy -q 3 -r 30 -t 00:05:00 DSC_8574_3.avi")


#USE FOR 10 MINUTE OFT, OR OTHER SPECIFIC DURATION TESTS
system("ffmpeg -ss 00:00:03 -i WASPER_1.avi -c -filter:v "setpts=2.0*PTS" copy -q 3 -r 30 -t 00:01:00 RI.avi")


#USE FOR 10 MINUTE OFT, OR OTHER SPECIFIC DURATION TESTS
system("ffmpeg -ss 00:00:00 -i test.mp4 -c copy -q 3 -r 20  -t 00:00:15 sample2.avi")

system("ffmpeg -ss 00:00:00 -i test.mp4 -c copy -q 3 -r 20  -t 00:00:15 sample2.avi")


#USE FOR SPECIFIC START AND STOP TIMES. 
system("ffmpeg -i 1hour.avi -c copy -q 3 -r 10 -ss 00:06:00 -to 00:07:00 1minlight.avi")

#USE FOR SPECIFIC START TIME ONLY
system("ffmpeg -i WASPER_1.avi -c copy -q 3 -r 15 -ss 00:07:52 WASPER_1_short.avi")

system("ffmpeg -i WASPER_2.avi -c copy -q 3 -r 15 -ss 00:09:06 WASPER_2_short.avi")


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





# BATCH CONVERT VIDEO FORMAT AND SCALING ---------------------------


# NOTE: THIS STEP IS NECESSARY FOR RE-ENCODING THE SHITTY ASS VIDEOS THAT PYTHON OPENCV SPITS THE FUCK OUT USING
# THE SHITTY FUCKING LAVF58.20.100 VIDEO ENCODER. NO IDEA WHY THE FUCK IT DOES THIS, BUT OPENCV SHOULD BE USING THE
# LAVF58.12.100 ENCODER TO EXPORT AVI FILES IN THE FMP4 CODEC. FUCKITY FUCK!
# system("ffmpeg -i T001_C57_F174.mp4 -c copy -q 3 -r 15 -vf scale=1920:1080 T001_C57_F174.avi")
#system('ffmpeg -i input.avi -b:v 2000k -r 8 -q 2 -vf scale=960:720 720_r.avi')



# INSTALL PACKAGES
library(stringr)

setwd(fp<-"E:/3_Liddell_2018/T002/W3") 
system("cd /d E:/3_Liddell_2018/T002/W3")
dir.create("R_proc")

fl <- list.files(fp, full.names=TRUE, pattern=".avi|.mp4")
fl.short <- list.files(fp,full.names=FALSE,pattern=".avi|.mp4")
fl.shorty <-str_remove(fl.short, ".avi|.mp4")
out_fp = "E:/3_Liddell_2018/T002/W3/R_proc"

setwd(fp)
#Loop #1
for(i in 1:length(fl)){   #-vf scale=-1:1080 # throws error, degrades quality
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -vcodec copy -an ',  '"', #-an removes audio
               paste(out_fp, paste(fl.shorty[i], ".avi", sep=""), sep='/'),'"',sep='')
  # print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}

#Loop #2, downgrade quality. 
for(i in 1:length(fl)){   #-vf scale=-1:1080 # throws error, degrades quality #-an removes audio
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -b:v 2000k -r 8 -q 2 -vf scale=960:720 -an ',  '"', 
               paste(out_fp, paste(fl.shorty[i], ".avi", sep=""), sep='/'),'"',sep='')
  # print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}







