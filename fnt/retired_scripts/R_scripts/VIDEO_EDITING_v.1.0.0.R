## VIDEO_EDITING_v.1.0.0.R
## Caleb C. Vogt, Ph.D Student Cornell University
## Updated on 10.10.2019


# BATCH ENCODE TIMESTAMPS AND ADJUST VIDEO FRAMERATE/QUALITY  -------------------------
#Remember you can batch convert all of your video files at once on an external HDD, but not append using the python script.
#Then place the converted videos you want to append in the same folder and copy to Desktop SDD. Go to step 2. 
# FOR OPEN FIELD TRIALS, SET FPS TO AT LEAST 15 FPS. 
# Need to change bitrate = 2000, -r to 8, and scale= 960x720

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

# Batch appending hardcoded avi videos and creating videos from frames ------------------------

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



# FFMPEG: Batch Split & PNG Thermal Overlay -------
setwd(fp<-"C:/Users/Caleb Vogt/Desktop/Pee_Validation")
setwd(fp)
system("cd /d C:/Users/Caleb Vogt/Desktop/Pee_Validation")
getwd()


#USE FOR 10 MINUTE OFT, OR OTHER SPECIFIC DURATION TEST
system("ffmpeg -ss 00:00:03 -i 1.avi -c copy -q 3 -r 15 -t 00:10:00 test.avi")



system("ffmpeg -ss 05:28:25 -i T003_A13_D1.avi -c copy -q 1 -r 1 -t 00:30:00 T003_A13_D1_trim.avi")



system("ffmpeg -ss 03:55:38 -i T003_A24_D1.avi -c copy -q 1 -r 1 -t 00:30:00 T003_A24_D1_trim.avi")


system("ffmpeg -ss 05:48:59 -i T004_A13_D1.avi -c copy -q 1 -r 1 -t 00:30:00 T004_A13_D1_trim.avi")




system("ffmpeg -ss 05:33:55 -i T004_A24_D1.avi -c copy -q 1 -r 1 -t 00:30:00 T004_A24_D1_trim.avi")





system("ffmpeg -ss 04:37:09 -i T009_A13_D1.avi -c copy -q 1 -r 1 -t 00:30:00 T009_A13_D1_trim.avi")




system("ffmpeg -ss 04:13:19 -i T009_A24_D1.avi -c copy -q 1 -r 1 -t 00:30:00 T009_A24_D1_trim.avi")





system("ffmpeg -ss 06:34:48 -i T010_A13_D1.avi -c copy -q 1 -r 1 -t 00:30:00 T010_A13_D1_trim.avi")




system("ffmpeg -ss 08:00:02 -i T010_A24_D1.avi -c copy -q 1 -r 1 -t 00:30:00 T010_A24_D1_trim.avi")




system("ffmpeg -ss 07:27:50 -i T012_A13_D1.avi -c copy -q 1 -r 1 -t 00:30:00 T012_A13_D1_trim.avi")




system("ffmpeg -ss 06:45:12 -i T012_A24_D1.avi -c copy -q 1 -r 1 -t 00:30:00 T012_A24_D1_trim.avi")



#USE FOR SPECIFIC START AND STOP TIMES. 
system("ffmpeg -i 1hour.avi -c copy -q 3 -r 10 -ss 00:06:00 -to 00:07:00 1minlight.avi")

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
#4:3 aspect ratio resolutions: 640×480, 800×600, 960×720, 1024×768, 1280×960, 1400×1050, 1440×1080, 
#1600×1200, 1856×1392, 1920×1440, and 2048×1536.


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



# FFMPEG: Batch Convert Liddell Video Formats & Rescale ---------------------------


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




# LIDDELL: Create OCR output csv file -------------------------------------------
library(tesseract)
library(magick)
library(beepr)
eng <- tesseract("eng")

#Set R and system working directories
# CHECK THAT ALL VIDEOS ARE 1080P. 
wd <- setwd("E:/3_Liddell_2018_PROC/T003/OCR")
system("cd /d E:/3_Liddell_2018_PROC/T003/OCR")
getwd()

#Create jpgs from video and list of all the generated jpgs. 
avi_list <- dir(wd, pattern = "*.avi") #Note that this code works with .avi's in mpeg4 codec
avi_list

## ffmpeg -i video.mp4 -vf "fps=1,crop=1540:1000:250:0" -q:v 2 outimage_%03d.jpeg Can use ffmpeg to crop the frame. Likely faster than using tesseract to crop the frame. Test with a short video. 

# BEST RUN ON FOLDERS CONTAINING 10-15 VIDEOS
for (aa in 1:length(avi_list)) {
  # EXTRACT FRAMES AT SET FRAMERATE. NOTE: FRAMES CREATED = 147,435
  cmdstring <- paste("ffmpeg -i", avi_list[aa], "-vf fps=1 frame%06d.jpg") # Keep at 1 FPS, for now. 
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


