# library(stringr)
# setwd(fp<-"Z:/20_8x8_2021/IR_DATA/T008_KOOBwt") #### CHANGE THIS
# system("cd /d Z:/20_8x8_2021/IR_DATA/T008_KOOBwt") #### CHANGE THIS
# dir.create("R_proc")
# fl <- list.files(fp, full.names=TRUE, pattern=".avi")
# fl.short <- list.files(fp,full.names=FALSE,pattern=".avi")
# fl.shorty <-str_remove(fl.short, ".avi")
# out_fp="Z:/20_8x8_2021/IR_DATA/T008_KOOBwt/R_proc"  #change this
# setwd(fp)
# #-crf 28 results in a video that is basically the same quality, but with an insanely smaller file size. use this.
# # note that forcing the original aspect ratio does NOT effect the transcoding speed. This code runs at about 2.07x speed unfortunately. 
# for(i in 1:length(fl)){ #Changes codec from H264 -> mpeg4
#   cmdstr=paste('ffmpeg -i "', fl.short[i], '" -an -r 30 -codec libx265 -crf 28 -vf "scale=1920:1080:force_original_aspect_ratio=decrease:eval=frame,pad=1920:1080:-1:-1:color=black,format=gray,subtitles=',
#                substr(as.character(fl.short[i]),1,(nchar(fl.short[i])-4)),
#                '.smi',
#                ":force_style='",
#                'FontSize=10,Alignment=1,BorderStyle=3,Outline=1,Shadow=0,MarginV=20" ',
#                ' -max_muxing_queue_size 10000000 ',  '"', 
#                paste(out_fp, paste(fl.shorty[i], ".mp4", sep=""), sep='/'),'"',sep='')
#   # print(cmdstr)#display in command window
#   system(cmdstr) #send to windows
# }
# 


## SPEED TEST 2
library(stringr)
setwd(fp<-"E:/T008_KOOBwt") #### CHANGE THIS
system("cd /d E:/T008_KOOBwt") #### CHANGE THIS
dir.create("R_proc")
fl <- list.files(fp, full.names=TRUE, pattern=".avi")
fl.short <- list.files(fp,full.names=FALSE,pattern=".avi")
fl.shorty <-str_remove(fl.short, ".avi")
out_fp="E:/T008_KOOBwt/R_proc"  #change this
setwd(fp)
# - note that running loop on server videos can limit top speeds. but this code should run at about 7x speed. 
# - converting from libx264 to libx265 drastically reduces the file size with minimal tradeoffs in quality, but cuts speed in half. 
#-crf 28 results in a video that is basically the same quality, but with an insanely smaller file size. use this.
# note that forcing the original aspect ratio does NOT effect the transcoding speed. This code runs at about 2.07x speed unfortunately. 
i = 1
for(i in 1:length(fl)){ #Changes codec from H264 -> mpeg4
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -analyzeduration 2147483647 -probesize 2147483647 -an -codec libx265 -crf 25 -preset ultrafast -vf "scale=1920:1080:force_original_aspect_ratio=decrease:eval=frame,pad=1920:1080:-1:-1:color=black,format=gray,subtitles=',
               substr(as.character(fl.short[i]),1,(nchar(fl.short[i])-4)),
               '.smi',
               ":force_style='",
               'FontSize=10,Alignment=1,BorderStyle=3,Outline=1,Shadow=0,MarginV=20" ',
               ' -max_muxing_queue_size 100000000 ',  '"', 
               paste(out_fp, paste(fl.shorty[i], ".mp4", sep=""), sep='/'),'"',sep='')
  print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}


# alternatively, open up a command prompt line in folder directory. 

