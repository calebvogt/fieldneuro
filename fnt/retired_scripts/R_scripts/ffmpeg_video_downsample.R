library(stringr)
setwd(fp<-"Z:/2022_CV_LID_mong/lorex_video_raw/raw_videos") #### CHANGE THIS
system("cd /d Z:/2022_CV_LID_mong/lorex_video_raw/raw_videos") #### CHANGE THIS
dir.create("R_proc")

fl <- list.files(fp, full.names=TRUE, pattern=".avi|.mp4|.mov")
fl.short <- list.files(fp,full.names=FALSE,pattern=".avi|.mp4|.mov")
fl.shorty <-str_remove(fl.short, ".avi|.mp4|.mov")
out_fp="Z:/2022_CV_LID_mong/lorex_video_raw/raw_videos/R_proc"  #change this

setwd(fp)
#-q should be set to 3. -r should be set to 10 or 15 fps for 8x8 long videos. Anything with fighting should be boosted to minimum 25fps 
## -crf default is set to 23. 51 is worst, 0, is original. 

for(i in 1:length(fl)){ #Changes codec from H264 -> mpeg4
  cmdstr=paste('ffmpeg -i "', fl.short[i], '" -vcodec libx265 -preset ultrafast -crf 23 -pix_fmt yuv420p -an -r 30 -max_muxing_queue_size 10000000 ',
               paste(out_fp, paste(fl.shorty[i], ".mp4", sep=""), sep='/'),'"',sep='')
  # print(cmdstr)#display in command window
  system(cmdstr) #send to windows
}
