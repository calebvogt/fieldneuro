## ocr_tesseract.R
## Caleb C. Vogt, Ph.D. Candidate Cornell University


# CREATE OCR FILE FROM VIDEO -------------------------------------------

# Need to learn how to turn this all into a function that can be called outright. 
library(tesseract)
library(magick)
library(beepr)
eng <- tesseract("eng")

#Set R and system working directories
# CHECK THAT ALL VIDEOS ARE 1080P. 
wd <- setwd("Z:/7_Liddell_2020/TRIALS/OCR/in_progress")
system("cd /d Z:/7_Liddell_2020/TRIALS/OCR/in_progress")
getwd()

#Create jpgs from video and list of all the generated jpgs. 
avi_list <- dir(wd, pattern = "*.avi") #Note that this code works with .avi's in mpeg4 codec
avi_list


# Need to look into making the cropped image bigger.
# follow the guidelines to make OCR detection better. https://cran.r-project.org/web/packages/tesseract/vignettes/intro.html



# BEST RUN ON FOLDERS CONTAINING 10-15 VIDEOS
for (aa in 1:length(avi_list)) {
  cmdstring <- paste("ffmpeg -i", avi_list[aa], "-vf fps=1,crop=315:35:50:970 frame%09d.jpg") #### ADJUST TO 1 OR 10 FPS. w:h:x:y
  shell(cmdstring)
  jpg_list <- dir(wd, pattern ="*.jpg")
  for (bb in 1:length(jpg_list)) {
    image <- image_read(jpg_list[bb])
    txt <- tesseract::ocr(image, engine = eng)
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

