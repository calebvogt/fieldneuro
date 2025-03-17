
# Load packages -----------------------------------------------------------
library(tuneR)
library(seewave)
library(ggplot2)
library(dplyr)
library(tcltk)
library(gridExtra)


# Visualize DAS annotations -----------------------------------------------

# Prompt user to select folder
system("net use V: \\\\128.138.126.202\\donaldsonlab /user:caleb Liquid.CUB.Ape.93 /persistent:yes") ## use if the folder is on a mounted server
# usv_folder <- "V:/caleb/2021_8x8/Data/usv_proc/T003_C57" # use if above isnt working
usv_folder <- tk_choose.dir(default = "", caption = "Select the folder containing USV annotations and WAV files")

# Set the selected folder as the working directory
if (!is.na(usv_folder)) {
  setwd(usv_folder)  
  cat("Working directory set to:", getwd(), "\n")
} else {
  stop("No folder selected. Exiting script.")
}


# Get list of annotation and wav files
annotation_files <- list.files(usv_folder, pattern = "_annotations.csv", full.names = TRUE)
wav_files <- list.files(usv_folder, pattern = "\\.wav$", full.names = TRUE)

# Function to match annotation file with wav file
match_files <- function(annotation_files, wav_files) {
  matched <- list()
  for (annot in annotation_files) {
    base_name <- gsub("_annotations.csv", "", basename(annot))
    wav_match <- wav_files[grepl(base_name, wav_files)]
    if (length(wav_match) > 0) {
      matched[[annot]] <- wav_match[1]
    }
  }
  return(matched)
}

# Get matched files
matched_files <- match_files(annotation_files, wav_files)

# Function to extract and plot spectrogram
plot_spectrogram <- function(start_time, stop_time, audio, sample_rate) {
  start_sample <- round(start_time * sample_rate)
  stop_sample <- round(stop_time * sample_rate)
  segment <- audio@left[start_sample:stop_sample]
  
  spectro_obj <- spectro(wave = Wave(segment, samp.rate = sample_rate, bit = audio@bit), 
                         f = sample_rate, 
                         ovlp = 75, 
                         wl = 512, 
                         flim = c(0, 125), 
                         collevels = seq(-40, 0, by = 5), 
                         plot = FALSE)
  
  return(ggplot() + annotation_custom(rasterGrob(spectro_obj$z, interpolate = TRUE), 
                                      xmin = 0, xmax = 1, ymin = 0, ymax = 1) + 
           theme_void() + ggtitle(paste("USV from", round(start_time, 2), "to", round(stop_time, 2), "sec")))
}

# Process first annotation file
if (length(matched_files) > 0) {
  first_annotation <- names(matched_files)[1]
  first_wav <- matched_files[[first_annotation]]
  df <- read.csv(first_annotation)
  audio <- readWave(first_wav)
  
  # Plot first 50 USVs
  usv_plots <- lapply(1:min(50, nrow(df)), function(i) {
    plot_spectrogram(df$start_seconds[i], df$stop_seconds[i], audio, audio@samp.rate)
  })
  
  grid.arrange(grobs = usv_plots, ncol = 5)
} else {
  cat("No matching files found in selected folder.")
}
