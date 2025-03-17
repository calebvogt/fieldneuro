## 8x8.2018.Analysis.v.1.0.0.R
## Caleb Clifton Vogt, PhD Cornell University
## Updated 10.9.2019




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


