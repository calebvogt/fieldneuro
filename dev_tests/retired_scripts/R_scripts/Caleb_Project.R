setwd("D:/Alex Data/T001_Bravo_Zone1_CAM2")

filelist = list.files(pattern = "*.smi")
myfiles = lapply(filelist, read.delim)

##clean data
size <- rep(0,2)
for (j in c("CAM21", "CAM22", "CAM23")) {
print(dim(j))
}

j <- j[10:(size[1]-2), ]
}

x <- rep(0,size[1]/2)

for (i in 1:size[1]){
  if ((i %% 2) == 0) {
    x[i/2] <- CAM21[i,]  
  } else {
  }
}

## clean data
size1 <- dim(myfiles[1][[1]])
myfiles[1][[1]] <- myfiles[1][[1]][9:(size1[1]-2), ]

y <- rep(0,size1[1])
for(i in 1:size1[1]) {
  y[i] <- myfiles[1][i] 
}

x1 <- rep("x",(size1[1]/2))
for (i in 1:size1[1]){
  if ((i %% 2) == 0) {
    x1[i/2] <- myfiles[[1]][i]  
  } else {
  }
}

size22 <- dim(CAM22)
CAM22 <- CAM22[10:(size22[1]-2), ]

size23 <- dim(CAM23)
CAM23 <- CAM21[10:(size23[1]-2), ]

##Select only even rows
x21 <- rep(0,(size21[1]/2))
for (i in 1:size21[1]){
  if ((i %% 2) == 0) {
    x21[i/2] <- CAM21[i,]  
  } else {
  }
}

x22 <- rep(0,(size22[1]/2))
for (i in 1:size22[1]){
  if ((i %% 2) == 0) {
    x22[i/2] <- CAM22[i,]  
  } else {
  }
}

x23 <- rep(0,(size23[1]/2))
for (i in 1:size23[1]){
  if ((i %% 2) == 1) {
    x23[i/2] <- CAM23[i,]  
  } else {
  }
}









## clean data
size21 <- dim(CAM21)
CAM21 <- CAM21[10:(size21[1]-2), ]

size22 <- dim(CAM22)
CAM22 <- CAM22[10:(size22[1]-2), ]

size23 <- dim(CAM23)
CAM23 <- CAM21[10:(size23[1]-2), ]

##Select only even rows
x21 <- rep(0,(size21[1]/2))
for (i in 1:size21[1]){
  if ((i %% 2) == 0) {
  x21[i/2] <- CAM21[i,]  
  } else {
  }
}

x22 <- rep(0,(size22[1]/2))
for (i in 1:size22[1]){
  if ((i %% 2) == 0) {
    x22[i/2] <- CAM22[i,]  
  } else {
  }
}

x23 <- rep(0,(size23[1]/2))
for (i in 1:size23[1]){
  if ((i %% 2) == 1) {
    x23[i/2] <- CAM23[i,]  
  } else {
  }
}
