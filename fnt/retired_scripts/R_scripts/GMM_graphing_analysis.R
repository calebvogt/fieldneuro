## GMM_graphing_analysis.R
## Caleb C. Vogt, Cornell University

## TO-DO LIST:


#####################
library(asnipe)

# READ IN GMM.RDATA FILES. 

# EXTRACT OUTPUT FROM INDIVIDUAL GMM FILES. 
gbi <- gmm_data$gbi
events <- gmm_data$metadata
observations_per_event <- gmm_data$B


#Turning into network
r_network <- get_network(association_data = gmm_data$gbi, data_format = "GBI")

r_network_ALL <- get_network(association_data = gmm_data$gbi, data_format = "GBI")
r_net_ALL <- graph.adjacency(r_network_ALL, mode = "undirected", weighted = TRUE, diag = FALSE)

#Making network MF
r_network_MF <- r_network_ALL
r_network_MF[which(global_ids =="*-M-*"),which(global_ids =="*-M-*")] <- 0
r_network_MF[which(global_ids =="*-F-*"),which(global_ids =="*-F-*")] <- 0
save(r_network, file = "All RFID Network")
save(r_network_MF, file = "MF RFID Network")
write.csv(r_network_MF, file = "MF RFID Network.csv")

#Unweighted
r_network_MF_uw <- r_network_MF
r_network_MF_uw[r_network_MF_uw > 0] <- 1
degree_rfidMF <- rowSums(r_network_MF_uw)
degree_rfidMF
save(r_network_MF_uw, file = "MF RFID Network UW") #?? as csv???


# IN PROGRESS -------------------------------------------------------------
# 

# RFID RESOURCES
# https://github.com/animalnexus/feedr
# https://animalnexus.github.io/feedr/
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5632613/

# install.packages("devtools") # if not already installed
# devtools::install_github("hadley/devtools")
# install.packages(c("dplyr", "ggplot2", "htmlwidgets", "scales", "shiny", "stringr"))
# devtools::install_github("animalnexus/feedr")
# library(feedr)

