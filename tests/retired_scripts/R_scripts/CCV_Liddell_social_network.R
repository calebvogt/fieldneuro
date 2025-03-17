# Create social network adjacency matrix from full_zone_df.csv ------------

setwd("G:/My Drive/Alex Liddell Data/Analysis files")


vog<-read.csv("full_zone_df_v3.csv")
vog<-vog[,c(2:6,13)]
vog$duration2<-vog$STOP-vog$START
# plot(duration~duration2,data=vog)

vog<-vog[order(vog$START, na.last=FALSE) , ]
zonetypes<-sort(unique(vog$zone))

flag<-0
for (zonation in 1:length(zonetypes)) {
  thiszone<-zonetypes[zonation]
  zonewise<-vog[which(vog$zone==thiszone),]
  print(paste("Processing zone ",thiszone," out of ",
              length(zonetypes),sep=''))
  for(rowwise1 in 1:(nrow(zonewise)-1)){
    c1<-zonewise[rowwise1,,drop=FALSE]
    for(rowwise2 in (rowwise1+1):(nrow(zonewise))){
      c2<-zonewise[rowwise2,,drop=FALSE]
      if(c1$Subject!=c2$Subject){
        if(c2$START<c1$STOP & c2$START>c1$START){
          xtemp<-matrix(c(c1$Subject,c2$Subject,c1$zone),nrow=1)
          colnames(xtemp)<-c("ID1","ID2","zone")
          
          if(flag<1){
            socialinteractions<-xtemp
          } else {
            socialinteractions<-rbind(socialinteractions,xtemp)
          }
          flag<-flag+1
        
        }
      }
      
    }
  }
  
}



# This will create a directed data frame as combinations are repeated
socialinteractions<-as.data.frame(socialinteractions)
socialinteractions$ID2<-factor(socialinteractions$ID2)
socialinteractions$ID1<-factor(socialinteractions$ID1)
socialinteractions$zone<-factor(socialinteractions$zone)
summary(socialinteractions)



# Create directed adjacency matrix ----------------------------------------


g.unit<-(table(socialinteractions))
caca<-as.data.frame(g.unit)
#caca[which(max(caca$Freq)==caca$Freq),]
  ids<-list()
  ids[[1]]<-sort(unique(caca$ID1))
  ids[[2]]<-sort(unique(caca$ID2))
for(z in 1:length(zonetypes)){
  diszone<-caca[which(caca$zone==z),]
  present<-matrix(diszone$Freq,nrow=10,ncol=10,dimnames = ids)
  if(z<2){
    totsmagoats<-present
  } else {
    totsmagoats<-totsmagoats+present
  }
}


  totsmagoats
View(totsmagoats)
sum(totsmagoats)
write.csv(totsmagoats, "directed_adjacency_matrix.csv")

# CCV code directed adjacency matrrix ----------------------------------------------------------------

library(igraph)

network_df <- totsmagoats

net_graph <- graph.adjacency(network_df, mode="directed", weighted = TRUE)

g_dir <- simplify(net_graph)

V(g)$label <- V(g)$name
V(g)$degree <- degree(g)

set.seed(3952)
layout1 <- layout.fruchterman.reingold(g)
plot(g, layout=layout1)

plot.igraph(g)


# Obtain unweighted adjacency matrix from socialinteractions--------------------------------------


# Undirected weighted social network --------------------------------------


library(igraph)

summary(socialinteractions)
df <- subset(socialinteractions, select = -c(3))



# coerces the data into a two-column matrix format that igraph likes
el=as.matrix(df)
el[,1]=as.character(el[,1])
el[,2]=as.character(el[,2])
# turns the edgelist into a 'graph object'
g=graph.edgelist(el,directed=FALSE) 


#create adjacency matrix from edgelist
g <- get.adjacency(g,sparse=FALSE) 

# create igraph object from undirected adjacency matrix
g <- graph.adjacency(g, mode="undirected", weighted =TRUE)

#simplify igraph object. removes mulitiple edges and loop edges
g <- simplify(g)

V(g)$label <- V(g)$name
V(g)$degree <- degree(g)

set.seed(3952)
layout1 <- layout.fruchterman.reingold(g)
# standard plot
plot(g, 
     layout=layout1,
     vertex.color = "green",
     vertex.size = 25,
     edge.color = 'black'
     )

# Pretty plot!
E(g)$weight <- edge.betweenness(g)
plot(g,
     vertex.color = "grey", # change color of nodes
     vertex.label.color = "black", # change color of labels
     vertex.label.cex = 3.0, # change size of labels to 75% of original size
     edge.curved=.25, # add a 25% curve to the edges
     edge.color="grey20",# change edge color to grey
     edge.width = E(g)$weight
     # edge.width=edge.betweenness(g)
     ) 




# Network Measures --------------------------------------------------------

degree.cent <- centr_degree(g, mode = "all")
degree.cent$res



degree(g_undir, mode='all')

degree(g_undir, mode='in')
