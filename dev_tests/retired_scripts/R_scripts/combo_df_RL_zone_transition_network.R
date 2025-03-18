
setwd("C:/Users/Rusty/Amazon Drive/MICE")

data<-read.csv("C:/Users/Rusty/Amazon Drive/MICE/Caleb.csv")


date<-unlist(lapply(data$Field.Time,function(x) (as.character(strsplit(as.character(x)," ")[[1]][1]))))
data$month<-unlist(lapply(date,function(x) (as.character(strsplit(as.character(x),"-")[[1]][2]))))
data$day<-unlist(lapply(date,function(x) (as.character(strsplit(as.character(x),"-")[[1]][3]))))

time<-unlist(lapply(data$Field.Time,function(x) (as.character(strsplit(as.character(x)," ")[[1]][2]))))
time<-gsub("\n","",time)
data$dailysecond<-dailysecondfinder(time)
data<-data[!is.na(data$dailysecond),]

#turn those newly acquired variables into numbers
data[,c(31:33)]<-apply(data[,c(31:33)],2,function(x){as.numeric(as.character(x))})

#order data by month, day, dailysecond
datasorted<-data[with(data,order(month,day,dailysecond)),]

micenames<-c("M1", "M2", 
             "M3", "M4", "M5", "M6", "F01", "F2", "F3", "F4", "F5", "F6", 
             "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15")
#this will be filled with the raw sequences of occupied zones for each mouse in this 'micenames' vector
mouse.sequence.list<-list()

#this will be filled with 2 items per mouse in the 'micenames' vector
#first, there are a number of summary network metrics obtained for the movement of each mouse
#second, there is the igraph group assignment
mouse.network.list<-list()

pdf("Caleb.pdf",width= 30, height= 15,family="NimbusRom")
op <- par(mfrow=c(2,5))# rows, columns #mfcol fills columns first, mfrow fills rows first
par(mar=c(4,8,4,8))
  for(m in 1:length(micenames)){
    mouse<-micenames[m]
    thismousedata<-datasorted[!is.na(datasorted[,mouse]),
                              c(1:9,which(colnames(datasorted)==mouse),31:33),drop=FALSE]
    
    #drop duplicate frames
    thismousedata$unique.time<-!duplicated(thismousedata$OCR.Frame)
    thismousedata<-thismousedata[thismousedata$unique.time,]
    
    if(nrow(thismousedata)>1){
      mouse.sequence.list[[m]]<-zz<-as.character(paste0("Z",thismousedata$Zone))
      names(mouse.sequence.list)[m]<-mouse
    
      mouse.network.list[[m]]<-Network.Quick(zz,plotpretty='liddell',
                                           individual.label=mouse,regionnames="full",
                                           cutselfs=TRUE)
      #set cutselfs=TRUE to minimize the impact of self-transitions (e.g. Z1->Z1)
      
      names(mouse.network.list)[m]<-mouse
    }

    
  }
dev.off()
#########################################################################################
#########################################################################################
rgb2hex <- function(r,g,b) rgb(r, g, b, maxColorValue = 255)


#Return Center locations of Zones
Zonecenters<-function(){
  xs<-c(20,50,20,50,20,50,20,50)
  ys<-c(25,25,65,65,105,105,145,145)
  labelslocations<-cbind(xs,ys)
  labelslocations<-data.frame(labelslocations)
  labelslocations$region<-c("Z1","Z2","Z3","Z4","Z5","Z6","Z7","Z8")
  labelslocations$shortnames<-c("1","2","3","4","5","6","7","8")
  labelslocations<-data.frame(labelslocations)
  labelslocations[,c(1,2)]<-apply(labelslocations[,c(1,2)],2,function(x){as.numeric(as.character(x))})
  
  return(labelslocations)
  
  
}

#MOUSE ZONE PLOTTER
PLOTZONES<-function(regionnames=c("none","full","short"),AL,lineweighting=10,vertweighting=10,
                    showme.looped,totalpees,LABEL=individual.label){
  
  require(grDevices)
  require(colorRamps)
  require(geometry)
  require(plyr)
  require(reshape)
  ####################################################################################################################################
  library(MASS)
  library(plotrix)
  
  


  
  vertlocations<-Zonecenters();rownames(vertlocations)<-vertlocations$region;vertlocations<-vertlocations[,c(1,2)]
  
  
  flagme<-0
  for(jb in 1:nrow(vertlocations)){
    matcher<-grep(rownames(vertlocations)[jb],V(showme.looped)$name)
    if(length(matcher)==0){
      
    } else {
      
      keep<-vertlocations[jb,,drop=FALSE]
      if(flagme==0){
        flagme<-1
        newvert<-keep
      } else {
        newvert<-rbind(newvert,keep)
      }
    }
  }

  
  l <- layout_in_circle(showme.looped)
  
  colormat<-c(rgb2hex(255,201,14),rgb2hex(34,177,76),rgb2hex(185,122,87),rgb2hex(237,28,36),
              rgb2hex(220,10,220),rgb2hex(153,217,234),rgb2hex(0,162,232),rgb2hex(239,228,176))
  
  names(colormat)<-rownames(vertlocations)
  
  totalpees2<-totalpees[c("Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8")]
  
  tp2<-unlist(lapply(totalpees2,function(x){if(is.na(x)){bb<-0}else{bb<-x}}))
  names(tp2)<-c("Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8")
  totalpees2<-tp2
  
  
  totalpees2<-totalpees2[totalpees2>0]
  
  #set color
  V(showme.looped)$color<-colormat[V(showme.looped)$name]
  #Rather Let’s color the edges of the graph based on their source node color. 
  edge.start <- ends(showme.looped, es=E(showme.looped), names=F)[,1]
  edge.col <- V(showme.looped)$color[edge.start]
  
  
  
  plot.igraph(showme.looped,
              edge.width = E(showme.looped)$weight*lineweighting,
              # vertex.label=c("osb","osc","osw","osh",
              #                "ssb","ssc","ssw","ssh",
              #                "c","cc","gm","fc"),
              
              vertex.color = V(showme.looped)$color,
              
              edge.color=V(showme.looped)$color[edge.start],
              #edge.color='black',
              edge.curved = 0.2, 
              edge.arrow.size=.4,
              vertex.size=log2(totalpees2)*vertweighting,
              edge.loop.angle=rep(5.7,length(E(showme.looped))),
              vertex.shape = "circle",
              label.col="black",
              vertex.frame.color="purple",
              layout=as.matrix(vertlocations),
              main=LABEL)
  

}





Network.Quick<-function(z,plotpretty=c('liddell','yes','arena','no'),
                        individual.label,regionnames=c("none","full","short"),
                        cutselfs=TRUE){
  # regionnames refers to options for ArenaPlotting
  summarytable<-table(z)
  totalpees<-c(summarytable)
  proportiontable<-prop.table(summarytable)
  uniquebehaviors<-length(summarytable)
  
  #TRUE SHANNON DIVERSITY
  dftable<-t(as.data.frame(summarytable))
  colnames(dftable)<-dftable[1,]
  dftable<-dftable[-1,,drop=FALSE]
  rownames(dftable) <- c()
  dftable<-data.frame(dftable)
  indx<-c(1:uniquebehaviors)
  dftable[indx]<- lapply(dftable[indx], function(x) as.numeric(as.character(x)))
  Shannon<-vegan::diversity(dftable,index = "shannon")
  trueShannon<-exp(Shannon)
  
  
  #TRANSITIONSSSSSSSS  
  ##########################################################################################################
  c.no.OFFs<-as.character(z)
  
  #uses "createSequenceMatrix" function (from *markovchain*) to calculate transition matrix
  TransitionMatrix.cum<-createSequenceMatrix(c.no.OFFs,sanitize=FALSE)
  
  if(cutselfs){
    noselfs<-TransitionMatrix.cum
    diag(noselfs)<-0
    TransitionMatrix.cum<-noselfs
  }

  
  
  
  prop.table.excludeNAs<- function(x) {x/sum(x, na.rm=TRUE)}
  
  proportiontable.trans<-prop.table.excludeNAs(TransitionMatrix.cum) 
  
  ###########TRUE SHANNON DIVERSITY FOR TRANSITIONS
  if(uniquebehaviors>1){
    val3.s<-matrix(nrow=(((uniquebehaviors^2)-uniquebehaviors)/2),ncol=2)
    rownum<-1
    for (u in 1:uniquebehaviors){
      for(v in u:uniquebehaviors){#iteratively reduces columns analyzed in next loop to avoid double counting transitions 
        if (u!=v){
          val3.s[rownum,2]<-as.numeric(TransitionMatrix.cum[u,v]+TransitionMatrix.cum[v,u])
          val3.s[rownum,1]<-paste(row.names(TransitionMatrix.cum)[u],colnames(TransitionMatrix.cum)[v],sep="-")
          rownum<-rownum+1
        }
      }
    }
    val4.s<-as.data.frame(val3.s)
    val4.s[,2]<-as.numeric(as.character(val4.s[,2])) 
    
    dftable.T<-t(val4.s)
    colnames(dftable.T)<-dftable.T[1,]
    dftable.T<-dftable.T[-1,,drop=FALSE]
    rownames(dftable.T) <- c()
    dftable.T<-data.frame(dftable.T)
    indx<-c(1:length(dftable.T))
    dftable.T[indx]<- lapply(dftable.T[indx], function(x) as.numeric(as.character(x)))
    Shannon.T<-diversity(dftable.T,index = "shannon")
    trueShannon.T<-exp(Shannon.T)
    propShannon.T<-Shannon.T/log(specnumber(dftable.T))
    redundancy.T<-1-propShannon.T
    uniquetransitions<-length(val3.s)
    
  } else {
    trueShannon.T<-NA
    propShannon.T<-NA
    redundancy.T<-NA
    uniquetransitions<-1
  }
  
  
  #########################################################
  PropSelfTransitioning<-sum(diag(TransitionMatrix.cum))/sum(TransitionMatrix.cum)
  
  
  network.TransitionMatrix.cum<-network(proportiontable.trans,directed=TRUE,loops=TRUE,hyper=FALSE,matrix.type="adjacency")
  
  
  #Calculate behavioral group membership network
  community.observed <- fastgreedy.community(graph.adjacency(proportiontable.trans,mode="undirected",weighted=TRUE))
  ncommunities<-length(unique(community.observed$membership))
  
  # NETWORK SUMMARY VARIABLES
  showme.looped<- graph_from_adjacency_matrix(proportiontable.trans,mode="directed",weighted=TRUE,diag=TRUE)
  
  
  selfprops<-diag(proportiontable.trans)
  centersfornetworks<-Zonecenters()
  adjacencyList <- reshape2::melt(TransitionMatrix.cum)  # Convert to dyadic info
  adjacencyList <- adjacencyList[adjacencyList$value > 0, ] # remove zeros along diagonal
  adjacencyList[,c(1,2)] <- apply(adjacencyList[,c(1,2)],2,function(x){as.character(x)})
  
  maxweight<-max(adjacencyList$value)
  E(showme.looped)$width <- E(showme.looped)$weight
  
  
  
  showme.looped2<- graph_from_adjacency_matrix(proportiontable.trans,mode="undirected",weighted=TRUE,diag=TRUE)
  
  
  
 
  
  if(plotpretty=='liddell'){
    PLOTZONES(regionnames="full",AL=adjacencyList,lineweighting=150,vertweighting=2,
              showme.looped,totalpees,LABEL=individual.label)
  }
  ########################################################
  
  # Klein DJ, Randić M. Resistance distance. Journal of Mathematical Chemistry 1993; 12: 81–95.
  # Ellens W, Kooij RE. 2013. Graph measures and network robustness. arXiv:1311.5064v1
  
  # Klein and Randić [30] found that the effective graph resistance of a connected network can be written as a function of all 
  # non-zero Laplacian eigenvalues of the network.
  
  # Yang et al. 2016. The Rationality of Four Metrics of Network Robustness: A Viewpoint of Robust Growth of 
  # Generalized Meshes. PLOS ONE. https://doi.org/10.1371/journal.pone.0161077
  
  lap.mat<-laplacian_matrix(showme.looped2, norm=FALSE, sparse=FALSE)
  ev.lm<-eigen(lap.mat)
  ev.lm<-unlist(ev.lm[1])
  sorted.Laplacian.eigenvalues<-sort(ev.lm)
  value.ER<-0
  for(lam in 2:length(sorted.Laplacian.eigenvalues)){
    value.ER<-value.ER+1/sorted.Laplacian.eigenvalues[lam]
  }
  
  EffectiveGraphResistance<-value.ER*length(sorted.Laplacian.eigenvalues)
  
  
  
  
  
  ############################################################################
  #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
  #6.1 Density
  
  #The proportion of present edges from all possible edges in the network.
  #Edge density: % of edges compared to maximum
  #  densityscore<-ecount(showme.looped)/(vcount(showme.looped)*(vcount(showme.looped)-1)) #for a directed network
  densityscore<-ecount(showme.looped)/(vcount(showme.looped)^2)#for a directed network WITH self-loops
  
  
  #
  deg <- igraph::degree(showme.looped,loops=TRUE)
  #Average degree: Avg. number of links
  mean.degree<-mean(deg)
  
  #MAverage path length: avg of shortest pathes between reachable nodes
  mean_path_length<-mean_distance(showme.looped, directed=T)
  
  #network diameter is the longest geodesic distance (length of the shortest path between two nodes) in the network.
  network.diameter<-diameter(showme.looped, directed=T,weights=NA)
  
  #average clustering coefficient
  avg.cluster.coeff<-transitivity(showme.looped)
  
  #Modularity
  Modularity.Value<-modularity(community.observed)
  
  
  
  ############################################
  #SMALL WORLDNESS --- video-wise, no filter
  #number of nodes/vertices in graph
  vertices<- uniquebehaviors
  #number of edges in G(n,m) graph
  edges<- sum(TransitionMatrix.cum!=0)
  
  rando.network<-sample_gnm(vertices, edges, directed = TRUE, loops = TRUE)
  Trobserved<-avg.cluster.coeff
  mean.Trrandom<-transitivity(rando.network)
  SPobserved<-mean_path_length
  mean.SPrandom<-mean_distance(rando.network, directed=T)  
  
  Smallworldness<- (Trobserved/mean.Trrandom)/(SPobserved/mean.SPrandom)
  ############################################
  
  orig.printme<-data.frame(ncommunities,uniquebehaviors,uniquetransitions,
                           Shannon,Shannon.T,
                           trueShannon,trueShannon.T,
                           EffectiveGraphResistance,
                           densityscore,mean.degree,mean_path_length,network.diameter,avg.cluster.coeff,
                           PropSelfTransitioning,Smallworldness,Modularity.Value)
  
  
  return(list(orig.printme,community.observed))
}



#function to return daily second from military time (xx-xx-xx)
dailysecondfinder<-function(militarytime){
  
  if(length(grep("_",militarytime[1]))==1){
    t.hour<-unlist(lapply(militarytime,function(x) as.numeric(as.character(strsplit(x,"-")[[1]][1]))))
    t.min <-unlist(lapply(militarytime,function(x) as.numeric(as.character(strsplit(x,"-")[[1]][2]))))
    t.sec <-unlist(lapply(militarytime,function(x) as.numeric(as.character(strsplit(x,"-")[[1]][3]))))
  }
  
  if(length(grep(":",militarytime[1]))==1){
    t.hour<-unlist(lapply(militarytime,function(x) as.numeric(as.character(strsplit(x,":")[[1]][1]))))
    t.min <-unlist(lapply(militarytime,function(x) as.numeric(as.character(strsplit(x,":")[[1]][2]))))
    t.sec <-unlist(lapply(militarytime,function(x) as.numeric(as.character(strsplit(x,":")[[1]][3]))))
  }

  
  #calculate daily.sec values
  dailysecond<-t.sec+(t.min*60)+(t.hour*60*60)
  return(dailysecond)
}
