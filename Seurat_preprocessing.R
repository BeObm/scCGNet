rm(list = ls())
if(!is.null(dev.list())) dev.off()
cat("\f")

library(dplyr)
library(Seurat)
library(textshape)

setwd("G:\\My Drive\\01 HUs LAB\\02 Research projects\\2023 GMCM-VGAE\\data-scAce")

D1200.sci.log<-read.csv("CSVs\\raw\\Turtle_brain_X.csv", row.names = 1)
D <- CreateSeuratObject(counts = D1200.sci.log)
D <- FindVariableFeatures(D, nfeatures = 1200)
all.genes.D <- rownames(D)
D <- ScaleData(D, features = all.genes.D)
dim(D) #1200 18664
D <- RunPCA(D, features = VariableFeatures(object = D))

D <- FindNeighbors(D, dims = 1:10)
D <- FindClusters(D, resolution = 0.5)

saveRDS(Idents(D), "Turtle_brain_HVG_1200_seurat.RDS")

#######################

# Halfsize
#n=10

setwd("G:\\My Drive\\01 HUs LAB\\02 Research projects\\2023 GMCM-VGAE\\simulatedData")
n=10
for (i in 1:n){
  D1200.sci.log<-read.csv(paste0("DSim\\DSim_",i,".csv"), row.names = 1)
  D <- CreateSeuratObject(counts = D1200.sci.log)
  D <- FindVariableFeatures(D, nfeatures = 1200)
  all.genes.D <- rownames(D)
  D <- ScaleData(D, features = all.genes.D)
  
  D <- RunPCA(D, features = VariableFeatures(object = D))
  
  D <- FindNeighbors(D, dims = 1:10)
  D <- FindClusters(D, resolution = 0.5)
  
  saveRDS(Idents(D), paste0('Seurat_Processing\\DSim_Seurat_',i,'.RDS'))
}



