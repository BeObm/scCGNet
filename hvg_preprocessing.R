rm(list = ls())
suppressWarnings(suppressMessages(library(dplyr)))
suppressWarnings(suppressMessages(library(Seurat)))
suppressWarnings(suppressMessages(library(textshape)))
library(data.table)

setwd("G:\\My Drive\\01 HUs LAB\\02 Research projects\\2023 GMCM-VGAE\\data-23Oct")
project_name <- "turtle_brain"
#project_name <- "sota"
#project_name <- "baron4"
#project_name <- "baron3"
# Use your own count matrix (might or might not have a header)
#counts <- read.csv('rawData\\sota_raw.csv', header=FALSE) #genes x cells
counts <- read.csv('rawData\\Turtle_brain_X.csv', header=FALSE) #genes+1 x cells
#counts <- read.csv('rawData\\baron4_raw.csv', header=FALSE) # cells x genes
#counts <- read.csv('rawData\\baron3_raw.csv', header=FALSE) # cells x genes
dim(counts) 

View(counts)
#for sota, and turtle_brain only
counts <- counts[ -1,]

#don't run for sota
labels <- counts[c(1,3)]
counts <- counts[ -c(1:3)]

dim(counts) #genes x cells
#don't run for sota or turtle_brain
counts <- transpose(counts)
# Use the column names as index (gene names)
counts <- column_to_rownames(counts, 'V1') 
#count in seurat needs to be genes x cells
counts_seurat <- CreateSeuratObject(counts = counts, project = project_name)
# Normalizing the dta
counts_norm <- NormalizeData(counts_seurat, 
                             normalization.method = "LogNormalize", scale.factor = 10000)
#600, 800, 1000, 1200, 1400, 1600
num_hvg <- 1600 # Number of top genes
#num_hvg <- 21025 # All genes

#selected <- FindVariableFeatures(counts_seurat, nfeatures = num_hvg) 
selected <- FindVariableFeatures(counts_norm, nfeatures = num_hvg) #15720 variable features
topfeat = selected@assays[["RNA"]]@var.features
saveRDS(topfeat,sprintf("processedData\\%s_topfeat_%d.rds",project_name,length(topfeat)))
#saveRDS(topfeat,"baron4_15720.rds")
# Output the dataframe to csv to be processed in knnn.r
hvg <- as.data.frame(GetAssayData(object = selected))[selected@assays$RNA@var.features, ]
dim(hvg) #1200x1303 --> genes x cells
write.csv(as.matrix(hvg), sprintf("processedData\\%s_HVG_%d.csv", project_name, length(topfeat)))

#output cluster label
#write.csv(data.frame(labels),sprintf("processedData\\%s_Labels.csv", project_name))
