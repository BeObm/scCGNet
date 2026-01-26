library(Matrix)

# Load sparse matrix from MatrixMarket format
expr_mat <- readMM("matrix.mtx")   # Sparse dgCMatrix (genes × cells usually)

# Transpose if needed — check dimensions first
# Usually scRNA-seq is cells × genes
if (nrow(expr_mat) > ncol(expr_mat)) {
  expr_mat <- t(expr_mat)
}

# Subset: randomly pick 2000 cells and 100 genes
set.seed(123)
num_cells <- nrow(expr_mat)
num_genes <- ncol(expr_mat)

cell_idx <- sample(num_cells, 2000)
gene_idx <- sample(num_genes, 100)

expr_sub <- expr_mat[cell_idx, gene_idx]

# Save the subset
writeMM(expr_sub, file = "subset_matrix.mtx")

# Optionally: write out row/column names
writeLines(as.character(cell_idx), "subset_cells.txt")
writeLines(as.character(gene_idx), "subset_genes.txt")
