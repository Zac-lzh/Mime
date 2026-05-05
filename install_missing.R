user_lib <- file.path(Sys.getenv("USERPROFILE"), "R", "win-library",
                       paste0(R.version$major, ".", as.integer(R.version$minor)))
.libPaths(c(user_lib, .libPaths()))

# 安装 plsRcox (CRAN)
if (!requireNamespace("plsRcox", quietly=TRUE)) {
  cat("安装 plsRcox...\n")
  install.packages("plsRcox", repos="https://cloud.r-project.org", lib=user_lib)
}

# 安装 mixOmics (Bioconductor)
if (!requireNamespace("mixOmics", quietly=TRUE)) {
  cat("安装 mixOmics (Bioconductor)...\n")
  if (!requireNamespace("BiocManager", quietly=TRUE))
    install.packages("BiocManager", repos="https://cloud.r-project.org", lib=user_lib)
  BiocManager::install("mixOmics", ask=FALSE, update=FALSE, lib=user_lib)
}

cat("plsRcox:", requireNamespace("plsRcox", quietly=TRUE), "\n")
cat("mixOmics:", requireNamespace("mixOmics", quietly=TRUE), "\n")
