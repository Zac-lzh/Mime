user_lib <- file.path(Sys.getenv("USERPROFILE"), "R", "win-library",
                       paste0(R.version$major, ".", as.integer(R.version$minor)))
.libPaths(c(user_lib, .libPaths()))

# 安装 survcomp (Bioconductor)
if (!requireNamespace("survcomp", quietly=TRUE)) {
  cat("安装 survcomp...\n")
  if (!requireNamespace("BiocManager", quietly=TRUE))
    install.packages("BiocManager", repos="https://cloud.r-project.org", lib=user_lib)
  BiocManager::install("survcomp", ask=FALSE, update=FALSE, lib=user_lib)
}

# 重新安装 plsRcox
cat("安装 plsRcox...\n")
install.packages("plsRcox", repos="https://cloud.r-project.org", lib=user_lib)

cat("survcomp:", requireNamespace("survcomp", quietly=TRUE), "\n")
cat("plsRcox:", requireNamespace("plsRcox", quietly=TRUE), "\n")
