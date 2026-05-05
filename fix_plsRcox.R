user_lib <- file.path(Sys.getenv("USERPROFILE"), "R", "win-library",
                       paste0(R.version$major, ".", as.integer(R.version$minor)))
.libPaths(c(user_lib, .libPaths()))
install.packages("plsRcox", repos="https://cloud.r-project.org", lib=user_lib)
cat("plsRcox:", requireNamespace("plsRcox", quietly=TRUE), "\n")
