user_lib <- file.path(Sys.getenv("USERPROFILE"), "R", "win-library",
                       paste0(R.version$major, ".", as.integer(R.version$minor)))
.libPaths(c(user_lib, .libPaths()))

pkgs <- c("mixOmics","ggplot2","survminer","randomForestSRC","xgboost",
          "ncvreg","obliqueRSF","party","mboost","stabs","mRMRe",
          "survivalROC","Boruta","survivalsvm","CoxBoost","plsRcox",
          "superpc","gbm","BART","R.utils")
for (p in pkgs) {
  cat(sprintf("%-20s %s\n", p, ifelse(requireNamespace(p, quietly=TRUE), "OK", "MISSING")))
}
