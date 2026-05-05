# 检查并安装缺失的R包
user_lib <- file.path(Sys.getenv("USERPROFILE"), "R", "win-library",
                       paste0(R.version$major, ".", as.integer(R.version$minor)))
if (!dir.exists(user_lib)) dir.create(user_lib, recursive = TRUE)
.libPaths(c(user_lib, .libPaths()))

cat("Library paths:\n")
cat(paste(.libPaths(), collapse="\n"), "\n\n")

# 核心依赖
required_pkgs <- c(
  "ggplot2", "survival", "dplyr", "tibble", "tidyr", "ggsci", "ggbreak",
  "Matrix", "randomForestSRC", "glmnet", "plsRcox", "superpc", "gbm",
  "CoxBoost", "survivalsvm", "BART", "miscTools", "compareC",
  "data.table", "mixOmics", "R.utils", "ncvreg", "obliqueRSF", "party",
  "mboost", "stabs", "mRMRe", "xgboost", "survminer", "survivalROC",
  "Boruta", "ggpubr", "viridis", "gridExtra", "forestploter",
  "UpSetR", "aplot", "scales", "Hmisc", "ROCR", "Ckmeans.1d.dp",
  "compositions", "pbapply", "reshape2", "recipes", "readr",
  "stringr", "meta", "kknn", "caret", "ROCit", "pROC"
)

missing <- c()
for (p in required_pkgs) {
  installed <- requireNamespace(p, quietly = TRUE)
  status <- ifelse(installed, "OK", "MISSING")
  cat(sprintf("  %-25s %s\n", p, status))
  if (!installed) missing <- c(missing, p)
}

if (length(missing) > 0) {
  cat("\n缺少", length(missing), "个包，正在安装...\n")
  cat("缺失包:", paste(missing, collapse=", "), "\n")
  install.packages(missing, repos = "https://cloud.r-project.org", lib = user_lib, dependencies = TRUE)
  cat("安装完成!\n")
} else {
  cat("\n所有包已安装!\n")
}
