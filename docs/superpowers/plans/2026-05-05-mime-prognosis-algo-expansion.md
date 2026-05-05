# Mime Prognosis Model Algorithm Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the Mime R package's prognosis model with 8 new survival algorithms, 4 new feature selectors, Top-N ensemble voting, error isolation, and optional extra metrics — pure R, no Python.

**Architecture:** Wrapper/adapter layer — existing code is untouched. New `safe_run_algo()` provides tryCatch + withTimeout isolation. New algorithms are appended as code blocks in `ML.Dev.Prog.Sig.R`'s `mode == "all"` branch. New predict blocks are appended in `cal_RS_ml_res.R`. Ensemble logic and extra metrics are new R files.

**Tech Stack:** R 4.1+, existing: `glmnet`, `randomForestSRC`, `gbm`, `CoxBoost`, `xgboost`, `survival`. New: `ncvreg`, `obliqueRSF`, `party`, `mboost`, `stabs`, `Boruta`, `mRMRe`, `R.utils`, `pec`, `survC1`, `survIDINRI`.

---

### Task 1: Update DESCRIPTION with new dependencies

**Files:**
- Modify: `DESCRIPTION:26-76`

- [ ] **Step 1: Add new Imports entries**

Open `DESCRIPTION`. In the `Imports:` block (lines 27–76), add the following entries in alphabetical order among the existing imports:

```
    ncvreg,
    obliqueRSF,
    party,
    mboost,
    stabs,
    mRMRe,
    R.utils,
    pec,
    survC1,
    survIDINRI,
```

Note: `Boruta` is already present in DESCRIPTION (line 74). `xgboost` is already present via `Ckmeans.1d.dp` dependency. Ensure `xgboost` is also listed explicitly if not already:

```
    xgboost,
```

- [ ] **Step 2: Verify DESCRIPTION parses correctly**

Run in R console:

```r
desc <- read.dcf("DESCRIPTION")
cat("Imports:", desc[1, "Imports"], "\n")
```

Expected: prints all imports including the new packages without error.

- [ ] **Step 3: Commit**

```bash
git add DESCRIPTION
git commit -m "feat: add new R package dependencies for algo expansion"
```

---

### Task 2: Add safe_run_algo() helper to ML.Dev.Prog.Sig.R

**Files:**
- Modify: `R/ML.Dev.Prog.Sig.R:37-38` (after function signature, before first `if`)

- [ ] **Step 1: Add safe_run_algo() at the top of the function body**

Insert the following code immediately after line 38 (the opening `{` of the function body), before the `if (is.null(alpha_for_Enet)` block:

```r
  # --- Error isolation wrapper (new) ---
  safe_run_algo <- function(algo_name, expr, timeout_sec = algo_timeout) {
    tryCatch(
      R.utils::withTimeout(
        expr,
        timeout = timeout_sec,
        onTimeout = "error"
      ),
      error = function(e) {
        warning(sprintf("[SKIP] %s failed: %s", algo_name, e$message))
        return(NULL)
      }
    )
  }
```

- [ ] **Step 2: Add new parameters to the function signature**

Modify the function signature (line 23) to add three new parameters after `cores_for_parallel`:

```r
ML.Dev.Prog.Sig = function(train_data,
                           list_train_vali_Data,
                           candidate_genes = NULL,
                           unicox.filter.for.candi = NULL,
                           unicox_p_cutoff = NULL,
                           mode = NULL,
                           single_ml = NULL,
                           alpha_for_Enet = NULL,
                           direction_for_stepcox = NULL,
                           double_ml1 = NULL,
                           double_ml2 = NULL,
                           nodesize = NULL,
                           seed = NULL,
                           cores_for_parallel = NULL,
                           ensemble_top_n = 0L,
                           algo_timeout = 1800L,
                           extra_metrics = FALSE
                           ){
```

- [ ] **Step 3: Add default handling for new parameters**

After the existing default-handling blocks (after the `direction_for_stepcox` default block, around line 57), add:

```r
  if (is.null(ensemble_top_n)) {
    ensemble_top_n <- 0L
  }

  if (is.null(algo_timeout) || algo_timeout <= 0) {
    algo_timeout <- 1800L
  }

  if (is.null(extra_metrics)) {
    extra_metrics <- FALSE
  }
```

- [ ] **Step 4: Add library() calls for new packages**

In the `if(T)` library-loading block (lines 62–88), add after the existing library calls:

```r
    library(R.utils)
    library(ncvreg)
    library(obliqueRSF)
    library(party)
    library(mboost)
    library(stabs)
    library(mRMRe)
```

Note: `Boruta` is already loaded elsewhere. Conditionally load extra_metrics packages later.

- [ ] **Step 5: Verify no syntax errors**

```bash
Rscript -e "tryCatch(parse(file='R/ML.Dev.Prog.Sig.R'), error=function(e) cat('PARSE ERROR:', e$message, '\n'))"
```

Expected: no output (successful parse).

- [ ] **Step 6: Commit**

```bash
git add R/ML.Dev.Prog.Sig.R
git commit -m "feat: add safe_run_algo() wrapper and new parameters to ML.Dev.Prog.Sig"
```

---

### Task 3: Add XGBoost-Survival (AFT) algorithm block

**Files:**
- Modify: `R/ML.Dev.Prog.Sig.R` — append after existing algorithm blocks in `mode == "all"`, before the result assembly at the end

**Context:** In the `mode == "all"` branch, each algorithm follows this pattern:
```r
message("---N-M AlgoName ---")
set.seed(seed)
# ... train model ...
rs <- lapply(val_dd_list, function(x){cbind(x[,1:2], RS = ...)})
rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
cc <- data.frame(Cindex = sapply(rs, function(x){...})) %>% rownames_to_column('ID')
cc$Model <- 'AlgoName'
result <- rbind(result, cc)
ml.res[['AlgoName']] <- fit
riskscore[['AlgoName']] <- rs
```

- [ ] **Step 1: Add XGBoost-Surv (AFT) block**

Find the last single-algorithm block in `mode == "all"` (should be Lasso, ending before the combination blocks start). Insert after it:

```r
      # NEW: XGBoost-Survival (AFT) --------------------------------------------------------------
      message("---NEW: XGBoost-Surv (AFT) ---")
      ml.res[["XGBoost-Surv"]] <- safe_run_algo("XGBoost-Surv", {
        set.seed(seed)
        x_train <- as.matrix(est_dd[, -c(1, 2)])
        y_lower <- ifelse(est_dd$OS == 1, est_dd$OS.time, est_dd$OS.time)
        y_upper <- ifelse(est_dd$OS == 1, est_dd$OS.time, Inf)
        dtrain <- xgboost::xgb.DMatrix(data = x_train)
        xgboost::setinfo(dtrain, "label_lower", y_lower)
        xgboost::setinfo(dtrain, "label_upper", y_upper)
        params <- list(
          objective = "survival:aft",
          eval_metric = "aft-nloglik",
          aft_loss_distribution = "normal",
          max_depth = 6,
          eta = 0.05,
          subsample = 0.8,
          colsample_bytree = 0.8
        )
        cv_res <- xgboost::xgb.cv(
          params = params, data = dtrain, nrounds = 500,
          nfold = 10, early_stopping_rounds = 50, verbose = 0
        )
        best_nrounds <- cv_res$best_iteration
        xgboost::xgb.train(params = params, data = dtrain, nrounds = best_nrounds)
      })

      if (!is.null(ml.res[["XGBoost-Surv"]])) {
        fit_xgbsurv <- ml.res[["XGBoost-Surv"]]
        feature.ac <- fit_xgbsurv$feature_names
        rs <- lapply(val_dd_list, function(x) {
          dtest <- xgboost::xgb.DMatrix(data = as.matrix(x[, feature.ac]))
          cbind(x[, 1:2], RS = -as.numeric(predict(fit_xgbsurv, dtest)))
        })
        rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
        cc <- data.frame(Cindex = sapply(rs, function(x) {
          as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, x))$concordance[1])
        })) %>% rownames_to_column('ID')
        cc$Model <- "XGBoost-Surv"
        result <- rbind(result, cc)
        riskscore[["XGBoost-Surv"]] <- rs
      }
```

Note: Risk score is negated because AFT predicts survival time (higher = better), but risk score convention is higher = worse.

- [ ] **Step 2: Verify parse**

```bash
Rscript -e "tryCatch(parse(file='R/ML.Dev.Prog.Sig.R'), error=function(e) cat('PARSE ERROR:', e$message, '\n'))"
```

- [ ] **Step 3: Commit**

```bash
git add R/ML.Dev.Prog.Sig.R
git commit -m "feat: add XGBoost-Surv (AFT) algorithm to prognosis models"
```

---

### Task 4: Add MCP-Cox and SCAD-Cox algorithm blocks

**Files:**
- Modify: `R/ML.Dev.Prog.Sig.R` — append after XGBoost-Surv block

- [ ] **Step 1: Add MCP-Cox block**

```r
      # NEW: MCP-Cox --------------------------------------------------------------
      message("---NEW: MCP-Cox ---")
      ml.res[["MCP"]] <- safe_run_algo("MCP", {
        set.seed(seed)
        x_train <- as.matrix(est_dd[, -c(1, 2)])
        y_train <- Surv(est_dd$OS.time, est_dd$OS)
        ncvreg::cv.ncvsurv(X = x_train, y = y_train, penalty = "MCP", nfolds = 10)
      })

      if (!is.null(ml.res[["MCP"]])) {
        fit_mcp <- ml.res[["MCP"]]
        feature.ac <- colnames(fit_mcp$fit$X)[-1]  # exclude intercept
        if (is.null(feature.ac)) feature.ac <- colnames(as.matrix(est_dd[, -c(1, 2)]))
        rs <- lapply(val_dd_list, function(x) {
          cbind(x[, 1:2], RS = as.numeric(
            predict(fit_mcp, X = as.matrix(x[, feature.ac]), type = "lp")
          ))
        })
        rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
        cc <- data.frame(Cindex = sapply(rs, function(x) {
          as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, x))$concordance[1])
        })) %>% rownames_to_column('ID')
        cc$Model <- "MCP"
        result <- rbind(result, cc)
        riskscore[["MCP"]] <- rs
      }
```

- [ ] **Step 2: Add SCAD-Cox block**

```r
      # NEW: SCAD-Cox --------------------------------------------------------------
      message("---NEW: SCAD-Cox ---")
      ml.res[["SCAD"]] <- safe_run_algo("SCAD", {
        set.seed(seed)
        x_train <- as.matrix(est_dd[, -c(1, 2)])
        y_train <- Surv(est_dd$OS.time, est_dd$OS)
        ncvreg::cv.ncvsurv(X = x_train, y = y_train, penalty = "SCAD", nfolds = 10)
      })

      if (!is.null(ml.res[["SCAD"]])) {
        fit_scad <- ml.res[["SCAD"]]
        feature.ac <- colnames(as.matrix(est_dd[, -c(1, 2)]))
        rs <- lapply(val_dd_list, function(x) {
          cbind(x[, 1:2], RS = as.numeric(
            predict(fit_scad, X = as.matrix(x[, feature.ac]), type = "lp")
          ))
        })
        rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
        cc <- data.frame(Cindex = sapply(rs, function(x) {
          as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, x))$concordance[1])
        })) %>% rownames_to_column('ID')
        cc$Model <- "SCAD"
        result <- rbind(result, cc)
        riskscore[["SCAD"]] <- rs
      }
```

- [ ] **Step 3: Verify parse and commit**

```bash
Rscript -e "tryCatch(parse(file='R/ML.Dev.Prog.Sig.R'), error=function(e) cat('PARSE ERROR:', e$message, '\n'))"
git add R/ML.Dev.Prog.Sig.R
git commit -m "feat: add MCP-Cox and SCAD-Cox algorithms"
```

---

### Task 5: Add Adaptive Lasso Cox block

**Files:**
- Modify: `R/ML.Dev.Prog.Sig.R`

- [ ] **Step 1: Add Adaptive Lasso Cox block**

```r
      # NEW: Adaptive Lasso Cox --------------------------------------------------------------
      message("---NEW: Adaptive Lasso Cox ---")
      ml.res[["Alasso"]] <- safe_run_algo("Alasso", {
        set.seed(seed)
        x_train <- as.matrix(est_dd[, -c(1, 2)])
        y_train <- Surv(est_dd$OS.time, est_dd$OS)
        # Step 1: Ridge for initial coefficient estimates
        init_fit <- glmnet::cv.glmnet(x_train, y_train, family = "cox", alpha = 0, nfolds = 10)
        init_coef <- as.numeric(coef(init_fit, s = "lambda.min"))
        penalty_wts <- 1 / (abs(init_coef) + 1e-8)
        # Step 2: Weighted Lasso
        glmnet::cv.glmnet(x_train, y_train, family = "cox", alpha = 1,
                           nfolds = 10, penalty.factor = penalty_wts)
      })

      if (!is.null(ml.res[["Alasso"]])) {
        fit_alasso <- ml.res[["Alasso"]]
        feature.ac <- fit_alasso$glmnet.fit$beta@Dimnames[[1]]
        rs <- lapply(val_dd_list, function(x) {
          cbind(x[, 1:2], RS = as.numeric(
            predict(fit_alasso, type = "link", newx = as.matrix(x[, feature.ac]),
                    s = fit_alasso$lambda.min)
          ))
        })
        rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
        cc <- data.frame(Cindex = sapply(rs, function(x) {
          as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, x))$concordance[1])
        })) %>% rownames_to_column('ID')
        cc$Model <- "Alasso"
        result <- rbind(result, cc)
        riskscore[["Alasso"]] <- rs
      }
```

- [ ] **Step 2: Verify and commit**

```bash
Rscript -e "tryCatch(parse(file='R/ML.Dev.Prog.Sig.R'), error=function(e) cat('PARSE ERROR:', e$message, '\n'))"
git add R/ML.Dev.Prog.Sig.R
git commit -m "feat: add Adaptive Lasso Cox algorithm"
```

---

### Task 6: Add Oblique RSF and Conditional Inference Forest blocks

**Files:**
- Modify: `R/ML.Dev.Prog.Sig.R`

- [ ] **Step 1: Add Oblique RSF block**

```r
      # NEW: Oblique Random Survival Forest --------------------------------------------------------------
      message("---NEW: Oblique RSF ---")
      ml.res[["ORSF"]] <- safe_run_algo("ORSF", {
        set.seed(seed)
        obliqueRSF::orsf(
          formula = Surv(OS.time, OS) ~ .,
          data = est_dd,
          n_tree = 500
        )
      })

      if (!is.null(ml.res[["ORSF"]])) {
        fit_orsf <- ml.res[["ORSF"]]
        feature.ac <- names(fit_orsf$data$types)  # variable names from orsf
        rs <- lapply(val_dd_list, function(x) {
          cbind(x[, 1:2], RS = as.numeric(predict(fit_orsf, new_data = x)))
        })
        rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
        cc <- data.frame(Cindex = sapply(rs, function(x) {
          as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, x))$concordance[1])
        })) %>% rownames_to_column('ID')
        cc$Model <- "ORSF"
        result <- rbind(result, cc)
        riskscore[["ORSF"]] <- rs
      }
```

- [ ] **Step 2: Add Conditional Inference Forest block**

```r
      # NEW: Conditional Inference Forest --------------------------------------------------------------
      message("---NEW: Conditional Inference Forest ---")
      ml.res[["CIF"]] <- safe_run_algo("CIF", {
        set.seed(seed)
        party::cforest(
          Surv(OS.time, OS) ~ .,
          data = est_dd,
          controls = party::cforest_control(ntree = 500)
        )
      })

      if (!is.null(ml.res[["CIF"]])) {
        fit_cif <- ml.res[["CIF"]]
        rs <- lapply(val_dd_list, function(x) {
          cbind(x[, 1:2], RS = as.numeric(predict(fit_cif, newdata = x, type = "response")))
        })
        rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
        cc <- data.frame(Cindex = sapply(rs, function(x) {
          as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, x))$concordance[1])
        })) %>% rownames_to_column('ID')
        cc$Model <- "CIF"
        result <- rbind(result, cc)
        riskscore[["CIF"]] <- rs
      }
```

- [ ] **Step 3: Verify and commit**

```bash
Rscript -e "tryCatch(parse(file='R/ML.Dev.Prog.Sig.R'), error=function(e) cat('PARSE ERROR:', e$message, '\n'))"
git add R/ML.Dev.Prog.Sig.R
git commit -m "feat: add Oblique RSF and Conditional Inference Forest algorithms"
```

---

### Task 7: Add mboost and XGB-Cox blocks

**Files:**
- Modify: `R/ML.Dev.Prog.Sig.R`

- [ ] **Step 1: Add mboost block**

```r
      # NEW: mboost (Model-based Boosting) --------------------------------------------------------------
      message("---NEW: mboost ---")
      ml.res[["mboost"]] <- safe_run_algo("mboost", {
        set.seed(seed)
        surv_obj <- Surv(est_dd$OS.time, est_dd$OS)
        # Build formula with all feature columns
        fmla <- as.formula(paste("surv_obj ~", paste(
          paste0("`", colnames(est_dd)[-c(1, 2)]", "`"), collapse = "+"
        )))
        fit_mboost <- mboost::mboost(
          fmla, data = est_dd[, -c(1, 2)],
          family = mboost::CoxPH(),
          control = mboost::boost_control(mstop = 200, nu = 0.1)
        )
        # CV for optimal mstop
        cv_res <- mboost::cvrisk(fit_mboost, folds = mboost::cv(model.weights(fit_mboost), type = "kfold", B = 10))
        fit_mboost[mstop(cv_res)]
      })

      if (!is.null(ml.res[["mboost"]])) {
        fit_mboost <- ml.res[["mboost"]]
        rs <- lapply(val_dd_list, function(x) {
          cbind(x[, 1:2], RS = as.numeric(predict(fit_mboost, newdata = x[, -c(1, 2)], type = "link")))
        })
        rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
        cc <- data.frame(Cindex = sapply(rs, function(x) {
          as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, x))$concordance[1])
        })) %>% rownames_to_column('ID')
        cc$Model <- "mboost"
        result <- rbind(result, cc)
        riskscore[["mboost"]] <- rs
      }
```

- [ ] **Step 2: Add XGB-Cox block**

```r
      # NEW: XGB-Cox (XGBoost survival:cox) --------------------------------------------------------------
      message("---NEW: XGB-Cox ---")
      ml.res[["XGB-Cox"]] <- safe_run_algo("XGB-Cox", {
        set.seed(seed)
        x_train <- as.matrix(est_dd[, -c(1, 2)])
        surv_label <- ifelse(est_dd$OS == 1, est_dd$OS.time, -est_dd$OS.time)
        dtrain <- xgboost::xgb.DMatrix(data = x_train, label = surv_label)
        params <- list(
          objective = "survival:cox",
          max_depth = 4,
          eta = 0.01,
          subsample = 0.8,
          colsample_bytree = 0.8
        )
        cv_res <- xgboost::xgb.cv(
          params = params, data = dtrain, nrounds = 1000,
          nfold = 10, early_stopping_rounds = 50, verbose = 0
        )
        best_nrounds <- cv_res$best_iteration
        xgboost::xgb.train(params = params, data = dtrain, nrounds = best_nrounds)
      })

      if (!is.null(ml.res[["XGB-Cox"]])) {
        fit_xgbcox <- ml.res[["XGB-Cox"]]
        feature.ac <- fit_xgbcox$feature_names
        rs <- lapply(val_dd_list, function(x) {
          dtest <- xgboost::xgb.DMatrix(data = as.matrix(x[, feature.ac]))
          cbind(x[, 1:2], RS = as.numeric(predict(fit_xgbcox, dtest)))
        })
        rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
        cc <- data.frame(Cindex = sapply(rs, function(x) {
          as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, x))$concordance[1])
        })) %>% rownames_to_column('ID')
        cc$Model <- "XGB-Cox"
        result <- rbind(result, cc)
        riskscore[["XGB-Cox"]] <- rs
      }
```

- [ ] **Step 3: Verify and commit**

```bash
Rscript -e "tryCatch(parse(file='R/ML.Dev.Prog.Sig.R'), error=function(e) cat('PARSE ERROR:', e$message, '\n'))"
git add R/ML.Dev.Prog.Sig.R
git commit -m "feat: add mboost and XGB-Cox algorithms"
```

---

### Task 8: Add 4 new first-stage feature selectors

**Files:**
- Modify: `R/ML.Dev.Prog.Sig.R` — in the combination model section of `mode == "all"`

**Context:** Existing first-stage selectors (RSF, StepCox[both], StepCox[backward], StepCox[forward]) extract features, then run each second-stage algorithm. Each new selector follows the same pattern: extract `rid` (selected feature names), then for each second-stage algorithm, subset data to `rid` features and train.

- [ ] **Step 1: Add Boruta first-stage selector**

Find the last combination block (should be a StepCox variant + some algo). After it, add:

```r
      # --- NEW: Boruta first-stage selector ---
      message("---NEW: Boruta feature selection ---")
      boruta_sel <- safe_run_algo("Boruta-feature-sel", {
        set.seed(seed)
        # Convert survival to binary classification for Boruta
        med_time <- median(est_dd$OS.time)
        y_class <- ifelse(est_dd$OS == 1 & est_dd$OS.time < med_time, "poor",
                   ifelse(est_dd$OS == 0 & est_dd$OS.time >= med_time, "good", NA))
        keep_idx <- !is.na(y_class)
        boruta_data <- est_dd[keep_idx, -c(1, 2)]
        boruta_y <- as.factor(y_class[keep_idx])
        boruta_res <- Boruta::Boruta(x = boruta_data, y = boruta_y, pValue = 0.01, maxRuns = 1000)
        names(boruta_res$finalDecision[boruta_res$finalDecision == "Confirmed"])
      })

      if (!is.null(boruta_sel) && length(boruta_sel) > 1) {
        rid <- boruta_sel
        est_dd2 <- train_data[, c('OS.time', 'OS', rid)]
        val_dd_list2 <- lapply(list_train_vali_Data, function(x){x[, c('OS.time', 'OS', rid)]})

        # Boruta + RSF
        message("---NEW: Boruta + RSF ---")
        ml.res[["Boruta + RSF"]] <- safe_run_algo("Boruta + RSF", {
          set.seed(seed)
          rfsrc(Surv(OS.time, OS) ~ ., data = est_dd2, ntree = 1000,
                nodesize = rf_nodesize, splitrule = 'logrank', importance = TRUE, seed = seed)
        })
        if (!is.null(ml.res[["Boruta + RSF"]])) {
          fit <- ml.res[["Boruta + RSF"]]
          rs <- lapply(val_dd_list2, function(x){cbind(x[,1:2], RS = predict(fit, newdata = x)$predicted)})
          rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
          cc <- data.frame(Cindex = sapply(rs, function(x){as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, x))$concordance[1])})) %>% rownames_to_column('ID')
          cc$Model <- 'Boruta + RSF'
          result <- rbind(result, cc)
          riskscore[['Boruta + RSF']] <- rs
        }

        # Boruta + CoxBoost
        message("---NEW: Boruta + CoxBoost ---")
        ml.res[["Boruta + CoxBoost"]] <- safe_run_algo("Boruta + CoxBoost", {
          set.seed(seed)
          pen <- CoxBoost::optimCoxBoostPenalty(est_dd2[,'OS.time'], est_dd2[,'OS'], as.matrix(est_dd2[,-c(1,2)]), trace=TRUE, start.penalty=500)
          cv.res <- CoxBoost::cv.CoxBoost(est_dd2[,'OS.time'], est_dd2[,'OS'], as.matrix(est_dd2[,-c(1,2)]), maxstepno=500, K=10, type="verweij", penalty=pen$penalty)
          CoxBoost::CoxBoost(est_dd2[,'OS.time'], est_dd2[,'OS'], as.matrix(est_dd2[,-c(1,2)]), stepno=cv.res$optimal.step, penalty=pen$penalty)
        })
        if (!is.null(ml.res[["Boruta + CoxBoost"]])) {
          fit <- ml.res[["Boruta + CoxBoost"]]
          rs <- lapply(val_dd_list2, function(x){cbind(x[,1:2], RS = as.numeric(predict(fit, newdata=x[,-c(1,2)], newtime=x[,1], newstatus=x[,2], type="lp")))})
          rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
          cc <- data.frame(Cindex = sapply(rs, function(x){as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, x))$concordance[1])})) %>% rownames_to_column('ID')
          cc$Model <- 'Boruta + CoxBoost'
          result <- rbind(result, cc)
          riskscore[['Boruta + CoxBoost']] <- rs
        }

        # Boruta + all Enet alphas
        message("---NEW: Boruta + Enet ---")
        for (alpha in seq(0.1, 0.9, 0.1)) {
          model_name <- paste0('Boruta + Enet[', alpha, ']')
          ml.res[[model_name]] <- safe_run_algo(model_name, {
            set.seed(seed)
            glmnet::cv.glmnet(as.matrix(est_dd2[,rid]), Surv(est_dd2$OS.time, est_dd2$OS), family="cox", alpha=alpha, nfolds=10)
          })
          if (!is.null(ml.res[[model_name]])) {
            fit <- ml.res[[model_name]]
            rs <- lapply(val_dd_list2, function(x){cbind(x[,1:2], RS = as.numeric(predict(fit, type="link", newx=as.matrix(x[,rid]), s=fit$lambda.min)))})
            rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
            cc <- data.frame(Cindex = sapply(rs, function(x){as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, x))$concordance[1])})) %>% rownames_to_column('ID')
            cc$Model <- model_name
            result <- rbind(result, cc)
            riskscore[[model_name]] <- rs
          }
        }

        # Boruta + GBM
        message("---NEW: Boruta + GBM ---")
        ml.res[["Boruta + GBM"]] <- safe_run_algo("Boruta + GBM", {
          set.seed(seed)
          fit <- gbm(Surv(OS.time,OS)~., data=est_dd2, distribution='coxph', n.trees=10000, interaction.depth=3, n.minobsinnode=10, shrinkage=0.001, cv.folds=10)
          best <- which.min(fit$cv.error)
          list(gbm(Surv(OS.time,OS)~., data=est_dd2, distribution='coxph', n.trees=best, interaction.depth=3, n.minobsinnode=10, shrinkage=0.001), best)
        })
        if (!is.null(ml.res[["Boruta + GBM"]])) {
          fit <- ml.res[["Boruta + GBM"]][[1]]
          best <- ml.res[["Boruta + GBM"]][[2]]
          rs <- lapply(val_dd_list2, function(x){cbind(x[,1:2], RS = as.numeric(predict(fit, x, n.trees=best, type="link")))})
          rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
          cc <- data.frame(Cindex = sapply(rs, function(x){as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, x))$concordance[1])})) %>% rownames_to_column('ID')
          cc$Model <- 'Boruta + GBM'
          result <- rbind(result, cc)
          riskscore[['Boruta + GBM']] <- rs
        }

        # Boruta + remaining single algos: plsRcox, SuperPC, survivalsvm, Ridge, Lasso
        # Boruta + plsRcox
        message("---NEW: Boruta + plsRcox ---")
        ml.res[["Boruta + plsRcox"]] <- safe_run_algo("Boruta + plsRcox", {
          set.seed(seed)
          plsRcox::plsRcox(Surv(est_dd2$OS.time, est_dd2$OS)~., data=est_dd2[,-c(1,2)], nt=as.integer(10))
        })
        if (!is.null(ml.res[["Boruta + plsRcox"]])) {
          fit <- ml.res[["Boruta + plsRcox"]]
          feature.ac <- colnames(fit$dataX)
          rs <- lapply(val_dd_list2, function(x){cbind(x[,1:2], RS = as.numeric(predict(fit, type="lp", newdata=x[,feature.ac])))})
          rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
          cc <- data.frame(Cindex = sapply(rs, function(x){as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, x))$concordance[1])})) %>% rownames_to_column('ID')
          cc$Model <- 'Boruta + plsRcox'
          result <- rbind(result, cc)
          riskscore[['Boruta + plsRcox']] <- rs
        }

        # Boruta + SuperPC
        message("---NEW: Boruta + SuperPC ---")
        ml.res[["Boruta + SuperPC"]] <- safe_run_algo("Boruta + SuperPC", {
          set.seed(seed)
          data_sp <- list(x=t(est_dd2[,-c(1,2)]), y=est_dd2$OS.time, censoring.status=est_dd2$OS, featurenames=colnames(est_dd2)[-c(1,2)])
          fit_sp <- superpc::superpc.train(data_sp, type="survival", s0.perc=0.5)
          cv_sp <- superpc::superpc.cv(fit_sp, data_sp, n.threshold=20, n.fold=10, n.components=3)
          list(fit_sp, cv_sp)
        })
        if (!is.null(ml.res[["Boruta + SuperPC"]])) {
          fit_sp <- ml.res[["Boruta + SuperPC"]][[1]]
          cv_sp <- ml.res[["Boruta + SuperPC"]][[2]]
          data_sp <- list(x=t(est_dd2[,-c(1,2)]), y=est_dd2$OS.time, censoring.status=est_dd2$OS, featurenames=colnames(est_dd2)[-c(1,2)])
          rs <- lapply(val_dd_list2, function(w){
            test_sp <- list(x=t(w[,-c(1,2)]), y=w$OS.time, censoring.status=w$OS, featurenames=colnames(w)[-c(1,2)])
            ff <- superpc::superpc.predict(fit_sp, data_sp, test_sp, threshold=cv_sp$thresholds[which.max(cv_sp$scor[1,])], n.components=1)
            cbind(w[,1:2], RS=as.numeric(ff$v.pred))
          })
          rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
          cc <- data.frame(Cindex = sapply(rs, function(x){as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, x))$concordance[1])})) %>% rownames_to_column('ID')
          cc$Model <- 'Boruta + SuperPC'
          result <- rbind(result, cc)
          riskscore[['Boruta + SuperPC']] <- rs
        }

        # Boruta + survivalsvm
        message("---NEW: Boruta + survivalsvm ---")
        ml.res[["Boruta + survivalsvm"]] <- safe_run_algo("Boruta + survivalsvm", {
          survivalsvm::survivalsvm(Surv(OS.time,OS)~., data=est_dd2, gamma.mu=1)
        })
        if (!is.null(ml.res[["Boruta + survivalsvm"]])) {
          fit <- ml.res[["Boruta + survivalsvm"]]
          rs <- lapply(val_dd_list2, function(x){cbind(x[,1:2], RS = as.numeric(predict(fit, x)$predicted))})
          rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
          cc <- data.frame(Cindex = sapply(rs, function(x){as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, x))$concordance[1])})) %>% rownames_to_column('ID')
          cc$Model <- 'Boruta + survivalsvm'
          result <- rbind(result, cc)
          riskscore[['Boruta + survivalsvm']] <- rs
        }

        # Boruta + Ridge
        message("---NEW: Boruta + Ridge ---")
        ml.res[["Boruta + Ridge"]] <- safe_run_algo("Boruta + Ridge", {
          set.seed(seed)
          list(glmnet::glmnet(as.matrix(est_dd2[,rid]), Surv(est_dd2$OS.time, est_dd2$OS), family="cox", alpha=0),
               glmnet::cv.glmnet(as.matrix(est_dd2[,rid]), Surv(est_dd2$OS.time, est_dd2$OS), family="cox", alpha=0, nfolds=10))
        })
        if (!is.null(ml.res[["Boruta + Ridge"]])) {
          fit <- ml.res[["Boruta + Ridge"]][[1]]
          cv.fit <- ml.res[["Boruta + Ridge"]][[2]]
          rs <- lapply(val_dd_list2, function(x){cbind(x[,1:2], RS = as.numeric(predict(fit, type="response", newx=as.matrix(x[,rid]), s=cv.fit$lambda.min)))})
          rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
          cc <- data.frame(Cindex = sapply(rs, function(x){as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, x))$concordance[1])})) %>% rownames_to_column('ID')
          cc$Model <- 'Boruta + Ridge'
          result <- rbind(result, cc)
          riskscore[['Boruta + Ridge']] <- rs
        }

        # Boruta + Lasso
        message("---NEW: Boruta + Lasso ---")
        ml.res[["Boruta + Lasso"]] <- safe_run_algo("Boruta + Lasso", {
          set.seed(seed)
          glmnet::cv.glmnet(as.matrix(est_dd2[,rid]), Surv(est_dd2$OS.time, est_dd2$OS), family="cox", alpha=1, nfolds=10)
        })
        if (!is.null(ml.res[["Boruta + Lasso"]])) {
          fit <- ml.res[["Boruta + Lasso"]]
          rs <- lapply(val_dd_list2, function(x){cbind(x[,1:2], RS = as.numeric(predict(fit, type="response", newx=as.matrix(x[,rid]), s=fit$lambda.min)))})
          rs <- returnIDtoRS(rs.table.list = rs, rawtableID = list_train_vali_Data)
          cc <- data.frame(Cindex = sapply(rs, function(x){as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, x))$concordance[1])})) %>% rownames_to_column('ID')
          cc$Model <- 'Boruta + Lasso'
          result <- rbind(result, cc)
          riskscore[['Boruta + Lasso']] <- rs
        }

        # Boruta + new algos: MCP, SCAD, Alasso, ORSF, CIF, mboost, XGBoost-Surv, XGB-Cox
        # (Same pattern as above for each — abbreviated here, implement all 8)
      } else {
        warning("Boruta feature selection returned <= 1 feature, skipping Boruta combinations")
      }
```

- [ ] **Step 2: Add XGB-imp first-stage selector**

Follow the exact same pattern as Boruta but with XGBoost importance:

```r
      # --- NEW: XGB-imp first-stage selector ---
      message("---NEW: XGBoost Importance feature selection ---")
      xgbimp_sel <- safe_run_algo("XGB-imp-feature-sel", {
        set.seed(seed)
        surv_label <- ifelse(est_dd$OS == 1, est_dd$OS.time, -est_dd$OS.time)
        dtrain <- xgboost::xgb.DMatrix(data = as.matrix(est_dd[,-c(1,2)]), label = surv_label)
        xgb_model <- xgboost::xgb.train(
          params = list(objective = "survival:cox", max_depth = 6, eta = 0.1),
          data = dtrain, nrounds = 100, verbose = 0
        )
        imp <- xgboost::xgb.importance(model = xgb_model)
        as.character(imp$Feature[imp$Gain > 0.01])
      })

      if (!is.null(xgbimp_sel) && length(xgbimp_sel) > 1) {
        rid <- xgbimp_sel
        est_dd2 <- train_data[, c('OS.time', 'OS', rid)]
        val_dd_list2 <- lapply(list_train_vali_Data, function(x){x[, c('OS.time', 'OS', rid)]})
        # ... same combination blocks as Boruta (RSF, CoxBoost, Enet×9, GBM, plsRcox, SuperPC, survivalsvm, Ridge, Lasso, + 8 new algos)
        # Replace "Boruta" with "XGB-imp" in all model names
      } else {
        warning("XGB-imp feature selection returned <= 1 feature, skipping XGB-imp combinations")
      }
```

- [ ] **Step 3: Add StabSel first-stage selector**

```r
      # --- NEW: Stability Selection first-stage selector ---
      message("---NEW: Stability Selection feature selection ---")
      stabsel_sel <- safe_run_algo("StabSel-feature-sel", {
        set.seed(seed)
        stab_res <- stabs::stabsel(
          x = as.matrix(est_dd[, -c(1, 2)]),
          y = Surv(est_dd$OS.time, est_dd$OS),
          fitfun = stabs::glmnet.lasso,
          cutoff = 0.75,
          PFER = 1
        )
        names(which(stab_res$max > stab_res$cutoff))
      })

      if (!is.null(stabsel_sel) && length(stabsel_sel) > 1) {
        rid <- stabsel_sel
        est_dd2 <- train_data[, c('OS.time', 'OS', rid)]
        val_dd_list2 <- lapply(list_train_vali_Data, function(x){x[, c('OS.time', 'OS', rid)]})
        # ... same combination blocks (replace "Boruta" with "StabSel" in all model names)
      } else {
        warning("StabSel feature selection returned <= 1 feature, skipping StabSel combinations")
      }
```

- [ ] **Step 4: Add MRMR first-stage selector**

```r
      # --- NEW: MRMR first-stage selector ---
      message("---NEW: MRMR feature selection ---")
      mrmr_sel <- safe_run_algo("MRMR-feature-sel", {
        set.seed(seed)
        # Encode survival info as continuous label
        surv_label <- ifelse(est_dd$OS == 1, est_dd$OS.time, -est_dd$OS.time)
        mrmr_df <- data.frame(surv_target = surv_label, est_dd[, -c(1, 2)])
        dd <- mRMRe::mRMR.data(data = mrmr_df)
        res <- mRMRe::mRMR.ensemble(data = dd, target_indices = 1, feature_count = min(20, ncol(est_dd)-2))
        selected_idx <- as.numeric(unlist(solutions(res)[[1]])) - 1  # -1 because col 1 is target
        colnames(mrmr_df)[selected_idx + 1]  # +1 to skip surv_target
      })

      if (!is.null(mrmr_sel) && length(mrmr_sel) > 1) {
        rid <- mrmr_sel
        est_dd2 <- train_data[, c('OS.time', 'OS', rid)]
        val_dd_list2 <- lapply(list_train_vali_Data, function(x){x[, c('OS.time', 'OS', rid)]})
        # ... same combination blocks (replace "Boruta" with "MRMR" in all model names)
      } else {
        warning("MRMR feature selection returned <= 1 feature, skipping MRMR combinations")
      }
```

- [ ] **Step 5: Verify and commit**

```bash
Rscript -e "tryCatch(parse(file='R/ML.Dev.Prog.Sig.R'), error=function(e) cat('PARSE ERROR:', e$message, '\n'))"
git add R/ML.Dev.Prog.Sig.R
git commit -m "feat: add 4 new first-stage feature selectors (Boruta, XGB-imp, StabSel, MRMR)"
```

**Note for implementor:** For each of the 4 selectors, you need to add combinations with ALL 18 second-stage algorithms (same as the Boruta block above). This is repetitive but follows an identical pattern — just change `rid` source and model name prefix. You can factor this into a helper function if desired, or copy-paste the pattern 4 × 18 times as the existing code does.

---

### Task 9: Add ensemble Top-N voting logic

**Files:**
- Modify: `R/ML.Dev.Prog.Sig.R` — at the end of `mode == "all"`, before the `return(result.cindex.fit)`

- [ ] **Step 1: Add ensemble logic after risk score collection, before return**

```r
      # --- NEW: Ensemble Top-N Voting ---
      ensemble_rs <- NULL
      if (ensemble_top_n > 0 && nrow(result) > 0) {
        message(paste0("--- Computing Top-", ensemble_top_n, " Ensemble Risk Scores ---"))

        # Get training C-index for each model
        train_cindex <- result[result$ID == names(list_train_vali_Data)[1], ]
        train_cindex <- train_cindex[order(-train_cindex$Cindex), ]

        # Select Top-N models
        n_select <- min(ensemble_top_n, nrow(train_cindex))
        top_models <- train_cindex$Model[1:n_select]
        top_weights <- train_cindex$Cindex[1:n_select]

        message(paste0("Top-", n_select, " models: ", paste(top_models, collapse=", ")))

        ensemble_rs <- list()
        for (cohort_name in names(list_train_vali_Data)) {
          # Initialize weighted risk score
          n_patients <- nrow(list_train_vali_Data[[cohort_name]])
          weighted_rs <- rep(0, n_patients)
          total_weight <- 0

          for (m_idx in seq_along(top_models)) {
            model_name <- top_models[m_idx]
            w <- top_weights[m_idx]

            if (!is.null(riskscore[[model_name]]) &&
                !is.null(riskscore[[model_name]][[cohort_name]])) {
              raw_rs <- riskscore[[model_name]][[cohort_name]]$RS
              rs_range <- max(raw_rs) - min(raw_rs)
              if (rs_range > 0) {
                norm_rs <- (raw_rs - min(raw_rs)) / rs_range
              } else {
                norm_rs <- rep(0, length(raw_rs))
              }
              weighted_rs <- weighted_rs + w * norm_rs
              total_weight <- total_weight + w
            }
          }

          if (total_weight > 0) {
            ensemble_rs[[cohort_name]] <- weighted_rs / total_weight
          }
        }

        # Compute ensemble C-index
        ensemble_cindex <- data.frame()
        for (cohort_name in names(ensemble_rs)) {
          tmp_df <- data.frame(
            ID = list_train_vali_Data[[cohort_name]]$ID,
            OS.time = as.numeric(list_train_vali_Data[[cohort_name]]$OS.time),
            OS = as.numeric(list_train_vali_Data[[cohort_name]]$OS),
            RS = ensemble_rs[[cohort_name]]
          )
          ci <- as.numeric(summary(coxph(Surv(OS.time, OS) ~ RS, tmp_df))$concordance[1])
          ensemble_cindex <- rbind(ensemble_cindex,
            data.frame(ID = cohort_name, Cindex = ci, Model = paste0("Ensemble-Top", n_select)))
        }
        result <- rbind(result, ensemble_cindex)
        message("Ensemble C-index: ", paste(ensemble_cindex$Cindex, collapse=", "))
      }
```

- [ ] **Step 2: Add ensemble_rs to the return list**

Modify the return statement in `mode == "all"` to include ensemble:

```r
      result.cindex.fit <- list('Cindex.res' = result, 'ml.res' = ml.res,
                                 'riskscore' = riskscore, 'Sig.genes' = pre_var)
      if (!is.null(ensemble_rs)) {
        result.cindex.fit[['Ensemble_riskscore']] <- ensemble_rs
      }
      return(result.cindex.fit)
```

- [ ] **Step 3: Verify and commit**

```bash
Rscript -e "tryCatch(parse(file='R/ML.Dev.Prog.Sig.R'), error=function(e) cat('PARSE ERROR:', e$message, '\n'))"
git add R/ML.Dev.Prog.Sig.R
git commit -m "feat: add Top-N ensemble voting with C-index weighting"
```

---

### Task 10: Add predict blocks for new algorithms in cal_RS_ml_res.R

**Files:**
- Modify: `R/cal_RS_ml_res.R` — in `mode == "all"` branch, after existing algorithm predict blocks

- [ ] **Step 1: Add library calls for new packages**

In the library loading block (lines 28–53), add:

```r
    library(ncvreg)
    library(obliqueRSF)
    library(party)
    library(mboost)
    library(xgboost)
```

- [ ] **Step 2: Add predict blocks for each new algorithm**

After the existing SuperPC predict block (around line 403), add:

```r
        # NEW: XGBoost-Surv predict
        for (i in ml.names[grep("XGBoost-Surv", ml.names, fixed = TRUE)]) {
          if (!is.null(res.by.ML.Dev.Prog.Sig[["ml.res"]][[i]])) {
            print(i)
            fit <- res.by.ML.Dev.Prog.Sig[["ml.res"]][[i]]
            feature.ac <- fit$feature_names
            rs <- lapply(val_dd_list, function(x) {
              dtest <- xgb.DMatrix(as.matrix(x[, feature.ac]))
              cbind(x[, 1:2], RS = -as.numeric(predict(fit, dtest)))
            })
            rs <- returnIDtoRS(rs.table.list = rs, rawtableID = inputmatrix.list)
            riskscore[[i]] <- rs
          }
        }

        # NEW: MCP predict
        for (i in ml.names[grep("MCP", ml.names, fixed = TRUE)]) {
          if (!is.null(res.by.ML.Dev.Prog.Sig[["ml.res"]][[i]])) {
            print(i)
            fit <- res.by.ML.Dev.Prog.Sig[["ml.res"]][[i]]
            feature.ac <- colnames(as.matrix(train_data[, -c(1:3)]))  # use all features
            rs <- lapply(val_dd_list, function(x) {
              cbind(x[, 1:2], RS = as.numeric(predict(fit, X = as.matrix(x[, feature.ac]), type = "lp")))
            })
            rs <- returnIDtoRS(rs.table.list = rs, rawtableID = inputmatrix.list)
            riskscore[[i]] <- rs
          }
        }

        # NEW: SCAD predict
        for (i in ml.names[grep("SCAD", ml.names, fixed = TRUE)]) {
          if (!is.null(res.by.ML.Dev.Prog.Sig[["ml.res"]][[i]])) {
            print(i)
            fit <- res.by.ML.Dev.Prog.Sig[["ml.res"]][[i]]
            feature.ac <- colnames(as.matrix(train_data[, -c(1:3)]))
            rs <- lapply(val_dd_list, function(x) {
              cbind(x[, 1:2], RS = as.numeric(predict(fit, X = as.matrix(x[, feature.ac]), type = "lp")))
            })
            rs <- returnIDtoRS(rs.table.list = rs, rawtableID = inputmatrix.list)
            riskscore[[i]] <- rs
          }
        }

        # NEW: Alasso predict (same interface as Lasso/Enet)
        for (i in ml.names[grep("Alasso", ml.names, fixed = TRUE)]) {
          if (!is.null(res.by.ML.Dev.Prog.Sig[["ml.res"]][[i]])) {
            print(i)
            fit <- res.by.ML.Dev.Prog.Sig[["ml.res"]][[i]]
            feature.ac <- fit$glmnet.fit$beta@Dimnames[[1]]
            rs <- lapply(val_dd_list, function(x) {
              cbind(x[, 1:2], RS = as.numeric(predict(fit, type = "link", newx = as.matrix(x[, feature.ac]), s = fit$lambda.min)))
            })
            rs <- returnIDtoRS(rs.table.list = rs, rawtableID = inputmatrix.list)
            riskscore[[i]] <- rs
          }
        }

        # NEW: ORSF predict
        for (i in ml.names[grep("ORSF", ml.names, fixed = TRUE)]) {
          if (!is.null(res.by.ML.Dev.Prog.Sig[["ml.res"]][[i]])) {
            print(i)
            fit <- res.by.ML.Dev.Prog.Sig[["ml.res"]][[i]]
            rs <- lapply(val_dd_list, function(x) {
              cbind(x[, 1:2], RS = as.numeric(predict(fit, new_data = x)))
            })
            rs <- returnIDtoRS(rs.table.list = rs, rawtableID = inputmatrix.list)
            riskscore[[i]] <- rs
          }
        }

        # NEW: CIF predict
        for (i in ml.names[grep("CIF", ml.names, fixed = TRUE)]) {
          if (!is.null(res.by.ML.Dev.Prog.Sig[["ml.res"]][[i]])) {
            print(i)
            fit <- res.by.ML.Dev.Prog.Sig[["ml.res"]][[i]]
            rs <- lapply(val_dd_list, function(x) {
              cbind(x[, 1:2], RS = as.numeric(predict(fit, newdata = x, type = "response")))
            })
            rs <- returnIDtoRS(rs.table.list = rs, rawtableID = inputmatrix.list)
            riskscore[[i]] <- rs
          }
        }

        # NEW: mboost predict
        for (i in ml.names[grep("mboost", ml.names, fixed = TRUE)]) {
          if (!is.null(res.by.ML.Dev.Prog.Sig[["ml.res"]][[i]])) {
            print(i)
            fit <- res.by.ML.Dev.Prog.Sig[["ml.res"]][[i]]
            rs <- lapply(val_dd_list, function(x) {
              cbind(x[, 1:2], RS = as.numeric(predict(fit, newdata = x[, -c(1, 2)], type = "link")))
            })
            rs <- returnIDtoRS(rs.table.list = rs, rawtableID = inputmatrix.list)
            riskscore[[i]] <- rs
          }
        }

        # NEW: XGB-Cox predict
        for (i in ml.names[grep("XGB-Cox", ml.names, fixed = TRUE)]) {
          if (!is.null(res.by.ML.Dev.Prog.Sig[["ml.res"]][[i]])) {
            print(i)
            fit <- res.by.ML.Dev.Prog.Sig[["ml.res"]][[i]]
            feature.ac <- fit$feature_names
            rs <- lapply(val_dd_list, function(x) {
              dtest <- xgb.DMatrix(as.matrix(x[, feature.ac]))
              cbind(x[, 1:2], RS = as.numeric(predict(fit, dtest)))
            })
            rs <- returnIDtoRS(rs.table.list = rs, rawtableID = inputmatrix.list)
            riskscore[[i]] <- rs
          }
        }
```

- [ ] **Step 3: Verify and commit**

```bash
Rscript -e "tryCatch(parse(file='R/cal_RS_ml_res.R'), error=function(e) cat('PARSE ERROR:', e$message, '\n'))"
git add R/cal_RS_ml_res.R
git commit -m "feat: add predict blocks for 8 new algorithms in cal_RS_ml_res"
```

---

### Task 11: Create extra_metrics.R

**Files:**
- Create: `R/extra_metrics.R`

- [ ] **Step 1: Write the file**

```r
#' Compute extra evaluation metrics for prognosis models
#'
#' @param res.by.ML.Dev.Prog.Sig The results of function ML.Dev.Prog.Sig
#' @param train_data The training data
#' @param inputmatrix.list A list of validation/training data frames
#' @param auc_time_vec Numeric vector of time points in years for AUC (default c(1,3,5))
#' @return A list with IBS, UnoC, Dindex results per model
#' @export
cal_extra_metrics <- function(res.by.ML.Dev.Prog.Sig,
                               train_data,
                               inputmatrix.list,
                               auc_time_vec = c(1, 3, 5)) {
  library(survival)

  result <- list(IBS = list(), UnoC = list(), Dindex = list())

  sig_genes <- res.by.ML.Dev.Prog.Sig$Sig.genes
  ml_names <- names(res.by.ML.Dev.Prog.Sig$ml.res)
  cohort_names <- names(inputmatrix.list)

  # Preprocess input data (same as other downstream functions)
  inputmatrix.list <- lapply(inputmatrix.list, function(x) {
    colnames(x) <- gsub("-", ".", colnames(x))
    x
  })
  colnames(train_data) <- gsub("-", ".", colnames(train_data))

  for (model_name in ml_names) {
    if (is.null(res.by.ML.Dev.Prog.Sig$ml.res[[model_name]])) next

    # Integrated Brier Score
    result$IBS[[model_name]] <- tryCatch({
      library(pec)
      # Build Cox model from risk scores for pec input
      rs_list <- res.by.ML.Dev.Prog.Sig$riskscore[[model_name]]
      if (!is.null(rs_list) && !is.null(rs_list[[1]])) {
        train_rs <- rs_list[[1]]  # first cohort = training
        df <- data.frame(
          OS.time = as.numeric(train_rs$OS.time),
          OS = as.numeric(train_rs$OS),
          RS = train_rs$RS
        )
        cox_fit <- coxph(Surv(OS.time, OS) ~ RS, data = df)
        pec_res <- pec::pec(
          object = list("Mime_Model" = cox_fit),
          formula = Surv(OS.time, OS) ~ 1,
          data = df,
          exact = FALSE,
          verbose = FALSE
        )
        list(IBS = pec_res$AppErr$Mime_Model, time = pec_res$time)
      } else {
        NULL
      }
    }, error = function(e) {
      message(paste0("[SKIP] IBS for ", model_name, ": ", e$message))
      NULL
    })

    # Uno's C-statistic
    result$UnoC[[model_name]] <- tryCatch({
      library(survC1)
      rs_list <- res.by.ML.Dev.Prog.Sig$riskscore[[model_name]]
      if (!is.null(rs_list) && !is.null(rs_list[[1]])) {
        train_rs <- rs_list[[1]]
        df <- data.frame(
          time = as.numeric(train_rs$OS.time),
          status = as.numeric(train_rs$OS),
          marker = train_rs$RS
        )
        tau <- max(df$time) * 0.9
        survC1::Est.Cval(df, tau = tau, nofit = TRUE)
      } else {
        NULL
      }
    }, error = function(e) {
      message(paste0("[SKIP] UnoC for ", model_name, ": ", e$message))
      NULL
    })

    # D-index
    result$Dindex[[model_name]] <- tryCatch({
      rs_list <- res.by.ML.Dev.Prog.Sig$riskscore[[model_name]]
      if (!is.null(rs_list) && !is.null(rs_list[[1]])) {
        train_rs <- rs_list[[1]]
        df <- data.frame(
          OS.time = as.numeric(train_rs$OS.time),
          OS = as.numeric(train_rs$OS),
          RS = train_rs$RS
        )
        cox_fit <- coxph(Surv(OS.time, OS) ~ RS, data = df)
        # D-index = exp(coef) from Cox model
        list(
          Dindex = as.numeric(exp(coef(cox_fit))),
          lower = as.numeric(exp(confint(cox_fit))[1]),
          upper = as.numeric(exp(confint(cox_fit))[2]),
          pvalue = summary(cox_fit)$coefficients[, "Pr(>|z|)"]
        )
      } else {
        NULL
      }
    }, error = function(e) {
      message(paste0("[SKIP] Dindex for ", model_name, ": ", e$message))
      NULL
    })
  }

  return(result)
}
```

- [ ] **Step 2: Verify and commit**

```bash
Rscript -e "tryCatch(parse(file='R/extra_metrics.R'), error=function(e) cat('PARSE ERROR:', e$message, '\n'))"
git add R/extra_metrics.R
git commit -m "feat: add cal_extra_metrics() for IBS, UnoC, D-index"
```

---

### Task 12: Create ensemble_viz.R

**Files:**
- Create: `R/ensemble_viz.R`

- [ ] **Step 1: Write the file**

```r
#' Plot Kaplan-Meier survival curves for ensemble risk groups
#'
#' @param res.by.ML.Dev.Prog.Sig The results of ML.Dev.Prog.Sig (must contain Ensemble_riskscore)
#' @param list_train_vali_Data The list of cohort data used in ML.Dev.Prog.Sig
#' @param seed Random seed for reproducibility
#' @return A ggplot object (or list of ggplot objects, one per cohort)
#' @export
ensemble_km_plot <- function(res.by.ML.Dev.Prog.Sig,
                              list_train_vali_Data,
                              seed = 42) {
  library(survival)
  library(survminer)
  library(ggplot2)

  ensemble_rs <- res.by.ML.Dev.Prog.Sig$Ensemble_riskscore
  if (is.null(ensemble_rs)) {
    warning("No Ensemble_riskscore found in results. Run ML.Dev.Prog.Sig with ensemble_top_n > 0.")
    return(NULL)
  }

  plot_list <- list()
  for (cohort_name in names(ensemble_rs)) {
    df <- data.frame(
      ID = list_train_vali_Data[[cohort_name]]$ID,
      OS.time = as.numeric(list_train_vali_Data[[cohort_name]]$OS.time),
      OS = as.numeric(list_train_vali_Data[[cohort_name]]$OS),
      RS = ensemble_rs[[cohort_name]]
    )
    df$RiskGroup <- ifelse(df$RS >= median(df$RS), "High Risk", "Low Risk")

    fit <- survfit(Surv(OS.time, OS) ~ RiskGroup, data = df)
    p <- survminer::ggsurvplot(
      fit, data = df,
      pval = TRUE, risk.table = TRUE,
      palette = c("#E7B800", "#2E9FDF"),
      title = paste0("Ensemble Model — ", cohort_name),
      xlab = "Time (Days)",
      ylab = "Overall Survival Probability"
    )
    plot_list[[cohort_name]] <- p
  }

  return(plot_list)
}

#' Plot C-index comparison bar chart including ensemble
#'
#' @param res.by.ML.Dev.Prog.Sig The results of ML.Dev.Prog.Sig
#' @param cohort_name Name of cohort to plot (default: first cohort)
#' @return A ggplot object
#' @export
ensemble_cindex_compare <- function(res.by.ML.Dev.Prog.Sig,
                                     cohort_name = NULL) {
  library(ggplot2)
  library(dplyr)

  result <- res.by.ML.Dev.Prog.Sig$Cindex.res
  if (is.null(cohort_name)) {
    cohort_name <- unique(result$ID)[1]
  }
  df <- result[result$ID == cohort_name, ]
  df <- df[order(-df$Cindex), ]

  # Highlight ensemble model
  df$IsEnsemble <- grepl("Ensemble", df$Model)

  p <- ggplot(df, aes(x = reorder(Model, Cindex), y = Cindex, fill = IsEnsemble)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    scale_fill_manual(values = c("FALSE" = "grey60", "TRUE" = "steelblue"), guide = "none") +
    labs(title = paste("C-index Comparison —", cohort_name), x = "Model", y = "C-index") +
    theme_minimal()

  return(p)
}
```

- [ ] **Step 2: Verify and commit**

```bash
Rscript -e "tryCatch(parse(file='R/ensemble_viz.R'), error=function(e) cat('PARSE ERROR:', e$message, '\n'))"
git add R/ensemble_viz.R
git commit -m "feat: add ensemble KM plot and C-index comparison visualization"
```

---

### Task 13: Update roxygen docs and NAMESPACE

**Files:**
- Modify: `R/ML.Dev.Prog.Sig.R` (roxygen header)
- Modify: `R/extra_metrics.R` (already has roxygen)
- Modify: `R/ensemble_viz.R` (already has roxygen)
- Modify: `NAMESPACE`

- [ ] **Step 1: Update roxygen header for ML.Dev.Prog.Sig**

Add the new parameter docs to the roxygen block (lines 1–22):

```r
#' @param ensemble_top_n Integer. Number of top models to combine via C-index weighted voting. 0 = disabled (default).
#' @param algo_timeout Integer. Timeout in seconds for each algorithm (default 1800 = 30 minutes).
#' @param extra_metrics Logical. Whether to compute additional metrics (IBS, Uno's C, D-index) (default FALSE).
```

- [ ] **Step 2: Add new exports to NAMESPACE**

Append to `NAMESPACE`:

```
export(cal_extra_metrics)
export(ensemble_km_plot)
export(ensemble_cindex_compare)
```

- [ ] **Step 3: Regenerate docs (if roxygen2 available)**

```bash
Rscript -e "roxygen2::roxygenise('.')"
```

- [ ] **Step 4: Commit**

```bash
git add R/ML.Dev.Prog.Sig.R NAMESPACE R/extra_metrics.R R/ensemble_viz.R
git commit -m "docs: update roxygen docs and NAMESPACE for new exports"
```

---

### Task 14: Integration test — verify mode="all" with new algorithms

**Files:**
- Create: `tests/testthat/test-new-algorithms.R`

- [ ] **Step 1: Write integration test**

```r
test_that("ML.Dev.Prog.Sig mode=all includes new algorithms", {
  # Skip if example data not available
  skip_if(!file.exists("External data/Example.cohort.Rdata"))

  load("External data/Example.cohort.Rdata")

  # Run with a small seed and all algorithms
  res <- ML.Dev.Prog.Sig(
    train_data = Example.cohort[[1]],
    list_train_vali_Data = Example.cohort,
    mode = "all",
    seed = 42,
    ensemble_top_n = 5,
    algo_timeout = 300
  )

  # Check that at least some new algorithms appear in results
  model_names <- unique(res$Cindex.res$Model)
  new_algos <- c("XGBoost-Surv", "MCP", "SCAD", "Alasso", "ORSF", "CIF", "mboost", "XGB-Cox")
  found_new <- intersect(model_names, new_algos)
  expect_true(length(found_new) > 0,
    info = paste("Expected at least one new algorithm, found:", paste(found_new, collapse=", ")))

  # Check ensemble results exist
  expect_true(!is.null(res$Ensemble_riskscore),
    info = "Ensemble_riskscore should be present when ensemble_top_n > 0")

  # Check backward compatibility: existing keys present
  expect_true("Cindex.res" %in% names(res))
  expect_true("ml.res" %in% names(res))
  expect_true("riskscore" %in% names(res))
  expect_true("Sig.genes" %in% names(res))
})
```

- [ ] **Step 2: Run the test**

```bash
cd D:/test/Mime-main/Mime-main
Rscript -e "testthat::test_file('tests/testthat/test-new-algorithms.R')"
```

Expected: PASS (some algorithms may be skipped due to missing packages, but no crash).

- [ ] **Step 3: Commit**

```bash
git add tests/testthat/test-new-algorithms.R
git commit -m "test: add integration test for new algorithms and ensemble"
```

---

### Task 15: Update cal_AUC_ml_res.R and cal_unicox_ml_res.R for new algorithms

**Files:**
- Modify: `R/cal_AUC_ml_res.R`
- Modify: `R/cal_unicox_ml_res.R`

**Context:** `cal_AUC_ml_res.R` uses `res.by.ML.Dev.Prog.Sig$riskscore` directly (same list that ML.Dev.Prog.Sig populates). It does NOT need model-specific predict logic — it works on pre-computed risk scores from the `$riskscore` list. Therefore, new algorithms are automatically supported as long as their risk scores are stored in `$riskscore` by ML.Dev.Prog.Sig (Tasks 3–7).

However, `cal_unicox_ml_res.R` runs Cox regression on each model's risk scores. It also works on `$riskscore` directly.

**Verify that both files iterate over `names(res$riskscore)` dynamically** (not hardcoded model lists). If they do, no code changes are needed — new algorithms are automatically included.

- [ ] **Step 1: Check cal_AUC_ml_res.R for hardcoded model lists**

Search for hardcoded algorithm names in `R/cal_AUC_ml_res.R`. If the file uses `names(res$riskscore)` or iterates over all keys dynamically, no change needed.

```bash
grep -n "RSF\|Enet\|Lasso\|CoxBoost" R/cal_AUC_ml_res.R | head -20
```

If hardcoded names found, refactor to use `names(res$riskscore)` loop.

- [ ] **Step 2: Check cal_unicox_ml_res.R similarly**

```bash
grep -n "RSF\|Enet\|Lasso\|CoxBoost" R/cal_unicox_ml_res.R | head -20
```

Same check — if hardcoded, refactor to dynamic iteration.

- [ ] **Step 3: Verify and commit**

```bash
Rscript -e "tryCatch(parse(file='R/cal_AUC_ml_res.R'), error=function(e) cat('PARSE ERROR:', e$message, '\n'))"
Rscript -e "tryCatch(parse(file='R/cal_unicox_ml_res.R'), error=function(e) cat('PARSE ERROR:', e$message, '\n'))"
git add R/cal_AUC_ml_res.R R/cal_unicox_ml_res.R
git commit -m "feat: ensure cal_AUC and cal_unicox support new algorithms dynamically"
```

---

## Summary of all tasks

| Task | Description | Files |
|------|-------------|-------|
| 1 | Update DESCRIPTION dependencies | `DESCRIPTION` |
| 2 | Add safe_run_algo() + new params | `R/ML.Dev.Prog.Sig.R` |
| 3 | Add XGBoost-Surv (AFT) | `R/ML.Dev.Prog.Sig.R` |
| 4 | Add MCP-Cox and SCAD-Cox | `R/ML.Dev.Prog.Sig.R` |
| 5 | Add Adaptive Lasso Cox | `R/ML.Dev.Prog.Sig.R` |
| 6 | Add Oblique RSF and CIF | `R/ML.Dev.Prog.Sig.R` |
| 7 | Add mboost and XGB-Cox | `R/ML.Dev.Prog.Sig.R` |
| 8 | Add 4 first-stage selectors | `R/ML.Dev.Prog.Sig.R` |
| 9 | Add ensemble Top-N voting | `R/ML.Dev.Prog.Sig.R` |
| 10 | Add predict blocks | `R/cal_RS_ml_res.R` |
| 11 | Create extra_metrics.R | `R/extra_metrics.R` |
| 12 | Create ensemble_viz.R | `R/ensemble_viz.R` |
| 13 | Update docs and NAMESPACE | `NAMESPACE`, roxygen |
| 14 | Integration test | `tests/testthat/test-new-algorithms.R` |
| 15 | Update cal_AUC + cal_unicox | `R/cal_AUC_ml_res.R`, `R/cal_unicox_ml_res.R` |

Total new code: ~2000–3000 lines added across all files.
