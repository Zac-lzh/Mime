# Mime Prognosis Model Algorithm Expansion Design

**Date**: 2026-05-05
**Scope**: Add cutting-edge ML algorithms, feature selection methods, ensemble strategies, and evaluation metrics to the Mime R package's prognosis (survival analysis) module. Pure R ecosystem — no deep learning / Python dependency.

---

## 1. Problem Statement

The Mime R package (`ML.Dev.Prog.Sig()`) currently implements 10 base survival algorithms and ~43 combination models. Key limitations:

- **No error isolation**: Any single algorithm failure crashes the entire function
- **No timeout protection**: Slow algorithms (e.g., GBM) can hang indefinitely
- **Limited algorithm diversity**: Missing modern boosting (XGBoost-Survival), advanced regularization (MCP/SCAD), and newer ensemble methods (Oblique RSF)
- **Limited feature selection**: Only 4 first-stage selectors (RSF + 3 StepCox variants)
- **No ensemble integration**: No mechanism to combine predictions from top-performing models
- **Basic evaluation**: Only C-index, time-dependent AUC, and HR

## 2. Design Approach: Wrapper/Adapter Layer

Keep existing code untouched. New functionality is added via:
- A `safe_run_algo()` wrapper for error isolation + timeout
- New algorithm blocks appended to the existing `mode == "all"` branch
- New first-stage selectors appended to the combination matrix
- New `cal_RS_ml_res.R` predict blocks for downstream compatibility
- Optional new evaluation metrics via `extra_metrics = TRUE`

## 3. Error Isolation Layer

### 3.1 safe_run_algo() Helper

Added at the top of `ML.Dev.Prog.Sig.R`:

```r
safe_run_algo <- function(algo_name, expr, timeout_sec = 1800) {
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

Default timeout: 30 minutes (1800 seconds). Configurable via new `algo_timeout` parameter.

### 3.2 Usage Pattern

Every algorithm block is wrapped:

```r
ml.res[["RSF"]] <- safe_run_algo("RSF", {
  rfsrc(Surv(OS.time, OS) ~ ., data = train_df, ntree = 1000, nodesize = nodesize,
        splitrule = "logrank", importance = TRUE)
}, timeout_sec = algo_timeout)
```

Failed algorithms return `NULL` and are silently skipped in downstream C-index calculation, risk score collection, and visualization.

### 3.3 Downstream Filtering

In `cal_RS_ml_res.R` and visualization functions, guard with:

```r
if (!is.null(ml.res[[algo_name]])) {
  # compute risk scores, C-index, etc.
}
```

## 4. New Survival Algorithms (8)

### 4.1 XGBoost-Survival (AFT Mode)

| Attribute | Value |
|-----------|-------|
| R Package | `xgboost` (already in DESCRIPTION) |
| Objective | `survival:aft` |
| Key Params | `aft_loss_distribution = "normal"`, `max_depth = 6`, `eta = 0.05` |
| CV | `xgb.cv()` with 10-fold, `early_stopping_rounds = 50` |
| Risk Score | `-predict(model, newdata)` (AFT predicts survival time; negate for risk) |
| Model Name | `"XGBoost-Surv"` |

### 4.2 MCP-Cox

| Attribute | Value |
|-----------|-------|
| R Package | `ncvreg` (new dependency) |
| Function | `ncvsurv()` with `penalty = "MCP"` |
| CV | `cv.ncvsurv()` with 10-fold |
| Risk Score | `predict(cvfit, X = ..., type = "lp")` |
| Model Name | `"MCP"` |

### 4.3 SCAD-Cox

| Attribute | Value |
|-----------|-------|
| R Package | `ncvreg` |
| Function | `ncvsurv()` with `penalty = "SCAD"` |
| CV | `cv.ncvsurv()` with 10-fold |
| Risk Score | `predict(cvfit, X = ..., type = "lp")` |
| Model Name | `"SCAD"` |

### 4.4 Adaptive Lasso Cox

| Attribute | Value |
|-----------|-------|
| R Package | `glmnet` (already in DESCRIPTION) |
| Method | Step 1: Ridge initial coefficients; Step 2: Weighted Lasso with `penalty.factor = 1/|coef|` |
| CV | `cv.glmnet()` with 10-fold, `alpha = 1` |
| Risk Score | `predict(fit, type = "link", newx = ..., s = "lambda.min")` |
| Model Name | `"Alasso"` |

### 4.5 Oblique Random Survival Forest

| Attribute | Value |
|-----------|-------|
| R Package | `obliqueRSF` (new dependency) |
| Function | `orsf()` |
| Key Params | `n_tree = 500` |
| Risk Score | `predict(model, newdata)` |
| Model Name | `"ORSF"` |

### 4.6 Conditional Inference Forest

| Attribute | Value |
|-----------|-------|
| R Package | `party` (new dependency) |
| Function | `cforest()` with `cforest_control(ntree = 500)` |
| Risk Score | `predict(model, newdata, type = "response")` |
| Model Name | `"CIF"` |

### 4.7 mboost (Model-based Boosting)

| Attribute | Value |
|-----------|-------|
| R Package | `mboost` (new dependency) |
| Function | `mboost()` with `family = CoxPH()` |
| CV | `cvrisk()` for optimal `mstop` selection |
| Risk Score | `predict(model, type = "link")` |
| Model Name | `"mboost"` |

### 4.8 XGB-Cox (XGBoost with survival:cox objective)

| Attribute | Value |
|-----------|-------|
| R Package | `xgboost` |
| Objective | `survival:cox` |
| Key Params | `max_depth = 4`, `eta = 0.01` |
| CV | `xgb.cv()` with 10-fold, `early_stopping_rounds = 50` |
| Risk Score | `predict(model, newdata)` (direct risk score) |
| Model Name | `"XGB-Cox"` |

### New Total Base Algorithms

10 (existing) + 8 (new) = **18 base algorithms**

## 5. New Feature Selection Methods (4 First-Stage Selectors)

### 5.1 Boruta

| Attribute | Value |
|-----------|-------|
| R Package | `Boruta` (new dependency) |
| Method | Convert survival to binary classification: patients with OS=1 and OS.time < median → "poor prognosis"; OS=0 and OS.time >= median → "good prognosis"; ties/discard ambiguous cases. Run `Boruta()` on the labeled subset. |
| Params | `pValue = 0.01`, `maxRuns = 1000` |
| Output | Features with `finalDecision == "Confirmed"` |
| Label | `"Boruta"` |

### 5.2 XGBoost Importance

| Attribute | Value |
|-----------|-------|
| R Package | `xgboost` |
| Method | Train XGBoost-Cox model; extract feature importance |
| Threshold | Features with `Gain > 0.01` |
| Label | `"XGB-imp"` |

### 5.3 Stability Selection

| Attribute | Value |
|-----------|-------|
| R Package | `stabs` (new dependency) |
| Method | `stabsel()` with Lasso base learner |
| Params | `cutoff = 0.75`, `PFER = 1` |
| Output | Features exceeding selection probability cutoff |
| Label | `"StabSel"` |

### 5.4 mRMRe (Maximum Relevance Minimum Redundancy)

| Attribute | Value |
|-----------|-------|
| R Package | `mRMRe` (new dependency) |
| Method | Encode survival info; run `mRMR.ensemble()` |
| Params | `feature_count = 20` |
| Label | `"MRMR"` |

### Combination Matrix

8 first-stage selectors x 18 second-stage algorithms = **144 double models**
Plus 18 single models = **162 total model configurations**

## 6. Ensemble Strategy: Top-N C-index Weighted Voting

### 6.1 Design

After all 162 models are trained, optionally generate an ensemble risk score:

1. Rank all models by training set C-index (descending)
2. Select Top-N models (configurable via `ensemble_top_n`)
3. For each cohort, compute weighted risk score:
   - Normalize each model's risk scores to [0, 1]
   - Weight by training C-index
   - Average weighted scores

### 6.2 New Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ensemble_top_n` | integer | 0 | 0 = disabled; >0 = Top-N ensemble |
| `algo_timeout` | integer | 1800 | Per-algorithm timeout in seconds |
| `extra_metrics` | logical | FALSE | Whether to compute IBS, Uno's C, D-index |

### 6.3 Output Extension

The return list is extended (backward-compatible — existing keys unchanged):

```r
result.cindex.fit <- list(
  'Cindex.res' = result,          # unchanged
  'ml.res' = ml.res,              # unchanged (new algorithms added)
  'riskscore' = riskscore,        # unchanged (new algorithms added)
  'Sig.genes' = pre_var,          # unchanged
  # New (conditional):
  'Ensemble_riskscore' = ensemble_rs,  # if ensemble_top_n > 0
  'IBS.res' = ibs_result,             # if extra_metrics = TRUE
  'UnoC.res' = unoc_result,           # if extra_metrics = TRUE
  'Dindex.res' = dindex_result        # if extra_metrics = TRUE
)
```

## 7. Evaluation Metrics Extension (Optional)

When `extra_metrics = TRUE`:

| Metric | R Package | Function | Output |
|--------|-----------|----------|--------|
| Integrated Brier Score | `pec` | `pec()` | Lower is better; `result$IBS.res` |
| Uno's C-statistic | `survC1` | `Est.Cval()` | Robust to censoring; `result$UnoC.res` |
| D-index | `survIDINRI` | `coxph()` based | Survival discrimination; `result$Dindex.res` |

Each metric computed with `tryCatch` — failure returns `NULL` for that metric.

## 8. Downstream Compatibility

### 8.1 cal_RS_ml_res.R

New algorithm predict blocks are appended, one per algorithm:

```r
for (i in ml.names[grep("MCP", ml.names, fixed = TRUE)]) {
  fit <- res.by.ML.Dev.Prog.Sig[["ml.res"]][[i]]
  # ... predict + store riskscore
}
```

### 8.2 Visualization Functions

Existing functions (`cindex_dis_all`, `auc_dis_all`, etc.) work unchanged because:
- They iterate over `result$Cindex.res` which auto-includes new algorithms
- They use `result$riskscore` which auto-includes new algorithm risk scores

New visualization function for ensemble:

```r
ensemble_km_plot <- function(result, list_data, ...) {
  # Kaplan-Meier plot for ensemble risk groups
}
```

## 9. New R Package Dependencies

| Package | Purpose | CRAN |
|---------|---------|------|
| `ncvreg` | MCP-Cox, SCAD-Cox | Yes |
| `obliqueRSF` | Oblique Random Survival Forest | Yes |
| `party` | Conditional Inference Forest | Yes |
| `mboost` | Model-based Boosting | Yes |
| `stabs` | Stability Selection | Yes |
| `Boruta` | Boruta feature selection | Yes |
| `mRMRe` | MRMR feature selection | Yes |
| `pec` | Integrated Brier Score | Yes |
| `survC1` | Uno's C-statistic | Yes |
| `survIDINRI` | D-index | Yes |
| `R.utils` | `withTimeout` | Yes |

## 10. Files to Modify

| File | Changes |
|------|---------|
| `DESCRIPTION` | Add new package dependencies |
| `R/ML.Dev.Prog.Sig.R` | Add `safe_run_algo()`, new algorithm blocks, new first-stage selectors, ensemble logic, new parameters |
| `R/cal_RS_ml_res.R` | Add predict blocks for 8 new algorithms, ensemble risk score handling |
| `R/cal_AUC_ml_res.R` | Add AUC calculation for new algorithms (follows existing pattern) |
| `R/cal_unicox_ml_res.R` | Add Cox regression for new algorithms (follows existing pattern) |
| New: `R/extra_metrics.R` | IBS, Uno's C, D-index calculation + visualization |
| New: `R/ensemble_viz.R` | Ensemble KM plot and comparison visualization |
| `NAMESPACE` | Export new functions |

## 11. Success Criteria

1. All 18 base algorithms and 144 double models are callable via `mode = "all"`
2. Single algorithm failure does not crash the function — logs warning and continues
3. Algorithm timeout (default 30 min) gracefully skips slow models
4. All new algorithms' risk scores are correctly computed in `cal_RS_ml_res()`
5. Existing visualization functions work unchanged with new algorithm results
6. `ensemble_top_n > 0` produces valid ensemble risk scores
7. `extra_metrics = TRUE` computes IBS, Uno's C, D-index without crashing on failure
8. All new R package dependencies are declared in DESCRIPTION

## 12. Scope Exclusions

- Deep learning algorithms (DeepSurv, DeepHit) — no Python dependency
- SuperLearner package integration — replaced by simpler Top-N weighted voting
- Binary classification module expansion — out of scope
- Core feature selection module expansion — out of scope (feature selection for prognosis only)
- Code refactoring of existing algorithm blocks — minimal changes to existing code
